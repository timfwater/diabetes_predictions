#!/usr/bin/env python3
"""
Neural-network CV-fold models: load them from S3, score locally, average.

Mirrors xgb_fold_ensemble.py. Used two ways:
  * as a library by predict_from_both.py when predict.nn_source=local_folds
    (the pipeline's NN predictions become the fold average; no endpoint);
  * as a standalone report (below) comparing every fold and the average.

Why an average instead of picking one fold: same reasoning as XGBoost's
fold ensemble - each fold's best model was chosen on its own validation
slice, so fold scores are not comparable, and picking on the test set
would use the final exam to make a decision.

Which jobs make up a run comes from the pointer run_tuning_nn.py writes:
    s3://<bucket>/<engineered>/tuning_runs/nn_latest.json

One extra step versus XGBoost: each NN model.tar.gz contains a TensorFlow
SavedModel directory PLUS an artifacts.json (feature order + imputation
means the training script computed). Normalization is baked into the
SavedModel itself (train_nn.py's build_serving_model), so this script
feeds it raw, imputed features and nothing else.

    python src/nn_fold_ensemble.py                    # report on latest run
    python src/nn_fold_ensemble.py --jobs a,b,c,d,e   # report on named jobs
    python src/nn_fold_ensemble.py --jobs a,b,c,d,e --seed-pointer
        # one-off: record named jobs as the latest run (for runs made before
        # tuning started writing the pointer)
"""
import argparse
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

REGION = cfg.get("aws.region")
BUCKET = cfg.get("storage.bucket")
PREFIX = cfg.prefix("engineered")
LABEL = cfg.get("data.label_col")
RUN_POINTER_KEY = f"{PREFIX}/tuning_runs/nn_latest.json"
SCORED_KEY = f"{cfg.prefix('scored')}/{Path(cfg.get('data.files.test_selected')).stem}_with_predictions.csv"


# ------------------------------------------------------------------ library
def clients():
    return (boto3.client("sagemaker", region_name=REGION),
            boto3.client("s3", region_name=REGION))


def fold_jobs(s3, explicit=None) -> list:
    """Tuning-job names for one run, in fold order."""
    if explicit:
        return list(explicit)
    try:
        body = s3.get_object(Bucket=BUCKET, Key=RUN_POINTER_KEY)["Body"].read()
    except Exception:
        raise SystemExit(
            f"❌ No fold-run pointer at s3://{BUCKET}/{RUN_POINTER_KEY}.\n"
            "   It is written by run_tuning_nn.py. For a run made before that, record it once:\n"
            "   python src/nn_fold_ensemble.py --jobs <job1>,...,<jobK> --seed-pointer")
    jobs = json.loads(body)["jobs"]
    if not jobs:
        raise SystemExit(f"❌ Fold-run pointer s3://{BUCKET}/{RUN_POINTER_KEY} lists no jobs.")
    return jobs


def _find_savedmodel_dir(root):
    """Walk the extracted tarball for the directory containing saved_model.pb."""
    for p in Path(root).rglob("saved_model.pb"):
        return p.parent
    return None


def _load_fold_artifact(tar_bytes: bytes):
    """Extract one fold's model.tar.gz: SavedModel dir + artifacts.json
    (feature order, imputation means, and the baked-normalization flag).
    Returns (model, feature_names, impute_means)."""
    import tensorflow as tf  # lazy import: xgb-only runs don't need TF installed

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(td)

        sm_dir = _find_savedmodel_dir(td)
        if sm_dir is None:
            found = [str(p.relative_to(td)) for p in Path(td).rglob("*")]
            raise SystemExit(f"❌ No SavedModel (saved_model.pb) in artifact. Contents: {found}")
        model = tf.keras.models.load_model(str(sm_dir))

        candidates = list(Path(td).rglob("artifacts.json"))
        if not candidates:
            raise SystemExit("❌ No artifacts.json found alongside the SavedModel.")
        artifacts = json.loads(candidates[0].read_text())

        if not artifacts.get("normalization_baked_into_model", False):
            raise SystemExit(
                "❌ This fold's model was exported WITHOUT baked-in normalization "
                "(normalization_baked_into_model=False in artifacts.json). Scoring it "
                "with raw features would silently produce wrong probabilities - stopping "
                "rather than guessing."
            )
        feature_names = artifacts["feature_names"]
        impute_means = dict(zip(artifacts["imputation"]["columns"], artifacts["imputation"]["means"]))
        return model, feature_names, impute_means


def load_folds(sm, s3, jobs: list) -> list:
    """Download each job's best model and its serving metadata."""
    folds = []
    for k, job in enumerate(jobs, start=1):
        best = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job)["BestTrainingJob"]
        trial = best["TrainingJobName"]
        uri = sm.describe_training_job(TrainingJobName=trial)["ModelArtifacts"]["S3ModelArtifacts"]
        bkt, key = uri[5:].split("/", 1)
        model, feature_names, impute_means = _load_fold_artifact(
            s3.get_object(Bucket=bkt, Key=key)["Body"].read()
        )
        folds.append({
            "fold": k, "job": job, "trial": trial,
            "val_aucpr": best.get("FinalHyperParameterTuningJobObjectiveMetric", {}).get("Value", np.nan),
            "model": model, "features": feature_names, "impute_means": impute_means,
        })
        print(f"📦 Fold {k}: {trial}  ({len(feature_names)} features)")
    return folds


def predict_folds(folds: list, df: pd.DataFrame) -> np.ndarray:
    """One column of predictions per fold. Applies each fold's OWN feature
    order and imputation means (they can differ slightly run to run) before
    calling its baked-normalization model on raw values."""
    cols = []
    for f in folds:
        missing = [c for c in f["features"] if c not in df.columns]
        if missing:
            raise SystemExit(f"❌ Fold {f['fold']}: input lacks features {missing[:5]}")
        X = df[f["features"]].apply(pd.to_numeric, errors="coerce").copy()
        for c in f["features"]:
            X[c] = X[c].fillna(f["impute_means"].get(c, 0.0))
        preds = f["model"].predict(X.to_numpy(dtype=np.float32), verbose=0).ravel()
        cols.append(np.clip(preds, 0.0, 1.0))
    return np.column_stack(cols)


# ------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", help="comma-separated tuning jobs in fold order (default: latest-run pointer)")
    ap.add_argument("--seed-pointer", action="store_true",
                    help="write --jobs to the latest-run pointer and exit")
    ap.add_argument("--scored", default=f"s3://{BUCKET}/{SCORED_KEY}")
    args = ap.parse_args()
    explicit = [j.strip() for j in args.jobs.split(",") if j.strip()] if args.jobs else None

    sm, s3 = clients()

    if args.seed_pointer:
        if not explicit:
            raise SystemExit("❌ --seed-pointer needs --jobs")
        for job in explicit:
            sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job)  # must exist
        record = {"jobs": explicit, "kfolds": len(explicit),
                  "created_utc": pd.Timestamp.utcnow().isoformat(), "seeded_manually": True}
        s3.put_object(Bucket=BUCKET, Key=RUN_POINTER_KEY, Body=json.dumps(record, indent=2).encode())
        print(f"📌 Wrote s3://{BUCKET}/{RUN_POINTER_KEY} with {len(explicit)} jobs")
        return 0

    jobs = fold_jobs(s3, explicit)
    scored = pd.read_csv(args.scored)
    y = scored[LABEL].astype(int).values
    print(f"📥 Test set: {len(scored):,} rows, prevalence={y.mean():.4f}")
    folds = load_folds(sm, s3, jobs)
    preds = predict_folds(folds, scored)
    avg = preds.mean(axis=1)

    # Correctness check against whatever produced the saved nn_prob: the
    # endpoint (= last fold, current production) in older runs, or this
    # average once it's wired in.
    if "nn_prob" in scored.columns:
        saved = scored["nn_prob"].values
        d_last = np.max(np.abs(preds[:, -1] - saved))
        d_avg = np.max(np.abs(avg - saved))
        source, d = ("fold average", d_avg) if d_avg < d_last else ("deployed last fold", d_last)
        print(f"\n🔎 Saved nn_prob matches the {source}: max |Δp| = {d:.2e}")

    rows = []
    for f in folds:
        p = preds[:, f["fold"] - 1]
        rows.append({"model": f"fold {f['fold']}", "val_aucpr": f["val_aucpr"],
                     "test_roc_auc": roc_auc_score(y, p), "test_pr_auc": average_precision_score(y, p),
                     "test_brier": brier_score_loss(y, p)})
    rows.append({"model": f"{len(folds)}-fold average", "val_aucpr": np.nan,
                 "test_roc_auc": roc_auc_score(y, avg), "test_pr_auc": average_precision_score(y, avg),
                 "test_brier": brier_score_loss(y, avg)})
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"))

    fold_auc = table["test_roc_auc"].iloc[:-1]
    print(f"\nSpread of single-fold test ROC-AUC: {fold_auc.min():.4f}-{fold_auc.max():.4f} "
          f"(range {fold_auc.max() - fold_auc.min():.4f})")
    print("\nPer-fold test rows are for information only. The average was chosen in advance;")
    print("picking a single fold because it scored best here would reuse the test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
