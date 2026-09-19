#!/usr/bin/env python3
"""
Score every XGBoost CV-fold model locally and build their 5-fold average.

Why an average instead of picking one fold: each fold's best model was chosen
on its own validation slice, so the fold scores are not comparable, and picking
the winner on the test set would use the final exam to make a decision. The
decision rule here is fixed in advance - "use the mean of all fold models" - so
the test set is only ever used to measure, never to choose. Per-fold test
scores are printed for information only: they show how much the fold choice
could have mattered, and must not be used to swap in a different fold.

Correctness check: the last fold is the one deployed in production. Its local
predictions are compared row-by-row against the endpoint's saved xgb_prob.

No endpoints, no training. Downloads the fold model files (a few MB) from S3.

    python src/xgb_fold_ensemble.py
    python src/xgb_fold_ensemble.py --jobs sagemaker-xgboost-260803-1517,...
"""
import argparse
import io
import pickle
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

REGION = cfg.get("aws.region")
BUCKET = cfg.get("storage.bucket")
PREFIX = cfg.prefix("engineered")
LABEL = cfg.get("data.label_col")
FEATURES_BY_TUNING_DIR = f"{PREFIX}/feature_lists/by_tuning_job"
SCORED_KEY = f"{cfg.prefix('scored')}/{Path(cfg.get('data.files.test_selected')).stem}_with_predictions.csv"

# The August 3 run (kfolds=5). Override with --jobs for a later run.
DEFAULT_JOBS = [
    "sagemaker-xgboost-260803-1517", "sagemaker-xgboost-260803-1524",
    "sagemaker-xgboost-260803-1532", "sagemaker-xgboost-260803-1539",
    "sagemaker-xgboost-260803-1546",
]


def load_booster(tar_bytes: bytes) -> xgb.Booster:
    """SageMaker's built-in XGBoost writes the booster as 'xgboost-model'
    inside model.tar.gz - native format in recent images, pickle in old ones."""
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(td)
        path = Path(td) / "xgboost-model"
        if not path.exists():
            found = [p.name for p in Path(td).rglob("*")]
            raise SystemExit(f"❌ No 'xgboost-model' in artifact. Contents: {found}")
        booster = xgb.Booster()
        try:
            booster.load_model(str(path))
        except xgb.core.XGBoostError:
            with open(path, "rb") as fh:
                booster = pickle.load(fh)
        return booster


def predict(booster: xgb.Booster, X: np.ndarray, use_best_iteration: bool) -> np.ndarray:
    dm = xgb.DMatrix(X)
    best = booster.attributes().get("best_iteration")
    if use_best_iteration and best is not None:
        return booster.predict(dm, iteration_range=(0, int(best) + 1))
    return booster.predict(dm)


def resolve_fold(sm, s3, job: str) -> dict:
    d = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job)
    best = d["BestTrainingJob"]
    tj = best["TrainingJobName"]
    uri = sm.describe_training_job(TrainingJobName=tj)["ModelArtifacts"]["S3ModelArtifacts"]
    bkt, key = uri[5:].split("/", 1)
    feats_txt = s3.get_object(Bucket=BUCKET, Key=f"{FEATURES_BY_TUNING_DIR}/{job}.txt")["Body"].read().decode()
    return {
        "job": job,
        "trial": tj,
        "val_aucpr": best.get("FinalHyperParameterTuningJobObjectiveMetric", {}).get("Value", np.nan),
        "tar": s3.get_object(Bucket=bkt, Key=key)["Body"].read(),
        "features": [ln.strip() for ln in feats_txt.splitlines() if ln.strip()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(DEFAULT_JOBS),
                    help="comma-separated XGB tuning jobs, fold order; last = deployed")
    ap.add_argument("--scored", default=f"s3://{BUCKET}/{SCORED_KEY}")
    args = ap.parse_args()
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    scored = pd.read_csv(args.scored)
    y = scored[LABEL].astype(int).values
    print(f"📥 Test set: {len(scored):,} rows, prevalence={y.mean():.4f}")

    folds = []
    for k, job in enumerate(jobs, start=1):
        f = resolve_fold(sm, s3, job)
        missing = [c for c in f["features"] if c not in scored.columns]
        if missing:
            raise SystemExit(f"❌ Fold {k}: test file lacks features {missing[:5]}")
        f["booster"] = load_booster(f["tar"])
        f["X"] = scored[f["features"]].astype(float).values
        folds.append(f)
        print(f"📦 Fold {k}: {f['trial']}  ({len(f['features'])} features)")

    # --- Correctness check: reproduce production's predictions for the deployed fold.
    # Also settles whether the endpoint scores with all trees or stops at the
    # early-stopping best iteration, so every fold is scored the same way.
    deployed = folds[-1]
    if "xgb_prob" not in scored.columns:
        raise SystemExit("❌ Scored file has no xgb_prob column to check against.")
    endpoint = scored["xgb_prob"].values
    diffs = {mode: np.max(np.abs(predict(deployed["booster"], deployed["X"], mode) - endpoint))
             for mode in (False, True)}
    use_best = diffs[True] < diffs[False]
    match = min(diffs.values())
    print(f"\n🔎 Deployed fold, local vs endpoint: max |Δp| = {match:.2e} "
          f"(scoring with {'best iteration' if use_best else 'all trees'})")
    if match > 1e-4:
        raise SystemExit("❌ Local scoring does not reproduce the endpoint. Stopping before "
                         "reporting anything built on it.")
    print("   ✅ Local scoring reproduces production.")

    # --- Score every fold, then the pre-committed 5-fold average
    preds = np.column_stack([predict(f["booster"], f["X"], use_best) for f in folds])
    rows = []
    for k, f in enumerate(folds, start=1):
        p = preds[:, k - 1]
        rows.append({"model": f"fold {k}" + (" (deployed)" if k == len(folds) else ""),
                     "val_aucpr": f["val_aucpr"],
                     "test_roc_auc": roc_auc_score(y, p), "test_pr_auc": average_precision_score(y, p),
                     "test_brier": brier_score_loss(y, p)})
    avg = preds.mean(axis=1)
    rows.append({"model": f"{len(folds)}-fold average", "val_aucpr": np.nan,
                 "test_roc_auc": roc_auc_score(y, avg), "test_pr_auc": average_precision_score(y, avg),
                 "test_brier": brier_score_loss(y, avg)})
    table = pd.DataFrame(rows)

    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"))
    fold_auc = table["test_roc_auc"].iloc[:-1]
    print(f"\nSpread of single-fold test ROC-AUC: {fold_auc.min():.4f}-{fold_auc.max():.4f} "
          f"(range {fold_auc.max() - fold_auc.min():.4f})")
    corr = np.corrcoef(preds, rowvar=False)[np.triu_indices(len(folds), 1)]
    print(f"Fold-model prediction correlation: min {corr.min():.3f}, mean {corr.mean():.3f}")
    print("\nPer-fold test rows are for information only. The average was chosen in advance;")
    print("picking a single fold because it scored best here would reuse the test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
