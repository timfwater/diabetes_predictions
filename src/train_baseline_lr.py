#!/usr/bin/env python3
"""
Logistic-regression baseline. Runs locally in seconds; no SageMaker.

Fits on the TRAIN split (same selected features the tuned models saw), scores
the already-scored TEST file, and adds an `lr_prob` column next to xgb_prob and
nn_prob so the evaluator compares all models on identical rows.

No class weighting on purpose: trained on the natural class balance, logistic
regression's outputs are probabilities out of the box. That makes it both the
bar the tuned models must clear and a calibration reference.

    python src/train_baseline_lr.py --no-write   # report only, touch nothing
    python src/train_baseline_lr.py              # also write lr_prob to S3
"""
import argparse
import io
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

LABEL = cfg.get("data.label_col")
BUCKET = cfg.get("storage.bucket")
SCORED_KEY = f"{cfg.prefix('scored')}/{Path(cfg.get('data.files.test_selected')).stem}_with_predictions.csv"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_s3_csv(df: pd.DataFrame, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    boto3.client("s3", region_name=cfg.get("aws.region")).put_object(
        Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv; charset=utf-8",
    )
    print(f"📤 Wrote lr_prob into s3://{BUCKET}/{key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=cfg.s3_uri("engineered", "train_selected"))
    ap.add_argument("--scored", default=f"s3://{BUCKET}/{SCORED_KEY}")
    ap.add_argument("--no-write", action="store_true", help="report metrics only")
    args = ap.parse_args()

    train = read_csv(args.train)
    scored = read_csv(args.scored)
    features = [c for c in train.columns if c != LABEL]

    missing = [c for c in features if c not in scored.columns]
    if missing:
        raise SystemExit(f"❌ Scored file lacks training features: {missing[:5]}")
    print(f"📦 Train {len(train):,} rows | test {len(scored):,} rows | {len(features)} features")

    y_tr = train[LABEL].astype(int).values
    y_te = scored[LABEL].astype(int).values

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(train[features].astype(float), y_tr)
    lr_prob = model.predict_proba(scored[features].astype(float))[:, 1]

    # --- Side-by-side on the identical test rows
    rows = {"lr (baseline)": lr_prob}
    for col in ("xgb_prob", "nn_prob"):
        if col in scored.columns:
            rows[col.replace("_prob", "")] = scored[col].values
    if {"xgb", "nn"} <= rows.keys():
        rows["ensemble"] = (rows["xgb"] + rows["nn"]) / 2

    print(f"\nTest set, prevalence={y_te.mean():.4f} "
          f"(always-predict-average Brier ≈ {brier_score_loss(y_te, np.full(len(y_te), y_tr.mean())):.4f})")
    print(f"{'model':<15}{'roc_auc':>9}{'pr_auc':>9}{'brier':>9}{'mean_p':>9}")
    for name, p in rows.items():
        print(f"{name:<15}{roc_auc_score(y_te, p):>9.4f}{average_precision_score(y_te, p):>9.4f}"
              f"{brier_score_loss(y_te, p):>9.4f}{p.mean():>9.4f}")

    # --- What drives the baseline (coefficients are per 1 SD of each feature)
    coefs = pd.Series(model[-1].coef_[0], index=features)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    print("\nTop 10 drivers (log-odds per 1 SD; + raises risk):")
    for feat, c in top.items():
        print(f"  {c:+.3f}  {feat}")

    if args.no_write:
        print("\n--no-write: nothing written.")
        return 0

    scored["lr_prob"] = lr_prob
    if args.scored.startswith("s3://"):
        write_s3_csv(scored, args.scored[5:].split("/", 1)[1])
    else:
        scored.to_csv(args.scored, index=False)
        print(f"💾 Wrote lr_prob into {args.scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
