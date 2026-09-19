#!/usr/bin/env python3
"""
Paired bootstrap: is each model's ranking edge over the logistic baseline real,
or within the noise of which patients happened to land in the test set?

Reads saved predictions; writes nothing. If the scored file has no lr_prob
column, it fits the logistic baseline itself from the training file (~1 second).

Resampling is by PATIENT, not by row: a patient's encounters move together,
matching how the train/test split was built.

    python src/bootstrap_compare.py            # 1000 resamples
    python src/bootstrap_compare.py --n 2000
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN = "s3://diabetes-directory/02_engineered/prepared_diabetes_train_selected.csv"
SCORED = "s3://diabetes-directory/03_scored/prepared_diabetes_test_selected_with_predictions.csv"
TEST = "s3://diabetes-directory/02_engineered/prepared_diabetes_test.csv"
LABEL = "readmitted"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=SCORED)
    ap.add_argument("--test", default=TEST)
    ap.add_argument("--train", default=TRAIN, help="used only if lr_prob is missing")
    ap.add_argument("--n", type=int, default=1000, help="number of resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s = pd.read_csv(args.scored)
    t = pd.read_csv(args.test)

    # Patient IDs live only in the test file; attach them by row position,
    # but only after proving the two files are in the same order.
    shared = [c for c in s.columns if c in t.columns]
    if len(s) != len(t) or any(not np.array_equal(s[c].values, t[c].values) for c in shared):
        raise SystemExit("❌ Row alignment failed between scored and test files.")
    if "lr_prob" not in s.columns:
        # Fit the logistic baseline here (about a second) instead of depending
        # on another script having written lr_prob to S3. Same recipe as
        # train_baseline_lr.py: standardized features, no class weighting.
        tr = pd.read_csv(args.train)
        feats = [c for c in tr.columns if c != LABEL]
        missing = [c for c in feats if c not in s.columns]
        if missing:
            raise SystemExit(f"❌ Scored file lacks training features: {missing[:5]}")
        lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        lr_model.fit(tr[feats].astype(float), tr[LABEL].astype(int))
        s["lr_prob"] = lr_model.predict_proba(s[feats].astype(float))[:, 1]
        print(f"ℹ️  No lr_prob column found; fitted logistic on {len(tr):,} training rows.")

    s["ensemble_prob"] = (s["xgb_prob"] + s["nn_prob"]) / 2
    models = {"xgb": "xgb_prob", "nn": "nn_prob", "ensemble": "ensemble_prob"}

    y = s[LABEL].astype(int).values
    lr = s["lr_prob"].values
    patient_code, _ = pd.factorize(t["patient_nbr"])
    n_pat = patient_code.max() + 1
    print(f"{len(s):,} rows | {n_pat:,} patients | prevalence={y.mean():.4f} | {args.n} resamples")

    # Each resample draws patients with replacement. A patient drawn k times
    # contributes all their rows with weight k (equivalent to duplicating them).
    rng = np.random.default_rng(args.seed)
    gaps = {m: np.empty(args.n) for m in models}
    for b in range(args.n):
        counts = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)
        w = counts[patient_code]
        keep = w > 0
        yb, wb = y[keep], w[keep]
        auc_lr = roc_auc_score(yb, lr[keep], sample_weight=wb)
        for m, col in models.items():
            gaps[m][b] = roc_auc_score(yb, s[col].values[keep], sample_weight=wb) - auc_lr

    auc_lr_obs = roc_auc_score(y, lr)
    print(f"\nlogistic ROC-AUC = {auc_lr_obs:.4f}\n")
    print(f"{'model':<10}{'ROC-AUC':>8}{'gap vs lr':>11}{'95% interval':>22}{'beats lr':>10}")
    for m, col in models.items():
        auc = roc_auc_score(y, s[col].values)
        lo, hi = np.percentile(gaps[m], [2.5, 97.5])
        print(f"{m:<10}{auc:>8.4f}{auc - auc_lr_obs:>+11.4f}"
              f"      [{lo:+.4f}, {hi:+.4f}]{(gaps[m] > 0).mean():>9.1%}")

    print("\nHow to read 'beats lr': the share of resamples in which the model out-ranked")
    print("logistic. Near 100% and an interval above zero = a real edge. An interval that")
    print("crosses zero = cannot distinguish the model from logistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
