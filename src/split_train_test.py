#!/usr/bin/env python3
"""
Split the full engineered dataset into train and test.

Strategies (config: split.strategy)
    stratified : StratifiedShuffleSplit on the label. Current behaviour.
                 WARNING - encounters from the same patient may land on both
                 sides, which inflates held-out metrics.
    grouped    : GroupShuffleSplit on split.group_col (patient_nbr), keeping
                 all of a patient's encounters together. Requires that column
                 to survive data engineering.
"""

import io
import sys
from pathlib import Path

import boto3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

# ------------------------------------------------------------------- config
REGION = cfg.get("aws.region")
BUCKET = cfg.get("storage.bucket")
LABEL_COL = cfg.get("data.label_col")
FULL_KEY = cfg.s3_key("engineered", "full")
TRAIN_KEY = cfg.s3_key("engineered", "train")
TEST_KEY = cfg.s3_key("engineered", "test")
TEST_SIZE = cfg.get("split.test_size")
SEED = cfg.get("split.seed")
STRATEGY = cfg.get("split.strategy")
GROUP_COL = cfg.get("split.group_col")

s3 = boto3.client("s3", region_name=REGION)


def s3_read_csv(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def s3_write_csv(df: pd.DataFrame, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))
    print(f"Wrote s3://{BUCKET}/{key}  shape={df.shape}")


def normalize_label(series: pd.Series) -> pd.Series:
    # NOTE: '>30' maps to 0 - a readmission after 30 days is a negative for
    # this target. The old mapping table sent '>30' to 1, which contradicted
    # data_engineering.py. It never fired (values were already int by then),
    # but it was a live landmine for anyone feeding in raw labels.
    mapping = {
        "NO": 0, "No": 0, "no": 0, "0": 0,
        "FALSE": 0, "False": 0, "false": 0, ">30": 0,
        "YES": 1, "Yes": 1, "yes": 1, "1": 1,
        "TRUE": 1, "True": 1, "true": 1, "<30": 1,
    }
    if series.dtype == object:
        series = series.map(mapping)
    series = pd.to_numeric(series, errors="coerce")
    return series.astype("float32")


def report_patient_overlap(df: pd.DataFrame, tr_idx, te_idx) -> None:
    """Quantify the leakage this split does or doesn't introduce."""
    if GROUP_COL not in df.columns:
        return
    train_ids = set(df.iloc[tr_idx][GROUP_COL])
    test_ids = set(df.iloc[te_idx][GROUP_COL])
    shared = train_ids & test_ids
    pct = 100.0 * len(shared) / max(len(test_ids), 1)
    print(f"\nPatient overlap check ({GROUP_COL}):")
    print(f"  train patients : {len(train_ids):,}")
    print(f"  test patients  : {len(test_ids):,}")
    print(f"  appearing in both: {len(shared):,}  ({pct:.1f}% of test patients)")
    if shared:
        print("  -> LEAKAGE: these patients are in both splits.")


def main() -> int:
    print(f"Loading s3://{BUCKET}/{FULL_KEY}")
    df = s3_read_csv(BUCKET, FULL_KEY)

    if LABEL_COL not in df.columns:
        raise SystemExit(f"Missing label column '{LABEL_COL}' in full dataset")

    y = normalize_label(df[LABEL_COL])
    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype("int8").reset_index(drop=True)

    print(f"Rows: {len(df):,} | positives: {int(y.sum()):,} ({100*y.mean():.2f}%)")
    print(f"Strategy: {STRATEGY}")

    if STRATEGY == "grouped":
        if GROUP_COL not in df.columns:
            raise SystemExit(
                f"split.strategy=grouped requires column '{GROUP_COL}', which is "
                f"not present.\nIt is currently dropped in data_engineering.py - "
                f"add it to the passthrough columns first."
            )
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(splitter.split(df, y, groups=df[GROUP_COL]))
    elif STRATEGY == "stratified":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(splitter.split(df, y))
    else:
        raise SystemExit(f"Unknown split.strategy '{STRATEGY}' (expected grouped|stratified)")

    report_patient_overlap(df, tr_idx, te_idx)

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    print(f"\nLabel prevalence  train={df_tr[LABEL_COL].mean():.4f}  "
          f"test={df_te[LABEL_COL].mean():.4f}")

    s3_write_csv(df_tr, TRAIN_KEY)
    s3_write_csv(df_te, TEST_KEY)
    print(f"\n{STRATEGY.capitalize()} split complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
