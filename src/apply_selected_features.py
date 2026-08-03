#!/usr/bin/env python3
"""Project the train/test splits onto the selected feature list."""
import io
import os
import sys
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

AWS_REGION = cfg.get("aws.region")
BUCKET     = cfg.get("storage.bucket")
LABEL_COL  = cfg.get("data.label_col")

# These five have no config entry of their own - they are derived from the
# engineered prefix plus the filenames in data.files. The env vars are kept
# as escape hatches for one-off reruns against non-standard keys.
TRAIN_IN   = os.getenv("TRAIN_IN",  cfg.s3_key("engineered", "train"))
TEST_IN    = os.getenv("TEST_IN",   cfg.s3_key("engineered", "test"))
SEL_KEY    = os.getenv("SELECTED_FEATURES_KEY", cfg.s3_key("engineered", "selected_features"))

TRAIN_OUT  = os.getenv("TRAIN_OUT", cfg.s3_key("engineered", "train_selected"))
TEST_OUT   = os.getenv("TEST_OUT",  cfg.s3_key("engineered", "test_selected"))

s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_read_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def s3_write_csv(df, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))
    print(f"📤 Wrote s3://{BUCKET}/{key}  shape={df.shape}")

def load_selected_features():
    df = s3_read_csv(SEL_KEY)
    for col in ("selected_features","feature","features"):
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
    if df.shape[1] == 1:
        return df.iloc[:,0].dropna().astype(str).tolist()
    raise SystemExit("❌ Could not parse selected feature list")

sel = load_selected_features()
print(f"📌 Selected {len(sel)} features (first5={sel[:5]})")

for name, in_key, out_key in [
    ("TRAIN", TRAIN_IN, TRAIN_OUT),
    ("TEST",  TEST_IN,  TEST_OUT),
]:
    df = s3_read_csv(in_key)
    missing = [c for c in sel if c not in df.columns]
    if missing:
        print(f"ℹ️ {name}: adding {len(missing)} missing columns as 0.0 (e.g., {missing[:5]})")
        for c in missing: df[c] = 0.0
    keep = sel + ([LABEL_COL] if LABEL_COL in df.columns else [])
    df = df[keep]
    s3_write_csv(df, out_key)

print("✅ Applied selected features to both splits.")
