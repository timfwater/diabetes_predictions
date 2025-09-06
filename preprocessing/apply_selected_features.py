#!/usr/bin/env python3
# preprocessing/apply_selected_features.py
import os, io, boto3, pandas as pd

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET     = os.getenv("BUCKET", "diabetes-directory")
PREFIX     = os.getenv("PREFIX", "02_engineered")
LABEL_COL  = os.getenv("LABEL_COL", "readmitted")

TRAIN_IN   = os.getenv("TRAIN_IN", f"{PREFIX}/prepared_diabetes_train.csv")
TEST_IN    = os.getenv("TEST_IN",  f"{PREFIX}/prepared_diabetes_test.csv")
SEL_KEY    = os.getenv("SELECTED_FEATURES_KEY", f"{PREFIX}/selected_features.csv")

TRAIN_OUT  = os.getenv("TRAIN_OUT", f"{PREFIX}/prepared_diabetes_train_selected.csv")
TEST_OUT   = os.getenv("TEST_OUT",  f"{PREFIX}/prepared_diabetes_test_selected.csv")

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
