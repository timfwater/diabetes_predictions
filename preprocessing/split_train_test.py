#!/usr/bin/env python3
# preprocessing/split_train_test.py
import os, io, boto3, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET     = os.getenv("BUCKET", "diabetes-directory")
PREFIX     = os.getenv("PREFIX", "02_engineered")
LABEL_COL  = os.getenv("LABEL_COL", "readmitted")
FULL_KEY   = os.getenv("FULL_KEY",  f"{PREFIX}/prepared_diabetes_full.csv")
TRAIN_KEY  = os.getenv("TRAIN_KEY", f"{PREFIX}/prepared_diabetes_train.csv")
TEST_KEY   = os.getenv("TEST_KEY",  f"{PREFIX}/prepared_diabetes_test.csv")
TEST_SIZE  = float(os.getenv("TEST_SIZE", "0.20"))
SPLIT_SEED = int(os.getenv("SPLIT_SEED", "42"))

s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_read_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def s3_write_csv(df, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))
    print(f"📤 Wrote s3://{BUCKET}/{key}  shape={df.shape}")

def normalize_label(s: pd.Series) -> pd.Series:
    mapping = {"NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
               "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
               "<30":1,">30":1}
    if s.dtype == object: s = s.map(mapping)
    s = pd.to_numeric(s, errors="coerce")
    return s.astype("int8")

print(f"📥 Loading full: s3://{BUCKET}/{FULL_KEY}")
df = s3_read_csv(BUCKET, FULL_KEY)
if LABEL_COL not in df.columns:
    raise SystemExit(f"❌ Missing label '{LABEL_COL}' in full dataset")

y = normalize_label(df[LABEL_COL])
mask = y.notna()
df = df.loc[mask].reset_index(drop=True)
y = y.loc[mask]

sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
tr_idx, te_idx = next(sss.split(df, y))

df_tr = df.iloc[tr_idx].reset_index(drop=True)
df_te = df.iloc[te_idx].reset_index(drop=True)

s3_write_csv(df_tr, TRAIN_KEY)
s3_write_csv(df_te, TEST_KEY)
print("✅ Stratified split complete.")
