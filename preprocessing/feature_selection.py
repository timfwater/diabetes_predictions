# preprocessing/feature_selection.py
import os
import io
import boto3
import pandas as pd
import xgboost as xgb

# -------- Config (aligned with run_tuning) --------
BUCKET = os.environ.get("BUCKET", "diabetes-directory")
PREFIX = os.environ.get("PREFIX", "02_engineered")
# Use the SAME engineered input as training/tuning
INPUT_FILE = os.environ.get("FILTERED_INPUT_FILE", "5_perc.csv")
LABEL_COL = os.environ.get("LABEL_COL", "readmitted")

# Where to save list for run_tuning to consume
OUTPUT_FILENAME = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")
INPUT_KEY = f"{PREFIX}/{INPUT_FILE}"
OUTPUT_KEY = f"{PREFIX}/{OUTPUT_FILENAME}"

TOP_N = int(os.environ.get("TOP_N", "50"))

s3 = boto3.client("s3")

# -------- Load engineered data (no ad-hoc encoding here) --------
obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in {INPUT_KEY}")

# Keep numeric features only (assume upstream engineering handled encoding)
X = df.drop(columns=[LABEL_COL])
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
if not num_cols:
    raise ValueError("No numeric features found. Ensure upstream step encoded categoricals.")
X = X[num_cols].copy().astype("float32")
y_raw = df[LABEL_COL]

# Normalize label to {0,1}
y = pd.to_numeric(y_raw.replace(
    {"NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
     "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
     "<30":1, ">30":1}
), errors="coerce").astype("float32")
if not set(pd.unique(y.dropna())).issubset({0.0, 1.0}):
    raise ValueError(f"Label contains values outside {{0,1}}: {pd.unique(y)}")

# Drop rows with missing label; fill feature NaNs
mask = y.notna()
X = X[mask].fillna(0.0)
y = y[mask].astype("int8")

# -------- Train quick XGB for importances --------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=200
)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(TOP_N).index.tolist()

# -------- Save as CSV with header 'selected_features' --------
csv = "selected_features\n" + "\n".join(top_features) + "\n"
s3.put_object(Bucket=BUCKET, Key=OUTPUT_KEY, Body=csv.encode("utf-8"))

print(f"✅ Top {TOP_N} features saved to s3://{BUCKET}/{OUTPUT_KEY}")
print(f"ℹ️ First 5: {top_features[:5]}")
print(f"ℹ️ Using engineered file: s3://{BUCKET}/{INPUT_KEY}")
