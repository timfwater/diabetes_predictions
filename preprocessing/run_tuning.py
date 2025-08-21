# preprocessing/run_tuning.py
import os
import io
import json
import boto3
import pandas as pd
from sagemaker import Session, image_uris, estimator as sm_estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import KFold

# ========= Config & session =========
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
sess = Session(boto_session=boto3.Session(region_name=AWS_REGION))

role = os.environ.get("SAGEMAKER_TRAINING_ROLE")
print(f"✅ Detected SAGEMAKER_TRAINING_ROLE: {role}")
if not role:
    raise ValueError("❌ SAGEMAKER_TRAINING_ROLE env var is missing in container.")

bucket = os.environ.get("BUCKET", "diabetes-directory")
prefix = os.environ.get("PREFIX", "02_engineered")

# Inputs/outputs
features_file = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")
input_file = os.environ.get("FILTERED_INPUT_FILE", "5_perc.csv")
xgb_output_prefix = os.environ.get("XGB_OUTPUT_PREFIX", "xgb_output")
fold_prefix = f"{prefix}/kfolds"  # where fold CSVs will be written
xgb_output = f"s3://{bucket}/{prefix}/{xgb_output_prefix}"
label_col = os.getenv("LABEL_COL", "readmitted")

# Where we’ll write the features used for this run
FEATURES_USED_LATEST_KEY = f"{prefix}/features_used_latest.txt"
FEATURES_BY_TUNING_DIR = f"{prefix}/feature_lists/by_tuning_job"

s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_put_text(bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    print(f"📤 Uploaded s3://{bucket}/{key}")

# ========= Load selected features =========
features_path = f"s3://{bucket}/{prefix}/{features_file}"
print(f"📥 Loading selected features from {features_path}")
feat_df = pd.read_csv(features_path)
feature_cols = None
for col in ["selected_features", "feature", "features"]:
    if col in feat_df.columns:
        feature_cols = feat_df[col].dropna().astype(str).tolist()
        break
if feature_cols is None and feat_df.shape[1] == 1:
    feature_cols = feat_df.iloc[:, 0].dropna().astype(str).tolist()
if not feature_cols:
    raise ValueError("❌ No selected features found in features file.")
print(f"📌 Selected {len(feature_cols)} features (first 5): {feature_cols[:5]}")

# ========= Load and filter data =========
df = pd.read_csv(f"s3://{bucket}/{prefix}/{input_file}")
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Selected features missing from input dataset: {missing[:10]}{'...' if len(missing)>10 else ''}")
if label_col not in df.columns:
    raise ValueError(f"❌ Label column '{label_col}' not found in dataset.")

df = df[feature_cols + [label_col]].dropna()
print(f"📊 Filtered to {df.shape[0]} rows and {len(feature_cols)} features.")

# --- Persist the exact feature list used for this run (versioned + pointer) ---
ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
features_versioned_key = f"{prefix}/features_used_{ts}.txt"
features_text = "\n".join(feature_cols)
s3_put_text(bucket, features_versioned_key, features_text)
s3_put_text(bucket, FEATURES_USED_LATEST_KEY, features_text)
print(f"📎 Saved features list: s3://{bucket}/{features_versioned_key}")
print(f"📌 Updated pointer:     s3://{bucket}/{FEATURES_USED_LATEST_KEY}")

# ========= Label normalization & writer (label first, headerless CSV) =========
def _normalize_label(series: pd.Series) -> pd.Series:
    """Map common encodings to {0,1} exactly for XGBoost binary:logistic."""
    s = series.copy()
    if s.dtype == object:
        mapping = {
            "NO": 0, "No": 0, "no": 0, "0": 0, "FALSE": 0, "False": 0, "false": 0,
            "YES": 1, "Yes": 1, "yes": 1, "1": 1, "TRUE": 1, "True": 1, "true": 1,
            "<30": 1, ">30": 1
        }
        s = s.map(mapping)
    s = pd.to_numeric(s, errors="coerce")
    if not set(pd.unique(s.dropna())).issubset({0, 1}):
        raise ValueError(f"Label contains values outside {{0,1}}: {pd.unique(s)}")
    return s.astype("int8")

def upload_csv_to_s3_for_xgb(df_part: pd.DataFrame, key: str) -> str:
    """
    Write CSV with label FIRST and NO HEADER (required by SageMaker built-in XGBoost).
    """
    df_local = df_part.copy()
    y = _normalize_label(df_local[label_col])
    X = df_local.drop(columns=[label_col])
    out = pd.concat([y, X], axis=1)  # label first
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"

# ========= Build K folds and write train/val CSVs =========
folds = []
kf = KFold(n_splits=2, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(kf.split(df), start=1):
    train_key = f"{fold_prefix}/train_{k}.csv"
    val_key   = f"{fold_prefix}/val_{k}.csv"
    train_s3 = upload_csv_to_s3_for_xgb(df.iloc[tr], train_key)
    val_s3   = upload_csv_to_s3_for_xgb(df.iloc[va], val_key)
    folds.append((train_s3, val_s3))
print(f"🧩 Prepared {len(folds)} folds under s3://{bucket}/{fold_prefix}")

# ========= Estimator & tuner =========
xgb_image = image_uris.retrieve("xgboost", AWS_REGION, version="1.7-1")
xgb_est = sm_estimator.Estimator(
    image_uri=xgb_image,
    role=role,
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    output_path=xgb_output,
    sagemaker_session=sess,
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": "200",
    },
)

hp_ranges = {
    "eta": ContinuousParameter(0.0, 1.0),
    "alpha": ContinuousParameter(0.0, 2.0),
    "min_child_weight": ContinuousParameter(1.0, 10.0),
    "max_depth": IntegerParameter(1, 10),
}

# NOTE: Built-in XGBoost sometimes logs "validation-auc=" and sometimes "validation:auc".
metric_defs = [{"Name": "validation:auc", "Regex": r"validation[-:]auc[:=]([0-9\.]+)"}]

tuner = HyperparameterTuner(
    estimator=xgb_est,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    max_jobs=4,
    max_parallel_jobs=2,
    hyperparameter_ranges=hp_ranges,
    metric_definitions=metric_defs,
)

# ========= Launch tuning per fold =========
import traceback
import botocore

latest_job_name = None
started_jobs = []

for i, (train_s3, val_s3) in enumerate(folds, start=1):
    print(f"🚀 Starting tuning job for fold {i}:")
    try:
        tuner.fit(
            inputs={
                "train": TrainingInput(train_s3, content_type="text/csv"),
                "validation": TrainingInput(val_s3, content_type="text/csv"),
            },
            include_cls_metadata=False,
        )
        job_name = tuner.latest_tuning_job.name
        latest_job_name = job_name
        started_jobs.append(job_name)
        print(f"✅ Started tuning job (fold {i}): {job_name}")

        # ---- Save a job-scoped copy of the EXACT features used for THIS run ----
        job_scoped_key = f"{FEATURES_BY_TUNING_DIR}/{job_name}.txt"
        s3_put_text(bucket, job_scoped_key, features_text)
        print(f"🧬 Wrote job-scoped features: s3://{bucket}/{job_scoped_key}")

    except botocore.exceptions.ClientError as e:
        print("!!! FAILED to create tuning job (ClientError):", e)
        traceback.print_exc()
        raise
    except Exception as e:
        print("!!! FAILED to create tuning job:", e)
        traceback.print_exc()
        raise

# ========= Persist last job name (both paths) =========
if latest_job_name:
    # Primary location expected by many scripts
    with open("/app/preprocessing/latest_tuning_job.txt", "w") as f:
        f.write(latest_job_name)
    print("💾 Wrote /app/preprocessing/latest_tuning_job.txt ->", latest_job_name)

    # Secondary location for deploy scripts that read /app/
    try:
        with open("/app/latest_tuning_job.txt", "w") as f:
            f.write(latest_job_name)
        print("💾 Wrote /app/latest_tuning_job.txt ->", latest_job_name)
    except Exception as e:
        # Non-fatal if this path is not writable in some environments
        print("⚠️ Could not write /app/latest_tuning_job.txt:", repr(e))

    print("🧾 Tuning jobs this run:", json.dumps(started_jobs, indent=2))
    print("✅ Saved latest tuning job:", latest_job_name)
else:
    print("⚠️ No tuning jobs were started.")
