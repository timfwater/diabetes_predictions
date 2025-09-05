# preprocessing/run_tuning_nn.py
import os
import io
import json
from pathlib import Path

import boto3
import pandas as pd
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)
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
label_col = os.getenv("LABEL_COL", "readmitted")

features_file = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")
input_file = os.environ.get("FILTERED_INPUT_FILE", "5_perc.csv")

nn_output_prefix = os.environ.get("NN_OUTPUT_PREFIX", "nn_output")
fold_prefix = f"{prefix}/kfolds_nn"
nn_output = f"s3://{bucket}/{prefix}/{nn_output_prefix}"

# Optional instance override (e.g., NN_INSTANCE_TYPE=ml.m5.large)
nn_instance_type = os.getenv("NN_INSTANCE_TYPE", "ml.c5.2xlarge")

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
    raise ValueError(
        f"❌ Selected features missing from input dataset: {missing[:10]}{'...' if len(missing)>10 else ''}"
    )
if label_col not in df.columns:
    raise ValueError(f"❌ Label column '{label_col}' not found in dataset.")

df = df[feature_cols + [label_col]].dropna()
print(f"📊 Filtered to {df.shape[0]} rows and {len(feature_cols)} features.")

# ========= Write folds for NN (header kept, label stays as column) =========
def upload_csv_with_header(df_part: pd.DataFrame, key: str) -> str:
    buf = io.StringIO()
    df_part.to_csv(buf, index=False)  # header INCLUDED
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"

folds = []
kf = KFold(n_splits=2, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(kf.split(df), start=1):
    train_key = f"{fold_prefix}/train_{k}.csv"
    val_key = f"{fold_prefix}/val_{k}.csv"
    train_s3 = upload_csv_with_header(df.iloc[tr], train_key)
    val_s3 = upload_csv_with_header(df.iloc[va], val_key)
    folds.append((train_s3, val_s3))
print(f"🧩 Prepared {len(folds)} NN folds under s3://{bucket}/{fold_prefix}")

# ========= TF Estimator (script mode) =========
# Path-agnostic: point source_dir to the directory this file is in (i.e., .../preprocessing)
source_dir_path = str(Path(__file__).parent.resolve())

tf_est = TensorFlow(
    entry_point="train_nn.py",
    source_dir=source_dir_path,
    role=role,
    instance_count=1,
    instance_type=nn_instance_type,
    framework_version="2.13",
    py_version="py310",
    output_path=nn_output,
    code_location=f"s3://{bucket}/{prefix}/code",   # <-- ensures TF container can download your code
    sagemaker_session=sess,
    hyperparameters={
        "label-col": label_col,
        "epochs": 30,
        "batch-size": 256,
        "lr": 1e-3,
        "hidden-dim": 128,
        "hidden-layers": 2,
        "dropout": 0.3,
    },
)

# HPO ranges
hp_ranges = {
    "lr": ContinuousParameter(1e-4, 5e-3),
    "dropout": ContinuousParameter(0.1, 0.6),
    "hidden-dim": IntegerParameter(64, 512),
    "hidden-layers": IntegerParameter(1, 4),
    "batch-size": CategoricalParameter([128, 256, 512]),
}

metric_defs = [{"Name": "validation:auc", "Regex": r"validation[-:]auc[:=]([0-9\.]+)"}]

tuner = HyperparameterTuner(
    estimator=tf_est,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    max_jobs=4,
    max_parallel_jobs=2,
    hyperparameter_ranges=hp_ranges,
    metric_definitions=metric_defs,
    # early_stopping_type="Auto",
)

# ========= Launch tuning per fold =========
started_jobs = []
for i, (train_s3, val_s3) in enumerate(folds, start=1):
    print(f"🚀 Starting NN tuning job for fold {i}:")
    tuner.fit(
        inputs={
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(val_s3, content_type="text/csv"),
        },
        include_cls_metadata=False,
    )
    job_name = tuner.latest_tuning_job.name
    started_jobs.append(job_name)
    print(f"✅ Started NN tuning job (fold {i}): {job_name}")

print("🧾 NN tuning jobs this run:", json.dumps(started_jobs, indent=2))
print("✅ Launched NN HPO.")
