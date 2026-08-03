#!/usr/bin/env python3
"""Launch SageMaker HPO for the neural network, one tuning job per CV fold."""
import io
import json
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import (
    HyperparameterTuner, ContinuousParameter, IntegerParameter, CategoricalParameter,
)
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

# ========= Config & session =========
AWS_REGION = cfg.get("aws.region")
sess = Session(boto_session=boto3.Session(region_name=AWS_REGION))

role = cfg.get("infra.sagemaker_role")
print(f"✅ Detected SageMaker role: {role}")
if not role:
    raise ValueError(
        "❌ SageMaker role missing. Set infra.sagemaker_role in config.infra.yaml, "
        "or SAGEMAKER_TRAINING_ROLE in the container environment."
    )

bucket = cfg.get("storage.bucket")
prefix = cfg.prefix("engineered")
label_col = cfg.get("data.label_col")

features_file = cfg.get("data.files.selected_features")
input_file = os.environ.get("FILTERED_INPUT_FILE", cfg.get("data.files.train_selected"))

nn_output_prefix = cfg.get("tuning.nn.output_prefix")
fold_prefix = f"{prefix}/kfolds_nn"
nn_output = f"s3://{bucket}/{prefix}/{nn_output_prefix}"

nn_instance_type = cfg.get("tuning.nn.instance_type")
NN_FRAMEWORK_VERSION = cfg.get("tuning.nn.framework_version")
NN_PY_VERSION = cfg.get("tuning.nn.py_version")

KFOLDS = int(cfg.get("tuning.kfolds"))
HPO_MAX_JOBS = int(cfg.get("tuning.max_jobs"))
HPO_MAX_PARALLEL = int(cfg.get("tuning.max_parallel"))
OBJECTIVE_METRIC = cfg.get("tuning.objective_metric")

# Fold seed is deliberately fixed rather than configurable: changing it
# silently invalidates comparisons against saved runs.
FOLD_SEED = 42

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
df_full = pd.read_csv(f"s3://{bucket}/{prefix}/{input_file}")
missing = [c for c in feature_cols if c not in df_full.columns]
if missing:
    raise ValueError(f"❌ Selected features missing from input dataset: {missing[:10]}{'...' if len(missing)>10 else ''}")
if label_col not in df_full.columns:
    raise ValueError(f"❌ Label column '{label_col}' not found in dataset.")

df = df_full[feature_cols + [label_col]].dropna()
print(f"📊 Filtered to {df.shape[0]} rows and {len(feature_cols)} features.")

# ========= Label normalization (for stratification and class_weight) =========
def _normalize_label(series: pd.Series) -> pd.Series:
    mapping = {"NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
               "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
               "<30":1,">30":1}
    s = series.copy()
    if s.dtype == object: s = s.map(mapping)
    s = pd.to_numeric(s, errors="coerce")
    if not set(pd.unique(s.dropna())).issubset({0,1}):
        raise ValueError(f"Label contains values outside {{0,1}}: {pd.unique(s)}")
    return s.astype("int8")

y = _normalize_label(df[label_col])

# ========= Write folds for NN (header kept) =========
def upload_csv_with_header(df_part: pd.DataFrame, key: str) -> str:
    buf = io.StringIO()
    df_part.to_csv(buf, index=False)  # header INCLUDED
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"

folds = []
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=FOLD_SEED)
for k, (tr, va) in enumerate(skf.split(df.drop(columns=[label_col]), y), start=1):
    train_key = f"{fold_prefix}/train_{k}.csv"
    val_key   = f"{fold_prefix}/val_{k}.csv"
    train_s3 = upload_csv_with_header(df.iloc[tr], train_key)
    val_s3   = upload_csv_with_header(df.iloc[va], val_key)
    folds.append((train_s3, val_s3))
print(f"🧩 Prepared {len(folds)} NN folds under s3://{bucket}/{fold_prefix}")

# ========= TF Estimator (script mode) =========
source_dir_path = str(Path(__file__).parent.resolve())

# class weight for positive class (pass to train script)
pos = int(y.sum()); neg = int(len(y) - pos)
pos_weight = (neg / max(pos,1)) if pos > 0 else 1.0
print(f"⚖️ Class balance: pos={pos}, neg={neg}, pos_weight≈{pos_weight:.2f}")

tf_est = TensorFlow(
    entry_point="train_nn.py",
    source_dir=source_dir_path,
    role=role,
    instance_count=1,
    instance_type=nn_instance_type,
    framework_version=NN_FRAMEWORK_VERSION,
    py_version=NN_PY_VERSION,
    output_path=nn_output,
    code_location=f"s3://{bucket}/{prefix}/code",
    sagemaker_session=sess,
    hyperparameters={
        "label-col": label_col,
        # Stronger defaults; ES will stop early
        "epochs": 200,
        "batch-size": 128,
        "lr": 3e-4,
        "hidden-dim": 256,
        "hidden-layers": 3,
        "dropout": 0.2,
        "l2": 1e-4,
        "activation": "relu",
        "use-batchnorm": 1,
        "use-class-weights": 1,
        "class-weight-pos": float(pos_weight),
        # Scaling + objective wiring (pin these; don't tune)
        "standardize": 1,
        "metric-pref": "aucpr",
        "aucpr-objective": 1,
    },
)

# ======== HPO search space (no single-value categoricals) ========
spw_low  = max(1.0, pos_weight ** 0.5)
spw_high = max(pos_weight * 1.5, pos_weight + 0.1)

hp_ranges = {
    "lr": ContinuousParameter(1e-5, 3e-3),
    "dropout": ContinuousParameter(0.0, 0.6),
    "hidden-dim": IntegerParameter(64, 1024),
    "hidden-layers": IntegerParameter(1, 6),
    "batch-size": CategoricalParameter([64, 128, 256, 512]),
    "l2": ContinuousParameter(1e-6, 1e-2),
    "activation": CategoricalParameter(["relu", "gelu", "selu"]),
    "standardize": CategoricalParameter([0, 1]),
    # NOTE: removed "metric-pref" from HPO to avoid single-value categorical
    "class-weight-pos": ContinuousParameter(spw_low, spw_high),
}

metric_defs = [
    {"Name": "validation:auc",   "Regex": r"validation[-:]auc[:=]([0-9\.]+)"},
    {"Name": "validation:aucpr", "Regex": r"validation[-:]aucpr[:=]([0-9\.]+)"},
]

tuner = HyperparameterTuner(
    estimator=tf_est,
    objective_metric_name=OBJECTIVE_METRIC,   # "validation:aucpr"
    objective_type="Maximize",
    max_jobs=HPO_MAX_JOBS,
    max_parallel_jobs=HPO_MAX_PARALLEL,
    hyperparameter_ranges=hp_ranges,
    metric_definitions=metric_defs,
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
