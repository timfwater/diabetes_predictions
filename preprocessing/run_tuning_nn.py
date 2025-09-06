# preprocessing/run_tuning_nn.py
import os, io, json
from pathlib import Path
import boto3, pandas as pd
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter, CategoricalParameter
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import StratifiedKFold

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
input_file = os.environ.get("FILTERED_INPUT_FILE", "prepared_diabetes_train_selected.csv")

nn_output_prefix = os.environ.get("NN_OUTPUT_PREFIX", "nn_output")
fold_prefix = f"{prefix}/kfolds_nn"
nn_output = f"s3://{bucket}/{prefix}/{nn_output_prefix}"

nn_instance_type = os.getenv("NN_INSTANCE_TYPE", "ml.c5.2xlarge")

KFOLDS = int(os.getenv("KFOLDS", "5"))
HPO_MAX_JOBS = int(os.getenv("HPO_MAX_JOBS", "20"))
HPO_MAX_PARALLEL = int(os.getenv("HPO_MAX_PARALLEL", str(min(4, HPO_MAX_JOBS))))
OBJECTIVE_METRIC = os.getenv("OBJECTIVE_METRIC", "validation:aucpr")  # prefer PR AUC

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
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42)
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
    framework_version="2.13",
    py_version="py310",
    output_path=nn_output,
    code_location=f"s3://{bucket}/{prefix}/code",
    sagemaker_session=sess,
    hyperparameters={
        "label-col": label_col,
        "epochs": 60,                 # give early stopping room
        "batch-size": 256,
        "lr": 1e-3,
        "hidden-dim": 128,
        "hidden-layers": 2,
        "dropout": 0.3,
        "use-class-weights": 1,
        "class-weight-pos": float(pos_weight),  # ← used in train_nn.py
        "standardize": 1,                        # ← z-score in train_nn.py
        "aucpr-objective": 1,                    # ← log aucpr for tuner
    },
)

hp_ranges = {
    "lr": ContinuousParameter(1e-4, 3e-2),
    "dropout": ContinuousParameter(0.0, 0.7),
    "hidden-dim": IntegerParameter(64, 512),
    "hidden-layers": IntegerParameter(1, 5),
    "batch-size": CategoricalParameter([128, 256, 512, 1024]),
}

metric_defs = [
    {"Name": "validation:auc",   "Regex": r"validation[-:]auc[:=]([0-9\.]+)"},
    {"Name": "validation:aucpr", "Regex": r"validation[-:]aucpr[:=]([0-9\.]+)"},
]

tuner = HyperparameterTuner(
    estimator=tf_est,
    objective_metric_name=OBJECTIVE_METRIC,   # default validation:aucpr
    objective_type="Maximize",
    max_jobs=HPO_MAX_JOBS,
    max_parallel_jobs=HPO_MAX_PARALLEL,
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
