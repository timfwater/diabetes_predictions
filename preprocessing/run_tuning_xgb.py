# preprocessing/run_tuning_xgb.py
#!/usr/bin/env python3
import os, io, json, traceback
import boto3, pandas as pd, botocore
from sagemaker import Session, image_uris, estimator as sm_estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
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

# Inputs/outputs
features_file     = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")
input_file        = os.environ.get("FILTERED_INPUT_FILE", "prepared_diabetes_train_selected.csv")
xgb_output_prefix = os.environ.get("XGB_OUTPUT_PREFIX", "xgb_output")
xgb_output        = f"s3://{bucket}/{prefix}/{xgb_output_prefix}"
label_col         = os.getenv("LABEL_COL", "readmitted")

KFOLDS           = int(os.getenv("KFOLDS", "5"))
HPO_MAX_JOBS     = int(os.getenv("HPO_MAX_JOBS", "20"))
HPO_MAX_PARALLEL = int(os.getenv("HPO_MAX_PARALLEL", str(min(4, HPO_MAX_JOBS))))
EVAL_METRIC      = os.getenv("EVAL_METRIC", "aucpr")           # prefer PR-AUC
OBJECTIVE_METRIC = os.getenv("OBJECTIVE_METRIC", "validation:aucpr")
INSTANCE_TYPE    = os.getenv("XGB_INSTANCE_TYPE", "ml.c5.2xlarge")

# Persist the exact feature list used for this run
FEATURES_USED_LATEST_KEY = f"{prefix}/features_used_latest.txt"
FEATURES_BY_TUNING_DIR   = f"{prefix}/feature_lists/by_tuning_job"
FOLDS_PREFIX             = f"{prefix}/kfolds"

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

# ========= Label normalization =========
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
X = df.drop(columns=[label_col])

# --- Persist the exact feature list used for this run ---
ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
features_versioned_key = f"{prefix}/features_used_{ts}.txt"
features_text = "\n".join(feature_cols)
s3_put_text(bucket, features_versioned_key, features_text)
s3_put_text(bucket, FEATURES_USED_LATEST_KEY, features_text)
print(f"📎 Saved features list: s3://{bucket}/{features_versioned_key}")
print(f"📌 Updated pointer:     s3://{bucket}/{FEATURES_USED_LATEST_KEY}")

# ========= Write CSVs for XGB (label FIRST, NO header) =========
def upload_csv_to_s3_for_xgb(X_part: pd.DataFrame, y_part: pd.Series, key: str) -> str:
    out = pd.concat([y_part, X_part], axis=1)  # label first
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"

# ========= Build stratified K folds =========
folds = []
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(skf.split(X, y), start=1):
    train_key = f"{FOLDS_PREFIX}/train_{k}.csv"
    val_key   = f"{FOLDS_PREFIX}/val_{k}.csv"
    train_s3 = upload_csv_to_s3_for_xgb(X.iloc[tr], y.iloc[tr], train_key)
    val_s3   = upload_csv_to_s3_for_xgb(X.iloc[va], y.iloc[va], val_key)
    folds.append((train_s3, val_s3))
print(f"🧩 Prepared {len(folds)} folds under s3://{bucket}/{FOLDS_PREFIX}")

# ========= Imbalance report & fixed SPW =========
pos = int(y.sum()); neg = int(len(y) - pos)
spw_empirical = (neg / max(pos, 1)) if pos > 0 else 1.0
print(f"⚖️ Class balance: pos={pos}, neg={neg}, scale_pos_weight≈{spw_empirical:.4f}")

# ========= Estimator & tuner =========
xgb_image = image_uris.retrieve("xgboost", AWS_REGION, version="1.7-1")
xgb_est = sm_estimator.Estimator(
    image_uri=xgb_image,
    role=role,
    instance_count=1,
    instance_type=INSTANCE_TYPE,
    output_path=xgb_output,
    sagemaker_session=sess,
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": EVAL_METRIC,              # optimize PR-AUC
        "early_stopping_rounds": "50",
        "verbosity": "1",
        "scale_pos_weight": f"{spw_empirical:.6f}",  # ← FIXED (not tunable in SM built-in)
        # sensible defaults; tuner will explore around these
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
)

# Allowed tunables for SM built-in XGBoost
hp_ranges = {
    # learning dynamics
    "eta": ContinuousParameter(0.01, 0.3),
    "num_round": IntegerParameter(200, 1500),

    # tree complexity
    "max_depth": IntegerParameter(3, 10),
    "min_child_weight": ContinuousParameter(1.0, 10.0),
    "gamma": ContinuousParameter(0.0, 5.0),

    # subsampling
    "subsample": ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0),

    # regularization
    "lambda": ContinuousParameter(0.0, 10.0),
    "alpha": ContinuousParameter(0.0, 10.0),

    # optional stabilizer for imbalance/convergence
    "max_delta_step": ContinuousParameter(0.0, 10.0),
    # You may also add the following if you want broader search:
    # "colsample_bylevel": ContinuousParameter(0.5, 1.0),
    # "colsample_bynode": ContinuousParameter(0.5, 1.0),
    # "num_parallel_tree": IntegerParameter(1, 4),
}

# capture both auc and aucpr formats for logs
metric_defs = [
    {"Name": "validation:auc",   "Regex": r"validation[-:]auc[:=]([0-9\.]+)"},
    {"Name": "validation:aucpr", "Regex": r"validation[-:]aucpr[:=]([0-9\.]+)"},
]

tuner = HyperparameterTuner(
    estimator=xgb_est,
    objective_metric_name=OBJECTIVE_METRIC,   # "validation:aucpr"
    objective_type="Maximize",
    max_jobs=HPO_MAX_JOBS,
    max_parallel_jobs=HPO_MAX_PARALLEL,
    hyperparameter_ranges=hp_ranges,
    metric_definitions=metric_defs,
)

# ========= Launch tuning per fold =========
latest_job_name = None
started_jobs = []

for i, (train_s3, val_s3) in enumerate(folds, start=1):
    print(f"🚀 Starting XGB tuning job for fold {i}:")
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

        # Save a job-scoped copy of the exact features used for THIS run
        job_scoped_key = f"{FEATURES_BY_TUNING_DIR}/{job_name}.txt"
        s3_put_text(bucket, job_scoped_key, features_text)

    except botocore.exceptions.ClientError as e:
        print("!!! FAILED to create tuning job (ClientError):", e)
        traceback.print_exc(); raise
    except Exception as e:
        print("!!! FAILED to create tuning job:", e)
        traceback.print_exc(); raise

# ========= Persist last job name =========
if latest_job_name:
    try:
        with open("/app/preprocessing/latest_tuning_job.txt", "w") as f:
            f.write(latest_job_name)
        print("💾 Wrote /app/preprocessing/latest_tuning_job.txt ->", latest_job_name)
        with open("/app/latest_tuning_job.txt", "w") as f:
            f.write(latest_job_name)
        print("💾 Wrote /app/latest_tuning_job.txt ->", latest_job_name)
    except Exception as e:
        print("⚠️ Could not write local latest_tuning_job files:", repr(e))

    print("🧾 Tuning jobs this run:", json.dumps(started_jobs, indent=2))
    print("✅ Saved latest tuning job:", latest_job_name)
else:
    print("⚠️ No tuning jobs were started.")
