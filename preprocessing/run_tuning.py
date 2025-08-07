import os, io, boto3
import pandas as pd
from sagemaker import Session, image_uris, estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.inputs import TrainingInput
import sagemaker.amazon.common as smac
from sklearn.model_selection import KFold

# 🔧 Config
sess = Session()
role = os.environ.get("SAGEMAKER_TRAINING_ROLE")

print(f"✅ Detected SAGEMAKER_TRAINING_ROLE: {role}")
if not role:
    raise ValueError("❌ SAGEMAKER_TRAINING_ROLE environment variable is missing in container.")

# 🌐 Read from environment variables (with sensible defaults)
bucket = os.environ.get("BUCKET", "diabetes-directory")
prefix = os.environ.get("PREFIX", "02_engineered")
features_file = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")
input_file = os.environ.get("FILTERED_INPUT_FILE", "5_perc.csv")
xgb_output_prefix = os.environ.get("XGB_OUTPUT_PREFIX", "xgb_output")
proto_prefix = f"{prefix}/k"
xgb_output = f"s3://{bucket}/{prefix}/{xgb_output_prefix}"

# 📥 Load and filter data
df = pd.read_csv(f"s3://{bucket}/{prefix}/{input_file}")
features = pd.read_csv(f"s3://{bucket}/{prefix}/{features_file}")["selected_features"].tolist()
df = df[features + ["readmitted"]].dropna()
print(f"📊 Filtered to {df.shape[0]} rows and {len(features)} features.")

# 📤 Export folds to protobuf
def export_protobuf(df_part, key):
    buf = io.BytesIO()
    X = df_part.drop("readmitted", axis=1).astype("float32").values
    y = df_part["readmitted"].astype("float32").values
    smac.write_numpy_to_dense_tensor(buf, X, y)
    buf.seek(0)
    boto3.Session().resource("s3").Bucket(bucket).Object(key).upload_fileobj(buf)
    return f"s3://{bucket}/{key}"

folds = []
for k, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(df), 1):
    train_s3 = export_protobuf(df.iloc[tr], f"{proto_prefix}/train_{k}.data")
    val_s3 = export_protobuf(df.iloc[va], f"{proto_prefix}/val_{k}.data")
    folds.append((train_s3, val_s3))

# ⚙️ Estimator and tuner setup
container = image_uris.retrieve("xgboost", sess.boto_region_name, "1.7-1")
xgb_est = estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    output_path=xgb_output,
    sagemaker_session=sess,
    hyperparameters={"objective": "binary:logistic", "num_round": "100"},
)

hp_ranges = {
    "eta": ContinuousParameter(0, 1),
    "alpha": ContinuousParameter(0, 2),
    "min_child_weight": ContinuousParameter(1, 10),
    "max_depth": IntegerParameter(1, 10),
}

tuner = HyperparameterTuner(
    estimator=xgb_est,
    objective_metric_name="validation:auc",
    hyperparameter_ranges=hp_ranges,
    metric_definitions=[{"Name": "validation:auc", "Regex": "validation-auc=([0-9\\.]+)"}],
    max_jobs=4,
    max_parallel_jobs=5,
    objective_type="Maximize",
)

# 🚀 Launch tuning
for train_s3, val_s3 in folds:
    tuner.fit({
        "train": TrainingInput(train_s3, content_type="application/x-recordio-protobuf"),
        "validation": TrainingInput(val_s3, content_type="application/x-recordio-protobuf")
    })

# 📝 Save latest tuning job name
job_name = tuner.latest_tuning_job.name
with open("latest_tuning_job.txt", "w") as f:
    f.write(job_name)
print("✅ Saved tuning job:", job_name)
