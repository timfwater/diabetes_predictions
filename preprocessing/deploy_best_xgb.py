# preprocessing/deploy_best_xgb.py
import os
import time
import boto3
import sagemaker
from sagemaker.tuner import HyperparameterTuner
from botocore.exceptions import ClientError

# ------------ Config ------------
REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET", "diabetes-directory")
PREFIX = os.getenv("PREFIX", "02_engineered")

ENDPOINT_NAME = os.getenv("ENDPOINT", "diabetes-xgb-endpoint")
DEPLOY_INSTANCE_TYPE = os.getenv("DEPLOY_INSTANCE_TYPE", "ml.m5.large")

# How to pick a tuning job
TUNING_JOB_NAME   = os.getenv("TUNING_JOB_NAME", "").strip()               # optional hard pin
TUNING_JOB_FILE   = os.getenv("TUNING_JOB_FILE", "latest_tuning_job.txt")  # written by run_tuning.py
TUNING_JOB_PREFIX = os.getenv("TUNING_JOB_PREFIX", "sagemaker-xgboost-")
WAIT_FOR_TUNING   = os.getenv("WAIT_FOR_TUNING", "false").lower() == "true"

# Feature storage
FEATURES_USED_LATEST_KEY = f"{PREFIX}/features_used_latest.txt"
FEATURES_BY_TUNING_DIR   = f"{PREFIX}/feature_lists/by_tuning_job"  # optional job-scoped
MODEL_FEATURES_DIR       = f"{PREFIX}/model_feature_lists"          # versioned per model

# ------------ Sessions/clients ------------
boto_sess = boto3.Session(region_name=REGION)
sm  = boto_sess.client("sagemaker")
s3  = boto_sess.client("s3")
sess = sagemaker.Session(boto_session=boto_sess)

# ------------ Helpers ------------
def s3_get_text(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def s3_put_text(bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    print(f"📤 Uploaded s3://{bucket}/{key}")

def _object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def _job_status(name: str) -> str:
    try:
        resp = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=name)
        return resp["HyperParameterTuningJobStatus"]
    except ClientError:
        return "Unknown"

def _pick_tuning_job() -> str:
    """
    Priority:
      1) TUNING_JOB_NAME (env). If not complete and WAIT_FOR_TUNING=true, wait.
      2) TUNING_JOB_FILE (if exists). If not complete and WAIT_FOR_TUNING=true, wait.
      3) Latest Completed job matching TUNING_JOB_PREFIX.
    """
    if TUNING_JOB_NAME:
        print("📦 Loaded tuning job (env):", TUNING_JOB_NAME)
        status = _job_status(TUNING_JOB_NAME)
        print("   status:", status)
        if status not in ("Completed", "Stopped"):
            if WAIT_FOR_TUNING:
                print("⏳ Waiting for tuning job to complete...")
                sm.get_waiter("hyper_parameter_tuning_job_completed_or_stopped").wait(
                    HyperParameterTuningJobName=TUNING_JOB_NAME
                )
            else:
                raise RuntimeError(
                    f"Tuning job {TUNING_JOB_NAME} not completed (status={status}). "
                    f"Set WAIT_FOR_TUNING=true or pass a Completed job."
                )
        return TUNING_JOB_NAME

    if os.path.exists(TUNING_JOB_FILE):
        with open(TUNING_JOB_FILE) as f:
            name = f.read().strip()
        if name:
            print("📦 Loaded tuning job (file):", name)
            status = _job_status(name)
            print("   status:", status)
            if status in ("Completed", "Stopped"):
                return name
            if WAIT_FOR_TUNING:
                print("⏳ Waiting for tuning job to complete...")
                sm.get_waiter("hyper_parameter_tuning_job_completed_or_stopped").wait(
                    HyperParameterTuningJobName=name
                )
                return name
            print("ℹ️ File’s tuning job not completed; selecting latest Completed instead.")

    # Latest Completed by prefix
    jobs = []
    token = None
    while True:
        kw = dict(StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending", MaxResults=50)
        if token:
            kw["NextToken"] = token
        resp = sm.list_hyper_parameter_tuning_jobs(**kw)
        for j in resp.get("HyperParameterTuningJobSummaries", []):
            if TUNING_JOB_PREFIX in j["HyperParameterTuningJobName"]:
                jobs.append(j["HyperParameterTuningJobName"])
        token = resp.get("NextToken")
        if not token or jobs:
            break
    if not jobs:
        raise RuntimeError("No Completed tuning jobs found. Provide TUNING_JOB_NAME or wait for one to finish.")
    chosen = jobs[0]
    print("✅ Selected latest Completed tuning job:", chosen)
    return chosen

def endpoint_exists(name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg  = str(e)
        if "Could not find endpoint" in msg or code in ("ValidationException", "ResourceNotFound", "404", "NotFound"):
            return False
        raise

# ------------ Choose tuning job & best estimator ------------
tuning_job_name = _pick_tuning_job()
tuner = HyperparameterTuner.attach(tuning_job_name, sagemaker_session=sess)
best_estimator = tuner.best_estimator()
print("🚀 Using best estimator from:", tuner.best_training_job())

# ------------ Resolve features for THIS deployment ------------
job_scoped_key = f"{FEATURES_BY_TUNING_DIR}/{tuning_job_name}.txt"

if _object_exists(BUCKET, job_scoped_key):
    print(f"📌 Using job-scoped features: s3://{BUCKET}/{job_scoped_key}")
    features_text = s3_get_text(BUCKET, job_scoped_key).strip()
else:
    print(f"⚠️ Job-scoped features not found at s3://{BUCKET}/{job_scoped_key}")
    if not _object_exists(BUCKET, FEATURES_USED_LATEST_KEY):
        raise RuntimeError(f"Could not find features pointer: s3://{BUCKET}/{FEATURES_USED_LATEST_KEY}")
    fallback_ptr = f"s3://{BUCKET}/{FEATURES_USED_LATEST_KEY}"
    print(f"📌 Falling back to features pointer: {fallback_ptr}")
    features_text = s3_get_text(BUCKET, FEATURES_USED_LATEST_KEY).strip()

if not features_text:
    raise RuntimeError("Resolved feature list is empty.")

# Small sanity: count features
feature_count = sum(1 for ln in features_text.splitlines() if ln.strip())
print(f"📏 Feature list resolved for deployment: {feature_count} features")

# ------------ Register model ------------
role = (
    getattr(best_estimator, "role", None)
    or os.getenv("SAGEMAKER_ROLE")
    or os.getenv("SAGEMAKER_TRAINING_ROLE")
)
if not role:
    raise RuntimeError(
        "No role available for model registration. "
        "Set SAGEMAKER_ROLE or SAGEMAKER_TRAINING_ROLE to an IAM role ARN with SageMaker permissions."
    )

model_name = f"{ENDPOINT_NAME}-model-{int(time.time())}"
model = best_estimator.create_model(role=role, name=model_name)
print(f"🧱 Creating model: {model_name}")
model.create()

# Version & tag the feature list with this model
model_features_key = f"{MODEL_FEATURES_DIR}/{model_name}.txt"
s3_put_text(BUCKET, model_features_key, features_text)

desc = sm.describe_model(ModelName=model_name)
model_arn = desc["ModelArn"]
sm.add_tags(ResourceArn=model_arn, Tags=[{"Key": "FeaturesListS3Key", "Value": model_features_key}])
print(f"🏷️ Tagged model with FeaturesListS3Key={model_features_key}")

# Keep an endpoint-level pointer too (latest serving features)
endpoint_features_latest_key = f"{PREFIX}/endpoint_features_latest.txt"
s3_put_text(BUCKET, endpoint_features_latest_key, model_features_key)
print(f"📌 Updated endpoint features pointer: s3://{BUCKET}/{endpoint_features_latest_key}")

# ------------ Create fresh endpoint-config ------------
endpoint_config_name = f"{ENDPOINT_NAME}-{int(time.time())}"
print(f"🧩 Creating endpoint-config: {endpoint_config_name}")
variant = {
    "ModelName": model_name,
    "VariantName": "AllTraffic",
    "InitialInstanceCount": 1,
    "InitialVariantWeight": 1.0,
    "InstanceType": DEPLOY_INSTANCE_TYPE,
}
sm.create_endpoint_config(EndpointConfigName=endpoint_config_name, ProductionVariants=[variant])

# ------------ Create or update endpoint ------------
if endpoint_exists(ENDPOINT_NAME):
    print(f"♻️ Updating endpoint {ENDPOINT_NAME} -> {endpoint_config_name}")
    sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=endpoint_config_name)
else:
    print(f"🆕 Creating endpoint {ENDPOINT_NAME}")
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=endpoint_config_name)

print("⏳ Waiting for endpoint to become InService...")
sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT_NAME)
print("✅ Endpoint ready:", ENDPOINT_NAME)
