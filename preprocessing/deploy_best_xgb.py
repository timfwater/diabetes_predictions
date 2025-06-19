import sagemaker
from sagemaker.tuner import HyperparameterTuner
import boto3

# --- Config ---
bucket = "diabetes-directory"
prefix = "02_engineered"
endpoint_name = "diabetes-xgb-endpoint"

# --- SageMaker Setup ---
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# --- Load Best Tuning Job ---
with open("latest_tuning_job.txt") as f:
    tuning_job_name = f.read().strip()
print("📦 Loaded tuning job:", tuning_job_name)

tuner = HyperparameterTuner.attach(tuning_job_name, sagemaker_session=sess)
best_estimator = tuner.best_estimator()
print("🚀 Using best estimator from:", tuner.best_training_job())

# --- Deploy to Endpoint ---
predictor = best_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)
print("✅ Deployed endpoint:", predictor.endpoint_name)
