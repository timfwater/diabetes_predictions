import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.xgboost.estimator import XGBoost
from sagemaker import get_execution_role
import datetime

# Core settings
bucket = "diabetes-directory"
role = get_execution_role()
session = sagemaker.Session()

# S3 paths
train_input_s3 = f"s3://{bucket}/03_tuning/train_proto.data"
output_path = f"s3://{bucket}/03_tuning/tuning_output/"

# Unique job name
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"xgb-tune-job-{timestamp}"

# XGBoost Estimator
xgb_estimator = XGBoost(
    entry_point="train_script.py",
    framework_version="1.3-1",
    script_mode=True,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
    base_job_name="xgb-tune",
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100
    }
)

# Define hyperparameter ranges
hyperparameter_ranges = {
    "max_depth": IntegerParameter(3, 10),
    "eta": ContinuousParameter(0.01, 0.3),
    "min_child_weight": IntegerParameter(1, 10)
}

# Tuner setup
tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges=hyperparameter_ranges,
    objective_type="Maximize",
    max_jobs=10,
    max_parallel_jobs=2
)

# Launch
tuner.fit(
    {"train": TrainingInput(train_input_s3, content_type="application/x-recordio-protobuf")},
    job_name=job_name
)

print(f"✅ Launched tuning job: {job_name}")
