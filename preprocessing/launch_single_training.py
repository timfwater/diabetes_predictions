# launch_single_training.py

import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role
import datetime
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
bucket = "diabetes-directory"
role = get_execution_role()
session = sagemaker.Session()

# --- Paths ---
train_input = TrainingInput(
    s3_data=f"s3://{bucket}/03_tuning/train_proto.data",
    content_type="application/x-recordio-protobuf"
)
output_path = f"s3://{bucket}/03_tuning/manual_output/"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"xgb-manual-job-{timestamp}"

logger.info(f"🧪 Launching job: {job_name}")
logger.info(f"📁 Training data: {train_input.config['DataSource']['S3DataSource']['S3Uri']}")
logger.info(f"📁 Output path: {output_path}")

# --- Estimator setup ---
estimator = XGBoost(
    entry_point="train_script.py",
    framework_version="1.3-1",
    script_mode=True,
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=output_path,
    base_job_name="xgb-manual",
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "max_depth": 5,
        "eta": 0.2,
        "min_child_weight": 1
    }
)

# --- Launch Training Job ---
try:
    estimator.fit({"train": train_input}, job_name=job_name)
    logger.info(f"✅ Training job submitted successfully: {job_name}")
except Exception as e:
    logger.error("❌ Failed to launch training job")
    logger.exception(e)
