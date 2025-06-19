import pandas as pd
import logging
import boto3
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 locations
bucket = "diabetes-directory"
prepared_key = "02_engineered/prepared_diabetes.csv"
features_key = "02_engineered/selected_features.csv"
s3_output_prefix = "03_modeling"
s3_output_path = f"s3://{bucket}/{s3_output_prefix}/output"
s3_train_key = f"{s3_output_prefix}/train.csv"

local_model_path = "legacy_outputs/best_model.joblib"

def load_filtered_training_data():
    s3 = boto3.client("s3")

    # Load full dataset
    obj = s3.get_object(Bucket=bucket, Key=prepared_key)
    df = pd.read_csv(obj["Body"])

    # Load top selected features
    obj = s3.get_object(Bucket=bucket, Key=features_key)
    selected_features = pd.read_csv(obj["Body"], header=None)[0].tolist()

    # Subset and return
    df = df[selected_features + ["readmitted"]]
    return df

def run_local_model():
    logger.info("🧪 Running local XGBoost model for testing...")

    df = load_filtered_training_data()
    df = df.select_dtypes(include=["number", "bool"])

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    logger.info("✅ Local model evaluation:")
    print(classification_report(y_test, preds))

    joblib.dump(model
