
import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_path = "legacy_outputs/features_selected.csv"
local_model_path = "legacy_outputs/best_model.joblib"

def train_model(df):
    logger.info("🧪 Training model via train_model(df)...")

    # Ensure input is numeric only (just like run_local_model)
    df = df.select_dtypes(include=["number", "bool"])

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    logger.info("✅ Training complete. Returning model object.")
    return model


def run_local_model():
    logger.info("🧪 Running local XGBoost model for testing...")

    df = pd.read_csv(input_path)

    # ❗ Drop non-numeric columns for local testing
    df = df.select_dtypes(include=["number", "bool"])

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    logger.info("✅ Local model evaluation:")
    print(classification_report(y_test, preds))

    joblib.dump(model, local_model_path)
    logger.info(f"💾 Saved local model to: {local_model_path}")

def run_sagemaker_tuning():
    import boto3
    from sagemaker.session import Session
    from sagemaker import XGBoost
    from sagemaker.inputs import TrainingInput
    from sagemaker.tuner import HyperparameterTuner, IntegerParameter

    logger.info("🚀 Launching SageMaker tuning job...")

    session = Session()
    role = session.get_caller_identity_arn()
    bucket = session.default_bucket()

    prefix = "diabetes-xgboost"
    s3_input_path = f"s3://{bucket}/{prefix}/train.csv"
    s3_output_path = f"s3://{bucket}/{prefix}/output"

    df = pd.read_csv(input_path)
    df.to_csv("train.csv", index=False)
    boto3.client("s3").upload_file("train.csv", bucket, f"{prefix}/train.csv")
    logger.info(f"✅ Uploaded training data to {s3_input_path}")

    xgb_estimator = XGBoost(entry_point="train_script.py",
                            framework_version="1.3-1",
                            instance_type="ml.m5.large",
                            role=role,
                            output_path=s3_output_path,
                            hyperparameters={"objective": "binary:logistic", "num_round": 100})

    hyperparameter_ranges = {
        "max_depth": IntegerParameter(3, 6),
        "eta": IntegerParameter(1, 3)
    }

    tuner = HyperparameterTuner(estimator=xgb_estimator,
                                 objective_metric_name="validation:auc",
                                 hyperparameter_ranges=hyperparameter_ranges,
                                 objective_type="Maximize",
                                 max_jobs=5,
                                 max_parallel_jobs=2)

    tuner.fit({"train": TrainingInput(s3_input_path, content_type="csv")})
    logger.info("⏳ SageMaker tuning job launched.")

def main():
    use_sagemaker = False  # Flip this to True for SageMaker
    if use_sagemaker:
        run_sagemaker_tuning()
    else:
        run_local_model()

if __name__ == "__main__":
    main()
