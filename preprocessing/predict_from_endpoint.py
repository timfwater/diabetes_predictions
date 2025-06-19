import pandas as pd
import numpy as np
import boto3
from io import StringIO
from sagemaker.predictor import Predictor

# --- Config ---
bucket = "diabetes-directory"
prefix = "02_engineered"
endpoint_name = "diabetes-xgb-endpoint"
region = boto3.Session().region_name
s3 = boto3.client("s3")
predictor = Predictor(endpoint_name=endpoint_name)

# --- Load CSV from S3 ---
def load_csv(key):
    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/{key}")
    return pd.read_csv(obj["Body"])

# --- Save CSV to S3 ---
def upload_to_s3(df, filename):
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/{filename}", Body=csv_buf.getvalue())
    print(f"✅ Uploaded {filename} to s3://{bucket}/{prefix}/")

# --- Load selected feature list from S3 ---
# --- Load selected feature list from S3 ---
def load_selected_features():
    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/selected_features.csv")
    df = pd.read_csv(obj["Body"])
    return df["selected_features"].tolist()

selected_features = load_selected_features()

# --- Predict using CSV and append probability ---
def predict_csv(df: pd.DataFrame, label="predicted_proba") -> pd.DataFrame:
    try:
        df_clean = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        payload = df_clean.to_csv(header=False, index=False).strip()
        response = predictor.predict(payload, initial_args={"ContentType": "text/csv"})

        if isinstance(response, bytes):
            response = response.decode("utf-8")
        preds = [float(line) for line in response.strip().split("\n")]

        df_out = df.copy()
        df_out[label] = preds
        return df_out

    except Exception as e:
        print("❌ Error during prediction:", e)
        raise

# --- Predict and Upload ---
def process_set(csv_key, output_key):
    print(f"🔮 Predicting {csv_key}...")
    df = load_csv(csv_key)
    df_selected = df[selected_features]
    df_out = predict_csv(df_selected)
    df_out["readmitted"] = df["readmitted"]
    upload_to_s3(df_out, output_key)

# --- Run on test and train ---
process_set("prepared_diabetes_test.csv", "test_with_predictions.csv")
process_set("prepared_diabetes_train.csv", "train_with_predictions.csv")

print("\n🏁 Done! Predictions uploaded to S3.")
