import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3
import io

# --- Load Environment Variables or Use Defaults ---
bucket = os.environ.get("BUCKET", "diabetes-directory")
prefix = os.environ.get("PREFIX", "02_engineered")
input_filename = os.environ.get("FULL_DATA_FILE", "prepared_diabetes_full.csv")
output_filename = os.environ.get("SELECTED_FEATURES_FILE", "selected_features.csv")

input_key = f"{prefix}/{input_filename}"
output_key = f"{prefix}/{output_filename}"

# --- S3 client ---
s3 = boto3.client("s3")

# --- Read input CSV from S3 ---
obj = s3.get_object(Bucket=bucket, Key=input_key)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

# --- Preprocessing ---
df = df.dropna(subset=["readmitted"])
df = df.dropna(axis=1, thresh=int(len(df) * 0.8))  # Drop cols with >20% missing

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["readmitted"])
X = pd.get_dummies(df.drop(columns=["readmitted"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost ---
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# --- Select top N features ---
TOP_N = 50
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(TOP_N)

# ✅ Format with header for downstream compatibility
top_features_df = pd.DataFrame({"selected_features": top_features.index})

# --- Upload to S3 ---
csv_buffer = io.StringIO()
top_features_df.to_csv(csv_buffer, index=False)  # Includes 'selected_features' header
s3.put_object(Bucket=bucket, Key=output_key, Body=csv_buffer.getvalue())

print(f"✅ Top {TOP_N} features saved to s3://{bucket}/{output_key}")
