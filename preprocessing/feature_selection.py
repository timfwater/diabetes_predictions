# xgb_feature_selection.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3
import io

# S3 paths
bucket = "diabetes-directory"
input_key = "02_engineered/prepared_diabetes_full.csv"
output_key = "02_engineered/selected_features.csv"

# S3 client
s3 = boto3.client("s3")

# Read input CSV from S3
obj = s3.get_object(Bucket=bucket, Key=input_key)
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

# Drop rows with missing target or too many NaNs
df = df.dropna(subset=["readmitted"])
df = df.dropna(axis=1, thresh=int(len(df) * 0.8))  # Drop cols with >20% missing

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["readmitted"])

# Drop target from features
X = df.drop(columns=["readmitted"])

# Handle categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit XGBoost model
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# Get top N features
TOP_N = 50
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(TOP_N)

# Save result to S3
csv_buffer = io.StringIO()
top_features.index.to_series().to_csv(csv_buffer, index=False, header=False)
s3.put_object(Bucket=bucket, Key=output_key, Body=csv_buffer.getvalue())

print(f"✅ Top {TOP_N} features saved to s3://{bucket}/{output_key}")
