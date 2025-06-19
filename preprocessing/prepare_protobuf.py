import pandas as pd
import boto3
import io
import numpy as np
import sagemaker.amazon.common as smac

# S3 paths
bucket = "diabetes-directory"
prepared_key = "02_engineered/prepared_diabetes.csv"
features_key = "02_engineered/selected_features.csv"
output_prefix = "03_tuning"
output_key = f"{output_prefix}/train_proto.data"

# Initialize S3 client
s3 = boto3.client("s3")

# Load the full prepared dataset
prepared_obj = s3.get_object(Bucket=bucket, Key=prepared_key)
df = pd.read_csv(io.BytesIO(prepared_obj["Body"].read()))

# Load selected features
features_obj = s3.get_object(Bucket=bucket, Key=features_key)
selected_features = pd.read_csv(io.BytesIO(features_obj["Body"].read()), header=None)[0].tolist()

# Subset to selected features + target
df = df[selected_features + ["readmitted"]]

# Ensure numerical dtype
df = df.select_dtypes(include=["number", "bool"])

# Split into features and labels
X = df.drop("readmitted", axis=1).values.astype("float32")
y = df["readmitted"].values.astype("float32")

# Convert to protobuf
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X, y)
buf.seek(0)

# Upload to S3
s3.put_object(Bucket=bucket, Key=output_key, Body=buf.getvalue())
print(f"✅ Protobuf data saved to s3://{bucket}/{output_key}")
