import pandas as pd
import numpy as np
import argparse
import boto3
import os
from sklearn.model_selection import train_test_split
from sagemaker.amazon.common import write_numpy_to_dense_tensor

def upload_to_s3(local_file, bucket, s3_key):
    s3 = boto3.client("s3")
    s3.upload_file(local_file, bucket, s3_key)
    print(f"✅ Uploaded {local_file} to s3://{bucket}/{s3_key}")

def prepare_and_upload_protobuf(csv_path, bucket, prefix, label_column="readmitted", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    # Separate features and label
    y = df[label_column].astype("float32")
    X = df.drop(columns=[label_column]).astype("float32")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to Protobuf format
    os.makedirs("protobuf_data", exist_ok=True)
    train_file = "protobuf_data/train_proto.data"
    val_file = "protobuf_data/validation_proto.data"

    with open(train_file, "wb") as f:
        write_numpy_to_dense_tensor(f, X_train.to_numpy(), y_train.to_numpy())
    with open(val_file, "wb") as f:
        write_numpy_to_dense_tensor(f, X_val.to_numpy(), y_val.to_numpy())

    # Upload to S3
    upload_to_s3(train_file, bucket, f"{prefix}/train_proto.data")
    upload_to_s3(val_file, bucket, f"{prefix}/validation_proto.data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--prefix", type=str, required=True, help="S3 prefix/path to store protobuf files")
    parser.add_argument("--label_column", type=str, default="readmitted", help="Name of label column")
    args = parser.parse_args()

    prepare_and_upload_protobuf(
        csv_path=args.csv_path,
        bucket=args.bucket,
        prefix=args.prefix,
        label_column=args.label_column
    )