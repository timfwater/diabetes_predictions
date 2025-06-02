"""
This script implements the cleaned logic from `notebooks/data_engineering_eda.ipynb`.
For full exploratory justification and visualizations, refer to that notebook.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath, is_s3=False, bucket=None, subfolder=None):
    if is_s3:
        import boto3
        s3_path = f"s3://{bucket}/{subfolder}/{filepath}"
        logger.info(f"📥 Loading from S3: {s3_path}")
        return pd.read_csv(s3_path)
    else:
        logger.info(f"📥 Loading from local: {filepath}")
        return pd.read_csv(filepath)

def save_data(df, filename, is_s3=False, bucket=None, subfolder=None):
    if is_s3:
        import boto3
        df.to_csv(filename, index=False)
        s3_key = f"{subfolder}/{filename}"
        logger.info(f"📤 Uploading to S3: s3://{bucket}/{s3_key}")
        boto3.Session().resource('s3').Bucket(bucket).upload_file(Filename=filename, Key=s3_key)
        os.remove(filename)  # clean up local temp file
    else:
        logger.info(f"💾 Saving locally to: {filename}")
        df.to_csv(filename, index=False)

def clean_data(df):
    df['readmitted'] = df['readmitted'].replace(['NO', '>30', '<30'], [0, 0, 1])
    df['age'] = df['age'].replace(
        ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', 
         '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
        list(range(1, 11))
    )
    df['race'] = df['race'].fillna("Other")
    df['gender'] = df['gender'].replace(['Unknown/Invalid'], ['Female'])
    df['A1Cresult'] = df['A1Cresult'].replace(['None'], ['NotTaken'])
    df['max_glu_serum'] = df['max_glu_serum'].replace(['None'], ['NotTaken'])
    logger.info("✅ Basic data cleaning complete")
    return df

def engineer_features(df):
    # Placeholder — insert engineering steps later
    return df

def encode_variables(df):
    # Placeholder — insert encoding steps later
    return df

def main():
    # Toggle between local and S3
    use_s3 = False
    input_file = "Diabetes_Input.csv"
    output_file = "processed_output.csv"

    if use_s3:
        bucket = "your-bucket-name"
        subfolder = "your-subfolder"
        df = load_data(input_file, is_s3=True, bucket=bucket, subfolder=subfolder)
    else:
        df = load_data(f"legacy_outputs/{input_file}")

    df = clean_data(df)
    df = engineer_features(df)
    df = encode_variables(df)

    if use_s3:
        save_data(df, output_file, is_s3=True, bucket=bucket, subfolder=subfolder)
    else:
        save_data(df, f"legacy_outputs/{output_file}")

    logger.info("✅ Data engineering pipeline complete.")

if __name__ == "__main__":
    main()
