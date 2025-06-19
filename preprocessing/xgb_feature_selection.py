import pandas as pd
import numpy as np
import logging
import boto3
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_N = 30  # Top N features to keep
BUCKET = 'diabetes-directory'
INPUT_KEY = '02_engineered/prepared_diabetes.csv'
OUTPUT_KEY = '02_engineered/selected_features.csv'

def preprocess(df):
    df = df.copy()
    df = df.dropna(axis=1, thresh=int(0.5 * len(df)))  # drop cols with >50% missing
    df = df.select_dtypes(include=[np.number]).join(df.select_dtypes(exclude=[np.number]).fillna('Unknown'))

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    logger.info("✅ Encoding complete")

    return df

def train_xgb_get_importance(X, y):
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0
    }
    bst = xgb.train(params, dtrain, num_boost_round=50)
    importance = bst.get_score(importance_type='weight')
    return importance

def read_csv_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

def write_csv_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3 = boto3.resource('s3')
    s3.Object(bucket, key).put(Body=csv_buffer.getvalue())

def main():
    logger.info(f"📥 Loading input from s3://{BUCKET}/{INPUT_KEY}")
    df = read_csv_from_s3(BUCKET, INPUT_KEY)
    logger.info(f"📊 Loaded data with shape: {df.shape}")

    df = preprocess(df)
    y = df['readmitted']
    X = df.drop(columns=['readmitted'])

    logger.info(f"🚀 Training XGBoost model for feature selection...")
    importances = train_xgb_get_importance(X, y)
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in ranked[:TOP_N]]

    output_df = pd.DataFrame(top_features, columns=["selected_features"])
    logger.info(f"✅ Saving top {TOP_N} features to s3://{BUCKET}/{OUTPUT_KEY}")
    write_csv_to_s3(output_df, BUCKET, OUTPUT_KEY)
    logger.info("🎉 Feature selection complete.")

if __name__ == "__main__":
    main()
