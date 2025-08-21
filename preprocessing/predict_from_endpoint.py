# preprocessing/predict_from_endpoint.py
import os
import io
import boto3
import botocore
import pandas as pd
import numpy as np
from io import StringIO
from sagemaker.session import Session  # still used to discover model tag via SageMaker API

# -------- Config --------
BUCKET = os.getenv("BUCKET", "diabetes-directory")
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "02_engineered")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "03_scored")
ENDPOINT_NAME = os.getenv("ENDPOINT", "diabetes-xgb-endpoint")
LABEL_COL = os.getenv("LABEL_COL", "readmitted")
REGION = os.getenv("AWS_REGION", boto3.Session().region_name or "us-east-1")

# Feature list discovery (ordered priority)
FEATURE_LIST_KEY = os.getenv("FEATURE_LIST_KEY", "")  # explicit override
ENDPOINT_FEATURES_LATEST_KEY = f"{INPUT_PREFIX}/endpoint_features_latest.txt"
FEATURES_USED_LATEST_KEY = f"{INPUT_PREFIX}/features_used_latest.txt"
SELECTED_FEATURES_FALLBACK = f"{INPUT_PREFIX}/selected_features.csv"

# Input CSV keys
TEST_KEY = os.getenv("TEST_KEY", f"{INPUT_PREFIX}/prepared_diabetes_test.csv")
TRAIN_KEY = os.getenv("TRAIN_KEY", f"{INPUT_PREFIX}/prepared_diabetes_train.csv")

# -------- AWS clients --------
boto_sess = boto3.Session(region_name=REGION)
s3 = boto_sess.client("s3", region_name=REGION)
sm = boto_sess.client("sagemaker", region_name=REGION)
rt = boto_sess.client("sagemaker-runtime", region_name=REGION)
sess = Session(boto_session=boto_sess)

# -------- S3 helpers --------
def s3_read_csv(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def s3_write_csv(df: pd.DataFrame, bucket: str, key: str):
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buf.getvalue())
    print(f"✅ Uploaded s3://{bucket}/{key}")

def _object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def _load_features_from_text_key(key: str) -> list:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    txt = obj["Body"].read().decode("utf-8").strip()
    feats = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not feats:
        raise RuntimeError(f"No features found in s3://{BUCKET}/{key}")
    return feats

def _load_features_from_selected_csv() -> list:
    obj = s3.get_object(Bucket=BUCKET, Key=SELECTED_FEATURES_FALLBACK)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    for col in ["selected_features", "feature", "features"]:
        if col in df.columns:
            feats = df[col].dropna().astype(str).tolist()
            if feats:
                return feats
    if len(df.columns) == 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()
    raise RuntimeError("Could not determine feature list from selected_features.csv")

def _load_features_from_model_tag() -> list | None:
    """Use current endpoint's deployed model tag 'FeaturesListS3Key'."""
    try:
        ep = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        cfg_name = ep["EndpointConfigName"]
        cfg = sm.describe_endpoint_config(EndpointConfigName=cfg_name)
        variants = cfg.get("ProductionVariants", [])
        if not variants:
            return None
        model_name = variants[0]["ModelName"]
        mdl = sm.describe_model(ModelName=model_name)
        arn = mdl["ModelArn"]
        tags = sm.list_tags(ResourceArn=arn).get("Tags", [])
        kv = {t["Key"]: t["Value"] for t in tags}
        key = kv.get("FeaturesListS3Key")
        if key:
            print(f"🏷️ Using FeaturesListS3Key from model tag: s3://{BUCKET}/{key}")
            return _load_features_from_text_key(key)
        return None
    except botocore.exceptions.ClientError:
        return None

def load_feature_list() -> list:
    """
    Priority:
      1) FEATURE_LIST_KEY (explicit override)
      2) model tag FeaturesListS3Key (current endpoint's model)
      3) endpoint_features_latest.txt
      4) features_used_latest.txt
      5) selected_features.csv
    """
    if FEATURE_LIST_KEY:
        print(f"📌 Using FEATURE_LIST_KEY override: s3://{BUCKET}/{FEATURE_LIST_KEY}")
        return _load_features_from_text_key(FEATURE_LIST_KEY)

    feats = _load_features_from_model_tag()
    if feats:
        return feats

    if _object_exists(BUCKET, ENDPOINT_FEATURES_LATEST_KEY):
        print(f"📌 Using endpoint features pointer: s3://{BUCKET}/{ENDPOINT_FEATURES_LATEST_KEY}")
        target_key = s3.get_object(Bucket=BUCKET, Key=ENDPOINT_FEATURES_LATEST_KEY)["Body"].read().decode("utf-8").strip()
        if target_key:
            return _load_features_from_text_key(target_key)

    if _object_exists(BUCKET, FEATURES_USED_LATEST_KEY):
        print(f"📌 Using training features pointer: s3://{BUCKET}/{FEATURES_USED_LATEST_KEY}")
        return _load_features_from_text_key(FEATURES_USED_LATEST_KEY)

    print(f"📌 Falling back to: s3://{BUCKET}/{SELECTED_FEATURES_FALLBACK}")
    return _load_features_from_selected_csv()

FEATURES = load_feature_list()
print(f"🧭 Feature count for serving: {len(FEATURES)}")

# -------- Diff preview --------
def preview_diff(df_cols: list[str], features: list[str]) -> None:
    df_set = set(df_cols)
    f_set = set(features)
    missing = [c for c in features if c not in df_set]
    extra = [c for c in df_cols if c not in f_set]
    if missing:
        print(f"⚠️ Missing features ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"⚠️ Extra columns in data ({len(extra)}): {extra[:5]}{'...' if len(extra) > 5 else ''}")
    if features:
        head = features[:5]
        tail = features[-5:] if len(features) >= 5 else features
        print(f"🔢 Order check head: {head} ... tail: {tail}")

# -------- Frame prep --------
def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in [LABEL_COL, "label", "y", "id", "patient_id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    preview_diff(df.columns.tolist(), FEATURES)

    missing = [c for c in FEATURES if c not in df.columns]
    extras  = [c for c in df.columns if c not in FEATURES]

    if missing:
        print(f"🧩 Adding {len(missing)} missing feature columns as 0.0 (e.g., {missing[:5]}{'...' if len(missing)>5 else ''})")
        for c in missing:
            df[c] = 0.0

    if extras:
        print(f"🧹 Dropping {len(extras)} extra columns not in model schema (e.g., {extras[:5]}{'...' if len(extras)>5 else ''})")
        df = df.drop(columns=extras)

    X = df.reindex(columns=FEATURES, fill_value=0.0).copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    nonfinite_before = int((~np.isfinite(X.to_numpy())).sum())
    if nonfinite_before:
        print(f"🧯 Found {nonfinite_before} non-finite values; clamping to 0.0")
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    nonfinite_after = int((~np.isfinite(X.to_numpy())).sum())
    if nonfinite_after:
        print(f"⚠️ Still found {nonfinite_after} non-finite values after clamping (unexpected)")
    return X

# -------- Prediction helpers --------
def _invoke(payload: str) -> str:
    try:
        resp = rt.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Accept="text/csv",
            Body=payload.encode("utf-8"),
        )
        body = resp["Body"].read().decode("utf-8")
        return body
    except botocore.exceptions.ClientError:
        print("🚨 InvokeEndpoint error. Tail the endpoint logs for details:")
        print("    aws logs tail /aws/sagemaker/Endpoints/diabetes-xgb-endpoint --since 15m --region", REGION, "--follow")
        raise

def predict_df(X: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        payload = batch.to_csv(header=False, index=False, lineterminator="\n")
        if i == 0:
            # quick shape sanity
            cols_in_payload = payload.split("\n", 1)[0].count(",") + 1
            print(f"🧾 First batch: {batch.shape[0]} rows × {cols_in_payload} cols (expected {len(FEATURES)})")
        resp = _invoke(payload)
        preds.extend([float(line) for line in resp.strip().split("\n") if line.strip() != ""])
    return np.array(preds, dtype=np.float32)

def smoke_test(X: pd.DataFrame):
    one = X.iloc[0:1]
    print(f"🫖 Smoke test row has {one.shape[1]} features")
    payload = one.to_csv(header=False, index=False, lineterminator="\n")
    print("🧾 Smoke payload preview:", payload[:120].replace("\n", "\\n"))
    resp = _invoke(payload)
    print("🔎 Smoke test prediction:", resp.strip())

# -------- Orchestration --------
def process_set(input_key: str, output_name: str):
    print(f"\n🔮 Processing: s3://{BUCKET}/{input_key}")
    df = s3_read_csv(BUCKET, input_key)
    X = prepare_frame(df)
    smoke_test(X)
    yhat = predict_df(X)
    out = df.copy()
    out["predicted_proba"] = yhat
    out_key = f"{OUTPUT_PREFIX}/{output_name}"
    s3_write_csv(out, BUCKET, out_key)

if __name__ == "__main__":
    process_set(TEST_KEY, "test_with_predictions.csv")
    process_set(TRAIN_KEY, "train_with_predictions.csv")
    print("\n🏁 Done! Predictions uploaded.")
