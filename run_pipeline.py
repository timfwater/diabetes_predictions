#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
from typing import Optional, List

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# -------- Config (env with sensible defaults) --------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET", "diabetes-directory")
PREFIX = os.getenv("PREFIX", "02_engineered")
ENDPOINT = os.getenv("ENDPOINT", "diabetes-xgb-endpoint")

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "deploy_and_predict").lower()
PINNED_TUNING_JOB = os.getenv("TUNING_JOB_NAME", "").strip()

# Optional: in smoke mode, how many synthetic rows to send
SMOKE_ROWS = int(os.getenv("SMOKE_ROWS", "1"))

# -------- AWS clients --------
SM = boto3.client("sagemaker", region_name=AWS_REGION)
SM_RT = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
S3 = boto3.client("s3", region_name=AWS_REGION)

# -------- Helpers --------
def run_step(py_rel_path: str, extra_env: Optional[dict] = None):
    """Run a preprocessing/*.py step and stream output."""
    step_name = os.path.basename(py_rel_path)
    print(f"\n🔧 Running: {step_name}")
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})
    code = subprocess.run([sys.executable, py_rel_path], env=env).returncode
    if code != 0:
        print(f"❌ Step failed: {step_name} (exit {code})")
        sys.exit(code)
    print(f"✅ Completed: {step_name}")

def latest_completed_tuning_job() -> Optional[str]:
    try:
        resp = SM.list_hyper_parameter_tuning_jobs(
            StatusEquals="Completed",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        jobs = resp.get("HyperParameterTuningJobSummaries", [])
        return jobs[0]["HyperParameterTuningJobName"] if jobs else None
    except ClientError as e:
        print(f"⚠️ Could not list tuning jobs: {e}", file=sys.stderr)
        return None

def endpoint_status(name: str) -> Optional[str]:
    try:
        return SM.describe_endpoint(EndpointName=name)["EndpointStatus"]
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            return None
        raise

def wait_endpoint_in_service(name: str, timeout_s: int = 30 * 60):
    """Wait until endpoint is InService, else raise."""
    start = time.time()
    print(f"⏳ Waiting for endpoint {name} to be InService ...")
    while True:
        status = endpoint_status(name)
        print(f"   • status={status or 'NotCreated'}")
        if status == "InService":
            print("✅ Endpoint is InService.")
            return
        if status in {"Failed", "OutOfService"}:
            reason = SM.describe_endpoint(EndpointName=name).get("FailureReason", "Unknown")
            raise RuntimeError(f"❌ Endpoint entered terminal state: {status}. Reason: {reason}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"❌ Timed out waiting for endpoint {name} to be InService.")
        time.sleep(20)

def _read_text_s3(s3_uri: str) -> str:
    # expects s3://bucket/key
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {s3_uri}")
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    obj = S3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def _load_feature_list_for_model_tag(endpoint_name: str) -> List[str]:
    """
    Reads FeaturesListS3Key from the endpoint's model tag (as your logs show),
    then loads that text file from S3 into a list of column names.
    """
    # endpoint -> endpoint-config -> model(s) -> model package tags
    e = SM.describe_endpoint(EndpointName=endpoint_name)
    econf_name = e["EndpointConfigName"]
    econf = SM.describe_endpoint_config(EndpointConfigName=econf_name)
    model_names = [pv["ModelName"] for pv in econf["ProductionVariants"]]
    if not model_names:
        raise RuntimeError("No models found on endpoint.")

    # read tag from the first model
    m = SM.describe_model(ModelName=model_names[0])
    tags = SM.list_tags(ResourceArn=m["ModelArn"]).get("Tags", [])
    tag_map = {t["Key"]: t["Value"] for t in tags}
    s3_key = tag_map.get("FeaturesListS3Key")
    if not s3_key:
        raise RuntimeError("Model tag 'FeaturesListS3Key' not found.")
    s3_uri = f"s3://{BUCKET}/{s3_key}"

    txt = _read_text_s3(s3_uri)
    cols = [line.strip() for line in txt.splitlines() if line.strip()]
    if not cols:
        raise RuntimeError(f"Empty features file at {s3_uri}")
    print(f"📌 Loaded serving schema ({len(cols)} features) from {s3_uri}")
    return cols

def smoke_invoke(endpoint_name: str, rows: int = 1):
    """
    Build a tiny CSV payload with correct number/order of features,
    send to the endpoint, and print the prediction.
    """
    cols = _load_feature_list_for_model_tag(endpoint_name)
    # simple payload: all zeros, with a few plausible non-zeros to avoid degenerate input
    row = [0.0] * len(cols)
    payload_lines = []
    for i in range(rows):
        payload_lines.append(",".join(str(x) for x in row))
    payload = "\n".join(payload_lines)

    print(f"🫖 Smoke test payload: {rows} row(s) × {len(cols)} cols")
    try:
        res = SM_RT.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload.encode("utf-8"),
        )
        body = res["Body"].read().decode("utf-8").strip()
        print("🔎 Smoke prediction(s):")
        print(body[:500] + ("..." if len(body) > 500 else ""))
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"Smoke test invocation failed: {e}")

def cleanup_endpoint(endpoint_name: str):
    """Delete endpoint, endpoint-config, and all referenced models."""
    try:
        desc = SM.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"ℹ️ Endpoint not found: {endpoint_name}")
            return
        raise

    econf_name = desc["EndpointConfigName"]
    econf = SM.describe_endpoint_config(EndpointConfigName=econf_name)
    model_names = [pv["ModelName"] for pv in econf["ProductionVariants"]]

    print(f"🧹 Deleting endpoint: {endpoint_name}")
    SM.delete_endpoint(EndpointName=endpoint_name)

    print(f"🧹 Deleting endpoint-config: {econf_name}")
    SM.delete_endpoint_config(EndpointConfigName=econf_name)

    for mn in model_names:
        print(f"🧹 Deleting model: {mn}")
        SM.delete_model(ModelName=mn)

    print("✅ Cleanup completed.")

# -------- Banner / sanity info --------
print("🔍 Installed pandas version:")
subprocess.run(["pip", "show", "pandas"])

print("\n🧭 Pipeline configuration:")
print(f"• PIPELINE_MODE={PIPELINE_MODE}")
print(f"• AWS_REGION={AWS_REGION}  • BUCKET={BUCKET}  • PREFIX={PREFIX}  • ENDPOINT={ENDPOINT}\n")

# -------- Modeed pipeline --------
try:
    if PIPELINE_MODE == "full_train":
        print("🚀 Starting FULL TRAIN → DEPLOY → PREDICT\n")
        run_step("preprocessing/data_engineering.py")
        run_step("preprocessing/feature_selection.py")
        run_step("preprocessing/run_tuning.py")

        job = PINNED_TUNING_JOB or latest_completed_tuning_job()
        if not job:
            raise SystemExit("❌ No Completed tuning job found. Set TUNING_JOB_NAME or ensure run_tuning.py produced one.")
        run_step("preprocessing/deploy_best_xgb.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
            "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
        })
        wait_endpoint_in_service(ENDPOINT)

        run_step("preprocessing/predict_from_endpoint.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX, "ENDPOINT": ENDPOINT
        })

    elif PIPELINE_MODE == "deploy_and_predict":
        print("🚀 Starting DEPLOY → PREDICT\n")
        job = PINNED_TUNING_JOB or latest_completed_tuning_job()
        if not job:
            raise SystemExit("❌ No Completed tuning job found. Set TUNING_JOB_NAME or run full_train first.")
        run_step("preprocessing/deploy_best_xgb.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
            "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
        })
        wait_endpoint_in_service(ENDPOINT)

        run_step("preprocessing/predict_from_endpoint.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX, "ENDPOINT": ENDPOINT
        })

    elif PIPELINE_MODE == "deploy_only":
        print("🚀 Starting DEPLOY ONLY\n")
        job = PINNED_TUNING_JOB or latest_completed_tuning_job()
        if not job:
            raise SystemExit("❌ No Completed tuning job found. Set TUNING_JOB_NAME or run full_train first.")
        run_step("preprocessing/deploy_best_xgb.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
            "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
        })
        wait_endpoint_in_service(ENDPOINT)

    elif PIPELINE_MODE == "predict_only":
        print("🚀 Starting PREDICT ONLY\n")
        status = endpoint_status(ENDPOINT)
        if status != "InService":
            raise SystemExit("❌ predict_only selected but endpoint is not InService. Use deploy_and_predict.")
        run_step("preprocessing/predict_from_endpoint.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX, "ENDPOINT": ENDPOINT
        })

    elif PIPELINE_MODE == "smoke":
        print("🚀 Starting SMOKE TEST\n")
        # If endpoint missing or not InService, deploy first from best job
        status = endpoint_status(ENDPOINT)
        if status != "InService":
            job = PINNED_TUNING_JOB or latest_completed_tuning_job()
            if not job:
                raise SystemExit("❌ No Completed tuning job found. Set TUNING_JOB_NAME or run full_train first.")
            run_step("preprocessing/deploy_best_xgb.py", {
                "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
                "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
            })
            wait_endpoint_in_service(ENDPOINT)
        smoke_invoke(ENDPOINT, rows=SMOKE_ROWS)

    elif PIPELINE_MODE == "cleanup":
        print("🧹 Starting CLEANUP\n")
        cleanup_endpoint(ENDPOINT)

    else:
        raise SystemExit(f"❌ Unknown PIPELINE_MODE: {PIPELINE_MODE}")

    print("\n🏁 Pipeline complete.\n")

except Exception as e:
    print(f"\n💥 Pipeline error: {e}", file=sys.stderr)
    sys.exit(1)
