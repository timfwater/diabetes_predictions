#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from typing import Optional, List

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# -------- Config (env with sensible defaults) --------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET", "diabetes-directory")
PREFIX = os.getenv("PREFIX", "02_engineered")

# XGB endpoint (built-in algo path)
ENDPOINT = os.getenv("ENDPOINT", "diabetes-xgb-endpoint")
# NN endpoint (TF/Keras path)
ENDPOINT_NN = os.getenv("ENDPOINT_NN", "diabetes-nn-endpoint")

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "deploy_and_predict").lower()
PINNED_TUNING_JOB = os.getenv("TUNING_JOB_NAME", "").strip()

# Optional: in smoke mode, how many synthetic rows to send
SMOKE_ROWS = int(os.getenv("SMOKE_ROWS", "1"))

# Optional: after dual_tune, deploy XGB automatically?
DEPLOY_AFTER_DUAL = os.getenv("DEPLOY_AFTER_DUAL", "false").lower() == "true"

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
    """XGB HPO (built-in)"""
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

def wait_endpoint_in_service(name: str, timeout_s: int = 45 * 60):
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
    Reads FeaturesListS3Key from the endpoint's model tag, then loads the text file from S3.
    (Used by the smoke CSV builder for XGB endpoints.)
    """
    e = SM.describe_endpoint(EndpointName=endpoint_name)
    econf_name = e["EndpointConfigName"]
    econf = SM.describe_endpoint_config(EndpointConfigName=econf_name)
    model_names = [pv["ModelName"] for pv in econf["ProductionVariants"]]
    if not model_names:
        raise RuntimeError("No models found on endpoint.")

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
    row = [0.0] * len(cols)
    payload_lines = [",".join(str(x) for x in row) for _ in range(rows)]
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
print(f"• AWS_REGION={AWS_REGION}  • BUCKET={BUCKET}  • PREFIX={PREFIX}")
print(f"• ENDPOINT_XGB={ENDPOINT}  • ENDPOINT_NN={ENDPOINT_NN}\n")

# -------- Modeed pipeline --------
try:
    if PIPELINE_MODE == "full_train":
        print("🚀 Starting FULL TRAIN → DEPLOY → PREDICT\n")
        run_step("preprocessing/data_engineering.py")
        run_step("preprocessing/feature_selection.py")
        run_step("preprocessing/run_tuning.py")  # XGB

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

    # ---------------------------
    # dual_tune (XGB + NN)
    # ---------------------------
    elif PIPELINE_MODE == "dual_tune":
        print("🚀 Starting DUAL TUNE (XGB + NN)\n")
        run_step("preprocessing/feature_selection.py")
        run_step("preprocessing/run_tuning.py")      # XGB HPO
        run_step("preprocessing/run_tuning_nn.py")   # NN  HPO
        if DEPLOY_AFTER_DUAL:
            job = PINNED_TUNING_JOB or latest_completed_tuning_job()
            if not job:
                raise SystemExit("❌ No Completed tuning job found for deploy. Set TUNING_JOB_NAME or wait for XGB HPO.")
            run_step("preprocessing/deploy_best_xgb.py", {
                "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
                "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
            })
            wait_endpoint_in_service(ENDPOINT)
        print("\n✅ Dual tuning steps submitted. Monitor both HPO jobs in SageMaker.\n")

    # ---------------------------
    # nn-only modes
    # ---------------------------
    elif PIPELINE_MODE == "nn_tune_only":
        print("🚀 Starting NN TUNE ONLY (with feature_selection)\n")
        run_step("preprocessing/feature_selection.py")
        run_step("preprocessing/run_tuning_nn.py")
        print("\n✅ NN tuning submitted (with feature selection).\n")

    elif PIPELINE_MODE == "nn_tune_only_no_fs":
        print("🚀 Starting NN TUNE ONLY (skip feature_selection)\n")
        run_step("preprocessing/run_tuning_nn.py")
        print("\n✅ NN tuning submitted (no feature selection).\n")

    # ---------------------------
    # deploy+predict for NN and BOTH
    # ---------------------------
    elif PIPELINE_MODE == "nn_deploy_and_predict":
        print("🚀 NN DEPLOY → PREDICT\n")
        run_step("preprocessing/deploy_best_nn.py")
        wait_endpoint_in_service(ENDPOINT_NN)  # <-- ensure NN is ready
        run_step("preprocessing/predict_from_both.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX
        })

    elif PIPELINE_MODE == "both_deploy_and_predict":
        print("🚀 BOTH DEPLOY → PREDICT (XGB + NN)\n")
        job = PINNED_TUNING_JOB or latest_completed_tuning_job()
        if not job:
            raise SystemExit("❌ No Completed XGB tuning job found. Set TUNING_JOB_NAME or run XGB HPO.")
        run_step("preprocessing/deploy_best_xgb.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX,
            "ENDPOINT": ENDPOINT, "TUNING_JOB_NAME": job
        })
        wait_endpoint_in_service(ENDPOINT)
        run_step("preprocessing/deploy_best_nn.py")
        wait_endpoint_in_service(ENDPOINT_NN)  # <-- ensure NN is ready
        run_step("preprocessing/predict_from_both.py", {
            "AWS_REGION": AWS_REGION, "BUCKET": BUCKET, "PREFIX": PREFIX
        })

    else:
        raise SystemExit(f"❌ Unknown PIPELINE_MODE: {PIPELINE_MODE}")

    print("\n🏁 Pipeline complete.\n")

except Exception as e:
    print(f"\n💥 Pipeline error: {e}", file=sys.stderr)
    sys.exit(1)
