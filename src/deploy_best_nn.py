#!/usr/bin/env python3
# deploy_best_nn.py  — drop-in
import os, sys, time, io, tarfile, tempfile, shutil
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
BUCKET      = os.getenv("BUCKET", "diabetes-directory")
PREFIX      = os.getenv("PREFIX", "02_engineered")
ENDPOINT_NN = os.getenv("ENDPOINT_NN", "diabetes-nn-endpoint")

# Endpoint instance type
DEPLOY_INSTANCE_TYPE = os.getenv("DEPLOY_INSTANCE_TYPE_NN", "ml.m5.large")
# Role with SageMaker permissions
ROLE_ARN = os.getenv("SAGEMAKER_TRAINING_ROLE") or os.getenv("SAGEMAKER_ROLE")

SM = boto3.client("sagemaker", region_name=AWS_REGION)
S3 = boto3.client("s3", region_name=AWS_REGION)

FEATURES_USED_LATEST_KEY = f"{PREFIX}/features_used_latest.txt"
SELECTED_FEATURES_FALLBACK = f"{PREFIX}/selected_features.csv"
ENDPOINT_FEATURES_LATEST_KEY = f"{PREFIX}/endpoint_features_latest_nn.txt"  # optional NN pointer

# -------------------------------
# Helpers for tuner / model data
# -------------------------------
def _latest_completed_tf_tuning_job() -> Optional[str]:
    try:
        resp = SM.list_hyper_parameter_tuning_jobs(
            StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending", MaxResults=50
        )
        for s in resp.get("HyperParameterTuningJobSummaries", []):
            name = s["HyperParameterTuningJobName"]
            if name.startswith("tensorflow-training-"):
                return name
        return None
    except ClientError as e:
        print(f"⚠️ Could not list tuning jobs: {e}", file=sys.stderr)
        return None

def _best_training_job_from_tuner(tuner_name: str) -> str:
    desc = SM.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner_name)
    best = desc.get("BestTrainingJob", {})
    name = best.get("TrainingJobName")
    if not name:
        raise SystemExit(f"❌ Tuning job {tuner_name} has no BestTrainingJob yet.")
    return name

def _model_artifacts_from_training_job(job_name: str) -> str:
    tj = SM.describe_training_job(TrainingJobName=job_name)
    return tj["ModelArtifacts"]["S3ModelArtifacts"]

# -------------------------------
# S3 utils
# -------------------------------
def _s3_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://")
    p = uri[5:]
    bkt, key = p.split("/", 1)
    return bkt, key

def _download_and_extract_model_tar(s3_uri: str, workdir: str) -> str:
    bkt, key = _parse_s3_uri(s3_uri)
    local_tar = os.path.join(workdir, "model.tar.gz")
    S3.download_file(bkt, key, local_tar)
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
    return extract_dir

def _find_any_saved_model_dir(root_dir: str) -> Optional[str]:
    for r, _, files in os.walk(root_dir):
        if "saved_model.pb" in files:
            return r
    return None

def _find_keras_file(root_dir: str) -> Optional[str]:
    for r, _, files in os.walk(root_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".keras") or lf.endswith(".h5"):
                return os.path.join(r, f)
    return None

def _tar_dir(src_dir: str, out_tar_path: str):
    with tarfile.open(out_tar_path, "w:gz") as tar:
        for root, dirs, files in os.walk(src_dir):
            for d in dirs:
                full = os.path.join(root, d)
                arc = os.path.relpath(full, src_dir)
                tar.add(full, arcname=arc)
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, src_dir)
                tar.add(full, arcname=arc)

def _tree_preview(path: str, limit: int = 200) -> str:
    lines = []
    for root, dirs, files in os.walk(path):
        depth = len(os.path.relpath(root, path).split(os.sep))
        indent = "  " * max(0, depth)
        lines.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            lines.append(f"{indent}  {f}")
            if len(lines) >= limit:
                return "\n".join(lines)
    return "\n".join(lines)

def _ensure_savedmodel_in_s3(original_s3: str) -> str:
    """
    Ensure model.tar.gz contains TF-Serving layout under model/1/.
    If not, repackage (from any found SavedModel path or from a Keras file).
    """
    print(f"🔎 Inspecting model artifacts: {original_s3}")
    with tempfile.TemporaryDirectory() as td:
        extracted = _download_and_extract_model_tar(original_s3, td)

        sm_dir = _find_any_saved_model_dir(extracted)
        if sm_dir:
            print(f"✅ Found SavedModel at: {os.path.relpath(sm_dir, extracted)}")
            repkg_root = os.path.join(td, "repkg")
            os.makedirs(os.path.join(repkg_root, "model"), exist_ok=True)
            shutil.copytree(sm_dir, os.path.join(repkg_root, "model", "1"))
            out_tar = os.path.join(td, "model_savedmodel.tar.gz")
            _tar_dir(repkg_root, out_tar)

            bkt, key = _parse_s3_uri(original_s3)
            base, _ = os.path.splitext(key)      # .../model.tar
            base, _ = os.path.splitext(base)     # .../model
            new_key = f"{base}-savedmodel.tar.gz"
            print(f"☁️ Uploading repackaged SavedModel to s3://{bkt}/{new_key}")
            S3.upload_file(out_tar, bkt, new_key)
            return f"s3://{bkt}/{new_key}"

        keras_path = _find_keras_file(extracted)
        if keras_path:
            print(f"🔄 Converting Keras → SavedModel from: {os.path.relpath(keras_path, extracted)}")
            try:
                import tensorflow as tf
                from tensorflow import keras
            except Exception as e:
                raise SystemExit(
                    f"❌ TensorFlow not available in deploy image for conversion: {e}\n"
                    "Add 'tensorflow>=2.12' to the Fargate image, or export SavedModel during training."
                )
            model = keras.models.load_model(keras_path)
            repkg_root = os.path.join(td, "repkg")
            os.makedirs(os.path.join(repkg_root, "model", "1"), exist_ok=True)
            tf.saved_model.save(model, os.path.join(repkg_root, "model", "1"))
            out_tar = os.path.join(td, "model_savedmodel.tar.gz")
            _tar_dir(repkg_root, out_tar)

            bkt, key = _parse_s3_uri(original_s3)
            base, _ = os.path.splitext(key)
            base, _ = os.path.splitext(base)
            new_key = f"{base}-savedmodel.tar.gz"
            print(f"☁️ Uploading converted SavedModel to s3://{bkt}/{new_key}")
            S3.upload_file(out_tar, bkt, new_key)
            return f"s3://{bkt}/{new_key}"

        preview = _tree_preview(extracted)
        raise SystemExit(
            "❌ No SavedModel and no Keras file found in model.tar.gz.\n"
            "Expected either any '<dir>/saved_model.pb' or a '*.keras'/'*.h5' file.\n"
            "Artifact contents:\n" + preview
        )

# -------------------------------
# Endpoint helpers
# -------------------------------
def _wait_endpoint_status(name: str, target="InService", timeout_s=45*60):
    start = time.time()
    while True:
        try:
            st = SM.describe_endpoint(EndpointName=name)["EndpointStatus"]
        except ClientError as e:
            if "Could not find endpoint" in str(e):
                st = None
            else:
                raise
        print(f"⏳ Endpoint {name} status: {st or 'NotCreated'}")
        if st == target:
            return
        if st in {"Failed", "OutOfService"}:
            fr = SM.describe_endpoint(EndpointName=name).get("FailureReason", "Unknown")
            raise SystemExit(f"❌ Endpoint {name} entered terminal state {st}. Reason: {fr}")
        if time.time() - start > timeout_s:
            raise SystemExit(f"❌ Timed out waiting for endpoint {name} -> {target}.")
        time.sleep(20)

def _get_endpoint_status(name: str) -> Optional[str]:
    try:
        return SM.describe_endpoint(EndpointName=name)["EndpointStatus"]
    except ClientError as e:
        if "Could not find endpoint" in str(e) or "ValidationException" in str(e):
            return None
        raise

def _wait_until_not_updating(name: str, timeout_s: int = 30*60):
    start = time.time()
    while True:
        st = _get_endpoint_status(name)
        if st is None:
            return  # doesn't exist
        if st not in {"Creating", "Updating"}:
            return
        print(f"⏳ Endpoint {name} currently {st} — waiting to update…")
        if time.time() - start > timeout_s:
            raise SystemExit(f"❌ Timed out waiting for endpoint {name} to leave Creating/Updating.")
        time.sleep(20)

def _resolve_tf_inference_image() -> str:
    from sagemaker import image_uris
    return image_uris.retrieve(
        framework="tensorflow",
        region=AWS_REGION,
        version="2.13",
        instance_type=DEPLOY_INSTANCE_TYPE,
        image_scope="inference",
    )

def _ensure_model(endpoint_name: str, model_data_s3: str, role_arn: str) -> str:
    """
    Create a SageMaker Model using TF-Serving DLC and return its name.
    """
    from sagemaker.model import Model

    if not role_arn:
        raise SystemExit("❌ SAGEMAKER_TRAINING_ROLE (or SAGEMAKER_ROLE) not set in env.")

    model_data_s3 = _ensure_savedmodel_in_s3(model_data_s3)
    image_uri = _resolve_tf_inference_image()

    print(f"📦 Using model data: {model_data_s3}")
    print(f"🔐 Using role: {role_arn}")
    print(f"🖼️ Inference image: {image_uri}")

    model = Model(
        image_uri=image_uri,
        model_data=model_data_s3,
        role=role_arn,
        env={},
    )

    model_name = f"{endpoint_name}-model"
    try:
        SM.describe_model(ModelName=model_name)
        print(f"🧹 Deleting existing model: {model_name}")
        SM.delete_model(ModelName=model_name)
    except ClientError:
        pass

    print(f"🧬 Creating model: {model_name}")
    model.name = model_name
    model.create()

    # ------ NEW: tag model with FeaturesListS3Key ------
    # Prefer features_used_latest.txt; else selected_features.csv
    features_key = FEATURES_USED_LATEST_KEY if _s3_exists(BUCKET, FEATURES_USED_LATEST_KEY) else SELECTED_FEATURES_FALLBACK
    try:
        desc = SM.describe_model(ModelName=model_name)
        SM.add_tags(
            ResourceArn=desc["ModelArn"],
            Tags=[{"Key": "FeaturesListS3Key", "Value": features_key}],
        )
        print(f"🏷️ Tagged model with FeaturesListS3Key={features_key}")
        # Optional pointer for NN
        S3.put_object(Bucket=BUCKET, Key=ENDPOINT_FEATURES_LATEST_KEY, Body=features_key.encode("utf-8"))
        print(f"📌 Updated NN endpoint features pointer: s3://{BUCKET}/{ENDPOINT_FEATURES_LATEST_KEY}")
    except Exception as e:
        print(f"⚠️ Could not tag model/publish pointer: {e}")

    return model_name

def _create_or_update_endpoint(endpoint_name: str, model_name: str):
    config_name = f"{endpoint_name}-config"
    variant = {
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": DEPLOY_INSTANCE_TYPE,
        "InitialVariantWeight": 1.0,
    }

    try:
        SM.describe_endpoint_config(EndpointConfigName=config_name)
        print(f"🧹 Deleting existing endpoint-config: {config_name}")
        SM.delete_endpoint_config(EndpointConfigName=config_name)
    except ClientError:
        pass

    print(f"🛠️ Creating endpoint-config: {config_name}")
    SM.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=[variant])

    status = _get_endpoint_status(endpoint_name)
    if status is None:
        print(f"🆕 Creating endpoint: {endpoint_name}")
        SM.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        return

    print(f"🔁 Updating endpoint: {endpoint_name} (current status: {status})")
    _wait_until_not_updating(endpoint_name)
    SM.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

# -------------------------------
# Main
# -------------------------------
def main():
    print(f"🧭 NN Deploy Config -> region={AWS_REGION} bucket={BUCKET} prefix={PREFIX} endpoint={ENDPOINT_NN}")
    tuner = _latest_completed_tf_tuning_job()
    if not tuner:
        raise SystemExit("❌ No completed TensorFlow tuning jobs found (prefix 'tensorflow-training-').")

    best_job = _best_training_job_from_tuner(tuner)
    print(f"🏆 Best training job from {tuner}: {best_job}")
    model_s3 = _model_artifacts_from_training_job(best_job)

    model_name = _ensure_model(ENDPOINT_NN, model_s3, ROLE_ARN)
    _create_or_update_endpoint(ENDPOINT_NN, model_name)
    _wait_endpoint_status(ENDPOINT_NN, "InService", timeout_s=45*60)
    print(f"✅ NN endpoint ready: {ENDPOINT_NN}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 deploy_best_nn.py error: {e}", file=sys.stderr)
        sys.exit(1)
