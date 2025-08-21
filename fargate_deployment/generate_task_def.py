#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
from pathlib import Path

ECR_IMAGE_RE = re.compile(
    r"^\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+:[A-Za-z0-9._-]+$"
)

RUNTIME_ENV_KEYS = [
    # Core AWS / app settings used by your scripts
    "AWS_REGION",
    "BUCKET",
    "PREFIX",
    "INPUT_PREFIX",
    "ENDPOINT",
    "LABEL_COL",
    "FEATURE_LIST_KEY",
    "SELECTED_FEATURES_FILE",
    "FILTERED_INPUT_FILE",
    "XGB_OUTPUT_PREFIX",
    "SAGEMAKER_TRAINING_ROLE",
    # Optional: leave these if you want them visible inside the container
    "ECS_CLUSTER_NAME",
    "TASK_FAMILY",
]

def load_env(env_path: str) -> dict:
    env = {}
    with open(env_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = os.path.expandvars(v.strip())
            # Remove surrounding quotes if user added them by accident
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            env[k] = v
    return env

def resolve_image_uri(script_dir: str, env: dict) -> str:
    # Prefer a file dropped by the build step (exact tag)
    last = Path(script_dir) / ".last_image_uri"
    if last.exists():
        img = last.read_text().strip()
        if img:
            return img
    # Fallback to IMAGE_URI from env (deploy script sets this)
    return env.get("IMAGE_URI", "")

def upsert_env(obj: dict, name: str, value: str):
    """Upsert one env var into the first container's environment array."""
    if "containerDefinitions" not in obj or not obj["containerDefinitions"]:
        sys.exit("❌ Task def JSON missing containerDefinitions[0].")
    cdef = obj["containerDefinitions"][0]
    env_list = cdef.get("environment") or []
    # remove existing with same name
    env_list = [e for e in env_list if e.get("name") != name]
    env_list.append({"name": name, "value": value})
    cdef["environment"] = env_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Path to task-def template JSON")
    ap.add_argument("--out", required=True, help="Path to write the final task-def JSON")
    ap.add_argument("--env", default=str(Path(__file__).with_name("config.env")), help="Path to config.env")
    args = ap.parse_args()

    script_dir = str(Path(__file__).parent)
    env = load_env(args.env)

    # Load template text and perform simple token replacements (backwards compatible)
    with open(args.template, "r") as f:
        tpl = f.read()

    image_uri = resolve_image_uri(script_dir, env)
    if not image_uri or "REPLACE_" in image_uri:
        sys.exit("❌ IMAGE_URI is empty or a placeholder. Run build_and_push.sh / deploy script first.")

    if not ECR_IMAGE_RE.match(image_uri):
        sys.exit(
            f"❌ IMAGE_URI looks invalid for ECS/ECR:\n  {image_uri}\n"
            "Expected: <acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>"
        )

    replacements = {
        "REPLACE_TASK_FAMILY": env.get("TASK_FAMILY", "diabetes"),
        "REPLACE_EXEC_ROLE": env["TASK_EXECUTION_ROLE"],
        "REPLACE_TASK_ROLE": env["TASK_ROLE"],
        "REPLACE_IMAGE_URI": image_uri,
        "REPLACE_LOG_GROUP": env["LOG_GROUP"],
        "REPLACE_LOG_STREAM_PREFIX": env["LOG_STREAM_PREFIX"],
        "REPLACE_AWS_REGION": env["AWS_REGION"],
        "REPLACE_BUCKET": env["BUCKET"],
        "REPLACE_PREFIX": env["PREFIX"],
        "REPLACE_SELECTED_FEATURES_FILE": env.get("SELECTED_FEATURES_FILE", "selected_features.csv"),
        "REPLACE_FILTERED_INPUT_FILE": env.get("FILTERED_INPUT_FILE", "5_perc.csv"),
        "REPLACE_XGB_OUTPUT_PREFIX": env.get("XGB_OUTPUT_PREFIX", "xgb_output"),
        "REPLACE_SAGEMAKER_TRAINING_ROLE": env["SAGEMAKER_TRAINING_ROLE"],
    }

    for key, val in replacements.items():
        tpl = tpl.replace(key, val)

    # Parse JSON and validate image
    try:
        obj = json.loads(tpl)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Task-def template after replacements is not valid JSON: {e}")

    try:
        img_in_json = obj["containerDefinitions"][0]["image"]
    except Exception as e:
        sys.exit(f"❌ Could not read image from task def JSON: {e}")

    if not ECR_IMAGE_RE.match(img_in_json):
        sys.exit(f"❌ task-def image is invalid:\n  {img_in_json}")

    # ---- Upsert runtime env vars so the container always has them ----
    injected = []
    for k in RUNTIME_ENV_KEYS:
        v = env.get(k)
        if v is None or v == "":
            # Skip silently if not provided; not all are mandatory
            continue
        upsert_env(obj, k, v)
        injected.append(k)

    # Write out final task-def
    out_path = Path(args.out)
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"✅ Wrote {out_path}")
    print(f"🖼️  Image: {img_in_json}")
    if injected:
        print("🔧 Injected env vars into container:")
        for k in injected:
            print(f"   - {k}={env.get(k)}")
    else:
        print("ℹ️ No additional env vars injected (RUNTIME_ENV_KEYS absent in config.env)")

if __name__ == "__main__":
    main()
