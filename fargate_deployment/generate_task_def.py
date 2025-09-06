#!/usr/bin/env python3
import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict

# ECR image sanity check
ECR_IMAGE_RE = re.compile(
    r"^\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+:[A-Za-z0-9._-]+$"
)

# Legacy placeholder -> env var mapping (only used if the placeholders exist)
PLACEHOLDER_KEYS = {
    "REPLACE_TASK_FAMILY": "TASK_FAMILY",
    "REPLACE_EXEC_ROLE": "TASK_EXECUTION_ROLE",
    "REPLACE_TASK_ROLE": "TASK_ROLE",
    "REPLACE_IMAGE_URI": "IMAGE_URI",
    "REPLACE_LOG_GROUP": "LOG_GROUP",
    "REPLACE_LOG_STREAM_PREFIX": "LOG_STREAM_PREFIX",
    "REPLACE_AWS_REGION": "AWS_REGION",
    "REPLACE_BUCKET": "BUCKET",
    "REPLACE_PREFIX": "PREFIX",
    "REPLACE_SELECTED_FEATURES_FILE": "SELECTED_FEATURES_FILE",
    "REPLACE_FILTERED_INPUT_FILE": "FILTERED_INPUT_FILE",
    "REPLACE_XGB_OUTPUT_PREFIX": "XGB_OUTPUT_PREFIX",
    "REPLACE_SAGEMAKER_TRAINING_ROLE": "SAGEMAKER_TRAINING_ROLE",
}

def load_env(env_path: str) -> Dict[str, str]:
    env = {}
    with open(env_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            # expand ${VAR} in values
            v = os.path.expandvars(v.strip())
            # strip surrounding quotes
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if v != "":
                env[k] = v
    return env

def resolve_image_uri(script_dir: str, env: Dict[str, str]) -> str:
    # 1) file dropped by build/push step
    last = Path(script_dir) / ".last_image_uri"
    if last.exists():
        img = last.read_text().strip()
        if img:
            return img
    # 2) IMAGE_URI in env
    img = env.get("IMAGE_URI", "")
    if img:
        return img
    # 3) ECR_REPO_URI + IMAGE_TAG
    repo = env.get("ECR_REPO_URI", "")
    tag = env.get("IMAGE_TAG", "")
    if repo and tag:
        return f"{repo}:{tag}"
    return ""

def upsert_env(obj: dict, name: str, value: str):
    if "containerDefinitions" not in obj or not obj["containerDefinitions"]:
        sys.exit("❌ Task def JSON missing containerDefinitions[0].")
    cdef = obj["containerDefinitions"][0]
    env_list = cdef.get("environment") or []
    # remove existing entry with same name
    env_list = [e for e in env_list if e.get("name") != name]
    env_list.append({"name": name, "value": value})
    cdef["environment"] = env_list

def ensure_logging(obj: dict, log_group: str, log_region: str, log_prefix: str):
    cdef = obj["containerDefinitions"][0]
    cdef.setdefault("logConfiguration", {
        "logDriver": "awslogs",
        "options": {
            "awslogs-group": log_group,
            "awslogs-region": log_region,
            "awslogs-stream-prefix": log_prefix
        }
    })

def ensure_core_fields(obj: dict, family: str, exec_role: str, task_role: str):
    obj["family"] = family or obj.get("family", "diabetes")
    if exec_role:
        obj["executionRoleArn"] = exec_role
    if task_role:
        obj["taskRoleArn"] = task_role
    obj.setdefault("requiresCompatibilities", ["FARGATE"])
    obj.setdefault("networkMode", "awsvpc")

def apply_placeholder_replacements(tpl_text: str, env: Dict[str, str]) -> str:
    for placeholder, env_key in PLACEHOLDER_KEYS.items():
        if placeholder in tpl_text and env_key in env:
            tpl_text = tpl_text.replace(placeholder, env[env_key])
    return tpl_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Path to task-def template JSON")
    ap.add_argument("--out", required=True, help="Path to write the final task-def JSON")
    ap.add_argument("--env", default=str(Path(__file__).with_name("config.env")), help="Path to config.env")
    args = ap.parse_args()

    script_dir = str(Path(__file__).parent)
    env = load_env(args.env)

    # Read template as text so we can substitute REPLACE_* tokens
    tpl_text = Path(args.template).read_text()

    # Compute image URI (compatible with both old/new flows)
    image_uri = resolve_image_uri(script_dir, env)

    expects_image_placeholder = "REPLACE_IMAGE_URI" in tpl_text
    if expects_image_placeholder:
        if not image_uri or "REPLACE" in image_uri:
            sys.exit("❌ IMAGE_URI is empty or a placeholder. Build/push or set IMAGE_URI/ECR_REPO_URI+IMAGE_TAG first.")
        if not ECR_IMAGE_RE.match(image_uri):
            sys.exit(
                f"❌ IMAGE_URI looks invalid for ECS/ECR:\n  {image_uri}\n"
                "Expected: <acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>"
            )
        env = {**env, **{"IMAGE_URI": image_uri}}

    # Replace placeholders (only those present)
    tpl_text = apply_placeholder_replacements(tpl_text, env)

    # Parse resulting JSON
    try:
        obj = json.loads(tpl_text)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Task-def template after replacements is not valid JSON: {e}")

    # Validate existing image (best-effort)
    try:
        img_in_json = obj["containerDefinitions"][0]["image"]
    except Exception as e:
        sys.exit(f"❌ Could not read image from task def JSON: {e}")

    if isinstance(img_in_json, str) and img_in_json and "REPLACE" not in img_in_json:
        if img_in_json.startswith(("http", "https")):
            sys.exit(f"❌ task-def image should be an ECR URI, not a web URL: {img_in_json}")
        if img_in_json != "REPLACED_BY_SCRIPT" and not ECR_IMAGE_RE.match(img_in_json):
            sys.exit(f"❌ task-def image is invalid:\n  {img_in_json}")

    # Ensure family/roles/logging defaults
    ensure_core_fields(
        obj,
        family=env.get("TASK_FAMILY", obj.get("family", "diabetes")),
        exec_role=env.get("TASK_EXECUTION_ROLE", obj.get("executionRoleArn", "")),
        task_role=env.get("TASK_ROLE", obj.get("taskRoleArn", "")),
    )
    ensure_logging(
        obj,
        log_group=env.get("LOG_GROUP", "/ecs/diabetes"),
        log_region=env.get("AWS_REGION", "us-east-1"),
        log_prefix=env.get("LOG_STREAM_PREFIX", "ecs"),
    )

    # 🔑 Inject EVERY non-empty key from config.env into the container environment
    injected = []
    for k, v in env.items():
        if v is None or v == "":
            continue
        upsert_env(obj, k, v)
        injected.append(k)

    # Write final task definition
    out_path = Path(args.out)
    out_path.write_text(json.dumps(obj, indent=2))

    # Friendly output
    container_name = obj["containerDefinitions"][0].get("name", "diabetes")
    final_image = obj["containerDefinitions"][0].get("image", "")
    print("aws ecs register-task-definition \\")
    print(f"  --cli-input-json file://{out_path} \\")
    print(f"  --region {env.get('AWS_REGION','us-east-1')}\n")
    print(f"✅ Wrote {out_path}")
    print(f"🧩 Family: {obj.get('family')}  •  Container: {container_name}")
    print(f"🖼️  Image (pre-jq): {final_image}")
    if injected:
        print("🔧 Injected env vars into container:")
        for k in sorted(injected):
            print(f"   - {k}={env.get(k)}")
    else:
        print("ℹ️ No env vars injected (env file empty or all values blank).")

if __name__ == "__main__":
    main()
