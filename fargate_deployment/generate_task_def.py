#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from dotenv import dotenv_values

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
TEMPLATE = BASE / "task-def-template.json"
OUTPUT = BASE / "final-task-def.json"
CONFIG_FILE = BASE / "config.env"

if not CONFIG_FILE.exists():
    sys.stderr.write(f"❌ {CONFIG_FILE} not found\n")
    sys.exit(1)

config = dotenv_values(CONFIG_FILE)

required_vars = [
    "AWS_REGION", "TASK_EXECUTION_ROLE", "TASK_ROLE",
    "ECR_REPO_URI", "LOG_GROUP", "LOG_STREAM_PREFIX",
    "TASK_FAMILY"
]
missing = [v for v in required_vars if not config.get(v)]
if missing:
    sys.stderr.write(f"❌ Missing vars in config.env: {missing}\n")
    sys.exit(1)

with open(TEMPLATE) as f:
    task_def = json.load(f)

task_def["family"] = config["TASK_FAMILY"]
task_def["executionRoleArn"] = config["TASK_EXECUTION_ROLE"]
task_def["taskRoleArn"] = config["TASK_ROLE"]

container = task_def["containerDefinitions"][0]
container["name"] = config["TASK_FAMILY"]
container["image"] = f"{config['ECR_REPO_URI']}:latest"
container["logConfiguration"]["options"]["awslogs-group"] = config["LOG_GROUP"]
container["logConfiguration"]["options"]["awslogs-region"] = config["AWS_REGION"]
container["logConfiguration"]["options"]["awslogs-stream-prefix"] = config["LOG_STREAM_PREFIX"]

# ⬇️ Add environment variables to pass into the container
env_vars_to_include = [
    "BUCKET",
    "PREFIX",
    "SELECTED_FEATURES_FILE",
    "FILTERED_INPUT_FILE",
    "XGB_OUTPUT_PREFIX",
    "SAGEMAKER_TRAINING_ROLE"
]

container["environment"] = [
    {"name": var, "value": config[var]}
    for var in env_vars_to_include
    if var in config and config[var]
]

with open(OUTPUT, "w") as out:
    json.dump(task_def, out, indent=2)

print(f"✅ Task definition generated: {OUTPUT}")
