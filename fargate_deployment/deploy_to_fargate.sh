#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths & config.env
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CANDIDATES=(
  "$SCRIPT_DIR/config.env"
  "$ROOT_DIR/fargate_deployment/config.env"
  "$ROOT_DIR/config.env"
  "$(pwd)/config.env"
)
CONFIG_PATH=""
for p in "${CANDIDATES[@]}"; do
  [[ -f "$p" ]] && CONFIG_PATH="$p" && break
done
if [[ -z "$CONFIG_PATH" ]]; then
  echo "❌ Could not find config.env"
  printf '   - %s\n' "${CANDIDATES[@]}"
  exit 1
fi
echo "📄 Using config: $CONFIG_PATH"

# --- Load config.env WITHOUT clobbering already-set env vars ---
load_env_if_unset() {
  local file="$1"
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    # strip comments/blank
    local line="${raw%%#*}"; line="$(echo "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    [[ "$line" != *"="* ]] && continue
    local k="${line%%=*}"; k="$(echo "$k" | xargs)"
    local v="${line#*=}"; v="$(echo "$v" | xargs)"
    # dequote
    if [[ ( "$v" == \"*\" && "$v" == *\" ) || ( "$v" == \'*\' && "$v" == *\' ) ]]; then
      v="${v:1:${#v}-2}"
    fi
    # expand ${VAR} in the value using current env
    v="$(eval "echo \"$v\"")"
    # only export if not already set in process env
    if [[ -z "${!k-}" ]]; then
      export "$k=$v"
    fi
  done < "$file"
}
load_env_if_unset "$CONFIG_PATH"

# =========================
# CLI prereqs
# =========================
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing command: $1"; exit 1; }; }
need_cmd aws
need_cmd docker
need_cmd jq
need_cmd python

# =========================
# Required env
# =========================
: "${AWS_REGION:?Missing AWS_REGION}"
: "${AWS_ACCOUNT_ID:?Missing AWS_ACCOUNT_ID}"
: "${ECR_REPO_NAME:?Missing ECR_REPO_NAME}"
: "${ECS_CLUSTER_NAME:?Missing ECS_CLUSTER_NAME}"
: "${LOG_GROUP:?Missing LOG_GROUP}"
: "${LOG_STREAM_PREFIX:?Missing LOG_STREAM_PREFIX}"
: "${FARGATE_SUBNET_IDS:?Missing FARGATE_SUBNET_IDS (comma-separated)}"
: "${FARGATE_SECURITY_GROUP_IDS:?Missing FARGATE_SECURITY_GROUP_IDS (comma-separated)}"
: "${ASSIGN_PUBLIC_IP:?Missing ASSIGN_PUBLIC_IP (ENABLED|DISABLED)}"

# Compute ECR URI
ECR_REPO_URI="${ECR_REPO_URI:-${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}}"

# =========================
# Optional knobs
# =========================
: "${USE_OVERRIDES:=true}"                  # pass container env from shell/config
: "${PIPELINE_MODE:=dual_tune}"             # can be overridden at invocation
: "${DEPLOY_AFTER_DUAL:=false}"
: "${NO_CACHE:=false}"
: "${SKIP_IMAGE_BUILD:=false}"              # skip docker build/push if true
: "${IMAGE_URI:=}"                          # if SKIP_IMAGE_BUILD=true, you may set a specific image uri
: "${OUTPUT_PREFIX:=03_scored}"
: "${ENDPOINT:=diabetes-xgb-endpoint}"      # XGB endpoint
: "${ENDPOINT_XGB:=$ENDPOINT}"              # alias for clarity in run_pipeline banner
: "${ENDPOINT_NN:=diabetes-nn-endpoint}"    # NN endpoint

# macOS bash 3.2 lowercase helper
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# =========================
# Ensure ECR repo exists
# =========================
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "🧫 Creating ECR repository: $ECR_REPO_NAME"
  aws ecr create-repository \
    --repository-name "$ECR_REPO_NAME" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null
else
  echo "✅ ECR repository exists: $ECR_REPO_NAME"
fi

# =========================
# Build/push image (or skip)
# =========================
BUILD_TAG="$(date +"%Y%m%d-%H%M%S")"
IMAGE_URI_TAGGED="${ECR_REPO_URI}:${BUILD_TAG}"
IMAGE_URI_LATEST="${ECR_REPO_URI}:latest"

if [[ "$(lower "$SKIP_IMAGE_BUILD")" == "true" ]]; then
  if [[ -z "$IMAGE_URI" ]]; then
    IMAGE_URI="$IMAGE_URI_LATEST"
  fi
  echo "⏭️  SKIP_IMAGE_BUILD=true — using existing image: $IMAGE_URI"
else
  echo "🔧 Building Docker image..."
  if [[ "$NO_CACHE" == "true" ]]; then
    docker build --no-cache -t "$ECR_REPO_NAME" "$ROOT_DIR"
  else
    docker build -t "$ECR_REPO_NAME" "$ROOT_DIR"
  fi
  docker tag "$ECR_REPO_NAME" "$IMAGE_URI_LATEST"
  docker tag "$ECR_REPO_NAME" "$IMAGE_URI_TAGGED"

  push_with_retry() {
    local tag="$1"
    local attempt=1 max_attempts=3
    while (( attempt <= max_attempts )); do
      echo "🔄 ECR login (attempt $attempt)..."
      aws ecr get-login-password --region "$AWS_REGION" \
        | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" && \
      echo "📤 Pushing $tag (attempt $attempt)..." && \
      docker push "$tag" && { echo "✅ Pushed $tag"; return 0; }
      echo "⚠️ Push failed. Retrying in 5s..."
      sleep 5
      ((attempt++))
    done
    echo "❌ Failed to push $tag"; exit 1
  }
  push_with_retry "$IMAGE_URI_LATEST"
  push_with_retry "$IMAGE_URI_TAGGED"
  IMAGE_URI="$IMAGE_URI_TAGGED"
fi

echo "🖼️ Using image: $IMAGE_URI"

# =========================
# Ensure ECS cluster exists
# =========================
if ! aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --region "$AWS_REGION" \
   --query 'clusters[?status==`ACTIVE`].[clusterName]' --output text | grep -qx "$ECS_CLUSTER_NAME"; then
  echo "🧩 Creating ECS cluster: $ECS_CLUSTER_NAME"
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null
else
  echo "✅ ECS cluster exists: $ECS_CLUSTER_NAME"
fi

# =========================
# Task definition
# =========================
echo "📝 Generating task definition..."
python "$SCRIPT_DIR/generate_task_def.py" \
  --template "$SCRIPT_DIR/task-def-template.json" \
  --out "$SCRIPT_DIR/final-task-def.json"

# Stamp the image
tmp="$SCRIPT_DIR/final-task-def.json.tmp"
jq --arg img "$IMAGE_URI" '.containerDefinitions[0].image = $img' \
  "$SCRIPT_DIR/final-task-def.json" > "$tmp" && mv "$tmp" "$SCRIPT_DIR/final-task-def.json"

IMG_IN_TD=$(jq -r '.containerDefinitions[0].image' "$SCRIPT_DIR/final-task-def.json")
[[ "$IMG_IN_TD" == "$IMAGE_URI" ]] || { echo "❌ Task-def image mismatch"; exit 1; }
echo "📦 Task-def image confirmed: $IMG_IN_TD"

REV_ARN=$(aws ecs register-task-definition \
  --cli-input-json "file://$SCRIPT_DIR/final-task-def.json" \
  --region "$AWS_REGION" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)
echo "📎 Registered task definition: $REV_ARN"

# =========================
# Networking
# =========================
csv_to_json_array() {
  local csv="$1"; IFS=',' read -r -a arr <<< "$csv"
  local out="["; local i
  for i in "${!arr[@]}"; do
    local v; v="$(echo "${arr[$i]}" | xargs)"
    out+="\"$v\""; (( i < ${#arr[@]}-1 )) && out+=","
  done; out+="]"; echo "$out"
}
SUBNETS_JSON=$(csv_to_json_array "$FARGATE_SUBNET_IDS")
SGS_JSON=$(csv_to_json_array "$FARGATE_SECURITY_GROUP_IDS")
NETWORK_CFG=$(jq -n --argjson subs "$SUBNETS_JSON" --argjson sgs "$SGS_JSON" --arg assign "$ASSIGN_PUBLIC_IP" \
  '{awsvpcConfiguration:{subnets:$subs,securityGroups:$sgs,assignPublicIp:$assign}}')

# =========================
# Container overrides
# =========================
build_overrides_json() {
  if [[ "$(lower "$USE_OVERRIDES")" == "false" ]]; then
    echo "{}"; return 0
  fi

  local names=(
    PIPELINE_MODE DEPLOY_AFTER_DUAL
    AWS_REGION BUCKET PREFIX OUTPUT_PREFIX
    ENDPOINT ENDPOINT_XGB ENDPOINT_NN
    SELECTED_FEATURES_FILE FILTERED_INPUT_FILE
    XGB_OUTPUT_PREFIX NN_OUTPUT_PREFIX NN_INSTANCE_TYPE
    SAGEMAKER_TRAINING_ROLE
    TUNING_JOB_NAME TUNING_JOB_FILE WAIT_FOR_TUNING
    TRAIN_KEY TEST_KEY FEATURE_LIST_KEY
  )

  local env_elems=()
  for var in "${names[@]}"; do
    if [[ -n "${!var-}" ]]; then
      env_elems+=("{\"name\":\"$var\",\"value\":\"${!var}\"}")
    fi
  done

  # EXTRA overrides win: concat then reverse|unique_by(.name)|reverse
  local env_json
  if [[ -n "${EXTRA_CONTAINER_ENV:-}" ]] && echo "$EXTRA_CONTAINER_ENV" | jq -e 'type=="array"' >/dev/null 2>&1; then
    env_json="$(
      printf '[%s]' "$(IFS=, ; echo "${env_elems[*]}")" \
      | jq --argjson extra "$EXTRA_CONTAINER_ENV" \
           '. + $extra | (reverse | unique_by(.name) | reverse)'
    )"
  else
    env_json="$(printf '[%s]' "$(IFS=, ; echo "${env_elems[*]}")")"
  fi

  local cname; cname="$(jq -r '.containerDefinitions[0].name' "$SCRIPT_DIR/final-task-def.json")"
  jq -n --argjson env "$env_json" --arg name "$cname" \
    '{containerOverrides:[{"name":$name,"environment":$env}]}'
}

OVERRIDES_JSON="$(build_overrides_json)"
if [[ "$(lower "$USE_OVERRIDES")" != "false" ]]; then
  echo "🧩 Using container overrides:"
  echo "$OVERRIDES_JSON" | jq .
else
  echo "ℹ️ Container overrides disabled (USE_OVERRIDES=false)"
fi

# =========================
# Run task & tail logs
# =========================
echo "🚀 Running task on Fargate..."
if [[ "$(lower "$USE_OVERRIDES")" != "false" ]]; then
  TASK_ARN=$(aws ecs run-task \
    --cluster "$ECS_CLUSTER_NAME" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_CFG" \
    --task-definition "$REV_ARN" \
    --region "$AWS_REGION" \
    --overrides "$OVERRIDES_JSON" \
    --query 'tasks[0].taskArn' --output text)
else
  TASK_ARN=$(aws ecs run-task \
    --cluster "$ECS_CLUSTER_NAME" \
    --launch-type FARGATE \
    --network-configuration "$NETWORK_CFG" \
    --task-definition "$REV_ARN" \
    --region "$AWS_REGION" \
    --query 'tasks[0].taskArn' --output text)
fi

if [[ -z "$TASK_ARN" || "$TASK_ARN" == "None" ]]; then
  echo "❌ Failed to start task"; exit 1
fi
echo "✅ Task launched: $TASK_ARN"

TASK_ID="${TASK_ARN##*/}"
CONTAINER_NAME=$(jq -r '.containerDefinitions[0].name' "$SCRIPT_DIR/final-task-def.json")
STREAM_PREFIX="${LOG_STREAM_PREFIX}/${CONTAINER_NAME}/${TASK_ID}"

echo "⏳ Waiting for log stream: $STREAM_PREFIX"
for i in {1..12}; do
  STREAM=$(aws logs describe-log-streams \
    --log-group-name "$LOG_GROUP" \
    --log-stream-name-prefix "$STREAM_PREFIX" \
    --max-items 1 \
    --query 'logStreams[0].logStreamName' \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || true)
  if [[ -n "$STREAM" && "$STREAM" != "None" ]]; then
    echo "📜 Tailing logs for stream: $STREAM"
    aws logs tail "$LOG_GROUP" --log-stream-names "$STREAM" --follow --region "$AWS_REGION"
    exit 0
  fi
  sleep 5
done

echo "⚠️ Log stream not found yet. You can try:"
echo "   aws logs tail \"$LOG_GROUP\" --since 10m --follow --region \"$AWS_REGION\""
