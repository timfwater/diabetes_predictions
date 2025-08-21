#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Resolve paths & source config
# ---------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Robust config.env discovery (prefer alongside the script)
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
  echo "❌ Could not find config.env. Searched:"
  printf '   - %s\n' "${CANDIDATES[@]}"
  exit 1
fi
echo "📄 Using config: $CONFIG_PATH"

# Load and export config.env variables
set -a
# shellcheck disable=SC1090
source "$CONFIG_PATH"
set +a

# ---------------------------
# Preflight checks
# ---------------------------
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing required command: $1"; exit 1; }; }
need_cmd aws
need_cmd docker
need_cmd jq
need_cmd python

# Required env sanity
: "${AWS_REGION:?Missing AWS_REGION}"
: "${AWS_ACCOUNT_ID:?Missing AWS_ACCOUNT_ID}"
: "${ECR_REPO_NAME:?Missing ECR_REPO_NAME}"
: "${ECS_CLUSTER_NAME:?Missing ECS_CLUSTER_NAME}"
: "${LOG_GROUP:?Missing LOG_GROUP}"
: "${LOG_STREAM_PREFIX:?Missing LOG_STREAM_PREFIX}"
: "${FARGATE_SUBNET_IDS:?Missing FARGATE_SUBNET_IDS (comma-separated)}"
: "${FARGATE_SECURITY_GROUP_IDS:?Missing FARGATE_SECURITY_GROUP_IDS (comma-separated)}"
: "${ASSIGN_PUBLIC_IP:?Missing ASSIGN_PUBLIC_IP (ENABLED|DISABLED)}"

# Compute ECR_REPO_URI if not provided
ECR_REPO_URI="${ECR_REPO_URI:-${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}}"

# Optional knobs
: "${NO_CACHE:=false}"

# ---------------------------
# Ensure ECR repo exists
# ---------------------------
if ! aws ecr describe-repositories \
  --repository-names "$ECR_REPO_NAME" \
  --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "🧫 Creating ECR repository: $ECR_REPO_NAME"
  aws ecr create-repository \
    --repository-name "$ECR_REPO_NAME" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null
else
  echo "✅ ECR repository exists: $ECR_REPO_NAME"
fi

# ---------------------------
# Build & tag image
# ---------------------------
BUILD_TAG="$(date +"%Y%m%d-%H%M%S")"
IMAGE_LOCAL_TAG="$ECR_REPO_NAME"
IMAGE_URI_LATEST="${ECR_REPO_URI}:latest"
IMAGE_URI_TAGGED="${ECR_REPO_URI}:${BUILD_TAG}"

echo "🔧 Building Docker image..."
if [[ "$NO_CACHE" == "true" ]]; then
  docker build --no-cache -t "$IMAGE_LOCAL_TAG" "$ROOT_DIR"
else
  docker build -t "$IMAGE_LOCAL_TAG" "$ROOT_DIR"
fi
docker tag "$IMAGE_LOCAL_TAG" "$IMAGE_URI_LATEST"
docker tag "$IMAGE_LOCAL_TAG" "$IMAGE_URI_TAGGED"

# ---------------------------
# Login & push (with retry)
# ---------------------------
push_with_retry() {
  local full_tag="$1"
  local attempt=1 max_attempts=3
  while (( attempt <= max_attempts )); do
    echo "🔄 ECR login (attempt $attempt)..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" && \
    echo "📤 Pushing $full_tag (attempt $attempt)..." && \
    docker push "$full_tag" && { echo "✅ Pushed $full_tag"; return 0; }
    echo "⚠️ Push failed. Retrying in 5s..."
    sleep 5
    ((attempt++))
  done
  echo "❌ Failed to push $full_tag after $max_attempts attempts."
  exit 1
}
push_with_retry "$IMAGE_URI_LATEST"
push_with_retry "$IMAGE_URI_TAGGED"

# Ensure downstream uses this build
export IMAGE_TAG="$BUILD_TAG"
export IMAGE_URI="$IMAGE_URI_TAGGED"
echo "🖼️ Using image: $IMAGE_URI"

# ---------------------------
# Ensure ECS cluster exists
# ---------------------------
if ! aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --region "$AWS_REGION" \
   --query 'clusters[?status==`ACTIVE`].[clusterName]' --output text | grep -qx "$ECS_CLUSTER_NAME"; then
  echo "🧩 Creating ECS cluster: $ECS_CLUSTER_NAME"
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null
else
  echo "✅ ECS cluster exists: $ECS_CLUSTER_NAME"
fi

# ---------------------------
# Generate task definition JSON
# ---------------------------
echo "📝 Generating task definition..."
python "$SCRIPT_DIR/generate_task_def.py" \
  --template "$SCRIPT_DIR/task-def-template.json" \
  --out "$SCRIPT_DIR/final-task-def.json"

# Force the image field to our freshly pushed tag
jq --arg img "$IMAGE_URI" '.containerDefinitions[0].image = $img' \
  "$SCRIPT_DIR/final-task-def.json" > "$SCRIPT_DIR/final-task-def.json.tmp" && \
mv "$SCRIPT_DIR/final-task-def.json.tmp" "$SCRIPT_DIR/final-task-def.json"

IMG_IN_TD=$(jq -r '.containerDefinitions[0].image' "$SCRIPT_DIR/final-task-def.json")
if [[ "$IMG_IN_TD" != "$IMAGE_URI" ]]; then
  echo "❌ Task-def image mismatch. Expected $IMAGE_URI but got: $IMG_IN_TD"
  exit 1
fi
echo "📦 Task-def image confirmed: $IMG_IN_TD"

# ---------------------------
# Register task definition
# ---------------------------
REV_ARN=$(aws ecs register-task-definition \
  --cli-input-json "file://$SCRIPT_DIR/final-task-def.json" \
  --region "$AWS_REGION" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)
echo "📎 Registered task definition: $REV_ARN"

# ---------------------------
# Run task (awsvpc config)
# Support comma-separated IDs in env and convert to JSON arrays: ["subnet-1","subnet-2"]
# ---------------------------
csv_to_json_array() {
  local csv="$1"
  # trim spaces, split by comma, quote each
  IFS=',' read -r -a arr <<< "$csv"
  local out="["
  for i in "${!arr[@]}"; do
    val="$(echo "${arr[$i]}" | xargs)"  # trim
    out+="\"$val\""
    if (( i < ${#arr[@]}-1 )); then out+=","; fi
  done
  out+="]"
  echo "$out"
}

SUBNETS_JSON=$(csv_to_json_array "$FARGATE_SUBNET_IDS")
SGS_JSON=$(csv_to_json_array "$FARGATE_SECURITY_GROUP_IDS")

NETWORK_CFG=$(jq -n \
  --argjson subs "$SUBNETS_JSON" \
  --argjson sgs "$SGS_JSON" \
  --arg assign "$ASSIGN_PUBLIC_IP" \
  '{awsvpcConfiguration:{subnets:$subs,securityGroups:$sgs,assignPublicIp:$assign}}')

echo "🚀 Running task on Fargate..."
TASK_ARN=$(aws ecs run-task \
  --cluster "$ECS_CLUSTER_NAME" \
  --launch-type FARGATE \
  --network-configuration "$NETWORK_CFG" \
  --task-definition "$REV_ARN" \
  --region "$AWS_REGION" \
  --query 'tasks[0].taskArn' \
  --output text)

if [[ -z "$TASK_ARN" || "$TASK_ARN" == "None" ]]; then
  echo "❌ Failed to start task. Check IAM, subnets/SGs, and assignPublicIp."
  exit 1
fi

echo "✅ Task launched: $TASK_ARN"

# ---------------------------
# Tail CloudWatch logs for THIS task
# ---------------------------
TASK_ID="${TASK_ARN##*/}"   # e.g., a55a59690d7b4fbd...
CONTAINER_NAME=$(jq -r '.containerDefinitions[0].name' "$SCRIPT_DIR/final-task-def.json")
STREAM_PREFIX="${LOG_STREAM_PREFIX}/${CONTAINER_NAME}/${TASK_ID}"  # ecs/<container>/<taskId>

echo "⏳ Waiting for log stream: $STREAM_PREFIX"
for i in {1..12}; do  # up to ~60s
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
