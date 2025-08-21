#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1090
source "$ROOT_DIR/fargate_deployment/config.env"

echo "🚀 Running ECS task on Fargate..."

# Ensure cluster exists
if ! aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --region "$AWS_REGION" \
   --query 'clusters[?status==`ACTIVE`].[clusterName]' --output text | grep -qx "$ECS_CLUSTER_NAME"; then
  echo "🧩 Creating cluster: $ECS_CLUSTER_NAME"
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null
fi

# Paths
TASK_DEF_JSON="$ROOT_DIR/fargate_deployment/final-task-def.json"
ECR_REGEX='^[0-9]+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com\/[a-z0-9._\/-]+:[A-Za-z0-9._-]+$'

# Require final task def file
if [[ ! -f "$TASK_DEF_JSON" ]]; then
  echo "❌ Missing $TASK_DEF_JSON. Generate it first:"
  echo "   python fargate_deployment/generate_task_def.py --template fargate_deployment/task-def-template.json --out fargate_deployment/final-task-def.json"
  exit 1
fi

# Extract image string safely
IMAGE_VAL="$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j["containerDefinitions"][0]["image"])' "$TASK_DEF_JSON")" || {
  echo "❌ Failed to read image from $TASK_DEF_JSON"
  exit 1
}

if [[ -z "${IMAGE_VAL// }" ]]; then
  echo "❌ The image field in final-task-def.json is empty."
  exit 1
fi
if [[ "$IMAGE_VAL" == *"REPLACE_"* ]]; then
  echo "❌ The image contains placeholders ($IMAGE_VAL). Re-generate task def after building & pushing the image."
  exit 1
fi
if ! [[ "$IMAGE_VAL" =~ $ECR_REGEX ]]; then
  echo "❌ The image value looks invalid for ECS/ECR:"
  echo "   $IMAGE_VAL"
  echo "Expected format:"
  echo "   <acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>"
  exit 1
fi

echo "🖼️  Image OK: $IMAGE_VAL"

# Register task definition
REV_ARN=$(aws ecs register-task-definition --cli-input-json "file://$TASK_DEF_JSON" \
          --query 'taskDefinition.taskDefinitionArn' --output text --region "$AWS_REGION")
echo "📎 Registered task definition: $REV_ARN"

# Run latest revision explicitly
TASK_ARN=$(aws ecs run-task \
  --cluster "$ECS_CLUSTER_NAME" \
  --launch-type FARGATE \
  --task-definition "$REV_ARN" \
  --network-configuration "awsvpcConfiguration={subnets=[$FARGATE_SUBNET_IDS],securityGroups=[$FARGATE_SECURITY_GROUP_IDS],assignPublicIp=$ASSIGN_PUBLIC_IP}" \
  --count 1 \
  --region "$AWS_REGION" \
  --query 'tasks[0].taskArn' --output text)

echo "▶️  Started task: $TASK_ARN"
echo "💡 Tail logs: aws logs tail \"$LOG_GROUP\" --follow --region \"$AWS_REGION\""
