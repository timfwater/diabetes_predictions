#!/usr/bin/env bash
set -euo pipefail

# Load and export config.env variables
set -a
source fargate_deployment/config.env
set +a

# --- Create timestamp tag ---
BUILD_TAG=$(date +"%Y%m%d-%H%M%S")

# --- Build Docker image ---
docker build -t "$ECR_REPO_NAME" .
docker tag "$ECR_REPO_NAME" "$ECR_REPO_URI:latest"
docker tag "$ECR_REPO_NAME" "$ECR_REPO_URI:$BUILD_TAG"

# --- Function to push with retry ---
push_with_retry() {
    local tag=$1
    local attempt=1
    local max_attempts=3
    local success=0

    while [[ $attempt -le $max_attempts ]]; do
        echo "🔄 Logging into ECR (attempt $attempt)..."
        aws ecr get-login-password --region "$AWS_REGION" | \
            docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

        echo "📤 Pushing image tag: $tag (attempt $attempt)..."
        if docker push "$ECR_REPO_URI:$tag"; then
            success=1
            echo "✅ Successfully pushed $tag"
            break
        else
            echo "⚠️ Push failed for $tag. Retrying..."
            sleep 5
        fi
        attempt=$((attempt+1))
    done

    if [[ $success -ne 1 ]]; then
        echo "❌ Failed to push $tag after $max_attempts attempts."
        exit 1
    fi
}

# --- Push both tags with retry ---
push_with_retry "latest"
push_with_retry "$BUILD_TAG"

# --- Generate ECS task definition ---
python fargate_deployment/generate_task_def.py

# --- Register ECS task definition ---
aws ecs register-task-definition \
  --cli-input-json file://fargate_deployment/final-task-def.json \
  --region "$AWS_REGION"

# --- Run ECS task ---
TASK_ARN=$(aws ecs run-task \
  --cluster "$ECS_CLUSTER_NAME" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$FARGATE_SUBNET_IDS],securityGroups=[$FARGATE_SECURITY_GROUP_IDS],assignPublicIp=ENABLED}" \
  --task-definition "$TASK_FAMILY" \
  --region "$AWS_REGION" \
  --query 'tasks[0].taskArn' \
  --output text)

echo "✅ Task launched: $TASK_ARN"
echo "📦 Image tags pushed: latest and $BUILD_TAG"

# --- Wait for the task to start ---
echo "⏳ Waiting for task to start..."
sleep 10

# --- Tail latest CloudWatch logs ---
LOG_STREAM=$(aws logs describe-log-streams \
  --log-group-name "$LOG_GROUP" \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --query 'logStreams[0].logStreamName' \
  --output text \
  --region "$AWS_REGION")

if [[ "$LOG_STREAM" != "None" && -n "$LOG_STREAM" ]]; then
  echo "📜 Tailing logs for stream: $LOG_STREAM"
  aws logs tail "$LOG_GROUP" --log-stream-names "$LOG_STREAM" --follow --region "$AWS_REGION"
else
  echo "⚠️ No log stream found yet. You may need to check CloudWatch manually."
fi
