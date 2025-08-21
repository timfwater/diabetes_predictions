#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/config.env"

# Fallback tag if git isn't available or repo has no commits
if ! IMAGE_TAG_VAL=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null); then
  IMAGE_TAG_VAL="$(date +%Y%m%d%H%M%S)"
fi
# Overwrite IMAGE_TAG/IMAGE_URI for this build session (don’t edit config.env on disk)
IMAGE_TAG="$IMAGE_TAG_VAL"
IMAGE_URI="${ECR_REPO_URI}:${IMAGE_TAG}"

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
| docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "🐳 Building image ${ECR_REPO_NAME}:${IMAGE_TAG} ..."
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" "$ROOT_DIR"

echo "🏷️  Tagging -> ${IMAGE_URI}"
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"

echo "⬆️  Pushing ${IMAGE_URI} ..."
docker push "${IMAGE_URI}"

# Write the resolved tag to a temp file so the task-def generator can use it
echo "$IMAGE_URI" > "$SCRIPT_DIR/.last_image_uri"

echo "✅ Build & push complete: ${IMAGE_URI}"
