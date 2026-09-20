#!/usr/bin/env bash
# Build and push both no-endpoint batch-scoring images (Dockerfile.xgb-scorer,
# Dockerfile.nn-scorer) to the project's existing ECR repo, each under its
# own fixed tag. Run this once, and again any time src/predict_from_both.py,
# src/xgb_fold_ensemble.py, src/nn_fold_ensemble.py, src/config.py, or
# config.yaml changes.
#
# Usage:
#   bash sagemaker_processing/build_and_push_scorers.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

read_cfg() { python3 -c "from src.config import cfg; v = cfg.get('$1'); print(v if v is not None else '')"; }

AWS_REGION="$(read_cfg aws.region)"
AWS_ACCOUNT_ID="$(read_cfg infra.account_id)"
ECR_REPO_NAME="$(read_cfg infra.ecr_repo_name)"

if [[ -z "$AWS_ACCOUNT_ID" || -z "$ECR_REPO_NAME" ]]; then
  echo "❌ infra.account_id / infra.ecr_repo_name not resolved from config.infra.yaml." >&2
  exit 1
fi

ECR_REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "🔐 Logging into ECR (${ECR_REPO_URI})..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

for target in xgb nn; do
  TAG="${target}-scorer"
  echo ""
  echo "🐳 Building ${ECR_REPO_NAME}:${TAG} from Dockerfile.${TAG} ..."
  docker build -f "Dockerfile.${TAG}" -t "${ECR_REPO_NAME}:${TAG}" .

  echo "🏷️  Tagging -> ${ECR_REPO_URI}:${TAG}"
  docker tag "${ECR_REPO_NAME}:${TAG}" "${ECR_REPO_URI}:${TAG}"

  echo "⬆️  Pushing ${ECR_REPO_URI}:${TAG} ..."
  docker push "${ECR_REPO_URI}:${TAG}"

  echo "✅ Pushed ${ECR_REPO_URI}:${TAG}"
done

echo ""
echo "Done. Images ready for src/run_batch_scoring.py:"
echo "  ${ECR_REPO_URI}:xgb-scorer"
echo "  ${ECR_REPO_URI}:nn-scorer"
