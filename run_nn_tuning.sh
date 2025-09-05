#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

# --- Load config ---
if [[ -f "fargate_deployment/config.env" ]]; then
  # shellcheck disable=SC1091
  source "fargate_deployment/config.env"
elif [[ -f "config.env" ]]; then
  # shellcheck disable=SC1091
  source "config.env"
else
  echo "❌ config.env not found (looked in repo root and fargate_deployment/)."
  exit 1
fi

# --- Export critical vars so Python sees them ---
export AWS_REGION="${AWS_REGION:-}"
export AWS_DEFAULT_REGION="${AWS_REGION}"
export SAGEMAKER_TRAINING_ROLE="${SAGEMAKER_TRAINING_ROLE:-}"
export BUCKET="${BUCKET:-}"
export PREFIX="${PREFIX:-}"
export FILTERED_INPUT_FILE="${FILTERED_INPUT_FILE:-}"
export SELECTED_FEATURES_FILE="${SELECTED_FEATURES_FILE:-}"
export LABEL_COL="${LABEL_COL:-readmitted}"
export NN_OUTPUT_PREFIX="${NN_OUTPUT_PREFIX:-nn_output}"

# --- Sanity checks ---
: "${AWS_REGION:?Set AWS_REGION in config.env}"
: "${SAGEMAKER_TRAINING_ROLE:?Set SAGEMAKER_TRAINING_ROLE in config.env}"
: "${BUCKET:?Set BUCKET in config.env}"
: "${PREFIX:?Set PREFIX in config.env}"
: "${FILTERED_INPUT_FILE:?Set FILTERED_INPUT_FILE in config.env}"
: "${SELECTED_FEATURES_FILE:?Set SELECTED_FEATURES_FILE in config.env}"

echo "🚀 Launching NN HPO"
echo "   AWS_REGION              = ${AWS_REGION}"
echo "   SAGEMAKER_TRAINING_ROLE = ${SAGEMAKER_TRAINING_ROLE}"
echo "   BUCKET/PREFIX           = ${BUCKET}/${PREFIX}"
echo "   INPUT FILE              = ${FILTERED_INPUT_FILE}"
echo "   FEATURES FILE           = ${SELECTED_FEATURES_FILE}"
echo "   LABEL_COL               = ${LABEL_COL}"
echo "   NN_OUTPUT_PREFIX        = ${NN_OUTPUT_PREFIX}"

# Optional handoff vars (some code likes these names)
export S3_BUCKET="${BUCKET}"
export S3_PREFIX="${PREFIX}"

python preprocessing/run_tuning_nn.py

echo "✅ Submitted NN tuning jobs. Check progress in AWS Console → SageMaker → Hyperparameter tuning jobs."
