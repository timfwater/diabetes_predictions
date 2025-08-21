#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/config.env"

echo "🔧 Starting IAM/ECR/ECS setup..."

# ---- Sanity checks on required env ----
: "${AWS_ACCOUNT_ID:?Missing AWS_ACCOUNT_ID in config.env}"
: "${AWS_REGION:?Missing AWS_REGION in config.env}"
: "${ECR_REPO_NAME:?Missing ECR_REPO_NAME in config.env}"
: "${TASK_EXECUTION_ROLE:?Missing TASK_EXECUTION_ROLE (ARN or name) in config.env}"
: "${TASK_ROLE:?Missing TASK_ROLE (ARN or name) in config.env}"
: "${LOG_GROUP:?Missing LOG_GROUP in config.env}"
: "${ECS_CLUSTER_NAME:?Missing ECS_CLUSTER_NAME in config.env}"
: "${BUCKET:?Missing BUCKET in config.env}"
: "${PREFIX:?Missing PREFIX in config.env}"

if [[ -z "${SAGEMAKER_TRAINING_ROLE:-}" ]]; then
  echo "❌ SAGEMAKER_TRAINING_ROLE not set. Set this to the role SageMaker will assume for training/tuning."
  exit 1
fi

AWS_PARTITION="${AWS_PARTITION:-aws}"

EXEC_ROLE_NAME="$(basename "$TASK_EXECUTION_ROLE")"
TASK_ROLE_NAME="$(basename "$TASK_ROLE")"
SM_ROLE_NAME="$(basename "$SAGEMAKER_TRAINING_ROLE")"
if [[ "$SAGEMAKER_TRAINING_ROLE" == arn:${AWS_PARTITION}:iam::* ]]; then
  SM_ROLE_ARN="$SAGEMAKER_TRAINING_ROLE"
else
  SM_ROLE_ARN="arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/${SM_ROLE_NAME}"
fi

echo "ℹ️  Resolved execution role name: $EXEC_ROLE_NAME"
echo "ℹ️  Resolved task role name     : $TASK_ROLE_NAME"
echo "ℹ️  SageMaker exec role name    : $SM_ROLE_NAME"
echo "ℹ️  SageMaker exec role ARN     : $SM_ROLE_ARN"
echo "ℹ️  Target log group            : $LOG_GROUP"
echo "ℹ️  Target ECS cluster          : $ECS_CLUSTER_NAME"
echo

# --- ECR repo ---
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" >/dev/null
  echo "✅ Created ECR repository: $ECR_REPO_NAME"
else
  echo "✅ ECR repository exists: $ECR_REPO_NAME"
fi

# --- Roles trust policy path for ECS roles ---
TRUST="file://$SCRIPT_DIR/ecs-trust-policy.json"

# --- Execution role (pull image, logs, secrets) ---
if ! aws iam get-role --role-name "$EXEC_ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$EXEC_ROLE_NAME" --assume-role-policy-document "$TRUST" >/dev/null
  echo "✅ Created execution role $EXEC_ROLE_NAME"
else
  aws iam update-assume-role-policy --role-name "$EXEC_ROLE_NAME" --policy-document "$TRUST" >/dev/null
  echo "🔄 Updated execution role trust policy"
fi

aws iam attach-role-policy --role-name "$EXEC_ROLE_NAME" \
  --policy-arn arn:${AWS_PARTITION}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy >/dev/null || true
# If your task pulls secrets from Secrets Manager, uncomment:
# aws iam attach-role-policy --role-name "$EXEC_ROLE_NAME" \
#   --policy-arn arn:${AWS_PARTITION}:iam::aws:policy/SecretsManagerReadWrite >/dev/null || true
echo "🪝 Ensured AmazonECSTaskExecutionRolePolicy on $EXEC_ROLE_NAME"

# --- Task role (app permissions) ---
if ! aws iam get-role --role-name "$TASK_ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$TASK_ROLE_NAME" --assume-role-policy-document "$TRUST" >/dev/null
  echo "✅ Created task role $TASK_ROLE_NAME"
else
  aws iam update-assume-role-policy --role-name "$TASK_ROLE_NAME" --policy-document "$TRUST" >/dev/null
  echo "🔄 Updated task role trust policy"
fi

# Inline S3 policy (your file, scoped to your needs)
aws iam put-role-policy \
  --role-name "$TASK_ROLE_NAME" \
  --policy-name S3AccessPolicy \
  --policy-document "file://$SCRIPT_DIR/s3-access-policy.json" >/dev/null
echo "🪣 Attached S3 inline policy to task role"

# Keep broad SageMaker for now; trim later if only invoking endpoints
aws iam attach-role-policy \
  --role-name "$TASK_ROLE_NAME" \
  --policy-arn arn:${AWS_PARTITION}:iam::aws:policy/AmazonSageMakerFullAccess >/dev/null || true
echo "🧪 Ensured SageMakerFullAccess on task role (temporary)"

# --- SageMaker execution role (used BY SageMaker for training/tuning) ---
# Trust so SageMaker can assume the role
cat >/tmp/sm-trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Principal": { "Service": "sagemaker.amazonaws.com" }, "Action": "sts:AssumeRole" }
  ]
}
JSON

if ! aws iam get-role --role-name "$SM_ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$SM_ROLE_NAME" --assume-role-policy-document file:///tmp/sm-trust.json >/dev/null
  echo "✅ Created SageMaker execution role: $SM_ROLE_NAME"
else
  aws iam update-assume-role-policy --role-name "$SM_ROLE_NAME" --policy-document file:///tmp/sm-trust.json >/dev/null
  echo "🔄 Updated SageMaker exec role trust policy"
fi

# Broad SageMaker perms (okay for now), plus scoped S3 access to your data
aws iam attach-role-policy --role-name "$SM_ROLE_NAME" \
  --policy-arn arn:${AWS_PARTITION}:iam::aws:policy/AmazonSageMakerFullAccess >/dev/null || true

cat >/tmp/sm-s3-access.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::${BUCKET}",
      "Condition": { "StringLike": { "s3:prefix": ["${PREFIX}/*"] } } },
    { "Effect": "Allow", "Action": ["s3:GetObject","s3:PutObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/${PREFIX}/*" }
  ]
}
JSON
aws iam put-role-policy --role-name "$SM_ROLE_NAME" --policy-name DiabetesS3Access --policy-document file:///tmp/sm-s3-access.json >/dev/null
echo "🪣 Ensured S3 access for SageMaker role ${SM_ROLE_NAME}"

# --- Allow the ECS task role to PassRole the SageMaker execution role only ---
cat > /tmp/pass-sm-training-role.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PassTrainingRoleToSageMaker",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${SM_ROLE_ARN}",
      "Condition": { "StringEquals": { "iam:PassedToService": "sagemaker.amazonaws.com" } }
    }
  ]
}
JSON

aws iam put-role-policy \
  --role-name "$TASK_ROLE_NAME" \
  --policy-name PassSageMakerTrainingRole \
  --policy-document file:///tmp/pass-sm-training-role.json >/dev/null
echo "🎟️  Granted iam:PassRole for ${SM_ROLE_ARN} to task role (SageMaker only)"

# --- CloudWatch log group (with retention) ---
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION" 2>/dev/null || true
# 14 days retention; tweak as desired
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14 --region "$AWS_REGION" >/dev/null || true
echo "🪵 Ensured log group & retention: $LOG_GROUP"

# --- ECS cluster ---
if ! aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --region "$AWS_REGION" \
     --query 'clusters[?status==`ACTIVE`].[clusterName]' --output text | grep -qx "$ECS_CLUSTER_NAME"; then
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null
  echo "✅ Created ECS cluster: $ECS_CLUSTER_NAME"
else
  echo "✅ ECS cluster exists: $ECS_CLUSTER_NAME"
fi

echo "🎯 IAM/ECR/ECS setup complete."
echo "   ➤ If you created/updated the SageMaker role above, ensure config.env has:"
echo "     SAGEMAKER_TRAINING_ROLE=${SM_ROLE_ARN}"
