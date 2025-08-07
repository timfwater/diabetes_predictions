#!/bin/bash
set -e

# Load environment variables
source "$(dirname "$0")/config.env"

echo "🔧 Starting IAM and ECS setup..."

# Create ECR repo if not exists
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" > /dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION"
  echo "✅ Created ECR repository: $ECR_REPO_NAME"
else
  echo "✅ ECR repository already exists"
fi

# Set ECR repository policy
ECR_POLICY=$(cat <<EOF
{
  "Version": "2008-10-17",
  "Statement": [
    {
      "Sid": "AllowPushPullForAccount",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::${AWS_ACCOUNT_ID}:root"
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeImages"
      ]
    }
  ]
}
EOF
)

aws ecr set-repository-policy \
  --repository-name "$ECR_REPO_NAME" \
  --policy-text "$ECR_POLICY" \
  --region "$AWS_REGION" >/dev/null

echo "✅ ECR repository policy applied for account $AWS_ACCOUNT_ID"

# Create ECS Execution Role
if ! aws iam get-role --role-name "$(basename $TASK_EXECUTION_ROLE)" > /dev/null 2>&1; then
  aws iam create-role --role-name "$(basename $TASK_EXECUTION_ROLE)" \
    --assume-role-policy-document file://fargate_deployment/ecs-trust-policy.json

  aws iam attach-role-policy --role-name "$(basename $TASK_EXECUTION_ROLE)" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

  echo "✅ Created ECS Execution Role"
else
  aws iam update-assume-role-policy \
    --role-name "$(basename $TASK_EXECUTION_ROLE)" \
    --policy-document file://fargate_deployment/ecs-trust-policy.json

  echo "🔄 Updated trust policy for existing ECS Execution Role"
fi

# Create ECS Task Role
if ! aws iam get-role --role-name "$(basename $TASK_ROLE)" > /dev/null 2>&1; then
  aws iam create-role --role-name "$(basename $TASK_ROLE)" \
    --assume-role-policy-document file://fargate_deployment/ecs-trust-policy.json

  echo "✅ Created ECS Task Role"
else
  aws iam update-assume-role-policy \
    --role-name "$(basename $TASK_ROLE)" \
    --policy-document file://fargate_deployment/ecs-trust-policy.json

  echo "🔄 Updated trust policy for existing ECS Task Role"
fi


# Attach inline S3 access policy to task role
aws iam put-role-policy \
  --role-name "$(basename $TASK_ROLE)" \
  --policy-name S3AccessPolicy \
  --policy-document file://fargate_deployment/s3-access-policy.json

# Attach SageMaker and PassRole permissions to task role
aws iam attach-role-policy \
  --role-name "$(basename $TASK_ROLE)" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Optional: IAMFullAccess is very broad — only use if necessary
# aws iam attach-role-policy \
#   --role-name "$(basename $TASK_ROLE)" \
#   --policy-arn arn:aws:iam::aws:policy/IAMFullAccess

echo "✅ S3 and SageMaker permissions attached to Task Role"

# Detect current CLI user
USER_NAME=$(aws sts get-caller-identity --query Arn --output text | cut -d'/' -f2)
echo "🔍 Detected AWS CLI user: $USER_NAME"

# Attach permissions to current user
aws iam attach-user-policy \
  --user-name "$USER_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-user-policy \
  --user-name "$USER_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess

echo "✅ Attached ECR/ECS permissions to $USER_NAME"

# Create ECS Cluster if it doesn't exist
if ! aws ecs describe-clusters --clusters "$ECS_CLUSTER_NAME" --region "$AWS_REGION" | grep -q "\"status\": \"ACTIVE\""; then
  aws ecs create-cluster --cluster-name "$ECS_CLUSTER_NAME" --region "$AWS_REGION" > /dev/null
  echo "✅ ECS cluster created: $ECS_CLUSTER_NAME"
else
  echo "✅ ECS cluster already exists"
fi

echo "🎯 IAM setup complete — ECR repo, ECS cluster, IAM roles, and permissions are ready."
