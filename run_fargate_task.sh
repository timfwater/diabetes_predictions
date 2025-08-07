#!/bin/bash

# Load environment config
source ./fargate_deployment/config.env

echo "🚀 Running ECS task on Fargate..."

aws ecs run-task \
  --cluster $ECS_CLUSTER_NAME \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$FARGATE_SUBNET_IDS],securityGroups=[$FARGATE_SECURITY_GROUP_IDS],assignPublicIp=ENABLED}" \
  --task-definition $TASK_FAMILY \
  --region $AWS_REGION
