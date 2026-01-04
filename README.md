# Diabetes Readmissions --- Cost-Optimized ML Pipeline (AWS)

Problem: Predict 30-day hospital readmissions among diabetic patients
and choose a decision threshold that maximizes net cost savings, not
just ROC AUC.

Stack (end-to-end): S3 • PySpark/Pandas (EDA + prep) • Feature Selection
• XGBoost & Neural Network models • SageMaker HPO • Optional endpoint
deployment • Batch predictions to S3 • Evaluation & business metrics •
ECS/Fargate (optional orchestrator)

## What this project shows

-   Practical ML framing for an imbalanced clinical outcome (\~11%
    positives)
-   Feature selection and hyperparameter tuning using SageMaker
-   Evaluation beyond AUC, including net cost savings and avoided
    readmissions
-   Comparison of XGBoost vs Neural Network models
-   Production-style execution via ECS/Fargate or fully local runs

## Architecture (high-level)

Raw CSV → Preprocess / Feature Select → SageMaker HPO (XGB) → Best Model
→ Optional Endpoint → Batch Predict → Evaluate (AUC + Cost) → Report

## Repository layout

Dev/notebooks/: data_engineering_eda.ipynb, feature_selection_eda.ipynb,
model_tuning.ipynb, evaluation_visualization.ipynb\
preprocessing/: data_engineering.py, feature_selection.py,
run_tuning_xgb.py, deploy_best_xgb.py, predict_from_endpoint.py\
Other key files: run_pipeline.py, requirements.txt, Dockerfile,
fargate_deployment/

## How to run

Option A --- ECS/Fargate: run build_and_push.sh, deploy_to_fargate.sh,
then run_fargate_task.sh to execute the full pipeline as a batch job.

Option B --- Local execution: create a virtual environment, install
requirements, run preprocessing scripts, perform SageMaker tuning,
optionally deploy the best model, and evaluate outputs using the
provided notebook or eval_reports directory.

## Metrics you will see

Model metrics include ROC AUC, PR AUC, and confusion matrix at a
selected threshold.\
Operational KPIs include net cost savings, prevented readmissions,
precision, recall, F1, specificity, and accuracy.\
Feature-level insights include feature importances from XGBoost and
selected features from the feature-selection stage.

## Configuration

Common environment variables: AWS_REGION, S3_BUCKET, S3_PREFIX.\
The SageMaker execution role needs permissions for tuning/training and
S3 read/write access.

## Reproducing visuals

Use Dev/notebooks/evaluation_visualization.ipynb to regenerate confusion
matrices, ROC and PR curves, cost-versus-threshold analyses, and
feature-importance plots.\
Evaluation summaries are also written to eval_reports/.

## Notes on data, ethics, and limits

Dataset: UCI Diabetes Readmissions (no PHI).\
Imbalanced prediction problem (\~11% positive).\
Readmission risk is a proxy, not clinical need; cost estimates are
illustrative.\
Subgroup fairness analysis is future work.

## Portfolio walkthrough page

https://wbst-bkt.s3.amazonaws.com/index.html
