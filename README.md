# 🧠 Diabetes Readmission Prediction Pipeline

This project implements a fully automated, end-to-end machine learning pipeline for predicting diabetes-related hospital readmissions. It leverages:

- **AWS Fargate** for serverless execution  
- **SageMaker** for model training, tuning, and deployment  
- **Docker** for environment consistency  
- **GitHub** for reproducible version control

---

## 📁 Directory Structure

```
.
├── Dockerfile                         # Image used for Fargate task
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Master controller script for full pipeline
├── run_fargate_task.sh               # CLI launcher for the pipeline via Fargate
├── fargate_deployment/               # Deployment, IAM, and ECS configuration
├── preprocessing/                    # Modular scripts for each pipeline stage
├── Dev/                              # Local-only folder for notebooks, exploration
```

---

## ⚙️ Pipeline Overview

The pipeline consists of the following steps:

1. **Data Engineering**  
   `data_engineering.py`  
   Cleans and transforms raw data; splits into train/test sets.

2. **Feature Selection**  
   `feature_selection.py`  
   Identifies the top N most predictive features using model-driven ranking.

3. **Hyperparameter Tuning**  
   `run_tuning.py`  
   Launches a SageMaker XGBoost tuning job to find optimal model parameters.

4. **Model Deployment**  
   `deploy_best_xgb.py`  
   Deploys the best model to a SageMaker endpoint.

5. **Batch Prediction**  
   `predict_from_endpoint.py`  
   Uses the deployed model to generate predictions for both train/test datasets.

---

## 🚀 Deployment Steps

### 1. Set up AWS resources (ECR, IAM, ECS, SageMaker)
```bash
cd fargate_deployment
./setup_iam.sh
```

### 2. Build ECS task definition from template
```bash
python fargate_deployment/generate_task_def.py
```

### 3. Register task definition and push Docker image to ECR
```bash
./deploy_to_fargate.sh
```

### 4. Launch full pipeline on Fargate
```bash
./run_fargate_task.sh
```

---

## 📦 Dev Folder

The `Dev/` folder contains Jupyter notebooks used during development and experimentation:

- `data_engineering_eda.ipynb`
- `feature_selection_eda.ipynb`
- `model_tuning.ipynb`
- `evaluation_visualization.ipynb`

These are not required for running the pipeline but may be useful for further customization or inspection.

---

## ✅ Outputs

All intermediate and final outputs are saved to your configured S3 bucket, including:

- Cleaned datasets
- Feature lists
- Trained model artifacts
- Predictions:
  - `train_with_predictions.csv`
  - `test_with_predictions.csv`

---

## 🔐 IAM and Security

Roles and policies are configured automatically via `setup_iam.sh`, including:

- ECS task execution and runtime roles
- SageMaker trust and access
- Inline S3 access policies

---

## 🛠️ Requirements

- AWS account with permissions for Fargate, SageMaker, ECS, IAM, and ECR  
- Docker installed locally  
- Python 3.10+  
- AWS CLI configured (`aws configure`)  

---

## 📬 Questions or Contributions

This project was developed by **Timothy Waterman** as part of ongoing work in healthcare analytics and ML system deployment.  
PRs and suggestions welcome.