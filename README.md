# Diabetes Readmissions — Cost-Optimized ML Pipeline (AWS)

**Goal:** Predict 30-day readmissions for diabetic patients and choose an operating threshold that **maximizes cost savings** (not just ROC AUC).

**Stack (end-to-end):** S3 • PySpark/Pandas (EDA + prep) • Feature Selection • XGBoost (SageMaker HPO) • Best-model deploy (optional) • Batch predictions → S3 • Evaluation & business metrics • ECS/Fargate (optional orchestrator)

---

## 🔎 What this project shows
- Practical ML framing for an **imbalanced** clinical outcome.
- **Feature selection** + **hyperparameter tuning** for XGBoost via SageMaker.
- Evaluation beyond AUC: **net cost savings**, **prevented readmissions**, and confusion-matrix metrics.
- Production-style execution via **ECS/Fargate** (optional), or fully local.

---

## 🗺️ Architecture (high-level)

**Raw CSV → Preprocess/Feature Select → SageMaker HPO (XGB) → Best Model → (Optional) Endpoint → Batch Predict → Evaluate (AUC & Cost) → Report**


---

## 📂 Repository layout
```
.
├── Dev/notebooks/
│ ├── data_engineering_eda.ipynb # EDA + cleaning notes
│ ├── feature_selection_eda.ipynb # FS rationale & checks
│ ├── model_tuning.ipynb # HPO results exploration
│ └── evaluation_visualization.ipynb # confusion matrix, tables, plots
├── preprocessing/
│ ├── data_engineering.py # clean/encode/splits
│ ├── feature_selection.py # select informative features
│ ├── run_tuning_xgb.py # SageMaker HPO (XGBoost)
│ ├── deploy_best_xgb.py # (optional) deploy best model
│ ├── predict_from_endpoint.py # (optional) real-time predict
│ └── latest_tuning_job.txt # HPO job id cache
├── run_pipeline.py # single entrypoint (CLI)
├── fargate_deployment/ # optional containerized orchestration
│ ├── build_and_push.sh
│ ├── deploy_to_fargate.sh
│ ├── task-def-template.json
│ └── ...
├── requirements.txt
└── Dockerfile
```


---

## ▶️ How to run

### Option A — ECS/Fargate (one-command orchestration)

```
# Build & push image, update task def
./fargate_deployment/build_and_push.sh
./fargate_deployment/deploy_to_fargate.sh

# Launch one task run of the pipeline
./run_fargate_task.sh
```

### Option B — Local

```
# 1) Create a virtual env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Preprocess + feature select
python preprocessing/data_engineering.py
python preprocessing/feature_selection.py

# 3) Hyperparameter tune XGBoost on SageMaker
python preprocessing/run_tuning_xgb.py

# 4) (Optional) Deploy best model for real-time inference
python preprocessing/deploy_best_xgb.py

# 5) Generate predictions & evaluate
# - batch predict in your evaluation notebook OR via your own script
# - open Dev/notebooks/evaluation_visualization.ipynb to render tables/plots
```



Fargate simply wraps the same Python entrypoints in a containerized run.

## 🧪 Metrics you’ll see

ROC AUC (holdout)

Confusion matrix at the chosen threshold

Operational KPIs:

Net_Cost_Savings (program cost vs. prevented readmissions value)

Prevented_Readmissions

Recall / Precision / F1 / Specificity / Accuracy

Feature importances (model + FS stage)

## 🧰 Configuration

Set AWS region/bucket once (env or .env):

AWS_REGION=us-east-1
S3_BUCKET=your-bucket
S3_PREFIX=diabetes-ml/


SageMaker permissions: the role running tuning needs sagemaker:* for training jobs and s3:{Get,Put,List} on your prefixes.

## 📈 Reproducing the visuals

Open Dev/notebooks/evaluation_visualization.ipynb to render:

Confusion matrix for the final threshold

Model comparison table (AUC vs. cost)

Feature importances

Prediction score distributions

Screenshots from this notebook are included in the presentation.

## 🔒 Notes on data, ethics, and limits

Dataset: UCI Diabetes (tabular, imbalanced ~11% positives).

No PHI; academic dataset.

We report business outcomes (cost) alongside AUC.

Model bias can be assessed by stratifying metrics across subgroups (future work).

**Project Walkthrough:**
https://wbst-bkt.s3.us-east-1.amazonaws.com/patient_index.html