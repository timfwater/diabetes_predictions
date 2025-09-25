# syntax=docker/dockerfile:1.7

# ---- Base image ----
FROM python:3.10-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    MPLBACKEND=Agg

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
  && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Python deps ----
# Make sure requirements.txt includes: boto3, sagemaker, pandas, matplotlib, seaborn, s3fs (optional but handy)
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY preprocessing/ /app/preprocessing/
COPY run_pipeline.py /app/run_pipeline.py
COPY model_eval_xgb_vs_nn.py /app/model_eval_xgb_vs_nn.py
# ✅ include the exporter so it can run inside the container
COPY diabetes_eval_export.py /app/diabetes_eval_export.py

# Optional: placeholder so deploy scripts can read it before tuning writes it
RUN touch /app/latest_tuning_job.txt

# ---- Default command ----
CMD ["python", "run_pipeline.py"]
