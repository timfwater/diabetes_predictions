# syntax=docker/dockerfile:1.7

# ---- Base image ----
FROM python:3.10-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
  && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
# Use BuildKit cache to avoid re-downloading wheels each build
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY preprocessing/ /app/preprocessing/
COPY run_pipeline.py /app/run_pipeline.py

# Optional: placeholder so deploy scripts can read it before tuning writes it
RUN touch /app/latest_tuning_job.txt

# ---- Default command ----
CMD ["python", "run_pipeline.py"]
