# ---- Base image ----
FROM python:3.10-slim

# ---- Environment setup ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential curl git \
  && rm -rf /var/lib/apt/lists/*

# ---- Create working directory ----
WORKDIR /app

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY preprocessing/ /app/preprocessing/
COPY run_pipeline.py /app/run_pipeline.py
# If your latest_tuning_job.txt lives under preprocessing/, also drop a copy at /app
# so the default TUNING_JOB_FILE works without extra envs:
RUN if [ -f /app/preprocessing/latest_tuning_job.txt ]; then cp /app/preprocessing/latest_tuning_job.txt /app/latest_tuning_job.txt; fi

# ---- Default command (full pipeline) ----
CMD ["python", "run_pipeline.py"]
