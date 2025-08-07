# ---- Base image ----
FROM python:3.10-slim

# ---- Environment setup ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Create working directory ----
WORKDIR /app

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY preprocessing/ /app/preprocessing/
COPY run_pipeline.py /app/run_pipeline.py

# ---- Set entrypoint ----
CMD ["python", "run_pipeline.py"]
