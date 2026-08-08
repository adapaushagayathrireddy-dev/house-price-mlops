# ---- base image ----
FROM python:3.11-slim

# Prevent Python from writing pyc + buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Install system deps needed by scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# ---- python deps (cached layer) ----
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- app code ----
COPY src/ ./src
COPY api/ ./api
COPY models/ ./models


ENV PYTHONPATH=/app

ENV MODEL_PATH=/app/models/best_model.pkl

EXPOSE 8000


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]