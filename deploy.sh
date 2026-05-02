#!/bin/bash
set -e

# 1. Set project
gcloud config set project ant-workshop-handson

# 2. Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  storage.googleapis.com

# 3. Create GCS bucket (ignore error if already exists)
gsutil mb -p ant-workshop-handson -l asia-south1 \
  gs://ant-workshop-handson-synthetic-inspector 2>/dev/null || true

# 4. Grant Cloud Run service account access to GCS bucket
PROJECT_NUMBER=$(gcloud projects describe ant-workshop-handson \
  --format='value(projectNumber)')
gsutil iam ch \
  serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com:objectAdmin \
  gs://ant-workshop-handson-synthetic-inspector

# 5. Submit build and deploy via Cloud Build
gcloud builds submit --config cloudbuild.yaml .

# 6. Print the service URL
gcloud run services describe synthetic-inspector \
  --region=asia-south1 \
  --format='value(status.url)'
