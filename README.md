# Synthetic Inspector

A synthetic data evaluation MVP deployed on Google Cloud Run. This tool assesses how faithful synthetic tabular data is to real data, specializing in the Pima Indians Diabetes dataset.

## Prerequisites
- `gcloud` CLI installed and authenticated (`gcloud auth login` and `gcloud auth application-default login`).
- Active Google Cloud Platform project named `ant-workshop-handson`.

## Deployment
Use the provided `deploy.sh` script to set up APIs, GCS bucket, IAM permissions, and deploy to Cloud Run.

```bash
chmod +x deploy.sh
./deploy.sh
```

## Architecture
- All job data (real and synthetic CSVs) is stored in a GCS bucket (`ant-workshop-handson-synthetic-inspector`).
- The Cloud Run service operates entirely statelessly; all data ingestion and reading routes directly to GCS memory streams. Local filesystem usage is strictly forbidden.

## Usage / API Endpoints

Once deployed, use the Cloud Run service URL to interact with the endpoints.

**1. Upload Real Data**
```bash
curl -X POST \
  -F "file=@your_real_data.csv" \
  https://synthetic-inspector-<hash>-<region>.a.run.app/upload
```
*Returns `{"job_id": "<uuid>"}`*

**2. Generate Synthetic Data**
```bash
curl -X POST \
  https://synthetic-inspector-<hash>-<region>.a.run.app/generate/<job_id>
```
*Returns `{"job_id": "<uuid>", "synthetic_rows": 768}`*

**3. Get Report**
```bash
# This returns an HTML response. You can open it in your browser.
curl -X GET \
  https://synthetic-inspector-<hash>-<region>.a.run.app/report/<job_id> > report.html
open report.html
```

## Local Development
Use the `smoke_test.py` to run the evaluators purely in-memory using dummy data, bypassing HTTP routing and GCS.

```bash
pip install -r requirements.txt
python smoke_test.py
```
