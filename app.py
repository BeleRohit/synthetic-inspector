import io
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from inspector import storage, generate, report
from inspector.evaluate import marginals, correlations, subgroups, downstream, privacy

app = FastAPI(title="Synthetic Inspector")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
    job_id = str(uuid.uuid4())
    blob_name = f"{job_id}/real.csv"
    
    storage.upload_df(df, blob_name)
    
    return {"job_id": job_id}

@app.post("/generate/{job_id}")
def generate_data(job_id: str):
    real_blob = f"{job_id}/real.csv"
    
    if not storage.blob_exists(real_blob):
        raise HTTPException(status_code=404, detail=f"Real data not found for job_id {job_id}")
        
    real_df = storage.download_df(real_blob)
    synthetic_df = generate.generate_synthetic(real_df)
    
    synth_blob = f"{job_id}/synthetic.csv"
    storage.upload_df(synthetic_df, synth_blob)
    
    return {"job_id": job_id, "synthetic_rows": len(synthetic_df)}

@app.get("/report/{job_id}", response_class=HTMLResponse)
def get_report(job_id: str):
    real_blob = f"{job_id}/real.csv"
    synth_blob = f"{job_id}/synthetic.csv"
    
    if not storage.blob_exists(real_blob):
        raise HTTPException(status_code=404, detail="Real data not found.")
    if not storage.blob_exists(synth_blob):
        raise HTTPException(status_code=404, detail="Synthetic data not found. Did you call /generate?")
        
    real_df = storage.download_df(real_blob)
    synth_df = storage.download_df(synth_blob)
    
    # Run evaluators
    findings = [
        marginals.evaluate(real_df, synth_df),
        correlations.evaluate(real_df, synth_df),
        subgroups.evaluate(real_df, synth_df),
        downstream.evaluate(real_df, synth_df),
        privacy.evaluate(real_df, synth_df)
    ]
    
    html_content = report.assemble(findings)
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
