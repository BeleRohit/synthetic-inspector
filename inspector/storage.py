import io
import pandas as pd
from google.cloud import storage

BUCKET_NAME = "ant-workshop-handson-synthetic-inspector"
# Initialize the client. This uses Application Default Credentials on Cloud Run.
client = storage.Client()

def upload_df(df: pd.DataFrame, blob_name: str) -> None:
    """Serialize DataFrame to CSV bytes and upload to GCS."""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

def download_df(blob_name: str) -> pd.DataFrame:
    """Download blob bytes from GCS and read into DataFrame."""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    content = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(content))

def blob_exists(blob_name: str) -> bool:
    """Check if a blob exists in the GCS bucket."""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.exists()
