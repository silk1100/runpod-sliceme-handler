import boto3
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(".env")

bucket = os.environ.get("AWS_S3_BUCKET")
key = os.environ.get("AWS_S3_KEY")
secret = os.environ.get("AWS_S3_SECRET")

local_model = Path("models/best.pt")

s3 = boto3.client(
    "s3",
    aws_access_key_id=key,
    aws_secret_access_key=secret,
)

s3.upload_file(
    str(local_model),
    bucket,
    "models/best.pt",  # S3 key — same as what handler.py downloads via AWS_S3_MODELKEY
)