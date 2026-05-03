# handler.py
import runpod
from ultralytics import YOLO
import boto3
import base64
import os
import torch  # ← Add this import
from pathlib import Path
from PIL import Image
import io

MODEL_LOCAL_PATH = Path("models/best.pt")

def download_model_from_s3():
    """Download model from S3 if not already present"""
    if MODEL_LOCAL_PATH.exists():
        print(f"Model already exists at {MODEL_LOCAL_PATH}, skipping download.")
        return

    bucket = os.environ.get("AWS_S3_BUCKET")
    model_key = os.environ.get("AWS_S3_MODELKEY")
    aws_key = os.environ.get("AWS_S3_KEY")
    aws_secret = os.environ.get("AWS_S3_SECRET")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    
    if not all([bucket, model_key, aws_key, aws_secret]):
        raise ValueError(
            "Missing required environment variables: "
            "AWS_S3_BUCKET, AWS_S3_MODELKEY, AWS_S3_KEY, AWS_S3_SECRET"
        )
    
    os.makedirs(MODEL_LOCAL_PATH.parent, exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )

    print(f"Downloading model from s3://{bucket}/{model_key} -> {MODEL_LOCAL_PATH}")
    s3.download_file(bucket, model_key, str(MODEL_LOCAL_PATH))
    print(f"Model downloaded successfully to {MODEL_LOCAL_PATH}")


# ============================================================================
# INITIALIZATION - Runs once when container starts
# ============================================================================

print("=" * 60)
print("SliceMe Runpods Handler Initialization")
print("=" * 60)

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  WARNING: CUDA not available! Model will run on CPU (SLOW!)")

# Download model from S3
print("\nChecking for model...")
download_model_from_s3()

# Load model and move to GPU
print(f"\nLoading YOLO model from {MODEL_LOCAL_PATH}...")
model = YOLO(str(MODEL_LOCAL_PATH))

# Explicitly move model to GPU if available
if torch.cuda.is_available():
    model.to('cuda')
    print("✓ Model loaded on GPU")
else:
    print("✓ Model loaded on CPU")

print("=" * 60)
print("Initialization complete! Ready to process requests.")
print("=" * 60)



def handler(event):
    """
    Runpods serverless function handler
    
    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image",
            "confidence": 0.25
        }
    }
    """
    try:
        job_input = event["input"]
        image_b64 = job_input["image"]
        confidence = job_input.get("confidence", 0.25)
        
        print(f"Processing image with confidence threshold: {confidence}")
        
        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        # Run inference on GPU
        # YOLO will automatically use GPU if model is on GPU
        results = model(image, conf=confidence, task="segment", device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Format results
        detections = []
        for result in results:
            if result.masks is None:
                for box in result.boxes:
                    detection = {
                        "class_id": int(box.cls),
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "mask": None
                    }
                    detections.append(detection)
            else:
                for box, mask in zip(result.boxes, result.masks):
                    detection = {
                        "class_id": int(box.cls),
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "mask": mask.xy[0].tolist() if mask else None
                    }
                    detections.append(detection)
        
        print(f"✓ Inference complete: found {len(detections)} detections")
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
    except Exception as e:
        print(f"✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "detections": []
        }


if __name__ == "__main__":
    print("\nStarting Runpods serverless handler...")
    runpod.serverless.start({"handler": handler})