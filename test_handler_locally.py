#!/usr/bin/env python3
"""
Local Handler Testing Script

Tests the SAHI-based handler.py logic locally before deploying to RunPod.
Usage:
    python test_handler_locally.py /path/to/test_image.jpg [confidence]
"""

import sys
import base64
import json
from pathlib import Path
from PIL import Image
import io
import torch


def test_handler_with_image(image_path, confidence=0.25):
    """Simulate the SAHI-based handler logic directly."""
    print("=" * 60)
    print("Local Handler Test (SAHI sliding-window)")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Confidence: {confidence}")
    print()

    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    # Step 1: Encode image (simulating client payload)
    print("Step 1: Encoding image to base64...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"✓ Encoded: {len(image_b64)} chars")
    print()

    # Step 2: Decode and prepare
    print("Step 2: Decoding to PIL Image...")
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    print(f"✓ {image.size[0]}x{image.size[1]}, mode={image.mode}")
    print()

    # Step 3: Run SAHI sliced inference
    print("Step 3: Running SAHI sliced inference...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=confidence,
        device=device,
    )

    result = get_sliced_prediction(
        image=image,
        detection_model=sahi_model,
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type='NMS',
        postprocess_match_metric='IOS',
    )

    # Step 4: Parse detections
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
        detections.append({
            "class_id": pred.category.id,
            "class_name": pred.category.name,
            "confidence": pred.score.value,
            "bbox": bbox,
        })

    print(f"✓ Found {len(detections)} detections")
    print()

    # Step 5: Display results
    print("=" * 60)
    print("Results (matches RunPod response format)")
    print("=" * 60)

    output = {
        "success": True,
        "detections": detections,
        "count": len(detections),
        "device": device,
    }
    print(json.dumps(output, indent=2))
    print()

    if detections:
        from collections import Counter
        cls_counts = Counter(d['class_name'] for d in detections)
        print(f"✅ SUCCESS — {len(detections)} detections")
        print(f"   Per class: {dict(cls_counts)}")
        top = detections[0]
        print(f"   Top: {top['class_name']} @ {top['confidence']:.1%}")
    else:
        print("⚠️  No detections (try lowering confidence threshold)")

    return output


def test_actual_handler():
    """Import and call the runpods_handler.py directly."""
    print("\n" + "=" * 60)
    print("Testing Actual runpods_handler.py")
    print("=" * 60)

    try:
        import runpods_handler as handler

        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

        event = {
            "input": {
                "image": image_b64,
                "confidence": confidence,
                "tile_size": 1024,
                "tile_overlap": 0.2,
            }
        }

        print("Calling runpods_handler.handler(event)...")
        result = handler.handler(event)

        print("\n" + "=" * 60)
        print("Handler Response")
        print("=" * 60)
        print(json.dumps(result, indent=2))

        if result.get("success"):
            print(f"\n✅ Handler test passed — {result.get('count')} detections")
        else:
            print(f"\n❌ Handler error: {result.get('error')}")

        return result

    except ImportError as e:
        print(f"⚠️  Could not import runpods_handler.py: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_handler_locally.py <image_path> [confidence]")
        print("  image_path   Path to an email screenshot (PNG/JPG)")
        print("  confidence   Confidence threshold (default 0.25)")
        sys.exit(1)

    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    if not Path("models/best.pt").exists():
        print("⚠️  models/best.pt not found — run with test_actual_handler() only if deployed")

    print(f"Image:      {image_path}")
    print(f"Confidence: {confidence}")
    print()

    test_handler_with_image(image_path, confidence)
    test_actual_handler()


if __name__ == "__main__":
    main()
