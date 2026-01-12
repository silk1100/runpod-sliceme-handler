#!/usr/bin/env python3
"""
Local Handler Testing Script

Tests your handler.py logic locally before deploying to RunPod.
This simulates exactly what RunPod does when it receives a request.

Usage:
    python test_handler_locally.py /path/to/test_image.jpg
"""

import sys
import base64
import json
from pathlib import Path

# ============================================================
# Simulate the handler function
# ============================================================

def test_handler_with_image(image_path, confidence=0.25):
    """
    Test the handler function with a local image file.
    This simulates what RunPod does.
    """
    print("="*60)
    print("Local Handler Test")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Confidence: {confidence}")
    print()
    
    # Step 1: Load and encode image (simulating client)
    print("Step 1: Encoding image to base64 (simulating client)...")
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        print(f"✓ Encoded: {len(image_b64)} characters")
        print()
    except FileNotFoundError:
        print(f"❌ Error: Image not found: {image_path}")
        return
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return
    
    # Step 2: Create RunPod event structure
    print("Step 2: Creating RunPod event structure...")
    event = {
        "input": {
            "image": image_b64,
            "confidence": confidence
        }
    }
    print(f"✓ Event created")
    print()
    
    # Step 3: Test OLD method (raw bytes) - This will fail
    print("Step 3: Testing OLD method (passing raw bytes to YOLO)...")
    print("-" * 60)
    try:
        from ultralytics import YOLO
        
        # Load model
        model = YOLO("models/best.pt")
        
        # Decode image
        image_bytes_decoded = base64.b64decode(image_b64)
        
        # Try to pass raw bytes to YOLO (THIS WILL FAIL)
        print("Attempting: model(raw_bytes)")
        results = model(image_bytes_decoded, conf=confidence)
        
        print("❌ Unexpected: Raw bytes worked! (This shouldn't happen)")
        
    except Exception as e:
        print(f"✓ Expected error: {type(e).__name__}: {str(e)[:100]}")
        print("   This is the error you got on RunPod!")
    
    print()
    
    # Step 4: Test NEW method (PIL Image) - This will work
    print("Step 4: Testing NEW method (converting to PIL Image)...")
    print("-" * 60)
    try:
        from ultralytics import YOLO
        from PIL import Image
        import io
        
        # Load model
        model = YOLO("models/best.pt")
        
        # Decode image
        image_bytes_decoded = base64.b64decode(image_b64)
        
        # Convert to PIL Image (THE FIX!)
        print("Converting bytes to PIL Image...")
        image = Image.open(io.BytesIO(image_bytes_decoded))
        print(f"✓ PIL Image created: {image.size[0]}x{image.size[1]} pixels, mode={image.mode}")
        
        # Pass PIL Image to YOLO (THIS WORKS!)
        print("Running inference with PIL Image...")
        results = model(image, conf=confidence)
        
        # Parse results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        "class_id": int(box.cls),
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                    }
                    detections.append(detection)
        
        print(f"✓ Inference complete: {len(detections)} detections found")
        print()
        
        # Display results
        print("="*60)
        print("Results (What RunPod will return)")
        print("="*60)
        
        output = {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
        
        print(json.dumps(output, indent=2))
        print()
        
        # Summary
        if len(detections) > 0:
            print(f"✅ SUCCESS! Found {len(detections)} detections")
            print(f"\nTop detection:")
            top = detections[0]
            print(f"  Class: {top['class_name']}")
            print(f"  Confidence: {top['confidence']:.2%}")
            print(f"  BBox: {top['bbox']}")
        else:
            print("⚠️  No detections found (try lowering confidence threshold)")
        
        return output
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_actual_handler():
    """
    Test your actual handler.py file directly.
    This requires your handler.py to be importable.
    """
    print("\n" + "="*60)
    print("Testing Actual Handler Function")
    print("="*60)
    
    try:
        # Import your handler
        # Note: This assumes handler.py is in current directory or PYTHONPATH
        import handler
        
        print("✓ Handler imported successfully")
        print()
        
        # Create test event
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        event = {
            "input": {
                "image": image_b64,
                "confidence": 0.25
            }
        }
        
        # Call handler
        print("Calling handler.handler(event)...")
        result = handler.handler(event)
        
        print("\n" + "="*60)
        print("Handler Response")
        print("="*60)
        print(json.dumps(result, indent=2))
        print()
        
        if result.get("success"):
            print(f"✅ Handler test successful!")
            print(f"   Detections: {result.get('count')}")
        else:
            print(f"❌ Handler returned error: {result.get('error')}")
        
        return result
        
    except ImportError as e:
        print(f"⚠️  Could not import handler.py: {e}")
        print("   This is okay - test using Method 1 instead")
        return None
    except Exception as e:
        print(f"❌ Error testing handler: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "="*60)
    print("Handler Local Testing Tool")
    print("="*60)
    print()
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python test_handler_locally.py <image_path> [confidence]")
        print()
        print("Example:")
        print("  python test_handler_locally.py test_image.jpg")
        print("  python test_handler_locally.py test_image.jpg 0.5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    
    # Verify image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Verify model exists
    if not Path("models/best.pt").exists():
        print("❌ Error: Model not found at models/best.pt")
        print("   Make sure you're running this from your project directory")
        print("   Or download/copy your model to models/best.pt")
        sys.exit(1)
    
    print("Environment:")
    print(f"  Image: {image_path}")
    print(f"  Model: models/best.pt")
    print(f"  Confidence: {confidence}")
    print()
    
    # Test Method 1: Simulate handler logic
    test_handler_with_image(image_path, confidence)
    
    # Test Method 2: Test actual handler.py (if available)
    print("\n")
    test_actual_handler()


if __name__ == "__main__":
    main()