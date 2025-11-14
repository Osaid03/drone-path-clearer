#!/usr/bin/env python3
"""
Export trained YOLOv8 model to different formats for deployment
Supports: ONNX, TensorFlow Lite, CoreML, TensorRT, and more
"""
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print("=" * 70)
    print("YOLOv8 Model Export - Multiple Formats")
    print("=" * 70)
    
    # Load the best trained model
    model_path = Path("/root/workspace/drone-azooz/runs/detect/combined_yolo/weights/best.pt")
    
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("Please train the model first!")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    
    export_dir = model_path.parent.parent / "exports"
    export_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXPORT OPTIONS:")
    print("=" * 70)
    print("\n1. ONNX (Recommended for most platforms)")
    print("   - Cross-platform")
    print("   - Good performance")
    print("   - Works with: OpenCV, ONNX Runtime, TensorRT")
    
    print("\n2. TensorFlow Lite (Android/iOS/Edge devices)")
    print("   - Optimized for mobile")
    print("   - Smaller size")
    print("   - INT8 quantization available")
    
    print("\n3. TensorRT (NVIDIA GPUs - fastest)")
    print("   - Maximum performance on NVIDIA hardware")
    print("   - Requires NVIDIA GPU")
    
    print("\n4. CoreML (iOS/macOS)")
    print("   - Apple devices only")
    
    print("\n5. OpenVINO (Intel CPUs/GPUs)")
    print("   - Intel hardware optimization")
    
    print("\n" + "=" * 70)
    print("Starting exports...")
    print("=" * 70)
    
    # Export to ONNX (Universal format)
    print("\n[1/5] Exporting to ONNX...")
    try:
        onnx_path = model.export(format='onnx', simplify=True)
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # Export to TensorFlow Lite (Android)
    print("\n[2/5] Exporting to TensorFlow Lite (FP32)...")
    try:
        tflite_path = model.export(format='tflite')
        print(f"✓ TFLite FP32 export successful: {tflite_path}")
    except Exception as e:
        print(f"✗ TFLite export failed: {e}")
    
    # Export to TensorFlow Lite INT8 (Smaller, for Android)
    print("\n[3/5] Exporting to TensorFlow Lite (INT8 - quantized)...")
    try:
        tflite_int8_path = model.export(format='tflite', int8=True)
        print(f"✓ TFLite INT8 export successful: {tflite_int8_path}")
        print("   (Smaller size, faster inference, slight accuracy loss)")
    except Exception as e:
        print(f"✗ TFLite INT8 export failed: {e}")
    
    # Export to TensorRT (if NVIDIA GPU available)
    if torch.cuda.is_available():
        print("\n[4/5] Exporting to TensorRT (FP16)...")
        try:
            trt_path = model.export(format='engine', half=True)
            print(f"✓ TensorRT export successful: {trt_path}")
            print("   (Maximum speed on NVIDIA GPUs)")
        except Exception as e:
            print(f"✗ TensorRT export failed: {e}")
    else:
        print("\n[4/5] Skipping TensorRT (No GPU detected)")
    
    # Export to CoreML (iOS)
    print("\n[5/5] Exporting to CoreML (iOS/macOS)...")
    try:
        coreml_path = model.export(format='coreml')
        print(f"✓ CoreML export successful: {coreml_path}")
    except Exception as e:
        print(f"✗ CoreML export failed: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Export process complete!")
    print("=" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPORTED FILES SUMMARY:")
    print("=" * 70)
    print(f"\nExport directory: {model_path.parent}")
    print("\nFor Android development, use:")
    print("  - best.tflite (FP32) - Higher accuracy")
    print("  - best_int8.tflite (INT8) - Smaller, faster")
    print("  - best.onnx - Universal format")
    
    print("\nFor web/JavaScript:")
    print("  - Use ONNX with onnxruntime-web")
    
    print("\nFor NVIDIA Jetson/embedded:")
    print("  - Use TensorRT (.engine)")
    
    print("\nFor iOS:")
    print("  - Use CoreML (.mlpackage)")
    
    print("\n" + "=" * 70)
    print("ANDROID INTEGRATION GUIDE:")
    print("=" * 70)
    print("""
For Android (Java/Kotlin):
1. Copy best.tflite to app/src/main/assets/
2. Add TensorFlow Lite dependencies to build.gradle:
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
3. Use TFLite Interpreter API

For Android (Native):
1. Use ONNX Runtime Mobile
2. Or use MediaPipe with the TFLite model

Sample Android code available at:
https://github.com/ultralytics/ultralytics/tree/main/examples/android
    """)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE:")
    print("=" * 70)
    print("""
Validation Results:
  Overall mAP50: 81.7%
  Overall mAP50-95: 62.9%
  
Per-class performance:
  FOD:         mAP50=93.2%, Precision=93.9%, Recall=88.5%
  drill:       mAP50=83.3%, Precision=84.7%, Recall=74.6%
  hammer:      mAP50=76.5%, Precision=82.7%, Recall=64.3%
  pliers:      mAP50=82.6%, Precision=85.0%, Recall=70.3%
  screwdriver: mAP50=78.7%, Precision=78.0%, Recall=72.9%
  wrench:      mAP50=75.8%, Precision=79.3%, Recall=65.2%
    """)


if __name__ == "__main__":
    main()
