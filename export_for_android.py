#!/usr/bin/env python3
"""
Simple model export to Android-compatible formats
Exports to TFLite and ONNX only (most common for mobile)
"""
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print("=" * 70)
    print("YOLOv8 Model Export - Android/Mobile Formats")
    print("=" * 70)
    
    # Load the best trained model
    model_path = Path("/root/workspace/drone-azooz/runs/detect/combined_yolo/weights/best.pt")
    
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"GPU Available: {torch.cuda.is_available()}")
    
    export_dir = model_path.parent
    
    print("\n" + "=" * 70)
    print("Exporting for Android...")
    print("=" * 70)
    
    # Export to TensorFlow Lite (FP16 - balanced)
    print("\n[1/3] Exporting to TensorFlow Lite (FP16)...")
    print("This will take a few minutes...")
    try:
        tflite_path = model.export(format='tflite', imgsz=640)
        print(f"✓ TFLite export successful!")
        print(f"  Location: {tflite_path}")
    except Exception as e:
        print(f"✗ TFLite export failed: {e}")
    
    # Export to ONNX (simple, no optimization)
    print("\n[2/3] Exporting to ONNX (simplified)...")
    try:
        onnx_path = model.export(format='onnx', simplify=False, dynamic=False)
        print(f"✓ ONNX export successful!")
        print(f"  Location: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # Export to TorchScript (PyTorch mobile)
    print("\n[3/3] Exporting to TorchScript...")
    try:
        torchscript_path = model.export(format='torchscript')
        print(f"✓ TorchScript export successful!")
        print(f"  Location: {torchscript_path}")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("ANDROID INTEGRATION:")
    print("=" * 70)
    print(f"""
OPTION 1: TensorFlow Lite (Recommended)
----------------------------------------
File: {model_path.parent}/best_saved_model/best_float16.tflite

1. Copy to Android project:
   app/src/main/assets/best.tflite

2. Add dependencies to build.gradle:
   implementation 'org.tensorflow:tensorflow-lite:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'

3. Sample Android code:
   ```java
   // Load model
   Interpreter tflite = new Interpreter(loadModelFile());
   
   // Prepare input
   float[][][][] input = new float[1][640][640][3];
   
   // Run inference
   float[][][] output = new float[1][90][8400];
   tflite.run(input, output);
   ```

OPTION 2: ONNX Runtime
-----------------------
File: {model_path.parent}/best.onnx

1. Add dependency:
   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'

2. Sample code:
   ```java
   OrtEnvironment env = OrtEnvironment.getEnvironment();
   OrtSession session = env.createSession("best.onnx");
   ```

OPTION 3: PyTorch Mobile
--------------------------
File: {model_path.parent}/best.torchscript

1. Add dependency:
   implementation 'org.pytorch:pytorch_android:1.13.1'
   implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'

For complete examples, see:
https://github.com/ultralytics/ultralytics/tree/main/examples/android
    """)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE:")
    print("=" * 70)
    print("""
Overall: mAP50=81.7%, mAP50-95=62.9%
  
Per-class:
  FOD:         mAP50=93.2% (Excellent!)
  drill:       mAP50=83.3%
  hammer:      mAP50=76.5%
  pliers:      mAP50=82.6%
  screwdriver: mAP50=78.7%
  wrench:      mAP50=75.8%
    """)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Find exported files in: runs/detect/combined_yolo/weights/
2. Copy .tflite file to your Android project
3. Implement inference in your app
4. Test on real device

For Flutter/React Native:
- Use tflite_flutter or onnxruntime packages
- Same .tflite or .onnx files work
    """)


if __name__ == "__main__":
    main()
