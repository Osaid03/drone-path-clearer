#!/usr/bin/env python3
"""
Export trained model to TFLite format for Android deployment
Handles the conversion properly to ensure compatibility
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import subprocess
import sys

def export_to_tflite():
    print("=" * 80)
    print("Model Export to TFLite for Android")
    print("=" * 80)
    
    model_path = Path("runs/detect/combined_full_86classes/weights/best.pt")
    
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("Please train the model first using train_full_86.py")
        return False
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"✓ Model loaded with {model.model.nc} classes")
    
    # Step 1: Export to ONNX (always works)
    print("\n" + "=" * 80)
    print("Step 1: Exporting to ONNX")
    print("=" * 80)
    
    try:
        onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
        print(f"✅ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False
    
    # Step 2: Export to TorchScript (mobile format)
    print("\n" + "=" * 80)
    print("Step 2: Exporting to TorchScript")
    print("=" * 80)
    
    try:
        ts_path = model.export(format='torchscript', imgsz=640)
        print(f"✅ TorchScript export successful: {ts_path}")
    except Exception as e:
        print(f"⚠️  TorchScript export failed: {e}")
    
    # Step 3: Try TFLite export with better error handling
    print("\n" + "=" * 80)
    print("Step 3: Exporting to TFLite (for Android)")
    print("=" * 80)
    
    print("\n📦 Installing TFLite dependencies...")
    
    # Install required packages for TFLite export
    packages = [
        "tensorflow==2.16.1",  # Specific version for compatibility
        "onnx2tf>=1.20.0",
        "sng4onnx>=1.0.1",
        "onnxruntime>=1.17.0",
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", package],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Failed to install {package}")
            print(f"      {e.stderr}")
    
    print("\n🔄 Attempting TFLite export...")
    
    try:
        # Try full TFLite export
        tflite_path = model.export(
            format='tflite',
            imgsz=640,
            int8=False,  # FP32 for better accuracy
        )
        print(f"✅ TFLite export successful: {tflite_path}")
        return True
        
    except Exception as e:
        print(f"\n⚠️  Direct TFLite export failed: {e}")
        print("\n💡 Alternative: Convert ONNX to TFLite manually")
        print("\nYou can use these alternatives:")
        print("1. Use ONNX format (best.onnx) with ONNX Runtime Mobile")
        print("2. Use TorchScript format for PyTorch Mobile")
        print("3. Convert ONNX to TFLite using online tools")
        return False
    
    finally:
        print("\n" + "=" * 80)
        print("Export Summary")
        print("=" * 80)
        
        weights_dir = model_path.parent
        print(f"\n📁 Available model formats in: {weights_dir}")
        
        formats = {
            'best.pt': 'PyTorch (Python inference)',
            'best.onnx': 'ONNX (cross-platform, Android compatible)',
            'best.torchscript': 'TorchScript (PyTorch Mobile)',
            'best_saved_model': 'TensorFlow SavedModel',
            'best_float32.tflite': 'TFLite FP32 (Android)',
            'best_int8.tflite': 'TFLite INT8 (Android, smaller)',
        }
        
        print("\n✅ Successfully exported:")
        for filename, desc in formats.items():
            file_path = weights_dir / filename
            if file_path.exists() or (file_path.parent / file_path.stem).exists():
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  • {filename} ({size_mb:.1f} MB) - {desc}")
                elif (file_path.parent / file_path.stem).is_dir():
                    print(f"  • {filename}/ (dir) - {desc}")
        
        print("\n📱 For Android deployment:")
        print("  Recommended: best.onnx with ONNX Runtime Mobile")
        print("  Alternative: best.pt with Ultralytics Android SDK")
        print("=" * 80)


if __name__ == "__main__":
    success = export_to_tflite()
    exit(0 if success else 1)
