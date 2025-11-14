#!/usr/bin/env python3
"""
Android Model Export Guide
Exports YOLOv8 model to Android-compatible formats
"""
from pathlib import Path

def main():
    print("=" * 80)
    print("YOLOv8 Model Export for Android")
    print("=" * 80)
    
    weights_dir = Path("runs/detect/combined_yolo/weights")
    
    print("\n📦 AVAILABLE MODELS:\n")
    
    # Check what we have
    models = {
        "best.pt": "PyTorch format (training/inference on PC)",
        "best.onnx": "ONNX format (cross-platform, recommended for Android)",
        "best.torchscript": "TorchScript format (mobile deployment)",
    }
    
    print("✅ Successfully Exported Models:")
    for model_file, description in models.items():
        file_path = weights_dir / model_file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"\n  📄 {model_file}")
            print(f"     Size: {size_mb:.1f} MB")
            print(f"     Use: {description}")
            print(f"     Path: {file_path}")
    
    print("\n" + "=" * 80)
    print("📱 FOR ANDROID DEPLOYMENT:")
    print("=" * 80)
    
    print("\n✅ RECOMMENDED: ONNX Format (best.onnx)")
    print("   - Size: ~43 MB")
    print("   - Compatible with ONNX Runtime Mobile")
    print("   - Good performance on Android")
    print("   - Location: runs/detect/combined_yolo/weights/best.onnx")
    
    print("\n🔧 HOW TO USE ON ANDROID:")
    print("\n1. Using ONNX Runtime Mobile:")
    print("   - Add dependency: com.microsoft.onnxruntime:onnxruntime-android")
    print("   - Copy best.onnx to your Android app assets/")
    print("   - Use ONNX Runtime API for inference")
    print("   - Tutorial: https://onnxruntime.ai/docs/tutorials/mobile/")
    
    print("\n2. Using Ultralytics Android SDK (Easiest):")
    print("   - Use best.pt directly with Ultralytics Android library")
    print("   - Add: implementation 'com.github.ultralytics:ultralytics-android'")
    print("   - Copy best.pt to assets/")
    print("   - Code example:")
    print("""
    val yolo = YOLO("best.pt")
    val results = yolo.predict(bitmap)
    """)
    
    print("\n3. Using TensorFlow Lite (if you need .tflite):")
    print("   - TFLite export failed due to Python 3.13 compatibility")
    print("   - Alternative: Use ONNX to TFLite converter separately")
    print("   - Or use ONNX Runtime (recommended)")
    
    print("\n" + "=" * 80)
    print("📊 MODEL DETAILS:")
    print("=" * 80)
    print("\nInput:")
    print("  - Format: RGB image")
    print("  - Shape: [1, 3, 640, 640] (batch, channels, height, width)")
    print("  - Preprocessing: Resize to 640x640, normalize to [0, 1]")
    
    print("\nOutput:")
    print("  - Shape: [1, 90, 8400]")
    print("  - 8400 predictions with 90 values each")
    print("  - First 4 values: bounding box (x, y, w, h)")
    print("  - Next 86 values: class probabilities (86 classes)")
    
    print("\n86 Classes:")
    print("  - Classes 0-79: COCO objects (not trained, pretrained weights)")
    print("  - Class 80: FOD (Foreign Object Debris)")
    print("  - Class 81: drill")
    print("  - Class 82: hammer")
    print("  - Class 83: pliers")
    print("  - Class 84: screwdriver")
    print("  - Class 85: wrench")
    
    print("\n" + "=" * 80)
    print("🚀 QUICK START FOR ANDROID:")
    print("=" * 80)
    
    print("\nOption 1: Ultralytics Android (Simplest)")
    print("""
    // In your Android app build.gradle
    dependencies {
        implementation 'com.github.ultralytics:ultralytics-android:8.3.0'
    }
    
    // Copy best.pt to app/src/main/assets/
    
    // In your Activity/Fragment:
    val yolo = YOLO("best.pt")
    val results = yolo.predict(bitmap)
    
    // Process results
    for (result in results) {
        val boxes = result.boxes
        for (box in boxes) {
            val cls = box.cls  // Class ID
            val conf = box.conf  // Confidence
            val xyxy = box.xyxy  // Bounding box
        }
    }
    """)
    
    print("\nOption 2: ONNX Runtime (More control)")
    print("""
    // In your build.gradle
    dependencies {
        implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
    }
    
    // Copy best.onnx to app/src/main/assets/
    
    // Load and run model:
    val env = OrtEnvironment.getEnvironment()
    val session = env.createSession(modelBytes, OrtSession.SessionOptions())
    
    // Preprocess image to [1, 3, 640, 640] float array
    val input = preprocessImage(bitmap)
    
    // Run inference
    val output = session.run(mapOf("images" to OnnxTensor.createTensor(env, input)))
    
    // Post-process output to get detections
    val results = postprocess(output)
    """)
    
    print("\n" + "=" * 80)
    print("📦 FILES TO COPY TO ANDROID PROJECT:")
    print("=" * 80)
    print(f"\n✅ For Ultralytics SDK: {weights_dir}/best.pt (22 MB)")
    print(f"✅ For ONNX Runtime:    {weights_dir}/best.onnx (43 MB)")
    print("\n💡 Tip: Use best.pt with Ultralytics Android for easiest integration!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
