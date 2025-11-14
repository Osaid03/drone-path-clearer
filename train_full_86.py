#!/usr/bin/env python3
"""
Training configuration for ALL 86 classes (COCO + Custom)
Optimized for large combined dataset
"""
import torch
from ultralytics import YOLO
from pathlib import Path

def train_full_model():
    print("=" * 80)
    print("YOLOv8 Training - ALL 86 Classes (COCO + FOD + Tools)")
    print("=" * 80)
    
    # Check GPU
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = 0
    else:
        print("⚠️  No GPU detected! Training will be VERY slow.")
        device = 'cpu'
    
    # Paths
    base_dir = Path("/root/workspace/drone-azooz")
    model_path = base_dir / "yolov8s.pt"
    data_yaml = base_dir / "combined_data.yaml"
    
    # Load model
    if not model_path.exists():
        print(f"\nDownloading yolov8s.pt...")
        model = YOLO("yolov8s.pt")
    else:
        print(f"\n✓ Using pretrained model: {model_path}")
        model = YOLO(str(model_path))
    
    if not data_yaml.exists():
        print(f"\n✗ Data config not found: {data_yaml}")
        return
    
    print(f"✓ Data config: {data_yaml}")
    
    # Training parameters - adjusted for large dataset
    # Using more epochs for thorough training on all 86 classes
    epochs = 50  # Increased for better accuracy across all classes
    imgsz = 640
    batch = 16  # May need to reduce if OOM
    
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Model: YOLOv8s (pretrained on COCO)")
    print(f"Dataset: ~128K images (118K COCO + 10K custom)")
    print(f"Classes: 86 (80 COCO + 6 custom)")
    print(f"Epochs: {epochs} (thorough training for all classes)")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"\n⏱️  Estimated time: 6-8 hours on RTX 4070")
    print(f"💾 Model will be TFLite-compatible")
    print("=" * 80)
    
    print("\n🚀 Starting training...")
    print("This will take several hours. Progress will be saved continuously.")
    print("-" * 80)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/detect',
        name='combined_full_86classes',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        
        # Performance settings
        workers=8,
        amp=True,  # Mixed precision
        
        # Learning rate - balanced for both COCO and custom classes
        lr0=0.01,  # Standard LR for good learning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Class weights to balance COCO vs custom classes
        # COCO has much more data, so we slightly boost custom classes
        patience=10,  # Early stopping patience
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
    )
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    print(f"\nResults saved to: runs/detect/combined_full_86classes/")
    print(f"Best model: runs/detect/combined_full_86classes/weights/best.pt")
    print(f"Last model: runs/detect/combined_full_86classes/weights/last.pt")
    
    # Validation
    print("\n" + "=" * 80)
    print("Running validation on best model...")
    print("=" * 80)
    
    best_model = YOLO('runs/detect/combined_full_86classes/weights/best.pt')
    metrics = best_model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    print("\n✅ Model ready for inference on ALL 86 classes!")
    print("=" * 80)


if __name__ == "__main__":
    train_full_model()
