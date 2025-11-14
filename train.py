#!/usr/bin/env python3
"""
Train YOLOv8s on combined COCO + FOD + Mechanical Tools dataset
Uses transfer learning from pretrained yolov8s.pt
"""
import torch
from ultralytics import YOLO
from pathlib import Path

def main():
    print("=" * 70)
    print("YOLOv8 Training - Combined Dataset (86 classes)")
    print("=" * 70)
    
    # Check GPU
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = 0
    else:
        print("⚠️  No GPU detected! Training will be very slow on CPU.")
        device = 'cpu'
    
    # Paths
    base_dir = Path("/root/workspace/drone-azooz")
    model_path = base_dir / "yolov8s.pt"
    data_yaml = base_dir / "combined_data.yaml"
    
    # Check if files exist
    if not model_path.exists():
        print(f"\n✗ Pretrained model not found: {model_path}")
        print("Downloading yolov8s.pt...")
        model = YOLO("yolov8s.pt")  # Will auto-download
    else:
        print(f"\n✓ Using pretrained model: {model_path}")
        model = YOLO(str(model_path))
    
    if not data_yaml.exists():
        print(f"\n✗ Data config not found: {data_yaml}")
        print("Please run merge_datasets.py first!")
        return
    
    print(f"✓ Data config: {data_yaml}")
    
    # Training parameters
    epochs = 20
    imgsz = 640
    batch = 16
    
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Model: YOLOv8s (pretrained on COCO)")
    print(f"Classes: 86 (80 COCO + 6 custom)")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Train
    print("\nStarting training...")
    print("This will take a while. Training progress will be saved to runs/detect/train/")
    print("-" * 70)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/detect',
        name='combined_yolo',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        
        # Performance settings for RTX 4070
        workers=8,
        amp=True,  # Automatic Mixed Precision for faster training
        
        # Learning rate settings
        lr0=0.01,
        lrf=0.01,
        
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
    )
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    print(f"\nResults saved to: runs/detect/combined_yolo/")
    print(f"Best model: runs/detect/combined_yolo/weights/best.pt")
    print(f"Last model: runs/detect/combined_yolo/weights/last.pt")
    
    # Validation
    print("\n" + "=" * 70)
    print("Running validation on best model...")
    print("=" * 70)
    
    best_model = YOLO('runs/detect/combined_yolo/weights/best.pt')
    metrics = best_model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    print("\n✓ All done! You can now use the trained model for inference.")


if __name__ == "__main__":
    main()
