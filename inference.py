#!/usr/bin/env python3
"""
Run inference using trained YOLOv8 model
Detects all 86 classes: 80 COCO + 6 custom (FOD, drill, hammer, pliers, screwdriver, wrench)
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference on images or video')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, folder, or video file')
    parser.add_argument('--model', type=str, default='runs/detect/combined_yolo/weights/best.pt',
                        help='Path to trained model (default: best.pt from training)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    parser.add_argument('--show', action='store_true',
                        help='Show detection results')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use: 0 for GPU, cpu for CPU (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLOv8 Inference - Combined Model (86 classes)")
    print("=" * 70)
    
    # Check GPU
    if args.device != 'cpu':
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n✗ Model not found: {model_path}")
        print("Please train the model first using train.py")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Check source
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"\n✗ Source not found: {source_path}")
        return
    
    print(f"✓ Source: {source_path}")
    
    print("\n" + "=" * 70)
    print("Detection Configuration")
    print("=" * 70)
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    print(f"Save results: {args.save}")
    print(f"Show results: {args.show}")
    print("=" * 70)
    
    # Run inference
    print("\nRunning detection...")
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        show=args.show,
        verbose=True,
        project='runs/detect',
        name='inference',
        exist_ok=True,
    )
    
    print("\n" + "=" * 70)
    print("✓ Detection complete!")
    print("=" * 70)
    
    # Print detection summary
    if results:
        print("\nDetection Summary:")
        for i, result in enumerate(results):
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy()
                unique_classes = set(classes)
                print(f"\nImage {i+1}: {len(result.boxes)} detections")
                for cls_id in unique_classes:
                    cls_name = result.names[int(cls_id)]
                    count = sum(classes == cls_id)
                    print(f"  - {cls_name}: {count}")
            else:
                print(f"\nImage {i+1}: No detections")
    
    if args.save:
        print(f"\nResults saved to: runs/detect/inference/")


if __name__ == "__main__":
    main()
