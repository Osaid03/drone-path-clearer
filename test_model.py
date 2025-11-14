#!/usr/bin/env python3
"""
Test the trained model on sample images to see what it detects
Tests both custom classes and COCO classes
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_collage(results_list, output_path="test_results_collage.jpg"):
    """Create a collage of test results"""
    images = []
    max_width = 0
    total_height = 0
    
    for img_array, detections_text in results_list:
        h, w = img_array.shape[:2]
        # Add text overlay
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Add detection count at top
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), detections_text, fill=(0, 255, 0), font=font)
        img_array = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        images.append(img_array)
        max_width = max(max_width, w)
        total_height += h + 10
    
    # Create collage
    collage = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
    y_offset = 0
    
    for img in images:
        h, w = img.shape[:2]
        collage[y_offset:y_offset+h, 0:w] = img
        y_offset += h + 10
    
    cv2.imwrite(output_path, collage)
    print(f"\n📊 Test collage saved: {output_path}")


def test_model():
    print("=" * 80)
    print("YOLOv8 Model Testing - All 86 Classes")
    print("=" * 80)
    
    # Load model
    model_path = "runs/detect/combined_yolo/weights/best.pt"
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"✓ Model loaded with {model.model.nc} classes")
    
    # Get test images from validation set
    val_img_dir = Path("datasets/combined/images/val")
    
    if not val_img_dir.exists():
        print(f"\n✗ Validation images not found at: {val_img_dir}")
        return
    
    # Get sample images for each custom class
    print("\n" + "=" * 80)
    print("TESTING ON VALIDATION IMAGES")
    print("=" * 80)
    
    # Get all images
    image_files = list(val_img_dir.glob("*.jpg"))[:20]  # Test on 20 images
    
    if not image_files:
        print("\n✗ No images found in validation set")
        return
    
    print(f"\n✓ Testing on {len(image_files)} validation images...")
    
    # Run inference
    results = model.predict(
        source=image_files,
        conf=0.25,
        iou=0.45,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False,
        save=True,
        project='runs/detect',
        name='test_all_classes',
        exist_ok=True,
    )
    
    # Analyze results
    class_detections = {}
    total_detections = 0
    results_for_collage = []
    
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results):
        img_name = image_files[i].name
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            print(f"\n📷 Image {i+1}: {img_name}")
            print(f"   Total detections: {len(result.boxes)}")
            
            # Count by class
            unique_classes = {}
            for cls_id, conf in zip(classes, confs):
                cls_id = int(cls_id)
                cls_name = result.names[cls_id]
                
                if cls_name not in unique_classes:
                    unique_classes[cls_name] = []
                unique_classes[cls_name].append(conf)
                
                # Track global counts
                if cls_name not in class_detections:
                    class_detections[cls_name] = 0
                class_detections[cls_name] += 1
                total_detections += 1
            
            # Print detections for this image
            detections_text = f"{len(result.boxes)} detections"
            for cls_name, confidences in sorted(unique_classes.items()):
                avg_conf = np.mean(confidences)
                print(f"   - {cls_name}: {len(confidences)} (avg conf: {avg_conf:.2f})")
            
            # Get annotated image
            img_array = result.plot()
            results_for_collage.append((img_array, detections_text))
        else:
            print(f"\n📷 Image {i+1}: {img_name}")
            print(f"   No detections")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal images tested: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Unique classes detected: {len(class_detections)}")
    
    print("\n📊 Detections by class:")
    
    # Separate custom and COCO classes
    custom_classes = {}
    coco_classes = {}
    
    for cls_name, count in sorted(class_detections.items(), key=lambda x: x[1], reverse=True):
        # Find class ID
        cls_id = None
        for id, name in model.names.items():
            if name == cls_name:
                cls_id = id
                break
        
        if cls_id is not None and cls_id >= 80:
            custom_classes[cls_name] = count
        else:
            coco_classes[cls_name] = count
    
    if custom_classes:
        print("\n✅ Custom Classes (80-85) - TRAINED:")
        for cls_name, count in sorted(custom_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls_name}: {count} detections")
    
    if coco_classes:
        print("\n🔄 COCO Classes (0-79) - PRETRAINED (not fine-tuned):")
        for cls_name, count in sorted(coco_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls_name}: {count} detections")
    
    print(f"\n💾 Annotated results saved to: runs/detect/test_all_classes/")
    
    # Create collage of some results
    if results_for_collage[:10]:  # First 10 images
        create_test_collage(results_for_collage[:10], "test_results_collage.jpg")
    
    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80)
    
    return class_detections


if __name__ == "__main__":
    test_model()
