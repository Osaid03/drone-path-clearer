#!/usr/bin/env python3
"""
Download COCO dataset in YOLO format
Uses ultralytics built-in dataset downloading
"""
import os
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize a YOLO model - this will help with dataset download
    print("Downloading COCO dataset...")
    print("This may take a while (several GB)...")
    
    # The easiest way is to train on 'coco.yaml' which will auto-download
    # But we'll create a simple script that uses the ultralytics auto-download feature
    
    # Create a temporary yaml to trigger COCO download
    temp_yaml = Path("/root/workspace/drone-azooz/temp_coco.yaml")
    temp_yaml.write_text("""# COCO dataset - will trigger auto-download
path: ../datasets/coco
train: train2017.txt
val: val2017.txt

nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
""")
    
    print("\nCOCO dataset configuration created.")
    print("Note: COCO will be auto-downloaded when training starts.")
    print("Alternatively, download manually from: http://images.cocodataset.org/zips/")
    print("\nFor this project, we'll download it during the merge process to save time.")
