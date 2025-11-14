#!/usr/bin/env python3
"""
Merge COCO dataset with custom FOD and Mechanical Tools datasets
- COCO classes: 0-79 (keep as is)
- FOD class 0 → 80 (renamed to "FOD")
- Mechanical tools classes 0-4 → 81-85
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("/root/workspace/drone-azooz")
COCO_DIR = BASE_DIR / "datasets" / "coco"
FOD_DIR = BASE_DIR / "FOD.v2i.yolov8"
MECH_DIR = BASE_DIR / "Mechanical tools-10000.v1i.yolov8"
OUTPUT_DIR = BASE_DIR / "datasets" / "combined"

# Class remapping
# FOD: class 0 → class 80
# Mechanical tools: class 0-4 → class 81-85
FOD_CLASS_OFFSET = 80
MECH_CLASS_OFFSET = 81


def remap_label_file(input_file, output_file, class_offset):
    """
    Remap class IDs in a YOLO label file
    Each line: class_id x_center y_center width height
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Remap class ID
                old_class = int(parts[0])
                new_class = old_class + class_offset
                parts[0] = str(new_class)
                f_out.write(' '.join(parts) + '\n')


def copy_dataset(src_images, src_labels, dst_images, dst_labels, class_offset=0, desc="Copying"):
    """
    Copy images and labels from source to destination with optional class remapping
    """
    src_images = Path(src_images)
    src_labels = Path(src_labels)
    dst_images = Path(dst_images)
    dst_labels = Path(dst_labels)
    
    # Create destination directories
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(src_images.glob('*.[jp][pn][g]')) + list(src_images.glob('*.jpeg'))
    
    print(f"\n{desc}: {len(image_files)} images")
    
    for img_file in tqdm(image_files, desc=desc):
        # Copy image
        dst_img = dst_images / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)
        
        # Copy and remap label if exists
        label_file = src_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            dst_label = dst_labels / f"{img_file.stem}.txt"
            if class_offset == 0:
                # No remapping needed (COCO)
                if not dst_label.exists():
                    shutil.copy2(label_file, dst_label)
            else:
                # Remap class IDs
                remap_label_file(label_file, dst_label, class_offset)


def main():
    print("=" * 60)
    print("Merging COCO + FOD + Mechanical Tools datasets")
    print("=" * 60)
    
    # Check if COCO exists
    coco_exists = (COCO_DIR / "images" / "train2017").exists()
    
    if not coco_exists:
        print("\n⚠️  COCO dataset not found!")
        print(f"Expected location: {COCO_DIR}")
        print("\nOptions:")
        print("1. Skip COCO for now (train only on custom datasets)")
        print("2. Download COCO manually:")
        print("   - Images: http://images.cocodataset.org/zips/train2017.zip")
        print("   - Images: http://images.cocodataset.org/zips/val2017.zip")
        print("   - Labels: Download YOLO format labels from Ultralytics")
        print("\nFor now, proceeding with FOD + Mechanical Tools only...")
        use_coco = False
    else:
        use_coco = True
        print(f"\n✓ COCO dataset found at: {COCO_DIR}")
    
    # Check custom datasets
    if not FOD_DIR.exists():
        print(f"\n✗ FOD dataset not found at: {FOD_DIR}")
        return
    if not MECH_DIR.exists():
        print(f"\n✗ Mechanical Tools dataset not found at: {MECH_DIR}")
        return
    
    print(f"\n✓ FOD dataset found at: {FOD_DIR}")
    print(f"✓ Mechanical Tools dataset found at: {MECH_DIR}")
    
    # === TRAIN SET ===
    print("\n" + "=" * 60)
    print("Processing TRAIN set")
    print("=" * 60)
    
    train_img_dir = OUTPUT_DIR / "images" / "train"
    train_lbl_dir = OUTPUT_DIR / "labels" / "train"
    
    if use_coco:
        # Copy COCO train (no remapping)
        copy_dataset(
            COCO_DIR / "images" / "train2017",
            COCO_DIR / "labels" / "train2017",
            train_img_dir,
            train_lbl_dir,
            class_offset=0,
            desc="COCO train"
        )
    
    # Copy FOD train (remap to class 80)
    copy_dataset(
        FOD_DIR / "train" / "images",
        FOD_DIR / "train" / "labels",
        train_img_dir,
        train_lbl_dir,
        class_offset=FOD_CLASS_OFFSET,
        desc="FOD train"
    )
    
    # Copy Mechanical Tools train (remap to classes 81-85)
    copy_dataset(
        MECH_DIR / "train" / "images",
        MECH_DIR / "train" / "labels",
        train_img_dir,
        train_lbl_dir,
        class_offset=MECH_CLASS_OFFSET,
        desc="Mechanical Tools train"
    )
    
    # === VALIDATION SET ===
    print("\n" + "=" * 60)
    print("Processing VALIDATION set")
    print("=" * 60)
    
    val_img_dir = OUTPUT_DIR / "images" / "val"
    val_lbl_dir = OUTPUT_DIR / "labels" / "val"
    
    if use_coco:
        # Copy COCO val (no remapping)
        copy_dataset(
            COCO_DIR / "images" / "val2017",
            COCO_DIR / "labels" / "val2017",
            val_img_dir,
            val_lbl_dir,
            class_offset=0,
            desc="COCO val"
        )
    
    # Copy FOD valid (remap to class 80)
    copy_dataset(
        FOD_DIR / "valid" / "images",
        FOD_DIR / "valid" / "labels",
        val_img_dir,
        val_lbl_dir,
        class_offset=FOD_CLASS_OFFSET,
        desc="FOD valid"
    )
    
    # Copy Mechanical Tools valid (remap to classes 81-85)
    copy_dataset(
        MECH_DIR / "valid" / "images",
        MECH_DIR / "valid" / "labels",
        val_img_dir,
        val_lbl_dir,
        class_offset=MECH_CLASS_OFFSET,
        desc="Mechanical Tools valid"
    )
    
    print("\n" + "=" * 60)
    print("✓ Dataset merge complete!")
    print("=" * 60)
    print(f"\nCombined dataset location: {OUTPUT_DIR}")
    print(f"  - Train images: {train_img_dir}")
    print(f"  - Train labels: {train_lbl_dir}")
    print(f"  - Val images: {val_img_dir}")
    print(f"  - Val labels: {val_lbl_dir}")
    
    # Count files
    train_imgs = len(list(train_img_dir.glob('*.[jp][pn][g]'))) if train_img_dir.exists() else 0
    val_imgs = len(list(val_img_dir.glob('*.[jp][pn][g]'))) if val_img_dir.exists() else 0
    print(f"\nTotal train images: {train_imgs}")
    print(f"Total validation images: {val_imgs}")


if __name__ == "__main__":
    main()
