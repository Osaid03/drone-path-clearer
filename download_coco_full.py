#!/usr/bin/env python3
"""
Download COCO 2017 dataset in YOLO format
This is a large download (~20GB)
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_coco():
    print("=" * 80)
    print("DOWNLOADING COCO 2017 DATASET")
    print("=" * 80)
    print("\nThis will download:")
    print("  - Train images: ~19GB (118,287 images)")
    print("  - Val images: ~1GB (5,000 images)")
    print("  - YOLO format labels")
    print("\nTotal size: ~20GB")
    print("Time estimate: 20-60 minutes depending on internet speed")
    print("=" * 80)
    
    base_dir = Path("/root/workspace/drone-azooz/datasets/coco")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO dataset URLs
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    }
    
    # Download images
    for name, url in urls.items():
        zip_path = base_dir / f"{name}.zip"
        
        if zip_path.exists():
            print(f"\n✓ {name}.zip already exists, skipping download")
        else:
            print(f"\n📥 Downloading {name}...")
            print(f"   URL: {url}")
            try:
                download_file(url, zip_path)
                print(f"✓ Downloaded: {zip_path}")
            except Exception as e:
                print(f"✗ Error downloading {name}: {e}")
                return False
    
    # Extract images
    print("\n" + "=" * 80)
    print("EXTRACTING IMAGES")
    print("=" * 80)
    
    for name in ['train_images', 'val_images']:
        zip_path = base_dir / f"{name}.zip"
        
        if not zip_path.exists():
            print(f"\n✗ {name}.zip not found")
            continue
        
        extract_dir = base_dir / "images"
        extract_dir.mkdir(exist_ok=True)
        
        folder_name = "train2017" if "train" in name else "val2017"
        if (extract_dir / folder_name).exists():
            print(f"\n✓ {folder_name} already extracted")
        else:
            print(f"\n📦 Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✓ Extracted to: {extract_dir / folder_name}")
    
    # Download YOLO format labels
    print("\n" + "=" * 80)
    print("DOWNLOADING YOLO FORMAT LABELS")
    print("=" * 80)
    
    labels_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
    labels_zip = base_dir / "coco_labels.zip"
    
    if labels_zip.exists():
        print("\n✓ YOLO labels already downloaded")
    else:
        print(f"\n📥 Downloading YOLO labels...")
        try:
            download_file(labels_url, labels_zip)
            print(f"✓ Downloaded: {labels_zip}")
        except Exception as e:
            print(f"✗ Error downloading labels: {e}")
            return False
    
    # Extract labels
    if not (base_dir / "labels").exists() or not (base_dir / "labels" / "train2017").exists():
        print(f"\n📦 Extracting YOLO labels...")
        with zipfile.ZipFile(labels_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Labels might be in a subdirectory
        coco_labels_dir = base_dir / "coco" / "labels"
        if coco_labels_dir.exists():
            shutil.move(str(coco_labels_dir), str(base_dir / "labels"))
            shutil.rmtree(base_dir / "coco")
        
        print(f"✓ Extracted labels")
    else:
        print("\n✓ YOLO labels already extracted")
    
    # Verify structure
    print("\n" + "=" * 80)
    print("VERIFYING DATASET STRUCTURE")
    print("=" * 80)
    
    required_dirs = [
        base_dir / "images" / "train2017",
        base_dir / "images" / "val2017",
        base_dir / "labels" / "train2017",
        base_dir / "labels" / "val2017",
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if dir_path.exists():
            count = len(list(dir_path.iterdir()))
            print(f"✓ {dir_path.relative_to(base_dir)}: {count} files")
        else:
            print(f"✗ {dir_path.relative_to(base_dir)}: NOT FOUND")
            all_good = False
    
    if all_good:
        print("\n" + "=" * 80)
        print("✅ COCO DATASET READY!")
        print("=" * 80)
        print(f"\nLocation: {base_dir}")
        print("\nYou can now proceed with merging datasets:")
        print("  python merge_datasets.py")
        return True
    else:
        print("\n" + "=" * 80)
        print("⚠️  DATASET INCOMPLETE")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = download_coco()
    exit(0 if success else 1)
