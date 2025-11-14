#!/usr/bin/env python3
"""
Quick test to verify the combined dataset and show some statistics
"""
from pathlib import Path
import random

def count_classes_in_labels(label_dir):
    """Count occurrences of each class in label files"""
    class_counts = {}
    label_files = list(Path(label_dir).glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts, len(label_files)


def main():
    print("=" * 70)
    print("Combined Dataset Verification")
    print("=" * 70)
    
    base_dir = Path("/root/workspace/drone-azooz/datasets/combined")
    
    # Check train set
    train_labels = base_dir / "labels" / "train"
    if train_labels.exists():
        print("\nTRAIN SET:")
        class_counts, num_files = count_classes_in_labels(train_labels)
        print(f"  Total label files: {num_files}")
        print(f"  Total annotations: {sum(class_counts.values())}")
        print(f"  Classes found: {sorted(class_counts.keys())}")
        
        # Show custom classes
        custom_classes = {80: "FOD", 81: "drill", 82: "hammer", 83: "pliers", 84: "screwdriver", 85: "wrench"}
        print("\n  Custom class counts:")
        for cls_id, cls_name in custom_classes.items():
            count = class_counts.get(cls_id, 0)
            print(f"    {cls_id} ({cls_name}): {count} annotations")
    
    # Check val set
    val_labels = base_dir / "labels" / "val"
    if val_labels.exists():
        print("\nVALIDATION SET:")
        class_counts, num_files = count_classes_in_labels(val_labels)
        print(f"  Total label files: {num_files}")
        print(f"  Total annotations: {sum(class_counts.values())}")
        print(f"  Classes found: {sorted(class_counts.keys())}")
        
        # Show custom classes
        print("\n  Custom class counts:")
        for cls_id, cls_name in custom_classes.items():
            count = class_counts.get(cls_id, 0)
            print(f"    {cls_id} ({cls_name}): {count} annotations")
    
    print("\n" + "=" * 70)
    print("✓ Dataset verification complete!")
    print("=" * 70)
    print("\nYou can now proceed to training with:")
    print("  python train.py")


if __name__ == "__main__":
    main()
