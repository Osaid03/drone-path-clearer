#!/usr/bin/env python3
"""
Autonomous pipeline that monitors COCO download and executes full training workflow.
No user intervention required - handles everything automatically.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

class AutoPipeline:
    def __init__(self):
        self.base_path = Path("/root/workspace/drone-azooz")
        self.coco_path = self.base_path / "datasets/coco"
        self.log_file = self.base_path / "pipeline.log"
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")
    
    def check_download_complete(self):
        """Check if COCO dataset is fully downloaded and extracted"""
        train_images = self.coco_path / "images/train2017"
        val_images = self.coco_path / "images/val2017"
        train_labels = self.coco_path / "labels/train2017"
        val_labels = self.coco_path / "labels/val2017"
        
        if not all([train_images.exists(), val_images.exists(), 
                   train_labels.exists(), val_labels.exists()]):
            return False
        
        # Check if directories have content
        train_img_count = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
        val_img_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
        
        self.log(f"COCO images found - Train: {train_img_count}, Val: {val_img_count}")
        
        # COCO has 118,287 train and 5,000 val images
        return train_img_count > 100000 and val_img_count > 4000
    
    def run_command(self, cmd, description):
        """Run command and log output"""
        self.log(f"Starting: {description}")
        self.log(f"Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.base_path)
            )
            
            if result.returncode == 0:
                self.log(f"✓ Completed: {description}", "SUCCESS")
                if result.stdout:
                    self.log(f"Output: {result.stdout[:500]}")
                return True
            else:
                self.log(f"✗ Failed: {description}", "ERROR")
                self.log(f"Error: {result.stderr[:500]}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"✗ Exception: {description} - {str(e)}", "ERROR")
            return False
    
    def merge_datasets(self):
        """Step 1: Merge COCO + custom datasets"""
        self.log("=" * 80)
        self.log("STEP 1: MERGING DATASETS")
        self.log("=" * 80)
        
        cmd = "source venv/bin/activate && python merge_datasets.py"
        return self.run_command(cmd, "Dataset merging")
    
    def train_model(self):
        """Step 2: Train model on all 86 classes"""
        self.log("=" * 80)
        self.log("STEP 2: TRAINING MODEL (50 epochs, ~6-8 hours)")
        self.log("=" * 80)
        
        cmd = "source venv/bin/activate && python train_full_86.py"
        return self.run_command(cmd, "Model training")
    
    def validate_model(self):
        """Step 3: Validate model performance"""
        self.log("=" * 80)
        self.log("STEP 3: VALIDATING MODEL")
        self.log("=" * 80)
        
        # Test COCO classes
        cmd1 = "source venv/bin/activate && python test_pretrained.py"
        success1 = self.run_command(cmd1, "COCO class validation")
        
        # Test custom classes
        cmd2 = "source venv/bin/activate && python test_model.py"
        success2 = self.run_command(cmd2, "Custom class validation")
        
        return success1 and success2
    
    def export_formats(self):
        """Step 4: Export to all formats including TFLite"""
        self.log("=" * 80)
        self.log("STEP 4: EXPORTING TO ALL FORMATS")
        self.log("=" * 80)
        
        cmd = "source venv/bin/activate && python export_to_tflite.py"
        return self.run_command(cmd, "Multi-format export")
    
    def generate_report(self):
        """Step 5: Generate final report"""
        self.log("=" * 80)
        self.log("GENERATING FINAL REPORT")
        self.log("=" * 80)
        
        report_path = self.base_path / "TRAINING_COMPLETE.txt"
        
        # Gather statistics
        model_path = self.base_path / "runs/detect/combined_full_86classes/weights/best.pt"
        exports_path = self.base_path / "exports"
        
        report = []
        report.append("=" * 80)
        report.append("YOLOV8 86-CLASS MODEL TRAINING COMPLETE")
        report.append("=" * 80)
        report.append(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            report.append(f"✓ Model trained successfully: {model_path}")
            report.append(f"  Size: {size_mb:.1f} MB")
        else:
            report.append("✗ Model file not found!")
        
        report.append("")
        report.append("CLASSES (86 total):")
        report.append("  - Classes 0-79: COCO objects (person, car, dog, etc.)")
        report.append("  - Class 80: FOD (Foreign Object Debris)")
        report.append("  - Class 81: Drill")
        report.append("  - Class 82: Hammer")
        report.append("  - Class 83: Pliers")
        report.append("  - Class 84: Screwdriver")
        report.append("  - Class 85: Wrench")
        report.append("")
        
        if exports_path.exists():
            report.append("EXPORTED FORMATS:")
            for fmt in ["best.onnx", "best.torchscript", "best_float32.tflite", "best_int8.tflite"]:
                file_path = exports_path / fmt
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    report.append(f"  ✓ {fmt}: {size_mb:.1f} MB")
        
        report.append("")
        report.append("USAGE:")
        report.append("  # Inference on image/video")
        report.append("  python inference.py --source image.jpg")
        report.append("")
        report.append("  # Android deployment")
        report.append("  Use: exports/best_float32.tflite (recommended)")
        report.append("  Or:  exports/best.onnx (ONNX runtime)")
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(report_path, "w") as f:
            f.write(report_text)
        
        self.log(report_text)
        return True
    
    def run(self):
        """Main execution loop"""
        self.log("=" * 80)
        self.log("AUTONOMOUS PIPELINE STARTED")
        self.log("=" * 80)
        
        # Wait for COCO download
        self.log("Waiting for COCO dataset download to complete...")
        check_interval = 60  # Check every minute
        
        while not self.check_download_complete():
            self.log("COCO dataset not ready yet, checking again in 60 seconds...")
            time.sleep(check_interval)
        
        self.log("✓ COCO dataset ready!")
        
        # Execute pipeline
        steps = [
            ("Merge datasets", self.merge_datasets),
            ("Train model", self.train_model),
            ("Validate model", self.validate_model),
            ("Export formats", self.export_formats),
            ("Generate report", self.generate_report)
        ]
        
        for step_name, step_func in steps:
            self.log(f"\n{'=' * 80}")
            self.log(f"Executing: {step_name}")
            self.log(f"{'=' * 80}\n")
            
            success = step_func()
            
            if not success:
                self.log(f"✗ Pipeline failed at step: {step_name}", "ERROR")
                self.log("Check pipeline.log for details", "ERROR")
                sys.exit(1)
        
        self.log("=" * 80)
        self.log("✓ AUTONOMOUS PIPELINE COMPLETED SUCCESSFULLY")
        self.log("=" * 80)
        self.log("Check TRAINING_COMPLETE.txt for full report")


if __name__ == "__main__":
    pipeline = AutoPipeline()
    pipeline.run()
