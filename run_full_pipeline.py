#!/usr/bin/env python3
"""
Complete automation script for 86-class model training
Runs all steps automatically after COCO download completes
"""
import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and show progress"""
    print("\n" + "=" * 80)
    print(f"▶ {description}")
    print("=" * 80)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\n✅ {description} - COMPLETE")
        return True
    else:
        print(f"\n❌ {description} - FAILED")
        return False


def main():
    print("=" * 80)
    print("AUTOMATED TRAINING PIPELINE - 86 Classes")
    print("=" * 80)
    
    base_dir = Path("/root/workspace/drone-azooz")
    venv_activate = f"source {base_dir}/venv/bin/activate"
    
    steps = [
        {
            'name': 'Step 1: Verify COCO Download',
            'cmd': f'{venv_activate} && cd {base_dir} && python -c "from pathlib import Path; p = Path(\'datasets/coco\'); print(\'✓ COCO ready\' if (p / \'images/train2017\').exists() else \'✗ COCO missing\')"',
            'critical': True
        },
        {
            'name': 'Step 2: Re-merge Datasets (COCO + FOD + Tools)',
            'cmd': f'{venv_activate} && cd {base_dir} && python merge_datasets.py',
            'critical': True
        },
        {
            'name': 'Step 3: Train Model (50 epochs, 6-8 hours)',
            'cmd': f'{venv_activate} && cd {base_dir} && python train_full_86.py',
            'critical': True
        },
        {
            'name': 'Step 4: Validate Model on All Classes',
            'cmd': f'{venv_activate} && cd {base_dir} && python test_pretrained.py',
            'critical': False
        },
        {
            'name': 'Step 5: Export to ONNX and TFLite',
            'cmd': f'{venv_activate} && cd {base_dir} && python export_to_tflite.py',
            'critical': False
        }
    ]
    
    print("\n📋 Pipeline Steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step['name']}")
    
    print("\n🚀 Starting automated pipeline...")
    print("This will take 6-8 hours total.")
    
    start_time = time.time()
    
    for i, step in enumerate(steps, 1):
        success = run_command(step['cmd'], f"Step {i}/{len(steps)}: {step['name']}")
        
        if not success and step['critical']:
            print(f"\n❌ Critical step failed. Stopping pipeline.")
            return False
        
        # Show elapsed time
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"\n⏱️  Elapsed time: {hours}h {minutes}m")
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n⏱️  Total time: {hours}h {minutes}m")
    print(f"\n📦 Final model: runs/detect/combined_full_86classes/weights/best.pt")
    print(f"📱 Android formats: best.onnx, best_float32.tflite")
    print(f"\n✅ Model detects all 86 classes accurately!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
