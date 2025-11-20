# YOLOv8 Object Detection - 86 Classes

Multi-class object detection model trained on COCO dataset (80 classes) plus custom FOD and mechanical tools datasets (6 additional classes).

## Features

- **86 Classes**: 80 COCO classes + FOD + drill, hammer, pliers, screwdriver, wrench
- **YOLOv8s**: Fast and accurate detection
- **Real-time inference**: Webcam, images, and video support

## Classes

**COCO (0-79)**: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**Custom (80-85)**: FOD, drill, hammer, pliers, screwdriver, wrench

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 20GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/Osaid03/drone-azooz.git
cd drone-azooz

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

### 1. Prepare Dataset

The datasets are already included:
- `FOD.v2i.yolov8/` - Foreign Object Debris dataset
- `Mechanical tools-10000.v1i.yolov8/` - Tools dataset
- `datasets/coco/` - COCO dataset (download separately if needed)

Merge all datasets:

```bash
python merge_datasets.py
```

This creates `datasets/combined/` with all 86 classes properly labeled.

### 2. Verify Dataset

```bash
python verify_dataset.py
```

### 3. Train Model

```bash
python train_full_86.py
```

**Training parameters:**
- Model: YOLOv8s
- Epochs: 50
- Image size: 640x640
- Batch size: 16
- Device: GPU (CUDA 0)

**Training time:** ~6-8 hours on RTX 3060

**Output:**
- Best model: `runs/detect/combined_full_86classes/weights/best.pt`
- Last checkpoint: `runs/detect/combined_full_86classes/weights/last.pt`

## Inference

### Images or Video

```bash
python inference.py --source path/to/image.jpg
python inference.py --source path/to/video.mp4
python inference.py --source path/to/folder/
```

**Options:**
- `--model`: Path to trained model (default: best.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Save detection results
- `--show`: Display results

### Webcam (Real-time)

```bash
python webcam_detection.py
```

**Note for WSL users:** WSL cannot access hardware directly. Use one of these options:
1. Install [DroidCam](https://www.dev47apps.com/) and run: `python webcam_ip.py http://127.0.0.1:4747/video`
2. Run the script directly on Windows

## Project Structure

```
drone-azooz/
├── train_full_86.py          # Main training script
├── inference.py              # Run detection on images/videos
├── webcam_detection.py       # Real-time webcam detection
├── webcam_ip.py             # IP camera detection
├── merge_datasets.py        # Merge COCO + custom datasets
├── verify_dataset.py        # Verify dataset integrity
├── combined_data.yaml       # Dataset configuration
├── requirements.txt         # Python dependencies
├── FOD.v2i.yolov8/         # FOD dataset
├── Mechanical tools-10000.v1i.yolov8/  # Tools dataset
├── datasets/
│   ├── coco/               # COCO dataset
│   └── combined/           # Merged dataset (86 classes)
└── runs/
    └── detect/
        └── combined_full_86classes/
            └── weights/
                ├── best.pt  # Best model
                └── last.pt  # Last checkpoint
```

## Model Performance

Trained on ~130,000 images across 86 classes with pretrained YOLOv8s weights.

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- ultralytics
- torch
- opencv-python
- numpy

## License

This project uses datasets from:
- COCO: [COCO License](https://cocodataset.org/#termsofuse)
- FOD & Mechanical Tools: Check respective dataset licenses

## Author

Osaid
