# YOLO + D4SM TBD Pipeline - Quick Start Guide

## What is This?

A **Tracking-by-Detection (TBD)** pipeline combining:
- **YOLOv11**: Detects objects in each frame
- **D4SM (DAM4SAM)**: Tracks detected objects with segmentation masks

## Installation

```bash
# Install YOLOv11 (ultralytics package)
pip install ultralytics

# The YOLO model will auto-download on first run
```

## Basic Usage

```bash
cd /home/2020112001/mot_justin/d4sm

# Run on test dataset (n1)
python tests/scripts/yolo_d4sm_tbd.py \
  --images_dir tests/data/n1 \
  --output_dir output_tbd \
  --yolo_model yolo11n.pt \
  --d4sm_model_size tiny \
  --save_video
```

## Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--images_dir` | Input image directory | **required** |
| `--output_dir` | Output directory | `output_tbd` |
| `--yolo_model` | YOLO model (n/s/m/l/x) | `yolo11n.pt` |
| `--d4sm_model_size` | D4SM size (tiny/small/base/large) | `tiny` |
| `--det_conf` | Detection confidence threshold | `0.5` |
| `--iou_threshold` | IoU threshold for track matching | `0.3` |
| `--classes` | Filter object classes (e.g., `--classes 0 2`) | All |
| `--save_video` | Generate MP4 video | `False` |
| `--fps` | Video FPS | `10` |

## Output Structure

```
output_tbd/
├── frames/           # Annotated frames (detection boxes + tracking masks)
├── result.mp4        # Video output (if --save_video used)
├── tracking_log.json # Frame-by-frame tracking data
└── config.json       # Run configuration
```

## How It Works

```
For each frame:
  1. YOLO detects objects
  2. Match detections to existing tracks (IoU-based)
  3. Add new tracks for unmatched detections
  4. D4SM tracks all objects
  5. Visualize: green boxes (detections) + colored masks (tracks)
```

## Tuning Tips

- **More detections**: Lower `--det_conf` (e.g., 0.3)
- **Reduce false positives**: Raise `--det_conf` (e.g., 0.7)
- **Reduce duplicate tracks**: Raise `--iou_threshold` (e.g., 0.5)
- **Specific objects**: Use `--classes` (see [COCO classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml))
  - Person: `--classes 0`
  - Car: `--classes 2`
  - Person + Car: `--classes 0 2`

## Examples

### Detect only people
```bash
python tests/scripts/yolo_d4sm_tbd.py \
  --images_dir tests/data/n1 \
  --output_dir output_people \
  --classes 0 \
  --det_conf 0.6
```

### High-quality tracking
```bash
python tests/scripts/yolo_d4sm_tbd.py \
  --images_dir tests/data/n1 \
  --output_dir output_hq \
  --yolo_model yolo11s.pt \
  --d4sm_model_size small \
  --det_conf 0.6 \
  --iou_threshold 0.4 \
  --save_video
```

## Future Enhancements

This is **Option 1: Simple TBD** - detect every frame.

Future improvements:
- **Option 2**: Detect only on first frame, track thereafter
- **Option 3**: Re-detect every N frames or on track loss
- **Track association**: Hungarian algorithm, Kalman filtering
- **Track management**: Handle track disappearance/reappearance
