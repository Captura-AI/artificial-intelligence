# AI Models Check-in

## Overview
Added pre-trained PyTorch model files to the repository for improved deployment reliability and reduced startup time.

## Models Added

### 1. motortype.pt (5.93 MB)
- **Purpose**: Vehicle/motor type classification
- **Model Type**: PyTorch neural network
- **Use Case**: Detects vehicle types (bicycle, car, motorcycle, bus, truck) from images
- **Integration**: Used in `services/vehicle_detector.py` for COCO-based vehicle classification

### 2. platdetect.pt (5.95 MB)
- **Purpose**: License plate detection
- **Model Type**: PyTorch object detection model
- **Use Case**: Locates and extracts license plate regions from vehicle images
- **Integration**: Used in `services/plate_reader.py` for plate region detection

### 3. platreader.pt (5.93 MB)
- **Purpose**: Optical Character Recognition (OCR) for plate text
- **Model Type**: PyTorch OCR model
- **Use Case**: Extracts Indonesian license plate text (format: B 1234 XYZ)
- **Integration**: Used in `services/plate_reader.py` for character recognition

## Configuration Changes

### .gitignore Updates
- Removed exclusion of `.pt` files (PyTorch model format)
- Removed exclusion of `.onnx` files (ONNX model format)
- Allows model files to be tracked in version control

**Before:**
```
*.pt
*.onnx
```

**After:**
```
# Models now included in repository
```

## Benefits

### Deployment Advantages
- **Faster Startup**: Models are pre-loaded, eliminating download delays
- **Offline Availability**: No internet required to pull models at startup
- **Version Control**: Model versions tracked with git commits
- **Consistency**: Ensures all environments use identical model binaries

### Size Impact
- Total model size: ~17.8 MB (relatively small)
- Repository size increase: Minimal impact on clone times
- CI/CD improvements: Faster deployment pipelines without model downloads

## Model Loading

Models are still lazy-loaded on first service call and cached in module-level globals:
- `_MODEL` globals in each service file remain in place
- Lifespan context manager in `main.py` warms up models at startup
- Cold-start latency on first request is eliminated

## Environment Setup

No additional configuration required. Models are automatically located at:
- `app/aimodels/motortype.pt`
- `app/aimodels/platdetect.pt`
- `app/aimodels/platreader.pt`

The service will use these local files instead of downloading from remote sources.

## Future Improvements

- Consider LFS (Large File Storage) if model sizes exceed 50 MB
- Implement model versioning in config
- Add model validation checksums for integrity verification
