# Pig Eye Tracking

A computer vision system for tracking pig eye position in video footage, designed for research applications in animal behavior analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)

## Overview

This project extends eye tracking capabilities to non-human subjects (pigs), handling challenges like:
- Side-profile face detection
- Foreground occlusion (hands, tools)
- Varying lighting conditions
- Camera shake and subject movement

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy matplotlib

# Run complete pipeline
python pig_eye_tracking_pipeline.py
```

## Pipeline

The system uses a two-stage approach:

1. **Video Stabilization** - Removes camera shake using optical flow
2. **Eye Tracking** - Detects and tracks eye position with baseline bounding

See [PIG_EYE_TRACKING.md](PIG_EYE_TRACKING.md) for detailed documentation.

## Documentation

- [PIG_EYE_TRACKING.md](PIG_EYE_TRACKING.md) - Technical documentation
- [APPROACH_HISTORY.md](APPROACH_HISTORY.md) - Development history and failed attempts

## Output

The pipeline generates:
- `annotated_final.mp4` - Video with eye position marked
- `gaze_data_final.csv` - Tracking data (position, gaze direction, confidence)
- `eye_position_final.png` - Position plot for analysis

## Original Project

Based on [antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking) - adapted for pig eye tracking.

## License

MIT License
