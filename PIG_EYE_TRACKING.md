# Pig Eye Tracking - Documentation

This repository contains code for tracking pig eyes in video footage, specifically designed for research applications involving animal behavior analysis.

## Overview

The system performs real-time eye tracking on pig video footage, handling challenges such as:
- Side-profile pig face detection
- Foreground occlusion (hands, tools)
- Lighting variations
- Natural head movements

## Pipeline Architecture

The complete processing pipeline consists of two main stages:

### Stage 1: Video Stabilization

**File:** `video_stabilizer.py`

Video stabilization is performed first to reduce camera shake and ensure the pig's eye remains in a consistent position throughout the footage.

**Algorithm:**
1. Extract features from each frame using Good Features to Track
2. Track features between consecutive frames using Lucas-Kanade optical flow
3. Estimate affine transformation between frames
4. Apply Gaussian smoothing to smooth the motion trajectory
5. Apply inverse smoothing to compensate for camera movement

**Key Parameters:**
- `smoothing_window`: 45 frames - determines the smoothing strength
- Feature tracking uses ORB features with RANSAC for robust homography estimation

### Stage 2: Eye Tracking

**File:** `track_bounded.py`

After stabilization, the eye is tracked using adaptive thresholding with CLAHE (Contrast Limited Adaptive Histogram Equalization) for improved detection in varying lighting conditions.

**Algorithm:**
1. Apply CLAHE to enhance contrast of grayscale eye region
2. Gaussian blur and adaptive threshold to isolate dark regions (pupil/eye)
3. Find contours and filter by area, aspect ratio, and position
4. Apply bounding constraints based on baseline statistics
5. Interpolate missing frames when detection fails

**Key Parameters:**
- `baseline_x_mean`, `baseline_y_mean`: Initial eye position from first valid detection
- `baseline_x_std`, `baseline_y_std`: Natural eye movement variance (~150px horizontal, ~100px vertical)
- Bounds: mean ± 2.5 * standard deviation
- `stabilization_window`: 30 frames for temporal smoothing

## Usage

### 1. Stabilize Video

```bash
python video_stabilizer.py
```

This will create `stabilized_output.mp4` in the data directory.

### 2. Track Eye

```bash
python track_bounded.py
```

This will create:
- `annotated_final.mp4` - Video with eye position marked
- `gaze_data_final.csv` - CSV with tracking data
- `eye_position_final.png` - Position plot for analysis

## Output Format

### CSV Data (`gaze_data_final.csv`)

| Column | Description |
|--------|-------------|
| frame | Frame number |
| timestamp_sec | Timestamp in seconds |
| eye_x_raw | Raw detected X position |
| eye_y_raw | Raw detected Y position |
| eye_x_stabilized | Temporal smoothed X position |
| eye_y_stabilized | Temporal smoothed Y position |
| gaze_direction | left/center/right |
| eye_area | Detected eye contour area |
| confidence | detected/interpolated/missing |

### Gaze Direction Classification

Gaze direction is classified based on deviation from reference position:
- **Left**: X < reference_x - 100 pixels
- **Right**: X > reference_x + 100 pixels
- **Center**: Within ±100 pixels of reference

## Technical Details

### Why Video Stabilization?

The original video contains significant camera shake and pig movement. Without stabilization, the eye position varies significantly throughout the video (hundreds of pixels), making it impossible to track accurately.

### Why Bounding Constraints?

The eye tracking algorithm uses dark region detection which can be fooled by:
- Foreground objects (hands, gloves)
- Shadows
- Other dark regions in the frame

By establishing baseline statistics from the first 15-20 seconds (when the setup is stable), we can reject detections that fall outside the expected range, preventing rogue tracking.

### Why Interpolation?

When a detection falls outside bounds, the system interpolates from the last known valid position rather than accepting the wrong detection. This maintains tracking continuity during brief occlusions.

## File Structure

```
GazeTracking/
├── video_stabilizer.py      # Video stabilization module
├── track_bounded.py          # Eye tracking with bounds
├── gaze_tracking/            # Original human eye tracking (not used for pigs)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dependencies

- Python 3.9+
- OpenCV 4.10+
- NumPy 1.26+
- dlib 19.24+ (for original human tracking, not needed for pig tracking)
- matplotlib (for analysis plots)

## Authors

Extended from [AntoineLame/GazeTracking](https://github.com/antoinelame/GazeTracking) for pig eye tracking applications.

## License

MIT License
