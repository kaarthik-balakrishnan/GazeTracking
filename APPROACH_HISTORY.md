# Approach History - Pig Eye Tracking Development

This document chronicles the different approaches tried during the development of the pig eye tracking system, including failed attempts and the reasoning behind final choices.

## Problem Statement

Track pig eye position in video footage for research applications. Challenges:
- Side-profile pig face (not frontal human face)
- Foreground occlusion (purple glove during handling)
- Varying lighting conditions
- Camera shake

## Approaches Tried

### 1. Original GazeTracking Library (Failed)

**File:** N/A - Original `gaze_tracking/` module

**Approach:** Use Dlib's pre-trained 68-landmark face detector

**Why it failed:**
- Dlib's face detector is trained on human frontal faces
- Pig side-profile was not detected
- No faces detected at any resolution

**Lesson:** Generic human face detectors don't work for animals

---

### 2. Dark Region Detection - Original Video (Partial Success)

**File:** `track_pig_eye.py`

**Approach:** 
- Threshold to find dark regions (pigs have dark eyes)
- Filter contours by area and position
- Search in right-center region (side view)

**Results:**
- Detection rate: 94%
- Worked well on first ~20 seconds
- Failed after ~13 seconds due to tracking drift

**Why it partially worked:**
- Pig eye is darker than surroundings
- Simple thresholding works in controlled lighting

**Why it failed:**
- No motion stabilization
- Eye position drifted as pig moved head
- Detection jumped to wrong regions

---

### 3. Constrained Search Region (Partial Success)

**File:** `track_pig_eye.py` (early versions)

**Approach:**
- Search only within ±200px of previous detection
- Interpolate when detection fails

**Results:**
- Detection rate improved to ~99%
- Still drifted after ~20 seconds

**Why it failed:**
- Search region followed the wrong detection
- Once tracking jumped, constrained search locked onto wrong position

---

### 4. Stabilized Video + Original Tracking (Success)

**File:** `video_stabilizer.py` + `track_on_stabilized.py`

**Approach:**
- Apply optical flow-based video stabilization first
- Then run eye tracking on stabilized video

**Results:**
- Detection rate: 99.5%
- Eye position stayed stable throughout video

**Why it worked:**
- Video stabilization removed camera shake
- Pig's eye remained in consistent relative position
- Detection algorithm could reliably find eye

---

### 5. CLAHE Contrast Enhancement (No Improvement)

**File:** `track_pig_eye_v2_clahe.py`

**Approach:**
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) before thresholding
- Use 15% center region search

**Results:**
- Detection rate: 98.7%
- Still had position drift after 20 seconds

**Why it didn't help:**
- Contrast wasn't the limiting factor
- The issue was tracking wrong objects, not poor detection quality

---

### 6. Strict Center Region Only (Failed)

**File:** `track_strict_center.py`, `track_calibrated.py`

**Approach:**
- Restrict search to center 20% of frame
- Calibrate reference from first 50 frames

**Results:**
- Detection rate dropped to ~86%
- Reference point was wrong (tracked shadows, not eye)
- All gaze directions were incorrect

**Why it failed:**
- Using raw first-frame position as reference was wrong
- Eye wasn't in the center of first frame
- Shadows near eye were detected as the eye

---

### 7. Video Stabilization + Bounded Tracking (Final Success)

**File:** `video_stabilizer.py` + `track_bounded.py`

**Approach:**
1. Stabilize video using optical flow (Gaussian smoothing of transforms)
2. Detect eye using CLAHE + threshold
3. Reject detections outside baseline bounds (mean ± 2.5*std)
4. Interpolate missing frames from last valid position

**Results:**
- Detection rate: 91.3%
- Interpolated: 8.7%
- Eye position stable throughout video
- Gaze classification accurate

**Why it worked:**
1. **Stabilization** - Removed camera motion, eye stays in consistent position
2. **Bounding** - Established baseline from first 15-20 seconds, rejected rogue detections
3. **Interpolation** - Maintained tracking during brief occlusions

---

## Key Insights

### What Worked

1. **Video stabilization is essential** for moving subjects
   - Optical flow + Gaussian smoothing of transforms
   - 45-frame smoothing window balances stability vs responsiveness

2. **Baseline bounding prevents drift**
   - Calculate mean position from first 15-20 seconds
   - Reject any detection outside ±2.5 standard deviations
   - This prevents tracking from following occluding objects

3. **Interpolation maintains continuity**
   - When detection fails (occlusion), use last known position
   - Smoothed output reduces jitter from this approximation

### What Didn't Work

1. **Generic face detectors** - Don't generalize to animals
2. **Tight search constraints** - Lock onto wrong features when initial detection is off
3. **CLAHE alone** - Contrast enhancement doesn't fix tracking logic issues

## Final Architecture

```
Original Video (3840x2160, 29fps)
         │
         ▼
┌─────────────────────────┐
│  Video Stabilization     │
│  - Optical flow tracking │
│  - Gaussian smoothing    │
│  - 45-frame window       │
└─────────────────────────┘
         │
         ▼
Stabilized Video
         │
         ▼
┌─────────────────────────┐
│  Eye Detection          │
│  - CLAHE contrast       │
│  - Dark region threshold │
│  - Contour filtering    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Bounded Tracking        │
│  - Baseline stats        │
│  - Reject outliers       │
│  - Interpolate gaps      │
└─────────────────────────┘
         │
         ▼
Annotated Video + CSV Data
```

## Files Generated

| File | Purpose |
|------|---------|
| `video_stabilizer.py` | Optical flow video stabilization |
| `track_bounded.py` | Eye tracking with baseline bounds |
| `PIG_EYE_TRACKING.md` | Main documentation |
| `APPROACH_HISTORY.md` | This file |

## Dependencies

- OpenCV 4.10+
- NumPy 1.26+
- matplotlib (for plots)
- Python 3.9+
