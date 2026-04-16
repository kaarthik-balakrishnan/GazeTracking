#!/usr/bin/env python3
"""
Pig Eye Tracking Pipeline
Combines video stabilization and eye tracking into a single script.
"""

import cv2
import numpy as np
import csv
import os

# Configuration
INPUT_VIDEO = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/PXL_20260410_024928909.mp4"
OUTPUT_DIR = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"

STABILIZED_VIDEO = os.path.join(OUTPUT_DIR, "stabilized_output.mp4")
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_final.mp4")
GAZE_CSV = os.path.join(OUTPUT_DIR, "gaze_data_final.csv")
POSITION_PLOT = os.path.join(OUTPUT_DIR, "eye_position_final.png")

# Tracking parameters
STABILIZATION_WINDOW = 30
BASELINE_X_MEAN = 1876
BASELINE_Y_MEAN = 1060
BASELINE_X_STD = 150
BASELINE_Y_STD = 100
X_BOUNDS = (BASELINE_X_MEAN - 2.5 * BASELINE_X_STD, BASELINE_X_MEAN + 2.5 * BASELINE_X_STD)
Y_BOUNDS = (BASELINE_Y_MEAN - 2.5 * BASELINE_Y_STD, BASELINE_Y_MEAN + 2.5 * BASELINE_Y_STD)


def stabilize_video(input_path, output_path):
    """Stabilize video using optical flow."""
    print("=" * 60)
    print("STAGE 1: VIDEO STABILIZATION")
    print("=" * 60)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    transforms = []
    prev_gray = None
    prev_points = None
    
    # Phase 1: Analyze motion
    print("Analyzing motion...")
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            if prev_points is None or len(prev_points) < 20:
                prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            if prev_points is not None:
                curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)
                status = status.flatten()
                
                valid_prev = prev_points[status == 1].reshape(-1, 2)
                valid_curr = curr_points[status == 1].reshape(-1, 2)
                
                if len(valid_prev) >= 20:
                    transform, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
                    if transform is not None:
                        transforms.append(transform)
                        prev_points = valid_curr
                    else:
                        transforms.append(np.eye(2, 3, dtype=np.float32))
                        prev_points = None
                else:
                    transforms.append(np.eye(2, 3, dtype=np.float32))
                    prev_points = None
            else:
                transforms.append(np.eye(2, 3, dtype=np.float32))
        else:
            transforms.append(np.eye(2, 3, dtype=np.float32))
        
        prev_gray = curr_gray
        if frame_num % 300 == 0:
            print(f"  {frame_num}/{total_frames} frames analyzed")
    
    cap.release()
    
    while len(transforms) < total_frames:
        transforms.append(np.eye(2, 3, dtype=np.float32))
    
    # Phase 2: Smooth and apply
    print("Smoothing and applying transforms...")
    dx = np.array([t[0, 2] for t in transforms], dtype=np.float32)
    dy = np.array([t[1, 2] for t in transforms], dtype=np.float32)
    da = np.array([np.arctan2(t[1, 0], t[0, 0]) for t in transforms], dtype=np.float32)
    
    kernel = 45
    sigma = kernel // 3
    dx_smooth = cv2.GaussianBlur(dx.reshape(1, -1), (1, kernel), sigma)[0]
    dy_smooth = cv2.GaussianBlur(dy.reshape(1, -1), (1, kernel), sigma)[0]
    da_smooth = cv2.GaussianBlur(da.reshape(1, -1), (1, kernel), sigma)[0]
    
    smoothed = []
    for i in range(len(transforms)):
        transform = np.array([
            [np.cos(da_smooth[i]), -np.sin(da_smooth[i]), dx[i] - dx_smooth[i]],
            [np.sin(da_smooth[i]),  np.cos(da_smooth[i]), dy[i] - dy_smooth[i]]
        ], dtype=np.float32)
        smoothed.append(transform)
    
    print("Applying stabilization...")
    cap = cv2.VideoCapture(input_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        stabilized = cv2.warpAffine(frame, smoothed[frame_num - 1], (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        out.write(stabilized)
        if frame_num % 300 == 0:
            print(f"  {frame_num}/{total_frames} frames stabilized")
    
    cap.release()
    out.release()
    print(f"Stabilized video saved: {output_path}")
    print()


def track_eye(input_path, output_video, output_csv):
    """Track eye position in stabilized video."""
    print("=" * 60)
    print("STAGE 2: EYE TRACKING")
    print("=" * 60)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"Bounds: X=[{X_BOUNDS[0]:.0f}, {X_BOUNDS[1]:.0f}], Y=[{Y_BOUNDS[0]:.0f}, {Y_BOUNDS[1]:.0f}]")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'timestamp_sec', 'eye_x_raw', 'eye_y_raw', 
                        'eye_x_stabilized', 'eye_y_stabilized', 
                        'gaze_direction', 'eye_area', 'confidence'])
    
    x_buffer, y_buffer = [], []
    last_valid_x, last_valid_y, last_valid_area = None, None, None
    reference_x, reference_y = None, None
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    frame_num = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        timestamp = (frame_num - 1) / fps if fps > 0 else 0
        
        # Detect eye
        raw_x, raw_y, eye_area, bbox = detect_eye(frame, clahe, last_valid_x, last_valid_y, last_valid_area)
        is_valid = raw_x is not None and X_BOUNDS[0] <= raw_x <= X_BOUNDS[1] and Y_BOUNDS[0] <= raw_y <= Y_BOUNDS[1]
        
        if is_valid:
            detected_count += 1
            last_valid_x, last_valid_y, last_valid_area = raw_x, raw_y, eye_area
        elif last_valid_x is not None:
            raw_x, raw_y = last_valid_x, last_valid_y
        
        # Stabilize and classify
        if raw_x is not None:
            if reference_x is None:
                reference_x, reference_y = raw_x, raw_y
                print(f"Reference: ({reference_x}, {reference_y})")
            
            x_buffer.append(raw_x)
            y_buffer.append(raw_y)
            if len(x_buffer) > STABILIZATION_WINDOW:
                x_buffer.pop(0)
                y_buffer.pop(0)
            
            stab_x = int(sum(x_buffer) / len(x_buffer))
            stab_y = int(sum(y_buffer) / len(y_buffer))
            
            dx = raw_x - reference_x
            direction = "left" if dx < -100 else "right" if dx > 100 else "center"
        else:
            stab_x, stab_y, direction = None, None, "unknown"
        
        # Draw visualization
        new_frame = frame.copy()
        if raw_x is not None:
            color = (0, 255, 255) if is_valid else (0, 255, 0)
            cv2.circle(new_frame, (raw_x, raw_y), 6, (0, 0, 255), 2)
            cv2.circle(new_frame, (stab_x, stab_y), 10, color, 3)
            cv2.drawMarker(new_frame, (reference_x, reference_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(new_frame, f"Raw: ({raw_x}, {raw_y})", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(new_frame, f"Stab: ({stab_x}, {stab_y})", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            gaze_color = (0, 255, 0) if direction == "center" else (0, 165, 255) if direction == "left" else (255, 0, 0)
            cv2.putText(new_frame, f"Gaze: {direction}", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, gaze_color, 2)
        else:
            cv2.putText(new_frame, "Not detected", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.putText(new_frame, f"Frame: {frame_num}/{total_frames}", (60, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(new_frame)
        
        confidence = "detected" if is_valid else "interpolated"
        csv_writer.writerow([frame_num, f"{timestamp:.3f}", raw_x, raw_y, stab_x, stab_y, direction, eye_area, confidence])
        
        if frame_num % 200 == 0:
            print(f"Progress: {frame_num}/{total_frames} ({100*detected_count/frame_num:.1f}% detected)")
    
    cap.release()
    out.release()
    csv_file.close()
    
    print(f"\nTracking complete: {detected_count}/{total_frames} frames ({100*detected_count/total_frames:.1f}%)")
    print(f"Output: {output_video}")
    print(f"CSV: {output_csv}")
    print()


def detect_eye(frame, clahe, search_x, search_y, expected_area):
    """Detect pig eye using CLAHE and thresholding."""
    scale = 0.25
    small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    h, w = small.shape[:2]
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (9, 9), 2)
    _, dark = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.erode(dark, kernel, iterations=1)
    dark = cv2.dilate(dark, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Search region
    if search_x is not None and search_y is not None:
        margin = int(w * 0.15)
        min_x = max(0, int(search_x * scale) - margin)
        max_x = min(w, int(search_x * scale) + margin)
        min_y = max(0, int(search_y * scale) - margin)
        max_y = min(h, int(search_y * scale) + margin)
    else:
        min_x, max_x = int(w * 0.35), int(w * 0.65)
        min_y, max_y = int(h * 0.35), int(h * 0.65)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 10000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx, cy = x + cw // 2, y + ch // 2
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                aspect = cw / ch if ch > 0 else 0
                if 0.2 < aspect < 3.5:
                    candidates.append((cx, cy, area, (x, y, cw, ch)))
    
    if not candidates:
        return None, None, None, None
    
    candidates.sort(key=lambda c: c[2], reverse=True)
    
    # Prefer similar area
    if expected_area is not None:
        for c in candidates:
            if c[2] > expected_area / 3 and c[2] < expected_area * 3:
                cx, cy, area, bbox = c
                return int(cx * 4), int(cy * 4), int(area * 16), bbox
    
    cx, cy, area, bbox = candidates[0]
    return int(cx * 4), int(cy * 4), int(area * 16), bbox


def main():
    print("=" * 60)
    print("PIG EYE TRACKING PIPELINE")
    print("=" * 60)
    print()
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video not found: {INPUT_VIDEO}")
        return
    
    # Stage 1: Stabilize
    stabilize_video(INPUT_VIDEO, STABILIZED_VIDEO)
    
    # Stage 2: Track
    track_eye(STABILIZED_VIDEO, ANNOTATED_VIDEO, GAZE_CSV)
    
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
