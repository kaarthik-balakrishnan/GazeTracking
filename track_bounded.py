import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

video_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/stabilized_output.mp4"
output_video_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/annotated_final.mp4"
output_csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_data_final.csv"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

csv_file = open(output_csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'timestamp_sec', 'eye_x_raw', 'eye_y_raw', 
                     'eye_x_stabilized', 'eye_y_stabilized', 
                     'gaze_direction', 'eye_area', 'confidence'])

frame_num = 0

stabilization_window = 30
x_buffer = []
y_buffer = []

last_valid_x = None
last_valid_y = None
last_valid_area = None

reference_x = None
reference_y = None

# Baseline statistics (calculated from first 15 seconds)
baseline_x_mean = 1876
baseline_y_mean = 1060
baseline_x_std = 150  # Allow ~150 pixels natural movement
baseline_y_std = 100   # Allow ~100 pixels natural movement

# Acceptable range: mean ± 2.5 * std
x_min = baseline_x_mean - 2.5 * baseline_x_std
x_max = baseline_x_mean + 2.5 * baseline_x_std
y_min = baseline_y_mean - 2.5 * baseline_y_std
y_max = baseline_y_mean + 2.5 * baseline_y_std

print("=" * 60)
print("EYE TRACKING - BOUNDED BY BASELINE")
print("=" * 60)
print(f"Baseline X: {baseline_x_mean} ± {baseline_x_std}")
print(f"Baseline Y: {baseline_y_mean} ± {baseline_y_std}")
print(f"Acceptable X range: [{x_min:.0f}, {x_max:.0f}]")
print(f"Acceptable Y range: [{y_min:.0f}, {y_max:.0f}]")
print("=" * 60)

clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

def detect_pig_eye(frame, search_center_x=None, search_center_y=None, expected_area=None):
    """Detect pig eye with CLAHE - original working algorithm"""
    scale = 0.25
    small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(enhanced, (9, 9), 2)
    _, dark = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.erode(dark, kernel, iterations=1)
    dark = cv2.dilate(dark, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = small.shape[:2]
    
    # Search around last known position with margin
    if search_center_x is not None and search_center_y is not None:
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        min_x = max(0, int(search_center_x * scale) - margin_x)
        max_x = min(w, int(search_center_x * scale) + margin_x)
        min_y = max(0, int(search_center_y * scale) - margin_y)
        max_y = min(h, int(search_center_y * scale) + margin_y)
    else:
        min_x, max_x = int(w * 0.35), int(w * 0.65)
        min_y, max_y = int(h * 0.35), int(h * 0.65)
    
    eye_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 10000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cx, cy = x + cw//2, y + ch//2
            
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                aspect = float(cw) / ch if ch > 0 else 0
                if 0.2 < aspect < 3.5:
                    eye_candidates.append((cx, cy, area, (x, y, cw, ch)))
    
    if not eye_candidates:
        return None
    
    eye_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Prefer candidates with similar area to expected
    if expected_area is not None:
        for candidate in eye_candidates:
            cx, cy, area, bbox = candidate
            if area > expected_area / 3 and area < expected_area * 3:
                return candidate
    
    return eye_candidates[0]

def stabilize(val, buffer):
    buffer.append(val)
    if len(buffer) > stabilization_window:
        buffer.pop(0)
    return int(sum(buffer) / len(buffer))

def is_in_bounds(x, y):
    """Check if position is within acceptable range"""
    return x_min <= x <= x_max and y_min <= y <= y_max

def classify_gaze(x, y, ref_x, ref_y):
    if ref_x is None:
        return "unknown"
    dx = x - ref_x
    if dx < -100:
        return "left"
    elif dx > 100:
        return "right"
    return "center"

print(f"Resolution: {width}x{height}")
print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
print("=" * 60)

detection_count = 0
out_of_bounds_count = 0
interpolated_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    timestamp = (frame_num - 1) / fps if fps > 0 else 0
    
    eye_det = detect_pig_eye(frame, last_valid_x, last_valid_y, last_valid_area)
    new_frame = frame.copy()
    
    raw_x = raw_y = stab_x = stab_y = eye_area = direction = None
    is_valid = False
    is_interpolated = False
    
    if eye_det:
        ex, ey, area, bbox = eye_det
        raw_x = int(ex * 4)
        raw_y = int(ey * 4)
        eye_area = int(area * 16)
        
        # Check if detection is within acceptable bounds
        if is_in_bounds(raw_x, raw_y):
            is_valid = True
            detection_count += 1
            last_valid_x = raw_x
            last_valid_y = raw_y
            last_valid_area = area
        else:
            out_of_bounds_count += 1
            # Detection is out of bounds - reject it
            raw_x = None
            raw_y = None
    
    # Interpolate if no valid detection but we have previous position
    if raw_x is None:
        if last_valid_x is not None:
            raw_x = last_valid_x
            raw_y = last_valid_y
            is_interpolated = True
            interpolated_count += 1
        else:
            # No detection and no previous - skip
            pass
    
    if raw_x is not None:
        if reference_x is None:
            reference_x = raw_x
            reference_y = raw_y
            print(f"Reference: ({reference_x}, {reference_y})")
        
        stab_x = stabilize(raw_x, x_buffer)
        stab_y = stabilize(raw_y, y_buffer)
        
        direction = classify_gaze(stab_x, stab_y, reference_x, reference_y)
        
        if eye_det and is_valid:
            bx, by, bw, bh = bbox[0] * 4, bbox[1] * 4, bbox[2] * 4, bbox[3] * 4
            cv2.rectangle(new_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        
        cv2.circle(new_frame, (raw_x, raw_y), 6, (0, 0, 255), 2)
        cv2.circle(new_frame, (stab_x, stab_y), 10, (0, 255, 0), 3)
        cv2.drawMarker(new_frame, (reference_x, reference_y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        
        # Draw acceptable region
        cv2.rectangle(new_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 1)
        
        cv2.putText(new_frame, f"Raw: ({raw_x}, {raw_y})", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(new_frame, f"Stabilized: ({stab_x}, {stab_y})", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if is_interpolated:
            cv2.putText(new_frame, "INTERPOLATED", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        else:
            gaze_color = (0, 255, 0) if direction == "center" else (0, 165, 255) if direction == "left" else (255, 0, 0)
            cv2.putText(new_frame, f"Gaze: {direction.upper()}", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, gaze_color, 2)
    else:
        cv2.putText(new_frame, "Eye not detected", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    cv2.putText(new_frame, f"Frame: {frame_num}/{total_frames}", (60, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out.write(new_frame)
    
    confidence = "interpolated" if is_interpolated else ("detected" if is_valid else "missing")
    csv_writer.writerow([frame_num, f"{timestamp:.3f}", raw_x, raw_y, stab_x, stab_y, direction, eye_area, confidence])
    
    if frame_num % 200 == 0:
        print(f"Progress: {frame_num}/{total_frames} (det: {detection_count}, oob: {out_of_bounds_count}, interp: {interpolated_count})")

cap.release()
out.release()
csv_file.close()

print("=" * 60)
print(f"Done! {frame_num} frames")
print(f"Valid detections: {detection_count} ({100*detection_count/frame_num:.1f}%)")
print(f"Out of bounds rejected: {out_of_bounds_count}")
print(f"Interpolated: {interpolated_count}")
print(f"Output: {output_video_path}")
print("=" * 60)

# Plot results
timestamps = []
eye_x = []
eye_y = []
labels = []

with open(output_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamps.append(float(row['timestamp_sec']))
        eye_x.append(float(row['eye_x_raw']) if row['eye_x_raw'] else np.nan)
        eye_y.append(float(row['eye_y_raw']) if row['eye_y_raw'] else np.nan)
        labels.append(row['confidence'])

timestamps = np.array(timestamps)
eye_x = np.array(eye_x)
eye_y = np.array(eye_y)
labels = np.array(labels)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Color by confidence
colors_x = ['red' if l == 'interpolated' else 'blue' if not np.isnan(x) else 'gray' for x, l in zip(eye_x, labels)]
axes[0].scatter(timestamps, eye_x, c=colors_x, s=2, alpha=0.7)
axes[0].axhline(y=x_min, color='green', linestyle='--', alpha=0.5, label=f'Lower bound ({x_min:.0f})')
axes[0].axhline(y=x_max, color='green', linestyle='--', alpha=0.5, label=f'Upper bound ({x_max:.0f})')
axes[0].axhline(y=baseline_x_mean, color='orange', linestyle='-', alpha=0.5, label=f'Baseline ({baseline_x_mean:.0f})')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Eye X Position (pixels)')
axes[0].set_title('Eye X Position vs Time (Red = Interpolated)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

colors_y = ['red' if l == 'interpolated' else 'green' if not np.isnan(y) else 'gray' for y, l in zip(eye_y, labels)]
axes[1].scatter(timestamps, eye_y, c=colors_y, s=2, alpha=0.7)
axes[1].axhline(y=y_min, color='blue', linestyle='--', alpha=0.5, label=f'Lower bound ({y_min:.0f})')
axes[1].axhline(y=y_max, color='blue', linestyle='--', alpha=0.5, label=f'Upper bound ({y_max:.0f})')
axes[1].axhline(y=baseline_y_mean, color='orange', linestyle='-', alpha=0.5, label=f'Baseline ({baseline_y_mean:.0f})')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Eye Y Position (pixels)')
axes[1].set_title('Eye Y Position vs Time (Red = Interpolated)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/eye_position_final.png', dpi=150)
plt.close()

print(f"\nPlot saved to: eye_position_final.png")