"""
FOV Cone Video Visualization (Optimized)
========================================

Creates annotated video with FOV cone visualization.
Downscales for fast processing, outputs at reduced resolution.

Usage:
    python fov_cone_video_fast.py
"""

import cv2
import numpy as np
import csv
from dataclasses import dataclass
from typing import List
import math


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    confidence: str


def load_gaze_data(csv_path: str) -> List[GazeSample]:
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(GazeSample(
                frame=int(row['frame']),
                timestamp=float(row['timestamp']),
                azimuth=float(row['azimuth_deg']) if row['azimuth_deg'] else 0,
                elevation=float(row['elevation_deg']) if row['elevation_deg'] else 0,
                confidence=row['confidence']
            ))
    return samples


def draw_fov_overlay(frame: np.ndarray, az: float, el: float,
                     fov_h: float = 120.0, fov_v: float = 80.0) -> np.ndarray:
    """Draw all FOV visualizations on a frame"""
    h, w = frame.shape[:2]
    
    # Bird's eye view (top-left)
    bird_x, bird_y = 30, 30
    bird_size = 140
    draw_birds_eye(frame, az, el, fov_h, fov_v, bird_x, bird_y, bird_size)
    
    # Side view (below bird's eye)
    side_x, side_y = 30, bird_y + bird_size + 50
    draw_side_view(frame, az, el, fov_h, fov_v, side_x, side_y)
    
    # Compass (top-right)
    compass_x, compass_y = w - 100, 100
    draw_compass(frame, az, el, compass_x, compass_y, 70)
    
    # Info panel (bottom-left)
    draw_info_panel(frame, az, el, fov_h, fov_v)
    
    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 30), -1)
    cv2.putText(frame, "Pig Eye Gaze Tracking - FOV Visualization", 
               (w//2 - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
    
    return frame


def draw_birds_eye(frame, az, el, fov_h, fov_v, x, y, size):
    """Draw bird's eye FOV cone view"""
    cx, cy = x + size//2, y + size//2
    
    # Background
    cv2.rectangle(frame, (x, y), (x + size, y + size), (25, 25, 40), -1)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (80, 80, 100), 1)
    
    # Concentric circles
    for r in [20, 40, 60]:
        cv2.circle(frame, (cx, cy), r, (50, 50, 60), 1)
    
    # Pig direction (points right in this view)
    pig_dir = 0  # pointing right
    gaze_rad = math.radians(pig_dir - az)
    
    # FOV cone
    half_fov = math.radians(fov_h / 2)
    left_rad = gaze_rad - half_fov
    right_rad = gaze_rad + half_fov
    
    pts = np.array([
        [cx, cy],
        [int(cx + 70 * math.cos(left_rad)), int(cy + 70 * math.sin(left_rad))],
        [int(cx + 70 * math.cos(right_rad)), int(cy + 70 * math.sin(right_rad))]
    ], np.int32)
    cv2.fillPoly(frame, [pts], (0, 120, 80, 100))
    cv2.polylines(frame, [pts], True, (0, 200, 120), 2)
    
    # Gaze line
    cv2.line(frame, (cx, cy), 
             (int(cx + 60 * math.cos(gaze_rad)), int(cy + 60 * math.sin(gaze_rad))),
             (0, 255, 0), 2)
    
    # Pig marker
    cv2.circle(frame, (cx, cy), 6, (255, 200, 100), -1)
    
    # Labels
    cv2.putText(frame, "TOP VIEW", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_side_view(frame, az, el, fov_h, fov_v, x, y):
    """Draw side view of FOV"""
    width, height = 140, 100
    
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (25, 25, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 100), 1)
    
    cx, cy = x + 30, y + height - 20
    
    # Ground line
    cv2.line(frame, (x, cy), (x + width, cy), (60, 60, 80), 1)
    
    # Vertical line (eye level)
    cv2.line(frame, (cx, cy), (cx, y + 10), (60, 60, 80), 1)
    
    # Elevation angle
    elev_rad = math.radians(el)
    half_v = math.radians(fov_v / 2)
    
    # FOV cone (vertical slice)
    pts = np.array([
        [cx, cy],
        [int(cx + 80 * math.cos(elev_rad - half_v)), int(cy - 80 * math.sin(elev_rad - half_v))],
        [int(cx + 80 * math.cos(elev_rad + half_v)), int(cy - 80 * math.sin(elev_rad + half_v))]
    ], np.int32)
    cv2.fillPoly(frame, [pts], (80, 80, 160, 100))
    cv2.polylines(frame, [pts], True, (120, 120, 255), 2)
    
    # Gaze line
    cv2.line(frame, (cx, cy), 
             (int(cx + 70 * math.cos(elev_rad)), int(cy - 70 * math.sin(elev_rad))),
             (0, 255, 0), 2)
    
    # Eye marker
    cv2.circle(frame, (cx, cy), 4, (255, 200, 100), -1)
    
    cv2.putText(frame, "SIDE VIEW", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_compass(frame, az, el, x, y, r):
    """Draw compass with gaze direction"""
    # Circle
    cv2.circle(frame, (x, y), r, (60, 60, 80), 2)
    cv2.circle(frame, (x, y), r - 2, (40, 40, 60), 1)
    
    # Cardinal directions
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        rad = math.radians(angle - 90)
        lx = int(x + (r + 15) * math.cos(rad))
        ly = int(y + (r + 15) * math.sin(rad))
        cv2.putText(frame, label, (lx-4, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    
    # Gaze point (azimuth mapped to compass)
    # Map azimuth to screen coords
    gaze_x = x + int((az / 90) * r) if abs(az) <= 90 else x + int(np.sign(az) * r)
    gaze_y = y - int((el / 60) * r) if abs(el) <= 60 else y - int(np.sign(el) * r)
    
    # Clamp to circle
    dx, dy = gaze_x - x, gaze_y - y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > r - 5:
        gaze_x = int(x + (dx / dist) * (r - 5))
        gaze_y = int(y + (dy / dist) * (r - 5))
    
    # Draw
    cv2.line(frame, (x, y), (gaze_x, gaze_y), (0, 255, 0), 2)
    cv2.circle(frame, (gaze_x, gaze_y), 5, (0, 255, 0), -1)


def draw_info_panel(frame, az, el, fov_h, fov_v):
    """Draw information panel"""
    h, w = frame.shape[:2]
    x, y = 30, h - 120
    
    cv2.rectangle(frame, (x, y), (x + 200, y + 100), (20, 20, 35), -1)
    cv2.rectangle(frame, (x, y), (x + 200, y + 100), (80, 80, 100), 1)
    
    cv2.putText(frame, f"Azimuth:  {az:+.1f}°", (x + 10, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"Elevation: {el:+.1f}°", (x + 10, y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"FOV: {fov_h}° x {fov_v}°", (x + 10, y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
    cv2.putText(frame, f"Animal: Pig", (x + 10, y + 95),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)


def create_fov_video(video_path: str, output_path: str, gaze_data: List[GazeSample],
                    fov_h: float = 120.0, fov_v: float = 80.0,
                    sample_rate: int = 5, scale: float = 0.5):
    """Create annotated FOV video (optimized)"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Downscale for faster processing
    out_w = int(width * scale)
    out_h = int(height * scale)
    out_fps = fps / sample_rate
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    
    print(f"Input: {width}x{height} @ {fps}fps")
    print(f"Output: {out_w}x{out_h} @ {out_fps:.1f}fps (every {sample_rate} frames)")
    print("Processing...")
    
    frame_idx = 0
    gaze_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Get gaze
            while gaze_idx < len(gaze_data) - 1 and gaze_data[gaze_idx + 1].frame <= frame_idx:
                gaze_idx += 1
            
            gaze = gaze_data[gaze_idx] if gaze_idx < len(gaze_data) else None
            
            if gaze:
                # Downscale
                small = cv2.resize(frame, (out_w, out_h))
                # Draw overlay
                small = draw_fov_overlay(small, gaze.azimuth, gaze.elevation, fov_h, fov_v)
                out.write(small)
        
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total} frames ({100*frame_idx/total:.0f}%)")
    
    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def main():
    video = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/PXL_20260410_024928909.mp4"
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_fov_visualization.mp4"
    
    print("Loading gaze data...")
    gaze = load_gaze_data(csv_path)
    print(f"Loaded {len(gaze)} samples")
    
    create_fov_video(video, output, gaze, sample_rate=5, scale=0.4)
    
    print("\nDone! Open in browser or video player.")


if __name__ == "__main__":
    main()
