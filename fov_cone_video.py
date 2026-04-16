"""
FOV Cone Video Visualization
============================

Superimposes a 3D FOV cone on the original video to visualize what the pig is seeing.
Creates a bird's-eye view and side view showing the gaze cone.

Usage:
    python fov_cone_video.py
"""

import cv2
import numpy as np
import csv
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    confidence: str


def load_gaze_data(csv_path: str) -> List[GazeSample]:
    """Load gaze data from CSV"""
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


def draw_3d_cone_birds_eye(frame: np.ndarray, azimuth: float, elevation: float,
                           fov_h: float = 120.0, fov_v: float = 80.0,
                           scale: float = 8.0, offset_x: int = 50, offset_y: int = 50) -> np.ndarray:
    """Draw bird's eye view of FOV cone"""
    h, w = frame.shape[:2]
    
    # Create overlay for the mini map
    overlay = frame.copy()
    
    # Center of pig (origin)
    origin_x = offset_x + 60
    origin_y = offset_y + 60
    
    # Pig direction is "forward" (down in bird's eye, so pig is at top looking down)
    pig_forward = 90  # degrees, pointing right in our view
    
    # Convert gaze to radians
    gaze_rad = math.radians(pig_forward - azimuth)
    elev_rad = math.radians(elevation)
    
    # Draw concentric circles for distance reference
    for r in [40, 80, 120, 160]:
        cv2.circle(overlay, (origin_x, origin_y), r, (50, 50, 50), 1)
    
    # FOV cone boundaries
    half_fov_h = fov_h / 2
    cone_left = math.radians(pig_forward - azimuth - half_fov_h)
    cone_right = math.radians(pig_forward - azimuth + half_fov_h)
    
    # Draw cone (filled)
    cone_length = 180
    left_x = int(origin_x + cone_length * math.cos(cone_left))
    left_y = int(origin_y + cone_length * math.sin(cone_left))
    right_x = int(origin_x + cone_length * math.cos(cone_right))
    right_y = int(origin_y + cone_length * math.sin(cone_right))
    
    # Fill cone with semi-transparent color
    pts = np.array([[origin_x, origin_y], [left_x, left_y], [right_x, right_y]], np.int32)
    cv2.fillPoly(overlay, [pts], (0, 150, 100, 100))
    cv2.polylines(overlay, [pts], True, (0, 255, 150), 2)
    
    # Draw gaze direction line
    gaze_end_x = int(origin_x + 150 * math.cos(gaze_rad))
    gaze_end_y = int(origin_y + 150 * math.sin(gaze_rad))
    cv2.line(overlay, (origin_x, origin_y), (gaze_end_x, gaze_end_y), (0, 255, 0), 3)
    
    # Draw pig position (triangle pointing forward)
    pig_pts = np.array([
        [origin_x + 15 * math.cos(gaze_rad), origin_y + 15 * math.sin(gaze_rad)],
        [origin_x + 10 * math.cos(gaze_rad + 2.5), origin_y + 10 * math.sin(gaze_rad + 2.5)],
        [origin_x + 10 * math.cos(gaze_rad - 2.5), origin_y + 10 * math.sin(gaze_rad - 2.5)]
    ], np.int32)
    cv2.fillPoly(overlay, [pig_pts], (255, 200, 100))
    
    # Add text labels
    cv2.putText(overlay, "BIRD'S EYE VIEW", (offset_x, offset_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Az: {azimuth:.1f}°", (offset_x, offset_y + 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(overlay, f"El: {elevation:.1f}°", (offset_x, offset_y + 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(overlay, f"FOV: {fov_h}° × {fov_v}°", (offset_x, offset_y + 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
    
    # Border
    cv2.rectangle(overlay, (offset_x - 5, offset_y - 5), 
                  (offset_x + 130, offset_y + 200), (100, 100, 100), 1)
    
    return overlay


def draw_3d_cone_side_view(frame: np.ndarray, azimuth: float, elevation: float,
                           fov_h: float = 120.0, fov_v: float = 80.0,
                           offset_x: int = 50, offset_y: int = 250) -> np.ndarray:
    """Draw side view of FOV cone showing elevation"""
    overlay = frame.copy()
    
    # Center of pig
    origin_x = offset_x + 60
    origin_y = offset_y + 60
    
    # Side view: left-right is depth, up-down is elevation
    pig_forward = 0  # looking to the right
    
    # Draw ground line
    cv2.line(overlay, (offset_x, origin_y), (offset_x + 150, origin_y), (80, 80, 80), 1)
    
    # Draw vertical line (pig's eye level)
    cv2.line(overlay, (origin_x, origin_y - 80), (origin_x, origin_y + 20), (80, 80, 80), 1)
    
    # Convert elevation to radians
    elev_rad = math.radians(elevation)
    half_fov_v = (fov_v / 2) / 180 * math.pi * 60  # scale to pixels
    
    # Gaze line
    gaze_length = 120
    gaze_x = origin_x + gaze_length * math.cos(elev_rad)
    gaze_y = origin_y - gaze_length * math.sin(elev_rad)
    cv2.line(overlay, (int(origin_x), int(origin_y)), (int(gaze_x), int(gaze_y)), (0, 255, 0), 3)
    
    # FOV cone (vertical)
    left_x = origin_x + 100 * math.cos(elev_rad - half_fov_v/60)
    left_y = origin_y - 100 * math.sin(elev_rad - half_fov_v/60)
    right_x = origin_x + 100 * math.cos(elev_rad + half_fov_v/60)
    right_y = origin_y - 100 * math.sin(elev_rad + half_fov_v/60)
    
    pts = np.array([[origin_x, origin_y], [left_x, left_y], [right_x, right_y]], np.int32)
    cv2.fillPoly(overlay, [pts], (100, 100, 200, 100))
    cv2.polylines(overlay, [pts], True, (150, 150, 255), 2)
    
    # Pig eye marker
    cv2.circle(overlay, (int(origin_x), int(origin_y)), 5, (255, 200, 100), -1)
    
    # Labels
    cv2.putText(overlay, "SIDE VIEW", (offset_x, offset_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Elevation: {elevation:.1f}°", (offset_x, offset_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Border
    cv2.rectangle(overlay, (offset_x - 5, offset_y - 5), 
                  (offset_x + 160, offset_y + 120), (100, 100, 100), 1)
    
    return overlay


def draw_panoramic_fov(frame: np.ndarray, azimuth: float, elevation: float,
                       fov_h: float = 120.0, fov_v: float = 80.0,
                       offset_x: int = 180, offset_y: int = 250, width: int = 140, height: int = 120) -> np.ndarray:
    """Draw panoramic FOV representation (unfolded view)"""
    overlay = frame.copy()
    
    # Draw rectangle representing unfolded panoramic view
    x1, y1 = offset_x, offset_y
    x2, y2 = offset_x + width, offset_y + height
    
    # Background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 40), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)
    
    # Calculate where current gaze position is in the panoramic view
    # Map azimuth to x position
    normalized_az = (azimuth + 60) / 120  # -60 to +60 maps to 0 to 1
    gaze_x = int(x1 + normalized_az * width)
    
    # Map elevation to y position
    normalized_el = (elevation + 40) / 80  # -40 to +40 maps to 0 to 1
    gaze_y = int(y1 + normalized_el * height)
    
    # Draw visible region
    vis_width = int(width * fov_h / 120)
    vis_height = int(height * fov_v / 80)
    cv2.rectangle(overlay, 
                  (gaze_x - vis_width//2, gaze_y - vis_height//2),
                  (gaze_x + vis_width//2, gaze_y + vis_height//2),
                  (0, 150, 100, 100), -1)
    cv2.rectangle(overlay, 
                  (gaze_x - vis_width//2, gaze_y - vis_height//2),
                  (gaze_x + vis_width//2, gaze_y + vis_height//2),
                  (0, 255, 150), 2)
    
    # Draw gaze point
    cv2.circle(overlay, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
    
    # Labels
    cv2.putText(overlay, "PANORAMIC FOV", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Cardinal directions
    cv2.putText(overlay, "L", (x1 + 5, y1 + height//2 + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(overlay, "R", (x2 - 12, y1 + height//2 + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    return overlay


def draw_gaze_indicator(frame: np.ndarray, azimuth: float, elevation: float,
                        center_x: int, center_y: int, radius: int = 80) -> np.ndarray:
    """Draw gaze direction as an arrow on a circular compass"""
    overlay = frame.copy()
    
    # Draw compass circle
    cv2.circle(overlay, (center_x, center_y), radius, (50, 50, 50), 2)
    cv2.circle(overlay, (center_x, center_y), radius - 2, (30, 30, 30), 1)
    
    # Cardinal directions
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        rad = math.radians(angle - 90)
        x = int(center_x + (radius + 15) * math.cos(rad))
        y = int(center_y + (radius + 15) * math.sin(rad))
        cv2.putText(overlay, label, (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Convert gaze to screen coordinates
    # Azimuth: negative = left, positive = right
    # Elevation: positive = up, negative = down
    gaze_x = center_x + (azimuth / 60) * radius
    gaze_y = center_y - (elevation / 40) * radius
    
    # Clamp to circle
    dx = gaze_x - center_x
    dy = gaze_y - center_y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > radius - 10:
        gaze_x = center_x + (dx / dist) * (radius - 10)
        gaze_y = center_y + (dy / dist) * (radius - 10)
    
    # Draw gaze point
    cv2.circle(overlay, (int(gaze_x), int(gaze_y)), 8, (0, 255, 0), -1)
    cv2.circle(overlay, (int(gaze_x), int(gaze_y)), 12, (0, 255, 0), 2)
    
    # Line from center to gaze
    cv2.line(overlay, (center_x, center_y), (int(gaze_x), int(gaze_y)), (0, 255, 0), 2)
    
    return overlay


def create_fov_visualization(video_path: str, output_path: str, gaze_data: List[GazeSample],
                            fov_h: float = 120.0, fov_v: float = 80.0,
                            sample_rate: int = 1):
    """Create annotated video with FOV visualization (with sampling for speed)"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video - sample rate for faster processing
    output_fps = fps / sample_rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Output: {output_fps:.1f}fps (sampling every {sample_rate} frames)")
    print(f"FOV: {fov_h}° horizontal × {fov_v}° vertical")
    print("Processing...")
    
    frame_idx = 0
    gaze_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every N frames for speed
        if frame_idx % sample_rate == 0:
            # Get corresponding gaze data
            while gaze_idx < len(gaze_data) - 1 and gaze_data[gaze_idx + 1].frame <= frame_idx:
                gaze_idx += 1
            
            if gaze_idx < len(gaze_data) and gaze_data[gaze_idx].frame == frame_idx:
                gaze = gaze_data[gaze_idx]
            else:
                gaze = gaze_data[gaze_idx] if gaze_idx < len(gaze_data) else None
            
            if gaze:
                az = gaze.azimuth
                el = gaze.elevation
                
                # Apply visualizations
                frame = draw_3d_cone_birds_eye(frame, az, el, fov_h, fov_v, 
                                              offset_x=50, offset_y=50)
                frame = draw_3d_cone_side_view(frame, az, el, fov_h, fov_v,
                                              offset_x=50, offset_y=250)
                frame = draw_panoramic_fov(frame, az, el, fov_h, fov_v,
                                           offset_x=180, offset_y=250)
                frame = draw_gaze_indicator(frame, az, el, 
                                          center_x=width - 100, center_y=150, radius=60)
                
                # Add info overlay
                info_text = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Time: {gaze.timestamp:.2f}s",
                    f"Azimuth: {az:.1f}°",
                    f"Elevation: {el:.1f}°",
                    f"Confidence: {gaze.confidence}",
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (20, height - 80 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "No gaze data", (20, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
            
            # Add title
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
            cv2.putText(frame, "Pig Eye Gaze Tracking - FOV Visualization", 
                       (width//2 - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
            
            out.write(frame)
        
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")


def create_side_by_side_view(video_path: str, output_path: str, gaze_data: List[GazeSample],
                             fov_h: float = 120.0, fov_v: float = 80.0,
                             sample_rate: int = 3):
    """Create side-by-side video: original + FOV visualization"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output is 2x width
    out_width = width * 2
    out_height = height
    output_fps = fps / sample_rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
    
    print(f"Creating side-by-side view: {out_width}x{out_height} @ {output_fps:.1f}fps")
    
    frame_idx = 0
    gaze_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Get gaze data
            while gaze_idx < len(gaze_data) - 1 and gaze_data[gaze_idx + 1].frame <= frame_idx:
                gaze_idx += 1
            
            if gaze_idx < len(gaze_data) and gaze_data[gaze_idx].frame == frame_idx:
                gaze = gaze_data[gaze_idx]
            else:
                gaze = gaze_data[gaze_idx] if gaze_idx < len(gaze_data) else None
            
            # Create visualization frame
            vis_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if gaze:
                az = gaze.azimuth
                el = gaze.elevation
                
                # Bird's eye view
                vis_frame = draw_3d_cone_birds_eye(vis_frame, az, el, fov_h, fov_v,
                                                   offset_x=50, offset_y=50)
                
                # Side view
                vis_frame = draw_3d_cone_side_view(vis_frame, az, el, fov_h, fov_v,
                                                   offset_x=50, offset_y=250)
                
                # Panoramic FOV
                vis_frame = draw_panoramic_fov(vis_frame, az, el, fov_h, fov_v,
                                               offset_x=180, offset_y=250)
                
                # Large compass
                vis_frame = draw_gaze_indicator(vis_frame, az, el,
                                                center_x=width - 120, center_y=150, radius=80)
                
                # Add info
                cv2.putText(vis_frame, f"Azimuth: {az:.1f}°", (width - 250, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(vis_frame, f"Elevation: {el:.1f}°", (width - 250, 330),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(vis_frame, f"FOV: {fov_h}° × {fov_v}°", (width - 250, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            
            # Concatenate frames
            combined = np.hstack([frame, vis_frame])
            
            # Add labels
            cv2.rectangle(combined, (0, 0), (width - 1, 40), (0, 0, 0), -1)
            cv2.rectangle(combined, (width, 0), (width * 2 - 1, 40), (0, 0, 0), -1)
            cv2.putText(combined, "ORIGINAL VIDEO", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(combined, "3D FOV VISUALIZATION", (width + 20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 1)
            
            out.write(combined)
        
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"  Processed {frame_idx} frames...")
    
    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")


def main():
    video_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/PXL_20260410_024928909.mp4"
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading gaze data...")
    gaze_data = load_gaze_data(csv_path)
    print(f"Loaded {len(gaze_data)} gaze samples")
    
    # Create overlaid version (annotations on original video)
    print("\n1. Creating FOV-overlaid video...")
    output_overlay = f"{output_dir}/gaze_fov_overlay.mp4"
    create_fov_visualization(video_path, output_overlay, gaze_data, fov_h=120, fov_v=80, sample_rate=3)
    
    # Create side-by-side version
    print("\n2. Creating side-by-side view...")
    output_side = f"{output_dir}/gaze_fov_sidebyside.mp4"
    create_side_by_side_view(video_path, output_side, gaze_data, fov_h=120, fov_v=80, sample_rate=3)
    
    print("\n=== Done! ===")
    print(f"Overlay video: {output_overlay}")
    print(f"Side-by-side: {output_side}")


if __name__ == "__main__":
    main()
