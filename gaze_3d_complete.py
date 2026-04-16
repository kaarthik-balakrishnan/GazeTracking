"""
Complete 3D Gaze Tracking System
================================

This module provides a complete 3D gaze tracking system for pig eye analysis:
1. Camera calibration from video
2. Eye geometry estimation
3. 3D gaze vector calculation
4. Saccade detection
5. 3D splat visualization
6. FOV prediction

Author: Extended from GazeTracking
"""

import cv2
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class CameraParams:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclass  
class EyeGeometry:
    """3D eye geometry"""
    radius_mm: float
    pupil_radius_mm: float
    cornea_radius_mm: float = 0.0


@dataclass
class GazeSample:
    """Single gaze measurement"""
    frame: int
    timestamp: float
    gaze_azimuth: float  # Horizontal angle (degrees)
    gaze_elevation: float  # Vertical angle (degrees)
    azimuth_raw: float  # Raw azimuth before smoothing
    elevation_raw: float
    pupil_x: float  # Image coordinates
    pupil_y: float
    confidence: str  # detected/interpolated


@dataclass
class SaccadeEvent:
    """Detected saccade"""
    start_frame: int
    end_frame: int
    duration_sec: float
    peak_velocity: float  # deg/sec
    amplitude: float  # deg
    direction: str  # left/right/up/down
    start_azimuth: float
    end_azimuth: float
    start_elevation: float
    end_elevation: float


class CameraCalibrator:
    """Camera calibration from video geometry"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fx = width * 0.8
        self.fy = height * 0.8
        self.cx = width / 2
        self.cy = height / 2
        
    def calibrate_from_ellipse(self, eye_ellipse: Tuple[float, float, float],
                               known_radius_mm: float) -> CameraParams:
        """
        Calibrate using ellipse dimensions of visible eye.
        
        Args:
            eye_ellipse: (center_x, center_y, angle_deg) of eye outline
            known_radius_mm: Known radius of pig eye (typically 10-15mm)
        """
        # For perspective projection of a sphere:
        # The visible ellipse has semi-axes a, b where a = R (apparent radius)
        # and b = R * cos(theta) where theta is viewing angle
        
        # Assume the eye appears roughly circular in our setup
        # so viewing angle is approximately 0
        apparent_radius_pixels = self.width * 0.05  # ~5% of frame
        
        # Scale factor: pixels per mm
        scale = apparent_radius_pixels / known_radius_mm
        
        self.fx = scale * 1000  # Assume 1m focal length base
        self.fy = self.fx
        
        return CameraParams(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            width=self.width, height=self.height
        )
    
    def estimate_focal_length(self, object_size_mm: float, 
                              object_size_pixels: float,
                              distance_mm: float) -> float:
        """Estimate focal length from object dimensions."""
        return (object_size_pixels * distance_mm) / object_size_mm
    
    def get_params(self) -> CameraParams:
        return CameraParams(
            fx=self.fx, fy=self.fy,
            cx=self.cx, cy=self.cy,
            width=self.width, height=self.height
        )


class GazeTracker3D:
    """3D gaze tracking from 2D pupil positions"""
    
    def __init__(self, camera: CameraParams, eye: EyeGeometry):
        self.camera = camera
        self.eye = eye
        self.sphere_center = None
        self.reference_azimuth = 0.0
        self.reference_elevation = 0.0
        
    def set_reference_from_data(self, gaze_data: List[GazeSample], 
                                stable_frames: int = 100):
        """Set gaze reference from stable initial frames."""
        stable_samples = [g for g in gaze_data[:stable_frames] 
                        if g.confidence == 'detected']
        
        if stable_samples:
            self.reference_azimuth = np.mean([s.azimuth_raw for s in stable_samples])
            self.reference_elevation = np.mean([s.elevation_raw for s in stable_samples])
            
            # Set sphere center from first stable sample
            self.sphere_center = np.array([
                stable_samples[0].pupil_x - self.camera.cx,
                stable_samples[0].pupil_y - self.camera.cy,
                500.0  # Assumed distance
            ], dtype=np.float64)
            
    def compute_gaze_angles(self, pupil_x: float, pupil_y: float) -> Tuple[float, float]:
        """
        Compute gaze angles from pupil position.
        
        Returns:
            (azimuth_degrees, elevation_degrees)
        """
        if self.sphere_center is None:
            # Use center of frame as reference
            dx = pupil_x - self.camera.cx
            dy = pupil_y - self.camera.cy
        else:
            # Project pupil to sphere surface
            ray = np.array([dx / self.camera.fx, dy / self.camera.fy, 1.0])
            ray = ray / np.linalg.norm(ray)
            
            # Intersection with sphere
            t = self._ray_sphere_intersection(ray, self.sphere_center, self.eye.radius_mm)
            if t is not None:
                point = t * ray
                # Gaze direction from center to point
                dx = point[0] - self.sphere_center[0]
                dy = point[1] - self.sphere_center[1]
                dz = point[2] - self.sphere_center[2]
                
                azimuth = math.degrees(math.atan2(dy, dx))
                elevation = math.degrees(math.asin(dz / self.eye.radius_mm))
                
                return azimuth, elevation
        
        # Fallback: direct angular computation
        azimuth = math.degrees(math.atan2(dx, 500))  # dx relative to distance
        elevation = math.degrees(math.atan2(dy, 500))
        
        return azimuth, elevation
    
    def _ray_sphere_intersection(self, ray: np.ndarray, 
                                  center: np.ndarray, 
                                  radius: float) -> Optional[float]:
        """Find t where ray intersects sphere: |t*ray - center|^2 = radius^2"""
        oc = -center
        a = np.dot(ray, ray)
        b = 2 * np.dot(ray, oc)
        c = np.dot(oc, oc) - radius ** 2
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        
        # Return closest positive intersection
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None


class SaccadeDetector3D:
    """Detect saccades from 3D gaze angles"""
    
    def __init__(self, velocity_threshold: float = 20.0,  # deg/sec
                 min_duration_frames: int = 2,
                 smoothing_window: int = 3):
        self.velocity_threshold = velocity_threshold
        self.min_duration = min_duration_frames
        self.smoothing = smoothing_window
        
    def detect(self, samples: List[GazeSample], fps: float) -> List[SaccadeEvent]:
        """
        Detect saccades using velocity thresholding.
        
        Args:
            samples: List of gaze samples
            fps: Frame rate
            
        Returns:
            List of saccade events
        """
        # Smooth gaze angles
        azimuths = self._smooth([s.azimuth_raw for s in samples])
        elevations = self._smooth([s.elevation_raw for s in samples])
        
        # Compute angular velocity
        velocities_az = np.diff(azimuths) * fps
        velocities_el = np.diff(elevations) * fps
        velocities = np.sqrt(velocities_az**2 + velocities_el**2)
        
        saccades = []
        in_saccade = False
        saccade_start = 0
        peak_velocity = 0
        velocities_list = velocities.tolist()
        
        for i, vel in enumerate(velocities_list):
            if vel > self.velocity_threshold and not in_saccade:
                # Start saccade
                in_saccade = True
                saccade_start = i
                peak_velocity = vel
                
            elif vel > self.velocity_threshold and in_saccade:
                peak_velocity = max(peak_velocity, vel)
                
            elif vel <= self.velocity_threshold and in_saccade:
                # End saccade
                duration = i - saccade_start
                if duration >= self.min_duration:
                    # Determine direction
                    da = azimuths[i] - azimuths[saccade_start]
                    de = elevations[i] - elevations[saccade_start]
                    
                    if abs(da) > abs(de):
                        direction = 'right' if da > 0 else 'left'
                    else:
                        direction = 'up' if de > 0 else 'down'
                    
                    amplitude = math.sqrt(da**2 + de**2)
                    
                    saccades.append(SaccadeEvent(
                        start_frame=saccade_start,
                        end_frame=i,
                        duration_sec=duration / fps,
                        peak_velocity=peak_velocity,
                        amplitude=amplitude,
                        direction=direction,
                        start_azimuth=azimuths[saccade_start],
                        end_azimuth=azimuths[i],
                        start_elevation=elevations[saccade_start],
                        end_elevation=elevations[i]
                    ))
                
                in_saccade = False
                
        return saccades
    
    def _smooth(self, values: List[float]) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(values) < self.smoothing:
            return np.array(values)
        
        smoothed = np.convolve(values, np.ones(self.smoothing)/self.smoothing, mode='same')
        return smoothed


class GazeVisualizer3D:
    """Visualize 3D gaze data"""
    
    def __init__(self):
        self.fig = None
        self.ax3d = None
        self.ax_time = None
        
    def plot_gaze_3d_splat(self, samples: List[GazeSample], 
                           saccades: List[SaccadeEvent] = None,
                           output_path: str = None):
        """
        Create 3D splat visualization of gaze directions.
        
        Each gaze sample is plotted as a point in 3D space where:
        - X: azimuth angle
        - Y: time
        - Z: elevation angle
        """
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main 3D scatter plot
        ax1 = self.fig.add_subplot(221, projection='3d')
        
        times = [s.timestamp for s in samples]
        azimuths = [s.azimuth_raw for s in samples]
        elevations = [s.elevation_raw for s in samples]
        
        # Color by confidence
        colors = ['green' if s.confidence == 'detected' else 'red' for s in samples]
        
        scatter = ax1.scatter(azimuths, times, elevations, c=colors, s=5, alpha=0.5)
        ax1.set_xlabel('Azimuth (deg)')
        ax1.set_ylabel('Time (s)')
        ax1.set_zlabel('Elevation (deg)')
        ax1.set_title('3D Gaze Splat')
        
        # Gaze over time - azimuth
        ax2 = self.fig.add_subplot(222)
        ax2.plot(times, azimuths, 'b-', linewidth=0.5, alpha=0.7, label='Azimuth')
        
        # Mark saccades
        if saccades:
            for s in saccades:
                start_t = samples[s.start_frame].timestamp if s.start_frame < len(samples) else 0
                end_t = samples[s.end_frame].timestamp if s.end_frame < len(samples) else start_t
                ax2.axvspan(start_t, end_t, alpha=0.3, color='yellow')
                
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Azimuth (deg)')
        ax2.set_title('Gaze Azimuth Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Gaze over time - elevation
        ax3 = self.fig.add_subplot(223)
        ax3.plot(times, elevations, 'g-', linewidth=0.5, alpha=0.7, label='Elevation')
        
        if saccades:
            for s in saccades:
                start_t = samples[s.start_frame].timestamp if s.start_frame < len(samples) else 0
                end_t = samples[s.end_frame].timestamp if s.end_frame < len(samples) else start_t
                ax3.axvspan(start_t, end_t, alpha=0.3, color='yellow')
                
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Elevation (deg)')
        ax3.set_title('Gaze Elevation Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Gaze scatter (top-down view)
        ax4 = self.fig.add_subplot(224)
        detected = [(s.azimuth_raw, s.elevation_raw) for s in samples if s.confidence == 'detected']
        interp = [(s.azimuth_raw, s.elevation_raw) for s in samples if s.confidence == 'interpolated']
        
        if detected:
            d_az, d_el = zip(*detected)
            ax4.scatter(d_az, d_el, c='green', s=3, alpha=0.3, label='Detected')
        if interp:
            i_az, i_el = zip(*interp)
            ax4.scatter(i_az, i_el, c='red', s=3, alpha=0.5, label='Interpolated')
            
        ax4.set_xlabel('Azimuth (deg)')
        ax4.set_ylabel('Elevation (deg)')
        ax4.set_title('Gaze Position (Top View)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        
        return self.fig
    
    def plot_gaze_velocity(self, samples: List[GazeSample], saccades: List[SaccadeEvent],
                           fps: float, output_path: str = None):
        """Plot gaze velocity over time with saccade detection."""
        # Compute velocities
        azimuths = [s.azimuth_raw for s in samples]
        elevations = [s.elevation_raw for s in samples]
        times = [s.timestamp for s in samples]
        
        vel_az = np.diff(azimuths) * fps
        vel_el = np.diff(elevations) * fps
        vel_mag = np.sqrt(vel_az**2 + vel_el**2)
        vel_times = times[1:]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Velocity magnitude
        axes[0].plot(vel_times, vel_mag, 'b-', linewidth=0.5)
        axes[0].axhline(y=20, color='r', linestyle='--', label='Saccade threshold')
        axes[0].set_ylabel('Velocity (deg/s)')
        axes[0].set_title('Gaze Velocity Magnitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Azimuth velocity
        axes[1].plot(vel_times, vel_az, 'g-', linewidth=0.5)
        axes[1].set_ylabel('Azimuth Vel (deg/s)')
        axes[1].set_title('Horizontal Gaze Velocity')
        axes[1].grid(True, alpha=0.3)
        
        # Elevation velocity
        axes[2].plot(vel_times, vel_el, 'r-', linewidth=0.5)
        axes[2].set_ylabel('Elevation Vel (deg/s)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Vertical Gaze Velocity')
        axes[2].grid(True, alpha=0.3)
        
        # Mark saccades
        for ax in axes:
            for s in saccades:
                start_t = samples[s.start_frame].timestamp if s.start_frame < len(samples) else 0
                end_t = samples[s.end_frame].timestamp if s.end_frame < len(samples) else start_t
                ax.axvspan(start_t, end_t, alpha=0.2, color='yellow')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved velocity plot to: {output_path}")
        
        return fig


def process_gaze_data(csv_path: str, fps: float = 29.0,
                      eye_radius_mm: float = 12.0,
                      output_dir: str = None) -> Tuple[List[GazeSample], List[SaccadeEvent], dict]:
    """
    Complete 3D gaze processing pipeline.
    
    Returns:
        Tuple of (samples, saccades, statistics)
    """
    print("=" * 60)
    print("3D GAZE TRACKING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading tracking data...")
    timestamps, eye_x, eye_y, confidences = [], [], [], []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp_sec']))
            eye_x.append(float(row['eye_x_raw']) if row['eye_x_raw'] else float('nan'))
            eye_y.append(float(row['eye_y_raw']) if row['eye_y_raw'] else float('nan'))
            confidences.append(row['confidence'])
    
    print(f"   Loaded {len(timestamps)} frames")
    
    # Camera calibration
    print("\n2. Camera calibration...")
    width, height = 3840, 2160
    calibrator = CameraCalibrator(width, height)
    
    # Estimate from eye size in frame (assuming eye ~5% of frame width)
    eye_size_pixels = width * 0.05
    scale = eye_size_pixels / eye_radius_mm
    calibrator.fx = scale * 500  # Approximate focal length
    calibrator.fy = scale * 500
    camera = calibrator.get_params()
    print(f"   Estimated focal length: {camera.fx:.1f} pixels")
    
    # Eye geometry
    eye_geometry = EyeGeometry(radius_mm=eye_radius_mm, pupil_radius_mm=2.0)
    
    # 3D gaze tracking
    print("\n3. Computing 3D gaze vectors...")
    tracker = GazeTracker3D(camera, eye_geometry)
    
    samples = []
    for i in range(len(timestamps)):
        az, el = tracker.compute_gaze_angles(eye_x[i], eye_y[i])
        
        samples.append(GazeSample(
            frame=i,
            timestamp=timestamps[i],
            gaze_azimuth=az,
            gaze_elevation=el,
            azimuth_raw=az,
            elevation_raw=el,
            pupil_x=eye_x[i],
            pupil_y=eye_y[i],
            confidence=confidences[i]
        ))
    
    # Set reference from stable initial frames
    tracker.set_reference_from_data(samples, stable_frames=50)
    
    # Recompute with reference
    ref_az = tracker.reference_azimuth
    ref_el = tracker.reference_elevation
    
    for s in samples:
        s.azimuth_raw = s.azimuth_raw - ref_az
        s.elevation_raw = s.elevation_raw - ref_el
    
    detected_count = sum(1 for s in samples if s.confidence == 'detected')
    print(f"   Computed {len(samples)} gaze vectors ({detected_count} detected)")
    
    # Saccade detection
    print("\n4. Detecting saccades...")
    saccade_detector = SaccadeDetector3D(velocity_threshold=20.0, min_duration_frames=2)
    saccades = saccade_detector.detect(samples, fps)
    print(f"   Detected {len(saccades)} saccades")
    
    if saccades:
        print("\n   First 5 saccades:")
        for s in saccades[:5]:
            print(f"   - Frame {s.start_frame}-{s.end_frame}: "
                  f"{s.amplitude:.1f}° {s.direction} @ {s.peak_velocity:.1f}°/s")
    
    # Statistics
    print("\n5. Computing statistics...")
    detected = [s for s in samples if s.confidence == 'detected']
    
    if detected:
        az_mean = np.mean([s.azimuth_raw for s in detected])
        az_std = np.std([s.azimuth_raw for s in detected])
        el_mean = np.mean([s.elevation_raw for s in detected])
        el_std = np.std([s.elevation_raw for s in detected])
        
        print(f"   Mean azimuth: {az_mean:.2f}° ± {az_std:.2f}°")
        print(f"   Mean elevation: {el_mean:.2f}° ± {el_std:.2f}°")
    
    stats = {
        'total_frames': len(samples),
        'detected_frames': detected_count,
        'saccade_count': len(saccades),
        'saccades': saccades,
        'mean_azimuth': az_mean if detected else 0,
        'std_azimuth': az_std if detected else 0,
        'mean_elevation': el_mean if detected else 0,
        'std_elevation': el_std if detected else 0
    }
    
    # Visualization
    if output_dir:
        print("\n6. Creating visualizations...")
        viz = GazeVisualizer3D()
        
        viz.plot_gaze_3d_splat(
            samples, saccades,
            output_path=f"{output_dir}/gaze_3d_splat.png"
        )
        
        viz.plot_gaze_velocity(
            samples, saccades, fps,
            output_path=f"{output_dir}/gaze_velocity.png"
        )
        
        # Save CSV
        csv_out = f"{output_dir}/gaze_3d_data.csv"
        with open(csv_out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp', 'azimuth_deg', 'elevation_deg', 'confidence'])
            for s in samples:
                writer.writerow([s.frame, f"{s.timestamp:.3f}", 
                              f"{s.azimuth_raw:.2f}", f"{s.elevation_raw:.2f}", 
                              s.confidence])
        print(f"   Saved 3D gaze data to: {csv_out}")
        
        # Save saccade report
        saccade_csv = f"{output_dir}/saccades.csv"
        with open(saccade_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_frame', 'end_frame', 'duration_sec', 'amplitude_deg', 
                          'peak_velocity', 'direction'])
            for s in saccades:
                writer.writerow([s.start_frame, s.end_frame, f"{s.duration_sec:.3f}",
                              f"{s.amplitude:.2f}", f"{s.peak_velocity:.1f}", s.direction])
        print(f"   Saved saccade report to: {saccade_csv}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return samples, saccades, stats


# Main execution
if __name__ == "__main__":
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_data_final.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    samples, saccades, stats = process_gaze_data(
        csv_path=csv_path,
        fps=29.0,
        eye_radius_mm=12.0,
        output_dir=output_dir
    )
