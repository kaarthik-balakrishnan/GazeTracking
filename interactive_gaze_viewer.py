"""
Interactive 3D Gaze Viewer
==========================

An interactive visualization tool for exploring 3D gaze data.
Features:
- 3D scatter plot with mouse rotation/zoom
- FOV prediction visualization
- Statistical summaries
- Saccade analysis

Usage:
    python interactive_gaze_viewer.py
"""

import cv2
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    pupil_x: float
    pupil_y: float
    confidence: str


@dataclass
class Saccade:
    start_frame: int
    end_frame: int
    duration_sec: float
    amplitude: float
    peak_velocity: float
    direction: str


@dataclass
class FOVPrediction:
    center_azimuth: float
    center_elevation: float
    fov_width: float
    fov_height: float


class InteractiveGazeViewer:
    """Interactive 3D gaze visualization"""
    
    def __init__(self, samples: List[GazeSample], saccades: List[Saccade]):
        self.samples = samples
        self.saccades = saccades
        self.fps = 29.0
        
        # Compute statistics
        self.stats = self._compute_statistics()
        
        # FOV predictions
        self.fov_predictions = self._predict_fov()
        
        # Figure setup
        self.fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, height_ratios=[2, 1, 1])
        
        # Main 3D plot
        self.ax3d = self.fig.add_subplot(gs[0, :], projection='3d')
        
        # Control panel
        self.ax_slider_time = self.fig.add_subplot(gs[1, 0])
        self.ax_slider_vel = self.fig.add_subplot(gs[1, 1])
        
        # Stats text
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.axis('off')
        
        # 2D projections
        self.ax_xy = self.fig.add_subplot(gs[2, 0])
        self.ax_time = self.fig.add_subplot(gs[2, 1])
        self.ax_hist = self.fig.add_subplot(gs[2, 2])
        
        # Current frame
        self.current_frame = 0
        self.velocity_threshold = 20.0
        
        self._setup_plots()
        self._setup_controls()
        self._update_display()
        
        plt.tight_layout()
        
    def _compute_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        detected = [s for s in self.samples if s.confidence == 'detected']
        
        if not detected:
            return {}
        
        azimuths = np.array([s.azimuth for s in detected])
        elevations = np.array([s.elevation for s in detected])
        times = np.array([s.timestamp for s in detected])
        
        # Velocity
        vel_az = np.diff(azimuths) * self.fps
        vel_el = np.diff(elevations) * self.fps
        vel_mag = np.sqrt(vel_az**2 + vel_el**2)
        
        # Statistics
        stats = {
            'n_frames': len(self.samples),
            'n_detected': len(detected),
            'n_interpolated': len(self.samples) - len(detected),
            'n_saccades': len(self.saccades),
            
            # Position statistics
            'az_mean': np.mean(azimuths),
            'az_std': np.std(azimuths),
            'az_min': np.min(azimuths),
            'az_max': np.max(azimuths),
            'az_range': np.max(azimuths) - np.min(azimuths),
            
            'el_mean': np.mean(elevations),
            'el_std': np.std(elevations),
            'el_min': np.min(elevations),
            'el_max': np.max(elevations),
            'el_range': np.max(elevations) - np.min(elevations),
            
            # Velocity statistics
            'vel_mean': np.mean(vel_mag),
            'vel_std': np.std(vel_mag),
            'vel_max': np.max(vel_mag),
            
            # Saccade statistics
            'saccade_rate': len(self.saccades) / (times[-1] if len(times) > 0 else 1),
            'mean_saccade_amplitude': np.mean([s.amplitude for s in self.saccades]) if self.saccades else 0,
            'mean_saccade_velocity': np.mean([s.peak_velocity for s in self.saccades]) if self.saccades else 0,
            'total_saccade_time': sum(s.duration_sec for s in self.saccades),
            
            # Fixation statistics
            'fixation_time_sec': times[-1] - sum(s.duration_sec for s in self.saccades) if len(times) > 0 else 0,
        }
        
        return stats
    
    def _predict_fov(self) -> List[FOVPrediction]:
        """Predict field of view at each frame"""
        predictions = []
        
        # Assuming pig has ~120° horizontal FOV, ~80° vertical
        fov_h = 120.0
        fov_v = 80.0
        
        for s in self.samples:
            predictions.append(FOVPrediction(
                center_azimuth=s.azimuth,
                center_elevation=s.elevation,
                fov_width=fov_h,
                fov_height=fov_v
            ))
        
        return predictions
    
    def _setup_plots(self):
        """Setup all plot elements"""
        self.ax3d.clear()
        
        # Extract data
        times = np.array([s.timestamp for s in self.samples])
        azimuths = np.array([s.azimuth for s in self.samples])
        elevations = np.array([s.elevation for s in self.samples])
        colors = ['green' if s.confidence == 'detected' else 'red' for s in self.samples]
        
        # 3D scatter
        scatter = self.ax3d.scatter(azimuths, times, elevations, 
                                    c=colors, s=10, alpha=0.6)
        
        # Mark saccades
        for saccade in self.saccades:
            if saccade.start_frame < len(self.samples):
                s = self.samples[saccade.start_frame]
                self.ax3d.scatter([s.azimuth], [s.timestamp], [s.elevation],
                                 c='yellow', s=100, marker='*')
        
        self.ax3d.set_xlabel('Azimuth (°)', fontsize=10)
        self.ax3d.set_ylabel('Time (s)', fontsize=10)
        self.ax3d.set_zlabel('Elevation (°)', fontsize=10)
        self.ax3d.set_title('3D Gaze Splat (drag to rotate)', fontsize=12)
        
        # 2D top view
        self.ax_xy.clear()
        self.ax_xy.scatter(azimuths, elevations, c=colors, s=5, alpha=0.5)
        self.ax_xy.set_xlabel('Azimuth (°)')
        self.ax_xy.set_ylabel('Elevation (°)')
        self.ax_xy.set_title('Gaze Position (Top View)')
        self.ax_xy.set_aspect('equal')
        self.ax_xy.grid(True, alpha=0.3)
        
        # Add FOV rectangle
        if self.samples:
            mean_az = self.stats.get('az_mean', 0)
            mean_el = self.stats.get('el_mean', 0)
            rect = plt.Rectangle((mean_az - 60, mean_el - 40), 120, 80,
                                  fill=False, edgecolor='blue', linestyle='--', linewidth=2)
            self.ax_xy.add_patch(rect)
            self.ax_xy.text(mean_az, mean_el, 'FOV', ha='center', va='center',
                           fontsize=8, color='blue')
        
        # Time series
        self.ax_time.clear()
        self.ax_time.plot(times, azimuths, 'b-', linewidth=0.5, alpha=0.7, label='Azimuth')
        self.ax_time.plot(times, elevations, 'g-', linewidth=0.5, alpha=0.7, label='Elevation')
        
        # Mark saccades
        for saccade in self.saccades:
            if saccade.start_frame < len(self.samples):
                t_start = self.samples[saccade.start_frame].timestamp
                t_end = self.samples[saccade.end_frame].timestamp if saccade.end_frame < len(self.samples) else t_start
                self.ax_time.axvspan(t_start, t_end, alpha=0.2, color='yellow')
        
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Angle (°)')
        self.ax_time.set_title('Gaze Over Time')
        self.ax_time.legend(loc='upper right')
        self.ax_time.grid(True, alpha=0.3)
        
        # Histogram
        self.ax_hist.clear()
        detected_az = [s.azimuth for s in self.samples if s.confidence == 'detected']
        self.ax_hist.hist(detected_az, bins=50, alpha=0.7, color='blue', edgecolor='black')
        self.ax_hist.axvline(x=self.stats.get('az_mean', 0), color='red', linestyle='--',
                            label=f'Mean: {self.stats.get("az_mean", 0):.1f}°')
        self.ax_hist.set_xlabel('Azimuth (°)')
        self.ax_hist.set_ylabel('Frequency')
        self.ax_hist.set_title('Azimuth Distribution')
        self.ax_hist.legend()
        self.ax_hist.grid(True, alpha=0.3)
        
        # Stats text
        self._update_stats_text()
    
    def _update_stats_text(self):
        """Update statistics display"""
        stats = self.stats
        
        text = f"""STATISTICAL SUMMARY
{'='*40}

POSITIONS:
  Mean Azimuth: {stats.get('az_mean', 0):.2f}° ± {stats.get('az_std', 0):.2f}°
  Mean Elevation: {stats.get('el_mean', 0):.2f}° ± {stats.get('el_std', 0):.2f}°
  
  Azimuth Range: [{stats.get('az_min', 0):.1f}°, {stats.get('az_max', 0):.1f}°]
  Elevation Range: [{stats.get('el_min', 0):.1f}°, {stats.get('el_max', 0):.1f}°]

DETECTION:
  Detected: {stats.get('n_detected', 0)} frames
  Interpolated: {stats.get('n_interpolated', 0)} frames
  Detection Rate: {100*stats.get('n_detected', 0)/max(1,stats.get('n_frames', 1)):.1f}%

SACCADES:
  Count: {stats.get('n_saccades', 0)}
  Rate: {stats.get('saccade_rate', 0):.2f}/sec
  Mean Amplitude: {stats.get('mean_saccade_amplitude', 0):.1f}°
  Mean Velocity: {stats.get('mean_saccade_velocity', 0):.1f}°/s
  Total Time: {stats.get('total_saccade_time', 0):.2f}s

FIXATIONS:
  Fixation Time: {stats.get('fixation_time_sec', 0):.2f}s
  ({100*stats.get('fixation_time_sec', 0)/max(1,self.samples[-1].timestamp if self.samples else 1):.1f}% of total)
"""
        self.ax_stats.text(0.05, 0.95, text, transform=self.ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    def _setup_controls(self):
        """Setup interactive controls"""
        pass
    
    def _update_display(self):
        """Update display based on current state"""
        self._setup_plots()
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the interactive viewer"""
        plt.show()


class FOVPredictor:
    """FOV prediction based on gaze direction"""
    
    def __init__(self, camera_fov_h=120.0, camera_fov_v=80.0):
        self.camera_fov_h = camera_fov_h
        self.camera_fov_v = camera_fov_v
        
    def predict_visible_region(self, gaze_azimuth: float, 
                              gaze_elevation: float) -> Dict:
        """
        Predict what region is visible given gaze direction.
        
        Returns dictionary with corner coordinates of visible FOV.
        """
        # FOV boundaries relative to gaze center
        half_w = self.camera_fov_h / 2
        half_h = self.camera_fov_v / 2
        
        return {
            'center': (gaze_azimuth, gaze_elevation),
            'corners': {
                'top_left': (gaze_azimuth - half_w, gaze_elevation + half_h),
                'top_right': (gaze_azimuth + half_w, gaze_elevation + half_h),
                'bottom_left': (gaze_azimuth - half_w, gaze_elevation - half_h),
                'bottom_right': (gaze_azimuth + half_w, gaze_elevation - half_h),
            },
            'width': self.camera_fov_h,
            'height': self.camera_fov_v,
            'area': self.camera_fov_h * self.camera_fov_v
        }
    
    def project_point_to_fov(self, world_azimuth: float, world_elevation: float,
                            gaze_azimuth: float, gaze_elevation: float) -> Tuple[float, float]:
        """
        Project a world point into the visual FOV.
        
        Returns normalized position (-1 to 1) within FOV.
        """
        dx = world_azimuth - gaze_azimuth
        dy = world_elevation - gaze_elevation
        
        norm_x = dx / (self.camera_fov_h / 2)
        norm_y = dy / (self.camera_fov_v / 2)
        
        return (norm_x, norm_y)
    
    def is_in_fov(self, world_azimuth: float, world_elevation: float,
                  gaze_azimuth: float, gaze_elevation: float) -> bool:
        """Check if a world point is within the visual FOV."""
        norm_x, norm_y = self.project_point_to_fov(
            world_azimuth, world_elevation, gaze_azimuth, gaze_elevation
        )
        return abs(norm_x) <= 1 and abs(norm_y) <= 1


class GazeStatistics:
    """Comprehensive gaze statistics"""
    
    def __init__(self, samples: List[GazeSample], saccades: List[Saccade]):
        self.samples = samples
        self.saccades = saccades
        self.fps = 29.0
        
    def compute_all(self) -> Dict:
        """Compute all statistics"""
        detected = [s for s in self.samples if s.confidence == 'detected']
        
        if not detected:
            return {}
        
        azimuths = np.array([s.azimuth for s in detected])
        elevations = np.array([s.elevation for s in detected])
        
        # Velocity computation
        vel_az = np.diff(azimuths) * self.fps
        vel_el = np.diff(elevations) * self.fps
        vel_mag = np.sqrt(vel_az**2 + vel_el**2)
        
        stats = {
            # Basic counts
            'total_frames': len(self.samples),
            'valid_frames': len(detected),
            'detection_rate': len(detected) / len(self.samples) if self.samples else 0,
            
            # Position - azimuth
            'azimuth': {
                'mean': float(np.mean(azimuths)),
                'std': float(np.std(azimuths)),
                'min': float(np.min(azimuths)),
                'max': float(np.max(azimuths)),
                'median': float(np.median(azimuths)),
                'range': float(np.max(azimuths) - np.min(azimuths)),
                'iqr': float(np.percentile(azimuths, 75) - np.percentile(azimuths, 25)),
            },
            
            # Position - elevation
            'elevation': {
                'mean': float(np.mean(elevations)),
                'std': float(np.std(elevations)),
                'min': float(np.min(elevations)),
                'max': float(np.max(elevations)),
                'median': float(np.median(elevations)),
                'range': float(np.max(elevations) - np.min(elevations)),
                'iqr': float(np.percentile(elevations, 75) - np.percentile(elevations, 25)),
            },
            
            # Velocity
            'velocity': {
                'mean': float(np.mean(vel_mag)),
                'std': float(np.std(vel_mag)),
                'max': float(np.max(vel_mag)),
                'median': float(np.median(vel_mag)),
            },
            
            # Saccades
            'saccades': self._saccade_statistics(),
            
            # Fixations
            'fixations': self._fixation_statistics(),
            
            # Gaze distribution clusters
            'clusters': self._find_gaze_clusters(detected),
        }
        
        return stats
    
    def _saccade_statistics(self) -> Dict:
        """Saccade-specific statistics"""
        if not self.saccades:
            return {}
        
        amplitudes = [s.amplitude for s in self.saccades]
        velocities = [s.peak_velocity for s in self.saccades]
        durations = [s.duration_sec for s in self.saccades]
        
        # Direction distribution
        directions = {}
        for s in self.saccades:
            directions[s.direction] = directions.get(s.direction, 0) + 1
        
        return {
            'count': len(self.saccades),
            'rate_per_sec': len(self.saccades) / (self.samples[-1].timestamp if self.samples else 1),
            'rate_per_min': len(self.saccades) / (self.samples[-1].timestamp / 60 if self.samples else 1),
            'total_duration_sec': sum(durations),
            'total_duration_pct': sum(durations) / (self.samples[-1].timestamp if self.samples else 1),
            
            'amplitude': {
                'mean': float(np.mean(amplitudes)),
                'std': float(np.std(amplitudes)),
                'min': float(np.min(amplitudes)),
                'max': float(np.max(amplitudes)),
            },
            
            'velocity': {
                'mean': float(np.mean(velocities)),
                'std': float(np.std(velocities)),
                'max': float(np.max(velocities)),
            },
            
            'duration': {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
            },
            
            'directions': directions,
        }
    
    def _fixation_statistics(self) -> Dict:
        """Fixation-specific statistics"""
        if not self.samples:
            return {}
        
        total_time = self.samples[-1].timestamp if self.samples else 0
        saccade_time = sum(s.duration_sec for s in self.saccades)
        fixation_time = total_time - saccade_time
        
        return {
            'total_duration_sec': fixation_time,
            'total_duration_pct': fixation_time / total_time if total_time > 0 else 0,
            'fixation_rate_per_sec': (len(self.samples) - len(self.saccades)) / fixation_time if fixation_time > 0 else 0,
        }
    
    def _find_gaze_clusters(self, detected: List[GazeSample], 
                           threshold: float = 10.0) -> List[Dict]:
        """Find clusters of similar gaze positions (fixation regions)"""
        if len(detected) < 10:
            return []
        
        azimuths = np.array([s.azimuth for s in detected])
        elevations = np.array([s.elevation for s in detected])
        
        # Simple clustering using distance threshold
        clusters = []
        current_cluster = [0]
        
        for i in range(1, len(detected)):
            dist = math.sqrt(
                (azimuths[i] - azimuths[i-1])**2 + 
                (elevations[i] - elevations[i-1])**2
            )
            
            if dist < threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) > 5:  # Minimum cluster size
                    cluster_az = azimuths[current_cluster]
                    cluster_el = elevations[current_cluster]
                    clusters.append({
                        'center_azimuth': float(np.mean(cluster_az)),
                        'center_elevation': float(np.mean(cluster_el)),
                        'size': len(current_cluster),
                        'time_sec': detected[current_cluster[-1]].timestamp - detected[current_cluster[0]].timestamp,
                    })
                current_cluster = [i]
        
        # Don't forget last cluster
        if len(current_cluster) > 5:
            cluster_az = azimuths[current_cluster]
            cluster_el = elevations[current_cluster]
            clusters.append({
                'center_azimuth': float(np.mean(cluster_az)),
                'center_elevation': float(np.mean(cluster_el)),
                'size': len(current_cluster),
                'time_sec': detected[current_cluster[-1]].timestamp - detected[current_cluster[0]].timestamp,
            })
        
        # Sort by size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return clusters[:10]  # Top 10 clusters
    
    def print_summary(self, stats: Dict):
        """Print formatted statistics summary"""
        print("\n" + "="*60)
        print("GAZE ANALYSIS STATISTICS")
        print("="*60)
        
        print("\n### POSITION STATISTICS ###")
        print(f"Azimuth:  {stats['azimuth']['mean']:.2f}° ± {stats['azimuth']['std']:.2f}°")
        print(f"          Range: [{stats['azimuth']['min']:.1f}°, {stats['azimuth']['max']:.1f}°]")
        print(f"Elevation: {stats['elevation']['mean']:.2f}° ± {stats['elevation']['std']:.2f}°")
        print(f"          Range: [{stats['elevation']['min']:.1f}°, {stats['elevation']['max']:.1f}°]")
        
        print("\n### SACCADE ANALYSIS ###")
        sacc = stats.get('saccades', {})
        if sacc:
            print(f"Saccade Count: {sacc['count']}")
            print(f"Saccade Rate: {sacc['rate_per_min']:.1f} /min")
            print(f"Amplitude: {sacc['amplitude']['mean']:.1f}° ± {sacc['amplitude']['std']:.1f}°")
            print(f"Peak Velocity: {sacc['velocity']['mean']:.1f}°/s ± {sacc['velocity']['std']:.1f}°/s")
            print(f"Duration: {sacc['duration']['mean']*1000:.0f}ms ± {sacc['duration']['std']*1000:.0f}ms")
            print(f"Total Saccade Time: {sacc['total_duration_pct']*100:.1f}%")
            print(f"Directions: {sacc.get('directions', {})}")
        
        print("\n### FIXATION ANALYSIS ###")
        fix = stats.get('fixations', {})
        if fix:
            print(f"Fixation Time: {fix['total_duration_sec']:.1f}s ({fix['total_duration_pct']*100:.1f}%)")
        
        print("\n### FIXATION REGIONS (Top 5) ###")
        clusters = stats.get('clusters', [])
        for i, c in enumerate(clusters[:5]):
            print(f"  {i+1}. Az: {c['center_azimuth']:.1f}°, El: {c['center_elevation']:.1f}° "
                  f"(spent {c['time_sec']:.1f}s)")
        
        print("\n" + "="*60)


def load_data(csv_path: str) -> Tuple[List[GazeSample], List[Saccade]]:
    """Load gaze data from CSV"""
    samples = []
    saccades = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(GazeSample(
                frame=int(row['frame']),
                timestamp=float(row['timestamp']),
                azimuth=float(row['azimuth_deg']) if row['azimuth_deg'] else 0,
                elevation=float(row['elevation_deg']) if row['elevation_deg'] else 0,
                pupil_x=0,  # Not in this CSV
                pupil_y=0,
                confidence=row['confidence']
            ))
    
    # Load saccades
    saccade_path = csv_path.replace('gaze_3d_data.csv', 'saccades.csv')
    try:
        with open(saccade_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                saccades.append(Saccade(
                    start_frame=int(row['start_frame']),
                    end_frame=int(row['end_frame']),
                    duration_sec=float(row['duration_sec']),
                    amplitude=float(row['amplitude_deg']),
                    peak_velocity=float(row['peak_velocity']),
                    direction=row['direction']
                ))
    except FileNotFoundError:
        pass
    
    return samples, saccades


def main():
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading data...")
    samples, saccades = load_data(csv_path)
    print(f"Loaded {len(samples)} samples, {len(saccades)} saccades")
    
    # Compute statistics
    stats_computer = GazeStatistics(samples, saccades)
    stats = stats_computer.compute_all()
    stats_computer.print_summary(stats)
    
    # FOV prediction example
    print("\n### FOV PREDICTION EXAMPLE ###")
    fov_predictor = FOVPredictor(camera_fov_h=120, camera_fov_v=80)
    
    if samples:
        example = samples[len(samples)//2]
        fov = fov_predictor.predict_visible_region(example.azimuth, example.elevation)
        print(f"Frame {example.frame} (t={example.timestamp:.2f}s):")
        print(f"  Gaze Center: ({example.azimuth:.1f}°, {example.elevation:.1f}°)")
        print(f"  Visible Region:")
        print(f"    Top-Left: ({fov['corners']['top_left'][0]:.1f}°, {fov['corners']['top_left'][1]:.1f}°)")
        print(f"    Bottom-Right: ({fov['corners']['bottom_right'][0]:.1f}°, {fov['corners']['bottom_right'][1]:.1f}°)")
    
    # Save statistics to file
    import json
    stats_path = f"{output_dir}/gaze_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")
    
    # Launch interactive viewer
    print("\nLaunching interactive viewer...")
    viewer = InteractiveGazeViewer(samples, saccades)
    viewer.show()


if __name__ == "__main__":
    main()
