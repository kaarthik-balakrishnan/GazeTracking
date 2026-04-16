"""
Static Visualization for 3D Gaze Analysis
========================================

Creates comprehensive static visualizations of 3D gaze data.
"""

import cv2
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict
from dataclasses import dataclass
import json


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    confidence: str


@dataclass
class Saccade:
    start_frame: int
    end_frame: int
    duration_sec: float
    amplitude: float
    peak_velocity: float
    direction: str


class FOVVisualizer:
    """Visualize FOV predictions"""
    
    def __init__(self, fov_h=120.0, fov_v=80.0):
        self.fov_h = fov_h
        self.fov_v = fov_v
    
    def plot_fov_over_time(self, samples: List[GazeSample], 
                          saccades: List[Saccade],
                          output_path: str):
        """Plot FOV regions over time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        detected = [s for s in samples if s.confidence == 'detected']
        if not detected:
            return
        
        # 1. FOV positions over time
        ax = axes[0, 0]
        times = [s.timestamp for s in detected]
        az = [s.azimuth for s in detected]
        el = [s.elevation for s in detected]
        
        ax.scatter(az, el, c=times, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Elevation (°)')
        ax.set_title('Gaze Position with Time (color = time)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add FOV rectangle
        mean_az = np.mean(az)
        mean_el = np.mean(el)
        rect = plt.Rectangle((mean_az - self.fov_h/2, mean_el - self.fov_v/2),
                             self.fov_h, self.fov_v,
                             fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(mean_az - self.fov_h, mean_az + self.fov_h)
        ax.set_ylim(mean_el - self.fov_v, mean_el + self.fov_v)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Time (s)')
        
        # 2. FOV heatmap
        ax = axes[0, 1]
        h, xedges, yedges = np.histogram2d(az, el, bins=50)
        im = ax.imshow(h.T, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='hot')
        plt.colorbar(im, ax=ax, label='Density')
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Elevation (°)')
        ax.set_title('Gaze Density Heatmap')
        
        # 3. Saccade directions
        ax = axes[1, 0]
        directions = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
        for s in saccades:
            directions[s.direction] = directions.get(s.direction, 0) + 1
        
        colors = {'right': 'green', 'left': 'blue', 'up': 'red', 'down': 'orange'}
        bars = ax.bar(directions.keys(), directions.values(), 
                     color=[colors[d] for d in directions.keys()])
        ax.set_ylabel('Count')
        ax.set_title('Saccade Direction Distribution')
        
        for bar, count in zip(bars, directions.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', fontsize=12)
        
        # 4. FOV center over time with visible region
        ax = axes[1, 1]
        times = [s.timestamp for s in samples]
        az = [s.azimuth for s in samples]
        el = [s.elevation for s in samples]
        
        ax.fill_between(times, 
                       [a - self.fov_h/2 for a in az],
                       [a + self.fov_h/2 for a in az],
                       alpha=0.2, color='blue', label='Horizontal FOV')
        ax.fill_between(times,
                       [e - self.fov_v/2 for e in el],
                       [e + self.fov_v/2 for e in el],
                       alpha=0.2, color='green', label='Vertical FOV')
        ax.plot(times, az, 'b-', linewidth=0.5, label='Gaze Azimuth')
        ax.plot(times, el, 'g-', linewidth=0.5, label='Gaze Elevation')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (°)')
        ax.set_title('FOV Center Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark saccades
        for s in saccades:
            if s.start_frame < len(samples):
                t = samples[s.start_frame].timestamp
                ax.axvline(x=t, alpha=0.3, color='yellow')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved FOV visualization to: {output_path}")


class ComprehensiveGazePlotter:
    """Create comprehensive gaze analysis plots"""
    
    def __init__(self, samples: List[GazeSample], saccades: List[Saccade]):
        self.samples = samples
        self.saccades = saccades
        self.fps = 29.0
        
    def create_comprehensive_plot(self, output_path: str):
        """Create comprehensive multi-panel visualization"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        detected = [s for s in self.samples if s.confidence == 'detected']
        times = np.array([s.timestamp for s in self.samples])
        az = np.array([s.azimuth for s in self.samples])
        el = np.array([s.elevation for s in self.samples])
        
        # 1. 3D Gaze Splat
        ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        colors = ['green' if s.confidence == 'detected' else 'red' 
                   for s in self.samples]
        ax3d.scatter(az, times, el, c=colors, s=5, alpha=0.5)
        ax3d.set_xlabel('Azimuth (°)')
        ax3d.set_ylabel('Time (s)')
        ax3d.set_zlabel('Elevation (°)')
        ax3d.set_title('3D Gaze Splat')
        
        # Mark saccades
        for s in self.saccades[:20]:  # Top 20
            if s.start_frame < len(self.samples):
                p = self.samples[s.start_frame]
                ax3d.scatter([p.azimuth], [p.timestamp], [p.elevation],
                            c='yellow', s=100, marker='*')
        
        # 2. Top View (Azimuth vs Elevation)
        ax = fig.add_subplot(gs[0, 2])
        ax.scatter(az, el, c=times, cmap='viridis', s=5, alpha=0.5)
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Elevation (°)')
        ax.set_title('Gaze Position (Top View)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 3. Time series
        ax = fig.add_subplot(gs[0, 3])
        ax.plot(times, az, 'b-', linewidth=0.5, alpha=0.7, label='Azimuth')
        ax.plot(times, el, 'g-', linewidth=0.5, alpha=0.7, label='Elevation')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (°)')
        ax.set_title('Gaze Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Azimuth histogram
        ax = fig.add_subplot(gs[1, 2])
        ax.hist(az[~np.isnan(az)], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=np.nanmean(az), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.nanmean(az):.1f}°')
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Frequency')
        ax.set_title('Azimuth Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Elevation histogram
        ax = fig.add_subplot(gs[1, 3])
        ax.hist(el[~np.isnan(el)], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=np.nanmean(el), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.nanmean(el):.1f}°')
        ax.set_xlabel('Elevation (°)')
        ax.set_ylabel('Frequency')
        ax.set_title('Elevation Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Velocity magnitude
        ax = fig.add_subplot(gs[2, 0])
        vel_az = np.diff(az) * self.fps
        vel_el = np.diff(el) * self.fps
        vel_mag = np.sqrt(vel_az**2 + vel_el**2)
        vel_times = times[1:]
        ax.plot(vel_times, vel_mag, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=20, color='red', linestyle='--', label='Saccade threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (°/s)')
        ax.set_title('Gaze Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Saccade amplitude distribution
        ax = fig.add_subplot(gs[2, 1])
        amplitudes = [s.amplitude for s in self.saccades]
        if amplitudes:
            ax.hist(amplitudes, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(x=np.mean(amplitudes), color='red', linestyle='--',
                      label=f'Mean: {np.mean(amplitudes):.1f}°')
        ax.set_xlabel('Amplitude (°)')
        ax.set_ylabel('Frequency')
        ax.set_title('Saccade Amplitude Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Saccade duration distribution
        ax = fig.add_subplot(gs[2, 2])
        durations = [s.duration_sec * 1000 for s in self.saccades]
        if durations:
            ax.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(x=np.mean(durations), color='red', linestyle='--',
                      label=f'Mean: {np.mean(durations):.0f}ms')
        ax.set_xlabel('Duration (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Saccade Duration Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Saccade direction pie chart
        ax = fig.add_subplot(gs[2, 3])
        directions = {'Right': 0, 'Left': 0, 'Up': 0, 'Down': 0}
        for s in self.saccades:
            directions[s.direction.capitalize()] += 1
        colors_pie = ['green', 'blue', 'red', 'orange']
        wedges, texts, autotexts = ax.pie(directions.values(), labels=directions.keys(),
                                          colors=colors_pie, autopct='%1.0f%%',
                                          startangle=90)
        ax.set_title('Saccade Directions')
        
        # 10. Summary statistics text
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')
        
        stats_text = self._create_stats_text()
        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Title
        fig.suptitle('Comprehensive 3D Gaze Analysis Report', fontsize=16, fontweight='bold')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive plot to: {output_path}")
    
    def _create_stats_text(self) -> str:
        """Create statistics summary text"""
        detected = [s for s in self.samples if s.confidence == 'detected']
        if not detected:
            return "No valid data"
        
        az = np.array([s.azimuth for s in detected])
        el = np.array([s.elevation for s in detected])
        
        vel_az = np.diff(az) * self.fps
        vel_el = np.diff(el) * self.fps
        vel_mag = np.sqrt(vel_az**2 + vel_el**2)
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    GAZE ANALYSIS STATISTICAL SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  POSITION STATISTICS                                                                                  ║
║  ────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  Azimuth:   Mean = {np.mean(az):7.2f}° ± {np.std(az):6.2f}°  |  Range: [{np.min(az):6.1f}°, {np.max(az):6.1f}°]         ║
║  Elevation: Mean = {np.mean(el):7.2f}° ± {np.std(el):6.2f}°  |  Range: [{np.min(el):6.1f}°, {np.max(el):6.1f}°]         ║
║                                                                                                       ║
║  VELOCITY STATISTICS                                                                                 ║
║  ────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  Mean Velocity:   {np.mean(vel_mag):6.1f}°/s  |  Max: {np.max(vel_mag):6.1f}°/s                                           ║
║                                                                                                       ║
║  SACCADE STATISTICS          ({len(self.saccades)} saccades detected)                                                          ║
║  ────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  Rate: {len(self.saccades)/(self.samples[-1].timestamp/60):6.1f} /min  |  Total Time: {sum(s.duration_sec for s in self.saccades):5.1f}s ({100*sum(s.duration_sec for s in self.saccades)/self.samples[-1].timestamp:.1f}%)                              ║
║  Mean Amplitude: {np.mean([s.amplitude for s in self.saccades]):5.1f}°  |  Mean Peak Velocity: {np.mean([s.peak_velocity for s in self.saccades]):5.1f}°/s                ║
║                                                                                                       ║
║  DATA QUALITY                                                                                         ║
║  ────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  Detected Frames: {len(detected):4d} / {len(self.samples):4d} ({100*len(detected)/len(self.samples):5.1f}%)                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""


def main():
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading data...")
    samples, saccades = load_data(csv_path)
    print(f"Loaded {len(samples)} samples, {len(saccades)} saccades")
    
    # Create FOV visualization
    print("\nCreating FOV visualization...")
    fov_viz = FOVVisualizer(fov_h=120, fov_v=80)
    fov_viz.plot_fov_over_time(samples, saccades, 
                               f"{output_dir}/fov_visualization.png")
    
    # Create comprehensive plot
    print("\nCreating comprehensive plot...")
    plotter = ComprehensiveGazePlotter(samples, saccades)
    plotter.create_comprehensive_plot(f"{output_dir}/comprehensive_gaze_analysis.png")
    
    print("\nDone!")


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
                confidence=row['confidence']
            ))
    
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


if __name__ == "__main__":
    main()
