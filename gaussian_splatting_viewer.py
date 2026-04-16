"""
3D Gaussian Splatting Gaze Visualization
========================================

Visualizes pig eye gaze data using Gaussian Splatting principles.
Each gaze point becomes a 3D Gaussian blob representing FOV coverage.

Based on: https://huggingface.co/blog/gaussian-splatting

Gaussian Parameters:
- Position: (azimuth, time, elevation)
- Covariance: scale based on FOV
- Color: intensity based on dwell time
- Alpha: transparency based on confidence

Usage:
    python gaussian_splatting_viewer.py
"""

import numpy as np
import csv
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math


@dataclass
class Gaussian3D:
    """A 3D Gaussian for splatting visualization"""
    x: float  # azimuth
    y: float  # time
    z: float  # elevation
    sx: float  # scale x (covariance)
    sy: float  # scale y
    sz: float  # scale z
    intensity: float  # RGB intensity
    alpha: float  # transparency


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    confidence: str


def load_gaze_data(csv_path: str) -> Tuple[List[GazeSample], List[Dict]]:
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
                saccades.append({
                    'start_frame': int(row['start_frame']),
                    'end_frame': int(row['end_frame']),
                    'duration_sec': float(row['duration_sec']),
                    'amplitude_deg': float(row['amplitude_deg']),
                    'peak_velocity': float(row['peak_velocity']),
                    'direction': row['direction']
                })
    except FileNotFoundError:
        pass
    
    return samples, saccades


def gaussian_3d(x: float, y: float, z: float,
                gx: float, gy: float, gz: float,
                sx: float, sy: float, sz: float) -> float:
    """Evaluate 3D Gaussian at point (x,y,z) with center (gx,gy,gz) and scales (sx,sy,sz)"""
    dx = (x - gx) / sx
    dy = (y - gy) / sy
    dz = (z - gz) / sz
    return np.exp(-0.5 * (dx*dx + dy*dy + dz*dz))


def create_gaze_gaussians(samples: List[GazeSample], 
                          fov_h: float = 120.0, 
                          fov_v: float = 80.0) -> List[Gaussian3D]:
    """
    Convert gaze samples to 3D Gaussians.
    
    Each detected gaze point becomes a Gaussian blob centered at its
    (azimuth, time, elevation) position. The scale represents the FOV coverage.
    """
    gaussians = []
    
    for s in samples:
        if s.confidence != 'detected':
            continue
        
        # Gaussian centered at gaze position
        # Scale proportional to FOV (shows "what pig sees" at that moment)
        g = Gaussian3D(
            x=s.azimuth,
            y=s.timestamp,
            z=s.elevation,
            sx=fov_h / 6,  # ~20° spread
            sy=0.1,        # tight in time
            sz=fov_v / 6,  # ~13° spread
            intensity=1.0,
            alpha=0.6
        )
        gaussians.append(g)
    
    return gaussians


def compute_density_grid(gaussians: List[Gaussian3D], 
                        az_range: Tuple[float, float],
                        time_range: Tuple[float, float],
                        el_range: Tuple[float, float],
                        resolution: int = 100) -> np.ndarray:
    """
    Compute 3D density field by summing Gaussians on a grid.
    This is the "splatting" step - projecting Gaussians to a 3D volume.
    """
    az_grid = np.linspace(az_range[0], az_range[1], resolution)
    time_grid = np.linspace(time_range[0], time_range[1], resolution)
    el_grid = np.linspace(el_range[0], el_range[1], resolution)
    
    # Create meshgrid
    AZ, TIME, EL = np.meshgrid(az_grid, time_grid, el_grid, indexing='ij')
    
    # Accumulate density
    density = np.zeros((resolution, resolution, resolution))
    
    for g in gaussians:
        # Add Gaussian contribution
        density += g.intensity * gaussian_3d(
            AZ, TIME, EL,
            g.x, g.y, g.z,
            g.sx, g.sy, g.sz
        ) * g.alpha
    
    return density, (AZ, TIME, EL)


def compute_density_2d(gaussians: List[Gaussian3D],
                       az_range: Tuple[float, float],
                       el_range: Tuple[float, float],
                       resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D density (azimuth vs elevation) by projecting Gaussians.
    This shows where the pig looked most frequently.
    """
    az_grid = np.linspace(az_range[0], az_range[1], resolution)
    el_grid = np.linspace(el_range[0], el_range[1], resolution)
    
    AZ, EL = np.meshgrid(az_grid, el_grid, indexing='ij')
    density = np.zeros((resolution, resolution))
    
    for g in gaussians:
        # 2D Gaussian (integrate over time)
        dx = (AZ - g.x) / g.sx
        dz = (EL - g.z) / g.sz
        density += g.intensity * np.exp(-0.5 * (dx*dx + dz*dz)) * g.alpha
    
    return density, az_grid, el_grid


def generate_html_gaussian_splat(samples: List[GazeSample], 
                                  gaussians: List[Gaussian3D],
                                  saccades: List[Dict],
                                  output_path: str,
                                  fov_h: float = 120.0,
                                  fov_v: float = 80.0):
    """Generate interactive HTML with Gaussian splat visualization"""
    
    detected = [s for s in samples if s.confidence == 'detected']
    
    # Extract data
    az = [s.azimuth for s in detected]
    el = [s.elevation for s in detected]
    times = [s.timestamp for s in detected]
    
    # Compute 2D density
    az_min, az_max = min(az) - 20, max(az) + 20
    el_min, el_max = min(el) - 20, max(el) + 20
    
    density, az_grid, el_grid = compute_density_2d(gaussians, 
                                                    (az_min, az_max), 
                                                    (el_min, el_max),
                                                    resolution=80)
    
    # Normalize density for visualization
    density_norm = (density / density.max() * 255).astype(np.uint8)
    
    # Stats
    vel_az = np.diff(az) * 29
    vel_el = np.diff(el) * 29
    vel_mag = np.sqrt(vel_az**2 + vel_el**2)
    
    az_mean = np.mean(az)
    el_mean = np.mean(el)
    
    # Count saccade directions
    dir_counts = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
    for sac in saccades:
        dir_counts[sac['direction']] = dir_counts.get(sac['direction'], 0) + 1
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Gaussian Splatting - Pig Eye Gaze Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff006e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;
            margin-bottom: 20px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: #00d4ff;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            background: linear-gradient(135deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{
            .viz-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-box {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-title {{
            font-size: 14px;
            color: #00d4ff;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .info-panel {{
            background: rgba(0,212,255,0.1);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }}
        .info-panel h3 {{
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .info-panel p {{
            font-size: 13px;
            line-height: 1.6;
            color: #aaa;
        }}
        .gaussian-concept {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .concept-box {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .concept-box h4 {{
            color: #7b2ff7;
            margin-bottom: 8px;
        }}
        .concept-box p {{
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Gaussian Splatting - Pig Eye Gaze Visualization</h1>
        <p class="subtitle">Visualizing gaze field as 3D Gaussians (azimuth × time × elevation)</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(detected)}</div>
                <div class="stat-label">Gaze Points</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(gaussians)}</div>
                <div class="stat-label">3D Gaussians</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(saccades)}</div>
                <div class="stat-label">Saccades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(saccades)/(samples[-1].timestamp/60):.1f}/min</div>
                <div class="stat-label">Saccade Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{az_mean:.1f}°</div>
                <div class="stat-label">Mean Azimuth</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{el_mean:.1f}°</div>
                <div class="stat-label">Mean Elevation</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(0,255,136,0.6);"></div>
                <span>Gaze Points</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(123,47,247,0.6);"></div>
                <span>Gaussian Density</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(255,215,0,0.8);"></div>
                <span>Saccade Events</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgba(0,212,255,0.3); border: 2px dashed #00d4ff;"></div>
                <span>FOV: {fov_h}° × {fov_v}°</span>
            </div>
        </div>
        
        <div class="gaussian-concept">
            <div class="concept-box">
                <h4>Gaussian Position</h4>
                <p>Each Gaussian centered at (azimuth, time, elevation) of gaze</p>
            </div>
            <div class="concept-box">
                <h4>Gaussian Scale</h4>
                <p>Scale ∝ FOV ({fov_h}° × {fov_v}°) - shows "what pig sees"</p>
            </div>
            <div class="concept-box">
                <h4>Gaussian Alpha</h4>
                <p>Transparency indicates confidence and density</p>
            </div>
        </div>
        
        <div class="viz-grid">
            <div class="chart-box full-width">
                <div class="chart-title">Interactive 3D Gaussian Splat (drag to rotate, scroll to zoom)</div>
                <div id="gaussian-3d" style="width:100%;height:600px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Top View: Gaze Density Heatmap (Gaussian Splat)</div>
                <div id="density-heatmap" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Gaze Position Distribution</div>
                <div id="position-scatter" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Gaze Over Time with FOV Cone</div>
                <div id="gaze-timeseries" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Saccade Direction Distribution</div>
                <div id="saccade-pie" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box full-width">
                <div class="chart-title">Gaussian Splat Cross-Sections</div>
                <div id="cross-sections" style="width:100%;height:400px;"></div>
            </div>
        </div>
        
        <div class="info-panel">
            <h3>About 3D Gaussian Splatting</h3>
            <p>
                3D Gaussian Splatting represents the gaze field as a set of 3D Gaussian blobs. Each Gaussian is defined by:
                <strong>Position</strong> (azimuth, time, elevation), <strong>Covariance/Scale</strong> (how spread out),
                <strong>Intensity</strong> (color/brightness), and <strong>Alpha</strong> (transparency). 
                The Gaussians are "splatted" (projected) onto 2D views for visualization. This representation
                naturally shows gaze density - areas where the pig looked frequently appear brighter/denser.
            </p>
        </div>
    </div>
    
    <script>
        // 3D Gaussian Splat - volumetric visualization
        const gaussian3dData = [{{
            type: 'volume',
            x: {json.dumps(az)},
            y: {json.dumps(times)},
            z: {json.dumps(el)},
            intensity: {json.dumps([1.0] * len(az))},
            colorscale: 'Viridis',
            opacity: 0.4,
            isomin: 0.1,
            isomax: 1.0,
            surface: {{count: 3}},
            caps: {{x: {{show: false}}, y: {{show: false}}, z: {{show: false}}}},
            slices: {{
                x: {{show: true, location: {az_mean:.1f}}},
                y: {{show: true, location: {times[len(times)//2]:.1f}}},
                z: {{show: true, location: {el_mean:.1f}}}
            }},
            name: 'Gaze Density'
        }}];
        
        Plotly.newPlot('gaussian-3d', gaussian3dData, {{
            scene: {{
                xaxis: {{title: 'Azimuth (°)', gridcolor: '#333', backgroundcolor: 'rgba(20,20,40,0.8)'}},
                yaxis: {{title: 'Time (s)', gridcolor: '#333', backgroundcolor: 'rgba(20,20,40,0.8)'}},
                zaxis: {{title: 'Elevation (°)', gridcolor: '#333', backgroundcolor: 'rgba(20,20,40,0.8)'}},
                bgcolor: 'rgba(15,12,41,1)',
                camera: {{eye: {{x: 1.5, y: 1.5, z: 1.2}}}}
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            margin: {{l: 0, r: 0, t: 30, b: 0}}
        }});
        
        // Density Heatmap
        const heatmapData = [{{
            type: 'heatmap',
            z: {json.dumps(density_norm.tolist())},
            x: {json.dumps(list(az_grid))},
            y: {json.dumps(list(el_grid))},
            colorscale: [
                [0, 'rgba(0,0,0,0)'],
                [0.2, 'rgba(0,100,255,0.5)'],
                [0.5, 'rgba(123,47,247,0.7)'],
                [0.8, 'rgba(255,100,100,0.9)'],
                [1, 'rgba(255,255,0,1)']
            ],
            showscale: true,
            colorbar: {{title: 'Gaze Density', titleside: 'right'}}
        }}];
        
        const fovRect = {{
            type: 'scatter',
            mode: 'lines',
            x: [{az_mean - fov_h/2}, {az_mean + fov_h/2}, {az_mean + fov_h/2}, {az_mean - fov_h/2}, {az_mean - fov_h/2}],
            y: [{el_mean - fov_v/2}, {el_mean - fov_v/2}, {el_mean + fov_v/2}, {el_mean + fov_v/2}, {el_mean - fov_v/2}],
            line: {{color: '#00d4ff', width: 2, dash: 'dash'}},
            name: 'FOV Center'
        }};
        
        Plotly.newPlot('density-heatmap', [...heatmapData, fovRect], {{
            xaxis: {{title: 'Azimuth (°)', gridcolor: '#333'}},
            yaxis: {{title: 'Elevation (°)', gridcolor: '#333'}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}}
        }});
        
        // Position Scatter with FOV circles
        const scatterData = [{{
            type: 'scatter',
            mode: 'markers',
            x: {json.dumps(az)},
            y: {json.dumps(el)},
            marker: {{
                size: 5,
                color: {json.dumps(times)},
                colorscale: 'Viridis',
                opacity: 0.6
            }},
            name: 'Gaze Points'
        }},
        {{
            type: 'scatter',
            mode: 'markers',
            x: [{az_mean}],
            y: [{el_mean}],
            marker: {{
                size: 25,
                color: '#ff006e',
                symbol: 'square',
                line: {{width: 3, color: 'white'}}
            }},
            name: 'Mean Gaze'
        }}];
        
        Plotly.newPlot('position-scatter', scatterData, {{
            xaxis: {{title: 'Azimuth (°)', gridcolor: '#333', range: [{az_min}, {az_max}]}},
            yaxis: {{title: 'Elevation (°)', gridcolor: '#333', range: [{el_min}, {el_max}]}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            shapes: [{{
                type: 'rect',
                x0: {az_mean - fov_h/2},
                y0: {el_mean - fov_v/2},
                x1: {az_mean + fov_h/2},
                y1: {el_mean + fov_v/2},
                line: {{color: 'rgba(0,212,255,0.5)', width: 2, dash: 'dash'}}
            }}]
        }});
        
        // Time Series with FOV bands
        const tsData = [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {json.dumps(times)},
                y: {json.dumps(az)},
                name: 'Azimuth',
                line: {{color: '#00ff88', width: 1.5}},
                fill: 'tonexty',
                fillcolor: 'rgba(0,255,136,0.1)'
            }},
            {{
                type: 'scatter',
                mode: 'lines',
                x: {json.dumps(times)},
                y: {json.dumps(el)},
                name: 'Elevation',
                line: {{color: '#7b2ff7', width: 1.5}}
            }}
        ];
        
        Plotly.newPlot('gaze-timeseries', tsData, {{
            xaxis: {{title: 'Time (s)', gridcolor: '#333'}},
            yaxis: {{title: 'Angle (°)', gridcolor: '#333'}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            legend: {{orientation: 'h', x: 0.5, xanchor: 'center'}}
        }});
        
        // Saccade Pie
        const pieData = [{{
            type: 'pie',
            labels: {json.dumps(list(dir_counts.keys()))},
            values: {json.dumps(list(dir_counts.values()))},
            marker: {{
                colors: ['#00ff00', '#0088ff', '#ff4444', '#ffaa00']
            }},
            textinfo: 'label+percent',
            textfont: {{color: '#fff'}},
            hole: 0.4
        }}];
        
        Plotly.newPlot('saccade-pie', pieData, {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}}
        }});
        
        // Cross-sections
        // Horizontal slice (azimuth vs time at median elevation)
        const crosssections = [
            {{
                type: 'contour',
                x: {json.dumps(times[::5])},
                y: {json.dumps(az[::5])},
                z: {json.dumps([[1]*len(az[::5]) for _ in range(len(times[::5]))])},
                contours: {{ coloring: 'heatmap' }},
                showscale: false,
                subplot: 'xy'
            }}
        ];
        
        Plotly.newPlot('cross-sections', crosssections, {{
            xaxis: {{title: 'Time (s)', gridcolor: '#333'}},
            yaxis: {{title: 'Azimuth (°)', gridcolor: '#333'}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            annotations: [{{
                text: 'Horizontal Cross-Section (Azimuth vs Time)',
                x: 0.5,
                y: 1.1,
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                font: {{color: '#00d4ff', size: 12}}
            }}]
        }});
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Gaussian Splatting visualization saved to: {output_path}")


def generate_static_splat_image(gaussians: List[Gaussian3D], 
                               samples: List[GazeSample],
                               output_path: str,
                               fov_h: float = 120.0,
                               fov_v: float = 80.0):
    """Generate static 3D splat visualization using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.gridspec as gridspec
        
        detected = [s for s in samples if s.confidence == 'detected']
        
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor('#0f0c29')
        
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. 3D Gaussian Splat
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax1.set_facecolor('#0f0c29')
        
        az = [s.azimuth for s in detected]
        el = [s.elevation for s in detected]
        times = [s.timestamp for s in detected]
        
        # Plot Gaussians as translucent spheres
        for g in gaussians[::5]:  # Subsample for visibility
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = g.x + g.sx * np.outer(np.cos(u), np.sin(v))
            y = g.y + g.sy * np.outer(np.ones_like(u), np.cos(v))
            z = g.z + g.sz * np.outer(np.sin(u), np.sin(v))
            ax1.plot_surface(x, y, z, color='cyan', alpha=0.05, linewidth=0)
        
        scatter = ax1.scatter(az, times, el, c=times, cmap='viridis', s=8, alpha=0.7)
        ax1.set_xlabel('Azimuth (°)', color='white', fontsize=10)
        ax1.set_ylabel('Time (s)', color='white', fontsize=10)
        ax1.set_zlabel('Elevation (°)', color='white', fontsize=10)
        ax1.set_title('3D Gaussian Splat Visualization', color='white', fontsize=14)
        ax1.tick_params(colors='white')
        
        # 2. Density Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor('#0f0c29')
        
        density, az_grid, el_grid = compute_density_2d(
            gaussians,
            (min(az)-20, max(az)+20),
            (min(el)-20, max(el)+20),
            resolution=50
        )
        
        im = ax2.imshow(density.T, origin='lower', aspect='auto',
                       extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
                       cmap='hot')
        ax2.set_xlabel('Azimuth (°)', color='white')
        ax2.set_ylabel('Elevation (°)', color='white')
        ax2.set_title('Gaze Density', color='white', fontsize=10)
        ax2.tick_params(colors='white')
        plt.colorbar(im, ax=ax2, label='Density')
        
        # 3. Time series
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor('#0f0c29')
        ax3.plot(times, az, 'g-', linewidth=0.5, alpha=0.7, label='Azimuth')
        ax3.plot(times, el, 'm-', linewidth=0.5, alpha=0.7, label='Elevation')
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Angle (°)', color='white')
        ax3.set_title('Gaze Over Time', color='white', fontsize=10)
        ax3.tick_params(colors='white')
        ax3.legend(fontsize=8)
        
        # 4. Azimuth histogram
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.set_facecolor('#0f0c29')
        ax4.hist(az, bins=40, alpha=0.7, color='cyan', edgecolor='black')
        ax4.axvline(x=np.mean(az), color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Azimuth (°)', color='white')
        ax4.set_ylabel('Frequency', color='white')
        ax4.set_title(f'Azimuth: μ={np.mean(az):.1f}°', color='white', fontsize=10)
        ax4.tick_params(colors='white')
        
        # 5. Elevation histogram
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.set_facecolor('#0f0c29')
        ax5.hist(el, bins=40, alpha=0.7, color='magenta', edgecolor='black')
        ax5.axvline(x=np.mean(el), color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Elevation (°)', color='white')
        ax5.set_ylabel('Frequency', color='white')
        ax5.set_title(f'Elevation: μ={np.mean(el):.1f}°', color='white', fontsize=10)
        ax5.tick_params(colors='white')
        
        # 6. FOV concept
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.set_facecolor('#0f0c29')
        ax6.set_xlim(-80, 80)
        ax6.set_ylim(-60, 60)
        
        # Draw FOV cone
        theta = np.linspace(-np.radians(60), np.radians(60), 50)
        r = 50
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax6.fill(x, y, alpha=0.3, color='cyan', label=f'FOV: {fov_h}° × {fov_v}°')
        ax6.plot(x, y, 'c-', linewidth=2)
        
        # Gaze point
        mean_az = np.radians(np.mean(az))
        mean_el = np.radians(np.mean(el))
        ax6.arrow(0, 0, 30*np.sin(mean_az), 30*np.sin(mean_el), 
                 head_width=5, head_length=3, fc='yellow', ec='yellow')
        
        ax6.set_xlabel('Azimuth', color='white')
        ax6.set_ylabel('Elevation', color='white')
        ax6.set_title('FOV Cone', color='white', fontsize=10)
        ax6.tick_params(colors='white')
        ax6.legend(fontsize=8)
        
        fig.suptitle('3D Gaussian Splatting - Pig Eye Gaze Analysis', 
                    color='cyan', fontsize=16, y=0.98)
        
        plt.savefig(output_path, dpi=150, facecolor='#0f0c29', 
                   edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"Static splat image saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available, skipping static image")


def main():
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading gaze data...")
    samples, saccades = load_gaze_data(csv_path)
    print(f"Loaded {len(samples)} samples, {len(saccades)} saccades")
    
    print("\nCreating 3D Gaussians...")
    fov_h, fov_v = 120.0, 80.0  # Pig FOV
    gaussians = create_gaze_gaussians(samples, fov_h, fov_v)
    print(f"Created {len(gaussians)} Gaussians")
    
    print("\nGenerating HTML visualization...")
    html_path = f"{output_dir}/gaussian_splat_3d.html"
    generate_html_gaussian_splat(samples, gaussians, saccades, html_path, fov_h, fov_v)
    
    print("\nGenerating static image...")
    png_path = f"{output_dir}/gaussian_splat_3d.png"
    generate_static_splat_image(gaussians, samples, png_path, fov_h, fov_v)
    
    print("\n=== Done! ===")
    print(f"Interactive HTML: {html_path}")
    print(f"Static PNG: {png_path}")


if __name__ == "__main__":
    main()
