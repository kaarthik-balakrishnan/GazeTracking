"""
3D Gaussian Splatting Gaze Visualization (Fixed)
================================================

Visualizes pig eye gaze data using Gaussian Splatting principles.
Each gaze point becomes a 3D Gaussian blob representing FOV coverage.

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
    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float
    alpha: float


@dataclass
class GazeSample:
    frame: int
    timestamp: float
    azimuth: float
    elevation: float
    confidence: str


def load_gaze_data(csv_path: str) -> Tuple[List[GazeSample], List[Dict]]:
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


def create_gaze_gaussians(samples: List[GazeSample], 
                          fov_h: float = 120.0, 
                          fov_v: float = 80.0) -> List[Gaussian3D]:
    gaussians = []
    for s in samples:
        if s.confidence != 'detected':
            continue
        g = Gaussian3D(
            x=s.azimuth,
            y=s.timestamp,
            z=s.elevation,
            sx=fov_h / 6,
            sy=0.1,
            sz=fov_v / 6,
            alpha=0.6
        )
        gaussians.append(g)
    return gaussians


def compute_2d_kde(xs, zs, x_range, z_range, resolution=80):
    """Fast 2D KDE using numpy broadcasting"""
    x_grid = np.linspace(x_range[0], x_range[1], resolution)
    z_grid = np.linspace(z_range[0], z_range[1], resolution)
    
    X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')
    density = np.zeros((resolution, resolution))
    
    # Compute density from all points
    for i in range(len(xs)):
        dx = (X - xs[i]) / 5.0
        dz = (Z - zs[i]) / 5.0
        density += np.exp(-0.5 * (dx*dx + dz*dz))
    
    return density, x_grid, z_grid


def compute_3d_kde_fast(az, times, el, n_samples=100):
    """Fast 3D KDE using subsampling and scipy if available"""
    try:
        from scipy import stats
        # Subsample for speed
        n = min(n_samples, len(az))
        idx = np.linspace(0, len(az)-1, n).astype(int)
        
        data = np.vstack([az[idx], times[idx], el[idx]])
        kde = stats.gaussian_kde(data, bw_method=0.1)
        
        return kde
    except ImportError:
        return None


def generate_html_gaussian_splat(samples: List[GazeSample], 
                                  gaussians: List[Gaussian3D],
                                  saccades: List[Dict],
                                  output_path: str,
                                  fov_h: float = 120.0,
                                  fov_v: float = 80.0):
    """Generate interactive HTML with Gaussian splat visualization"""
    
    detected = [s for s in samples if s.confidence == 'detected']
    
    az = [s.azimuth for s in detected]
    el = [s.elevation for s in detected]
    times = [s.timestamp for s in detected]
    
    az_arr = np.array(az)
    el_arr = np.array(el)
    times_arr = np.array(times)
    
    # Compute 2D density for heatmap
    az_min, az_max = az_arr.min() - 20, az_arr.max() + 20
    el_min, el_max = el_arr.min() - 20, el_arr.max() + 20
    
    print("Computing 2D density...")
    density, az_grid, el_grid = compute_2d_kde(az_arr, el_arr, 
                                               (az_min, az_max), 
                                               (el_min, el_max),
                                               resolution=60)
    
    density_norm = (density / density.max() * 255).astype(np.uint8).tolist()
    
    # Stats
    az_mean = float(az_arr.mean())
    el_mean = float(el_arr.mean())
    
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
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff006e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 28px;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(135deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; margin-top: 5px; }}
        .viz-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{ .viz-grid {{ grid-template-columns: 1fr; }} }}
        .chart-box {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-title {{ font-size: 14px; color: #00d4ff; margin-bottom: 15px; font-weight: 600; }}
        .full-width {{ grid-column: 1 / -1; }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
        .legend-dot {{ width: 16px; height: 16px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Gaussian Splatting - Pig Eye Gaze Visualization</h1>
        <p class="subtitle">Each gaze point rendered as a 3D Gaussian blob (azimuth × time × elevation)</p>
        
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
                <div class="legend-dot" style="background: rgba(0,255,136,0.7);"></div>
                <span>Gaze Points</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgba(123,47,247,0.7);"></div>
                <span>Gaussian Density</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgba(255,215,0,0.9);"></div>
                <span>Saccade Events</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="border: 2px dashed #00d4ff;"></div>
                <span>FOV: {fov_h}° × {fov_v}°</span>
            </div>
        </div>
        
        <div class="viz-grid">
            <div class="chart-box full-width">
                <div class="chart-title">Interactive 3D Gaussian Splat (drag to rotate, scroll to zoom)</div>
                <div id="gaussian-3d" style="width:100%;height:600px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Top View: Gaze Density Heatmap</div>
                <div id="density-heatmap" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Gaze Position Distribution</div>
                <div id="position-scatter" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Gaze Over Time</div>
                <div id="gaze-timeseries" style="width:100%;height:400px;"></div>
            </div>
            
            <div class="chart-box">
                <div class="chart-title">Saccade Direction Distribution</div>
                <div id="saccade-pie" style="width:100%;height:400px;"></div>
            </div>
        </div>
    </div>
    
    <script>
        // 3D Scatter Plot - Gaussian Splat visualization
        const scatterData = [{{
            type: 'scatter3d',
            mode: 'markers',
            x: {json.dumps(az)},
            y: {json.dumps(times)},
            z: {json.dumps(el)},
            marker: {{
                size: 4,
                color: {json.dumps(times)},
                colorscale: 'Viridis',
                opacity: 0.7,
                colorbar: {{
                    title: 'Time (s)',
                    titleside: 'right'
                }}
            }},
            name: 'Gaze Points'
        }}];
        
        Plotly.newPlot('gaussian-3d', scatterData, {{
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
            z: {json.dumps(density_norm)},
            x: {json.dumps(az_grid.tolist())},
            y: {json.dumps(el_grid.tolist())},
            colorscale: [
                [0, 'rgba(0,0,0,0)'],
                [0.2, 'rgba(0,100,255,0.5)'],
                [0.5, 'rgba(123,47,247,0.7)'],
                [0.8, 'rgba(255,100,100,0.9)'],
                [1, 'rgba(255,255,0,1)']
            ],
            showscale: true,
            colorbar: {{title: 'Density'}}
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
        
        // Position Scatter
        const scatter2Data = [{{
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
        
        Plotly.newPlot('position-scatter', scatter2Data, {{
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
        
        // Time Series
        const tsData = [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {json.dumps(times)},
                y: {json.dumps(az)},
                name: 'Azimuth',
                line: {{color: '#00ff88', width: 1.5}},
                fill: 'tozeroy',
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
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Gaussian Splatting visualization saved to: {output_path}")


def main():
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading gaze data...")
    samples, saccades = load_gaze_data(csv_path)
    print(f"Loaded {len(samples)} samples, {len(saccades)} saccades")
    
    print("\nCreating 3D Gaussians...")
    fov_h, fov_v = 120.0, 80.0
    gaussians = create_gaze_gaussians(samples, fov_h, fov_v)
    print(f"Created {len(gaussians)} Gaussians")
    
    print("\nGenerating HTML visualization...")
    html_path = f"{output_dir}/gaussian_splat_3d.html"
    generate_html_gaussian_splat(samples, gaussians, saccades, html_path, fov_h, fov_v)
    
    print("\n=== Done! ===")
    print(f"Open in browser: {html_path}")


if __name__ == "__main__":
    main()
