"""
Non-blocking HTML-based Interactive 3D Gaze Viewer
=================================================

Creates an interactive HTML visualization using Plotly.
Open in any web browser - no terminal blocking.

Usage:
    python gaze_viewer_html.py
    # Then open the generated HTML file in a browser
"""

import json
import csv
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path


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


def compute_statistics(samples: List[GazeSample], saccades: List[Saccade]) -> Dict:
    """Compute comprehensive statistics"""
    detected = [s for s in samples if s.confidence == 'detected']
    
    if not detected:
        return {}
    
    azimuths = np.array([s.azimuth for s in detected])
    elevations = np.array([s.elevation for s in detected])
    times = np.array([s.timestamp for s in detected])
    
    vel_az = np.diff(azimuths) * 29.0
    vel_el = np.diff(elevations) * 29.0
    vel_mag = np.sqrt(vel_az**2 + vel_el**2)
    
    dir_counts = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
    for s in saccades:
        dir_counts[s.direction] = dir_counts.get(s.direction, 0) + 1
    
    total_time = times[-1] if len(times) > 0 else 1
    saccade_time = sum(s.duration_sec for s in saccades)
    
    return {
        'n_frames': len(samples),
        'n_detected': len(detected),
        'n_interpolated': len(samples) - len(detected),
        'detection_rate': 100 * len(detected) / len(samples),
        
        'azimuth': {
            'mean': float(np.mean(azimuths)),
            'std': float(np.std(azimuths)),
            'min': float(np.min(azimuths)),
            'max': float(np.max(azimuths)),
        },
        
        'elevation': {
            'mean': float(np.mean(elevations)),
            'std': float(np.std(elevations)),
            'min': float(np.min(elevations)),
            'max': float(np.max(elevations)),
        },
        
        'velocity': {
            'mean': float(np.mean(vel_mag)),
            'std': float(np.std(vel_mag)),
            'max': float(np.max(vel_mag)),
        },
        
        'saccades': {
            'count': len(saccades),
            'rate_per_min': len(saccades) / (total_time / 60),
            'mean_amplitude': float(np.mean([s.amplitude for s in saccades])) if saccades else 0,
            'mean_velocity': float(np.mean([s.peak_velocity for s in saccades])) if saccades else 0,
            'mean_duration_ms': float(np.mean([s.duration_sec * 1000 for s in saccades])) if saccades else 0,
            'total_time_pct': 100 * saccade_time / total_time,
            'directions': dir_counts,
        },
        
        'fixation': {
            'time_sec': total_time - saccade_time,
            'time_pct': 100 * (total_time - saccade_time) / total_time,
        },
        
        'total_time': total_time,
    }


def generate_html(samples: List[GazeSample], saccades: List[Saccade], 
                  stats: Dict, output_path: str):
    """Generate interactive HTML visualization"""
    
    detected = [s for s in samples if s.confidence == 'detected']
    
    az = [s.azimuth for s in detected]
    el = [s.elevation for s in detected]
    times = [s.timestamp for s in detected]
    
    saccade_az = []
    saccade_el = []
    saccade_times = []
    for s in saccades:
        if s.start_frame < len(samples) and s.end_frame < len(samples):
            saccade_az.append(samples[s.start_frame].azimuth)
            saccade_el.append(samples[s.start_frame].elevation)
            saccade_times.append(samples[s.start_frame].timestamp)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pig Eye Gaze Tracking - Interactive 3D Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            padding: 20px 0;
            color: #4ecca3;
            text-shadow: 0 0 10px rgba(78, 204, 163, 0.5);
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #4ecca3;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .main-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 16px;
            margin-bottom: 10px;
            color: #4ecca3;
        }}
        .direction-chart {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }}
        .direction-bar {{
            text-align: center;
            padding: 10px;
        }}
        .direction-bar .bar {{
            width: 60px;
            height: {stats.get('saccades', {}).get('count', 1) * 2}px;
            background: linear-gradient(to top, #4ecca3, #45b393);
            border-radius: 5px 5px 0 0;
            margin: 0 auto;
            min-height: 10px;
        }}
        .direction-bar .label {{
            margin-top: 5px;
            font-size: 14px;
        }}
        .direction-bar .count {{
            font-size: 18px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐷 Pig Eye Gaze Tracking - Interactive 3D Analysis</h1>
        
        <div class="stats-bar">
            <div class="stat-card">
                <div class="value">{stats.get('n_detected', 0)}</div>
                <div class="label">Detected Frames</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('detection_rate', 0):.1f}%</div>
                <div class="label">Detection Rate</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('saccades', {}).get('count', 0)}</div>
                <div class="label">Saccades</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('saccades', {}).get('rate_per_min', 0):.1f}/min</div>
                <div class="label">Saccade Rate</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('saccades', {}).get('mean_amplitude', 0):.1f}°</div>
                <div class="label">Mean Amplitude</div>
            </div>
            <div class="stat-card">
                <div class="value">{stats.get('saccades', {}).get('mean_velocity', 0):.1f}°/s</div>
                <div class="label">Peak Velocity</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="chart-container">
                <div class="chart-title">3D Gaze Splat (drag to rotate, scroll to zoom)</div>
                <div id="gaze-3d" style="width:100%;height:500px;"></div>
            </div>
            
            <div>
                <div class="chart-container">
                    <div class="chart-title">Statistics</div>
                    <table style="width:100%;font-size:14px;">
                        <tr><td><b>Azimuth Mean</b></td><td>{stats.get('azimuth', {}).get('mean', 0):.2f}° ± {stats.get('azimuth', {}).get('std', 0):.2f}°</td></tr>
                        <tr><td><b>Azimuth Range</b></td><td>[{stats.get('azimuth', {}).get('min', 0):.1f}°, {stats.get('azimuth', {}).get('max', 0):.1f}°]</td></tr>
                        <tr><td><b>Elevation Mean</b></td><td>{stats.get('elevation', {}).get('mean', 0):.2f}° ± {stats.get('elevation', {}).get('std', 0):.2f}°</td></tr>
                        <tr><td><b>Elevation Range</b></td><td>[{stats.get('elevation', {}).get('min', 0):.1f}°, {stats.get('elevation', {}).get('max', 0):.1f}°]</td></tr>
                        <tr><td><b>Fixation Time</b></td><td>{stats.get('fixation', {}).get('time_sec', 0):.1f}s ({stats.get('fixation', {}).get('time_pct', 0):.1f}%)</td></tr>
                        <tr><td><b>Saccade Time</b></td><td>{stats.get('saccades', {}).get('total_time_pct', 0):.1f}%</td></tr>
                    </table>
                    
                    <div class="direction-chart">
                        <div class="direction-bar">
                            <div class="bar" style="background: linear-gradient(to top, #00ff00, #00cc00);"></div>
                            <div class="count">{stats.get('saccades', {}).get('directions', {}).get('right', 0)}</div>
                            <div class="label">Right</div>
                        </div>
                        <div class="direction-bar">
                            <div class="bar" style="background: linear-gradient(to top, #0088ff, #0055cc);"></div>
                            <div class="count">{stats.get('saccades', {}).get('directions', {}).get('left', 0)}</div>
                            <div class="label">Left</div>
                        </div>
                        <div class="direction-bar">
                            <div class="bar" style="background: linear-gradient(to top, #ff4444, #cc0000);"></div>
                            <div class="count">{stats.get('saccades', {}).get('directions', {}).get('up', 0)}</div>
                            <div class="label">Up</div>
                        </div>
                        <div class="direction-bar">
                            <div class="bar" style="background: linear-gradient(to top, #ffaa00, #cc8800);"></div>
                            <div class="count">{stats.get('saccades', {}).get('directions', {}).get('down', 0)}</div>
                            <div class="label">Down</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Top View: Azimuth vs Elevation (FOV = 120° × 80°)</div>
            <div id="gaze-topview" style="width:100%;height:400px;"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Gaze Over Time</div>
            <div id="gaze-timeseries" style="width:100%;height:300px;"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Velocity Profile</div>
            <div id="gaze-velocity" style="width:100%;height:250px;"></div>
        </div>
    </div>
    
    <script>
        // 3D Gaze Splat
        const gaze3dData = [{{
            type: 'scatter3d',
            mode: 'markers',
            x: {json.dumps(az)},
            y: {json.dumps(times)},
            z: {json.dumps(el)},
            marker: {{
                size: 3,
                color: {json.dumps(times)},
                colorscale: 'Viridis',
                opacity: 0.8
            }},
            name: 'Gaze Points'
        }}];
        
        if ({len(saccade_az)} > 0) {{
            gaze3dData.push({{
                type: 'scatter3d',
                mode: 'markers',
                x: {json.dumps(saccade_az)},
                y: {json.dumps(saccade_times)},
                z: {json.dumps(saccade_el)},
                marker: {{
                    size: 10,
                    color: 'yellow',
                    symbol: 'diamond'
                }},
                name: 'Saccades'
            }});
        }}
        
        Plotly.newPlot('gaze-3d', gaze3dData, {{
            scene: {{
                xaxis: {{title: 'Azimuth (°)', gridcolor: '#333', backgroundcolor: 'rgba(0,0,0,0.5)'}},
                yaxis: {{title: 'Time (s)', gridcolor: '#333', backgroundcolor: 'rgba(0,0,0,0.5)'}},
                zaxis: {{title: 'Elevation (°)', gridcolor: '#333', backgroundcolor: 'rgba(0,0,0,0.5)'}},
                bgcolor: 'rgba(20,20,40,1)'
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            margin: {{l: 0, r: 0, t: 30, b: 0}}
        }});
        
        // Top View
        const topviewData = [{{
            type: 'scatter',
            mode: 'markers',
            x: {json.dumps(az)},
            y: {json.dumps(el)},
            marker: {{
                size: 4,
                color: {json.dumps(times)},
                colorscale: 'Viridis',
                opacity: 0.7
            }}
        }}];
        
        const fovCenterX = {stats.get('azimuth', {}).get('mean', 0)};
        const fovCenterY = {stats.get('elevation', {}).get('mean', 0)};
        topviewData.push({{
            type: 'scatter',
            mode: 'markers',
            x: [fovCenterX],
            y: [fovCenterY],
            marker: {{
                size: 20,
                color: 'red',
                symbol: 'square',
                line: {{width: 3, color: 'white'}}
            }},
            name: 'FOV Center'
        }});
        
        Plotly.newPlot('gaze-topview', topviewData, {{
            xaxis: {{title: 'Azimuth (°)', gridcolor: '#333', zeroline: false}},
            yaxis: {{title: 'Elevation (°)', gridcolor: '#333', zeroline: false}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            shapes: [{{
                type: 'rect',
                x0: fovCenterX - 60,
                y0: fovCenterY - 40,
                x1: fovCenterX + 60,
                y1: fovCenterY + 40,
                line: {{color: 'rgba(78,204,163,0.5)', width: 2, dash: 'dash'}}
            }}]
        }});
        
        // Time Series
        const timeseriesData = [
            {{
                type: 'scatter',
                mode: 'lines',
                x: {json.dumps(times)},
                y: {json.dumps(az)},
                name: 'Azimuth',
                line: {{color: '#4ecca3', width: 1}}
            }},
            {{
                type: 'scatter',
                mode: 'lines',
                x: {json.dumps(times)},
                y: {json.dumps(el)},
                name: 'Elevation',
                line: {{color: '#ff6b6b', width: 1}}
            }}
        ];
        
        Plotly.newPlot('gaze-timeseries', timeseriesData, {{
            xaxis: {{title: 'Time (s)', gridcolor: '#333'}},
            yaxis: {{title: 'Angle (°)', gridcolor: '#333'}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            legend: {{orientation: 'h', x: 0.5, xanchor: 'center'}}
        }});
        
        // Velocity (precomputed in Python)
        const velMag = {json.dumps(list(np.sqrt(np.diff(az)**2 + np.diff(el)**2) * 29.0))};
        const velTimes = {json.dumps(list(times[1:]))};
        
        Plotly.newPlot('gaze-velocity', [{{
            type: 'scatter',
            mode: 'lines',
            x: velTimes,
            y: velMag,
            fill: 'tozeroy',
            line: {{color: '#4ecca3', width: 1}},
            name: 'Velocity'
        }}], {{
            xaxis: {{title: 'Time (s)', gridcolor: '#333'}},
            yaxis: {{title: 'Velocity (°/s)', gridcolor: '#333'}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{color: '#eee'}},
            shapes: [{{
                type: 'line',
                y0: 0,
                y1: 1,
                yref: 'paper',
                x0: 20,
                x1: 20,
                line: {{color: 'red', width: 2, dash: 'dot'}}
            }}],
            annotations: [{{
                x: 20,
                y: 1,
                yref: 'paper',
                text: 'Saccade Threshold',
                showarrow: false,
                xanchor: 'left'
            }}]
        }});
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Interactive HTML saved to: {output_path}")


def main():
    csv_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/gaze_3d_data.csv"
    output_dir = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data"
    
    print("Loading data...")
    samples, saccades = load_data(csv_path)
    print(f"Loaded {len(samples)} samples, {len(saccades)} saccades")
    
    print("Computing statistics...")
    stats = compute_statistics(samples, saccades)
    
    print("Generating interactive HTML...")
    output_path = f"{output_dir}/gaze_viewer_interactive.html"
    generate_html(samples, saccades, stats, output_path)
    
    print("\nDone! Open the HTML file in a browser to view the interactive visualization.")
    print(f"File: {output_path}")


if __name__ == "__main__":
    main()
