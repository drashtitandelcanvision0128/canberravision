"""
Parking Dashboard Web Application
Flask-based web dashboard for real-time parking monitoring with API endpoints
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
from pathlib import Path
import logging
from typing import Dict, List, Any

# Import parking modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modules.real_time_parking import ParkingDashboard, ParkingAlert, SystemStatus
from modules.parking_detection import ParkingDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global parking dashboard instance
parking_dashboard = None
dashboard_thread = None

class ParkingWebAPI:
    """Web API for parking system"""
    
    def __init__(self):
        self.dashboard = None
        self.is_initialized = False
        
    def initialize(self, config_path: str = None):
        """Initialize the parking dashboard"""
        try:
            self.dashboard = ParkingDashboard(config_path)
            
            # Set up alert callback for WebSocket updates
            def on_alert(alert: ParkingAlert):
                # This will be handled by WebSocket in production
                logger.info(f"Web Alert: {alert.message}")
                
            def on_update(results):
                # This will be handled by WebSocket in production
                pass
                
            self.dashboard.add_alert_callback(on_alert)
            self.dashboard.add_update_callback(on_update)
            
            self.is_initialized = True
            logger.info("Parking Web API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Web API: {e}")
            return False
            
    def start_monitoring(self, camera_configs: Dict[str, str]):
        """Start parking monitoring"""
        if not self.is_initialized:
            return False
            
        return self.dashboard.start_monitoring(camera_configs)
        
    def stop_monitoring(self):
        """Stop parking monitoring"""
        if self.dashboard:
            self.dashboard.stop_monitoring()

# Initialize API
parking_api = ParkingWebAPI()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('parking_dashboard.html')

@app.route('/api/status')
def get_system_status():
    """Get current system status"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        status = parking_api.dashboard.get_system_status()
        return jsonify({
            'success': True,
            'data': {
                'total_zones': status.total_zones,
                'active_zones': status.active_zones,
                'total_cameras': status.total_cameras,
                'active_cameras': status.active_cameras,
                'total_spots': status.total_spots,
                'occupied_spots': status.occupied_spots,
                'empty_spots': status.empty_spots,
                'overall_occupancy_rate': status.overall_occupancy_rate,
                'system_uptime': status.system_uptime,
                'last_update': status.last_update,
                'alerts_count': status.alerts_count
            }
        })
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/zones')
def get_zones_status():
    """Get detailed zone status"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        results = parking_api.dashboard.parking_system.get_current_results()
        
        zones_data = {}
        for zone_id, zone_result in results.items():
            zones_data[zone_id] = {
                'zone_id': zone_result.zone_id,
                'zone_name': zone_result.zone_name,
                'total_spots': zone_result.total_spots,
                'occupied_spots': zone_result.occupied_spots,
                'empty_spots': zone_result.empty_spots,
                'occupancy_rate': zone_result.occupancy_rate,
                'timestamp': zone_result.timestamp,
                'spot_details': [
                    {
                        'spot_id': spot.spot_id,
                        'status': spot.status,
                        'confidence': round(spot.confidence, 3),
                        'vehicle_type': spot.vehicle_type,
                        'bounding_box': spot.bounding_box
                    }
                    for spot in zone_result.spot_details
                ]
            }
            
        return jsonify({
            'success': True,
            'data': zones_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Zones API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/zone/<zone_id>')
def get_zone_details(zone_id):
    """Get specific zone details"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        results = parking_api.dashboard.parking_system.get_current_results()
        
        if zone_id not in results:
            return jsonify({'error': 'Zone not found'}), 404
            
        zone_result = results[zone_id]
        
        zone_data = {
            'zone_id': zone_result.zone_id,
            'zone_name': zone_result.zone_name,
            'total_spots': zone_result.total_spots,
            'occupied_spots': zone_result.occupied_spots,
            'empty_spots': zone_result.empty_spots,
            'occupancy_rate': zone_result.occupancy_rate,
            'timestamp': zone_result.timestamp,
            'spot_details': [
                {
                    'spot_id': spot.spot_id,
                    'status': spot.status,
                    'confidence': round(spot.confidence, 3),
                    'vehicle_type': spot.vehicle_type,
                    'bounding_box': spot.bounding_box,
                    'camera_id': spot.camera_id
                }
                for spot in zone_result.spot_details
            ]
        }
        
        return jsonify({
            'success': True,
            'data': zone_data
        })
        
    except Exception as e:
        logger.error(f"Zone details API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get active alerts"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        alerts = parking_api.dashboard.get_active_alerts()
        
        alerts_data = [
            {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'zone_id': alert.zone_id,
                'message': alert.message,
                'severity': alert.severity,
                'timestamp': alert.timestamp,
                'resolved': alert.resolved,
                'resolved_at': alert.resolved_at
            }
            for alert in alerts
        ]
        
        return jsonify({
            'success': True,
            'data': alerts_data,
            'count': len(alerts_data)
        })
        
    except Exception as e:
        logger.error(f"Alerts API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve an alert"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        success = parking_api.dashboard.resolve_alert(alert_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Alert {alert_id} resolved successfully'
            })
        else:
            return jsonify({'error': 'Alert not found or already resolved'}), 404
            
    except Exception as e:
        logger.error(f"Resolve alert API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends')
def get_trends():
    """Get occupancy trends"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        hours = request.args.get('hours', 24, type=int)
        trends = parking_api.dashboard.get_occupancy_trends(hours)
        
        return jsonify({
            'success': True,
            'data': trends
        })
        
    except Exception as e:
        logger.error(f"Trends API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def export_data():
    """Export parking data"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        format_type = request.args.get('format', 'json')
        hours = request.args.get('hours', 24, type=int)
        
        data = parking_api.dashboard.export_data(format_type, hours)
        
        if format_type.lower() == 'json':
            return jsonify({
                'success': True,
                'data': json.loads(data)
            })
        else:
            return data, 200, {
                'Content-Type': 'application/octet-stream',
                'Content-Disposition': f'attachment; filename=parking_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format_type}'
            }
            
    except Exception as e:
        logger.error(f"Export API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/snapshot', methods=['POST'])
def save_snapshot():
    """Save current parking snapshot"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        output_dir = request.json.get('output_dir', 'parking_detections') if request.json else 'parking_detections'
        snapshot_path = parking_api.dashboard.save_snapshot(output_dir)
        
        if snapshot_path:
            return jsonify({
                'success': True,
                'message': 'Snapshot saved successfully',
                'path': snapshot_path
            })
        else:
            return jsonify({'error': 'Failed to save snapshot'}), 500
            
    except Exception as e:
        logger.error(f"Snapshot API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def get_config():
    """Get system configuration"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        config = parking_api.dashboard.parking_system.detector.config
        
        return jsonify({
            'success': True,
            'data': {
                'zones': config.get('zones', {}),
                'detection_config': config.get('detection_config', {}),
                'vehicle_types': config.get('vehicle_types', []),
                'training_config': config.get('training_config', {})
            }
        })
        
    except Exception as e:
        logger.error(f"Config API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cameras')
def get_cameras():
    """Get camera status"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        cameras = {}
        for camera_id, cap in parking_api.dashboard.parking_system.camera_connections.items():
            cameras[camera_id] = {
                'camera_id': camera_id,
                'status': 'active' if cap.isOpened() else 'inactive',
                'last_update': parking_api.dashboard.last_camera_update.get(camera_id, 0)
            }
            
        return jsonify({
            'success': True,
            'data': cameras
        })
        
    except Exception as e:
        logger.error(f"Cameras API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start parking monitoring"""
    try:
        if not parking_api.is_initialized:
            return jsonify({'error': 'System not initialized'}), 503
            
        camera_configs = request.json.get('cameras', {}) if request.json else {}
        
        if not camera_configs:
            # Default camera configuration
            camera_configs = {
                'cam_01': 0,  # Default webcam
            }
            
        success = parking_api.start_monitoring(camera_configs)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Monitoring started successfully'
            })
        else:
            return jsonify({'error': 'Failed to start monitoring'}), 500
            
    except Exception as e:
        logger.error(f"Start monitoring API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop parking monitoring"""
    try:
        parking_api.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Monitoring stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Stop monitoring API error: {e}")
        return jsonify({'error': str(e)}), 500

def create_templates():
    """Create HTML templates for the dashboard"""
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Main dashboard template
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26 Parking Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--color-background-primary); color: var(--color-text-primary); }
        .header { background: var(--color-background-secondary); color: var(--color-text-primary); padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { font-size: 1.8rem; font-weight: 300; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .stat-card { background: var(--color-background-secondary); padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .stat-card h3 { color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; }
        .stat-card .value { font-size: 2rem; font-weight: bold; color: var(--color-text-primary); }
        .stat-card .change { font-size: 0.8rem; margin-top: 0.5rem; }
        .positive { color: var(--color-success); }
        .negative { color: var(--color-error); }
        .zones-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .zone-card { background: var(--color-background-secondary); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; }
        .zone-header { background: var(--color-background-tertiary); color: var(--color-text-primary); padding: 1rem; }
        .zone-content { padding: 1.5rem; }
        .occupancy-bar { background: var(--color-border-primary); height: 20px; border-radius: 10px; overflow: hidden; margin: 1rem 0; }
        .occupancy-fill { background: var(--color-accent-primary); height: 100%; transition: width 0.3s ease; }
        .spots-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(40px, 1fr)); gap: 4px; margin-top: 1rem; }
        .spot { width: 40px; height: 40px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: white; cursor: pointer; }
        .spot.empty { background: var(--color-success); }
        .spot.occupied { background: var(--color-error); }
        .alerts-section { background: var(--color-background-secondary); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 2rem; }
        .alert { padding: 1rem; margin: 0.5rem 0; border-radius: 4px; border-left: 4px solid; }
        .alert.critical { border-left-color: var(--color-error); background: var(--color-error-background); }
        .alert.high { border-left-color: var(--color-warning); background: var(--color-warning-background); }
        .alert.medium { border-left-color: var(--color-accent-primary); background: var(--color-accent-background); }
        .alert.low { border-left-color: var(--color-success); background: var(--color-success-background); }
        .controls { display: flex; gap: 1rem; margin-bottom: 2rem; }
        .btn { padding: 0.75rem 1.5rem; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; transition: background 0.3s; }
        .btn-primary { background: var(--color-accent-primary); color: white; }
        .btn-success { background: var(--color-success); color: white; }
        .btn-danger { background: var(--color-error); color: white; }
        .btn:hover { opacity: 0.9; }
        .chart-container { background: var(--color-background-secondary); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 2rem; }
        .loading { text-align: center; padding: 2rem; color: var(--color-text-secondary); }
        .error { background: var(--color-error); color: white; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 YOLO26 Parking Dashboard</h1>
    </div>
    
    <div class="container">
        <div class="controls">
            <button class="btn btn-success" onclick="startMonitoring()">Start Monitoring</button>
            <button class="btn btn-danger" onclick="stopMonitoring()">Stop Monitoring</button>
            <button class="btn btn-primary" onclick="exportData()">Export Data</button>
            <button class="btn btn-primary" onclick="saveSnapshot()">Save Snapshot</button>
        </div>
        
        <div id="error-message" class="error" style="display: none;"></div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Spots</h3>
                <div class="value" id="total-spots">-</div>
                <div class="change" id="spots-change">Loading...</div>
            </div>
            <div class="stat-card">
                <h3>Occupied</h3>
                <div class="value" id="occupied-spots">-</div>
                <div class="change positive" id="occupied-change">Loading...</div>
            </div>
            <div class="stat-card">
                <h3>Empty</h3>
                <div class="value" id="empty-spots">-</div>
                <div class="change" id="empty-change">Loading...</div>
            </div>
            <div class="stat-card">
                <h3>Occupancy Rate</h3>
                <div class="value" id="occupancy-rate">-</div>
                <div class="change" id="occupancy-change">Loading...</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Occupancy Trends (Last 24 Hours)</h3>
            <canvas id="trends-chart" width="400" height="100"></canvas>
        </div>
        
        <div class="zones-grid" id="zones-grid">
            <div class="loading">Loading zone data...</div>
        </div>
        
        <div class="alerts-section">
            <h3>Active Alerts</h3>
            <div id="alerts-container">
                <div class="loading">Loading alerts...</div>
            </div>
        </div>
    </div>

    <script>
        let trendsChart = null;
        let updateInterval = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            loadInitialData();
            startAutoUpdate();
        });
        
        function initializeChart() {
            const ctx = document.getElementById('trends-chart').getContext('2d');
            trendsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Occupancy Rate (%)',
                        data: [],
                        borderColor: 'var(--color-accent-primary)',
                        backgroundColor: 'var(--color-accent-background)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        async function loadInitialData() {
            try {
                await Promise.all([
                    loadSystemStatus(),
                    loadZones(),
                    loadAlerts(),
                    loadTrends()
                ]);
            } catch (error) {
                showError('Failed to load initial data: ' + error.message);
            }
        }
        
        async function loadSystemStatus() {
            try {
                const response = await axios.get('/api/status');
                const data = response.data.data;
                
                document.getElementById('total-spots').textContent = data.total_spots;
                document.getElementById('occupied-spots').textContent = data.occupied_spots;
                document.getElementById('empty-spots').textContent = data.empty_spots;
                document.getElementById('occupancy-rate').textContent = data.overall_occupancy_rate + '%';
                
                document.getElementById('spots-change').textContent = `${data.active_zones}/${data.total_zones} zones active`;
                document.getElementById('occupied-change').textContent = `${data.active_cameras}/${data.total_cameras} cameras`;
                document.getElementById('empty-change').textContent = `System uptime: ${data.system_uptime}`;
                document.getElementById('occupancy-change').textContent = `${data.alerts_count} alerts`;
                
            } catch (error) {
                console.error('Status load error:', error);
            }
        }
        
        async function loadZones() {
            try {
                const response = await axios.get('/api/zones');
                const zones = response.data.data;
                
                const zonesGrid = document.getElementById('zones-grid');
                zonesGrid.innerHTML = '';
                
                for (const [zoneId, zoneData] of Object.entries(zones)) {
                    const zoneCard = createZoneCard(zoneData);
                    zonesGrid.appendChild(zoneCard);
                }
                
            } catch (error) {
                console.error('Zones load error:', error);
            }
        }
        
        function createZoneCard(zoneData) {
            const card = document.createElement('div');
            card.className = 'zone-card';
            
            const occupancyPercentage = zoneData.occupancy_rate;
            
            card.innerHTML = `
                <div class="zone-header">
                    <h3>${zoneData.zone_name} (${zoneData.zone_id})</h3>
                </div>
                <div class="zone-content">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <span><strong>${zoneData.occupied_spots}</strong> occupied</span>
                        <span><strong>${zoneData.empty_spots}</strong> empty</span>
                        <span><strong>${occupancyPercentage}%</strong> full</span>
                    </div>
                    <div class="occupancy-bar">
                        <div class="occupancy-fill" style="width: ${occupancyPercentage}%"></div>
                    </div>
                    <div class="spots-grid">
                        ${zoneData.spot_details.map(spot => 
                            `<div class="spot ${spot.status.toLowerCase()}" title="${spot.spot_id}: ${spot.status}">${spot.spot_id.split('-')[1]}</div>`
                        ).join('')}
                    </div>
                </div>
            `;
            
            return card;
        }
        
        async function loadAlerts() {
            try {
                const response = await axios.get('/api/alerts');
                const alerts = response.data.data;
                
                const alertsContainer = document.getElementById('alerts-container');
                
                if (alerts.length === 0) {
                    alertsContainer.innerHTML = '<div style="color: var(--color-success);">✅ No active alerts</div>';
                } else {
                    alertsContainer.innerHTML = alerts.map(alert => `
                        <div class="alert ${alert.severity.toLowerCase()}">
                            <strong>${alert.alert_type}</strong> - ${alert.message}
                            <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
                
            } catch (error) {
                console.error('Alerts load error:', error);
            }
        }
        
        async function loadTrends() {
            try {
                const response = await axios.get('/api/trends?hours=24');
                const trends = response.data.data;
                
                if (trends.timestamps && trends.occupancy_rates) {
                    trendsChart.data.labels = trends.timestamps.map(ts => new Date(ts).toLocaleTimeString());
                    trendsChart.data.datasets[0].data = trends.occupancy_rates;
                    trendsChart.update();
                }
                
            } catch (error) {
                console.error('Trends load error:', error);
            }
        }
        
        function startAutoUpdate() {
            updateInterval = setInterval(async () => {
                await Promise.all([
                    loadSystemStatus(),
                    loadZones(),
                    loadAlerts()
                ]);
            }, 10000); // Update every 10 seconds
        }
        
        async function startMonitoring() {
            try {
                const response = await axios.post('/api/start', {
                    cameras: {
                        'cam_01': 0  // Use default webcam
                    }
                });
                
                if (response.data.success) {
                    showMessage('Monitoring started successfully!', 'success');
                } else {
                    showError('Failed to start monitoring');
                }
            } catch (error) {
                showError('Start monitoring error: ' + error.message);
            }
        }
        
        async function stopMonitoring() {
            try {
                const response = await axios.post('/api/stop');
                
                if (response.data.success) {
                    showMessage('Monitoring stopped successfully!', 'success');
                } else {
                    showError('Failed to stop monitoring');
                }
            } catch (error) {
                showError('Stop monitoring error: ' + error.message);
            }
        }
        
        async function exportData() {
            try {
                const response = await axios.get('/api/export?format=json');
                
                const blob = new Blob([JSON.stringify(response.data.data, null, 2)], {
                    type: 'application/json'
                });
                
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `parking_data_${new Date().toISOString().split('T')[0]}.json`;
                a.click();
                window.URL.revokeObjectURL(url);
                
                showMessage('Data exported successfully!', 'success');
            } catch (error) {
                showError('Export error: ' + error.message);
            }
        }
        
        async function saveSnapshot() {
            try {
                const response = await axios.post('/api/snapshot');
                
                if (response.data.success) {
                    showMessage('Snapshot saved successfully!', 'success');
                } else {
                    showError('Failed to save snapshot');
                }
            } catch (error) {
                showError('Snapshot error: ' + error.message);
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error-message');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
        
        function showMessage(message, type) {
            // Simple message display (could be enhanced with notifications)
            console.log(`${type}: ${message}`);
        }
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        });
    </script>
</body>
</html>
    """
    
    with open(templates_dir / "parking_dashboard.html", 'w') as f:
        f.write(dashboard_html)
        
    logger.info(f"HTML templates created in {templates_dir}")

def initialize_system():
    """Initialize the parking system"""
    global parking_api
    
    try:
        # Create templates
        create_templates()
        
        # Initialize API
        success = parking_api.initialize()
        
        if success:
            logger.info("Parking system initialized successfully")
            return True
        else:
            logger.error("Failed to initialize parking system")
            return False
            
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        return False

if __name__ == '__main__':
    print("=== YOLO26 Parking Dashboard Web Application ===")
    
    # Initialize system
    if initialize_system():
        print("✅ System initialized successfully")
        print("🌐 Starting web server on http://localhost:5000")
        print("📊 Dashboard available at: http://localhost:5000")
        print("🔗 API endpoints available at: http://localhost:5000/api/")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web server...")
            parking_api.stop_monitoring()
            print("✅ Server stopped")
        except Exception as e:
            print(f"❌ Server error: {e}")
            parking_api.stop_monitoring()
    else:
        print("❌ Failed to initialize system")
        exit(1)
