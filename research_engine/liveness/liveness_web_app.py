#!/usr/bin/env python3
"""
Web-based Real-time Liveness Detection
======================================

Flask web application for real-time liveness detection with webcam.
Provides a user-friendly web interface for liveness testing.
"""

import cv2
import json
import time
import base64
import logging
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import numpy as np
from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
detector = None
camera = None
detection_active = False
current_session = None
session_lock = threading.Lock()

class LivenessWebApp:
    """Web application for liveness detection"""
    
    def __init__(self):
        self.detector = None
        self.camera = None
        self.detection_active = False
        self.current_frame = None
        self.current_analysis = None
        self.session_results = []
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera"""
        try:
            if self.camera is not None:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Cannot open camera {camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {camera_index} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def initialize_detector(self, config=None):
        """Initialize liveness detector"""
        try:
            if config is None:
                config = create_detector_config(
                    strict_mode=False,
                    enable_challenges=True,
                    min_blinks=2,
                    liveness_threshold=0.7
                )
            
            self.detector = RealtimeLivenessDetector(config)
            logger.info("Liveness detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            return False
    
    def start_detection(self):
        """Start liveness detection"""
        with session_lock:
            if self.detector is None:
                return {'success': False, 'error': 'Detector not initialized'}
            
            if self.detection_active:
                return {'success': False, 'error': 'Detection already active'}
            
            self.detector.start_detection()
            self.detection_active = True
            self.current_session = {
                'start_time': time.time(),
                'session_id': int(time.time() * 1000)
            }
            
            logger.info("Detection session started")
            return {'success': True, 'session_id': self.current_session['session_id']}
    
    def stop_detection(self):
        """Stop liveness detection"""
        with session_lock:
            if not self.detection_active:
                return {'success': False, 'error': 'No active detection'}
            
            result = self.detector.stop_detection()
            self.detection_active = False
            
            # Store session result
            session_data = {
                'session_id': self.current_session['session_id'],
                'start_time': self.current_session['start_time'],
                'end_time': time.time(),
                'duration': time.time() - self.current_session['start_time'],
                'result': {
                    'is_live': result.is_live,
                    'confidence': result.confidence,
                    'score_breakdown': result.score_breakdown,
                    'challenges_passed': result.challenges_passed,
                    'challenges_failed': result.challenges_failed,
                    'frame_analysis': result.frame_analysis
                }
            }
            
            self.session_results.append(session_data)
            self.current_session = None
            
            logger.info(f"Detection session completed: {result.is_live} (confidence: {result.confidence:.3f})")
            return {'success': True, 'result': session_data}
    
    def get_frame(self):
        """Get current camera frame with annotations"""
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        try:
            if self.detector is not None and self.detection_active:
                # Process frame through detector
                annotated_frame, analysis = self.detector.process_frame(frame)
                self.current_frame = annotated_frame
                self.current_analysis = analysis
                
                # Check for automatic completion
                if not self.detector.detection_active and self.detector.final_result is not None:
                    # Session completed automatically
                    self.stop_detection()
                
                return annotated_frame
            else:
                # Just show raw frame with basic info
                self._add_basic_info(frame)
                self.current_frame = frame
                return frame
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def _add_basic_info(self, frame):
        """Add basic information to frame when not detecting"""
        h, w = frame.shape[:2]
        
        # Add title
        cv2.putText(frame, "Real-time Liveness Detection", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status
        cv2.putText(frame, "Ready - Click 'Start Detection' to begin", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_status(self):
        """Get current detection status"""
        if self.detector is None:
            return {'status': 'not_initialized'}
        
        if not self.detection_active:
            return {
                'status': 'ready',
                'sessions_completed': len(self.session_results)
            }
        
        detector_status = self.detector.get_current_status()
        
        return {
            'status': 'active',
            'session_id': self.current_session['session_id'] if self.current_session else None,
            'session_duration': detector_status['session_duration'],
            'frames_processed': detector_status['frames_processed'],
            'total_blinks': detector_status['total_blinks'],
            'current_challenge': detector_status['current_challenge'],
            'challenges_completed': detector_status['challenges_completed'],
            'average_score': detector_status['average_score'],
            'current_analysis': self.current_analysis
        }
    
    def get_session_results(self):
        """Get all session results"""
        return {
            'total_sessions': len(self.session_results),
            'sessions': self.session_results
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera is not None:
            self.camera.release()
        
        if self.detection_active and self.detector is not None:
            self.detector.stop_detection()

# Create global app instance
liveness_app = LivenessWebApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('liveness_detection.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize camera and detector"""
    try:
        data = request.json or {}
        camera_index = data.get('camera_index', 0)
        strict_mode = data.get('strict_mode', False)
        
        # Initialize camera
        if not liveness_app.initialize_camera(camera_index):
            return jsonify({'success': False, 'error': 'Failed to initialize camera'})
        
        # Initialize detector
        config = create_detector_config(
            strict_mode=strict_mode,
            enable_challenges=True,
            min_blinks=3 if strict_mode else 2,
            liveness_threshold=0.8 if strict_mode else 0.7
        )
        
        if not liveness_app.initialize_detector(config):
            return jsonify({'success': False, 'error': 'Failed to initialize detector'})
        
        return jsonify({'success': True, 'message': 'Initialized successfully'})
        
    except Exception as e:
        logger.error(f"Error in initialize: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    """Start liveness detection"""
    try:
        result = liveness_app.start_detection()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    """Stop liveness detection"""
    try:
        result = liveness_app.stop_detection()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def get_status():
    """Get current status"""
    try:
        status = liveness_app.get_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/api/results')
def get_results():
    """Get session results"""
    try:
        results = liveness_app.get_session_results()
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({'error': str(e)})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                frame = liveness_app.get_frame()
                if frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in video feed: {e}")
                break
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current frame as base64"""
    try:
        frame = liveness_app.current_frame
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    'success': True,
                    'image': f'data:image/jpeg;base64,{img_base64}',
                    'timestamp': time.time()
                })
        
        return jsonify({'success': False, 'error': 'No frame available'})
        
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return jsonify({'success': False, 'error': str(e)})

# HTML Template
template_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Liveness Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .video-section {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-container {
            flex: 2;
        }
        .controls-container {
            flex: 1;
        }
        .video-feed {
            width: 100%;
            border: 2px solid #ddd;
            border-radius: 10px;
        }
        .controls {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-panel {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .challenge-panel {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .results-panel {
            background: #f1f8e9;
            padding: 15px;
            border-radius: 8px;
        }
        .button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .button:hover {
            background: #0056b3;
        }
        .button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .button.success {
            background: #28a745;
        }
        .button.danger {
            background: #dc3545;
        }
        .button.warning {
            background: #ffc107;
            color: black;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-ready {
            background: #d4edda;
            color: #155724;
        }
        .status-active {
            background: #fff3cd;
            color: #856404;
        }
        .status-completed {
            background: #d1ecf1;
            color: #0c5460;
        }
        .score-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
            transition: width 0.3s ease;
        }
        .log {
            max-height: 200px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Real-time Liveness Detection</h1>
            <p>Advanced anti-spoofing system with multiple detection techniques</p>
        </div>
        
        <div class="video-section">
            <div class="video-container">
                <img id="videoFeed" class="video-feed" src="/video_feed" alt="Video Feed">
            </div>
            
            <div class="controls-container">
                <div class="controls">
                    <h3>Controls</h3>
                    <button id="initBtn" class="button">Initialize Camera</button>
                    <button id="startBtn" class="button success" disabled>Start Detection</button>
                    <button id="stopBtn" class="button danger" disabled>Stop Detection</button>
                    <br>
                    <label>
                        <input type="checkbox" id="strictMode"> Strict Mode
                    </label>
                </div>
                
                <div class="status-panel">
                    <h3>Status</h3>
                    <div id="statusInfo">
                        <span class="status-badge status-ready">Ready</span>
                        <p id="statusText">Click Initialize to begin</p>
                    </div>
                </div>
                
                <div class="challenge-panel" id="challengePanel" style="display: none;">
                    <h3>Current Challenge</h3>
                    <div id="challengeInfo">
                        <p id="challengeText">No active challenge</p>
                        <div class="score-bar">
                            <div class="score-fill" id="scoreBar" style="width: 0%"></div>
                        </div>
                        <p id="scoreText">Score: 0.00</p>
                    </div>
                </div>
                
                <div class="results-panel" id="resultsPanel" style="display: none;">
                    <h3>Detection Result</h3>
                    <div id="resultInfo"></div>
                </div>
            </div>
        </div>
        
        <div class="log" id="logArea"></div>
    </div>

    <script>
        let statusInterval;
        let detectionActive = false;
        
        // DOM elements
        const initBtn = document.getElementById('initBtn');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const strictMode = document.getElementById('strictMode');
        const statusInfo = document.getElementById('statusInfo');
        const statusText = document.getElementById('statusText');
        const challengePanel = document.getElementById('challengePanel');
        const challengeText = document.getElementById('challengeText');
        const scoreBar = document.getElementById('scoreBar');
        const scoreText = document.getElementById('scoreText');
        const resultsPanel = document.getElementById('resultsPanel');
        const resultInfo = document.getElementById('resultInfo');
        const logArea = document.getElementById('logArea');
        
        // Event listeners
        initBtn.addEventListener('click', initializeSystem);
        startBtn.addEventListener('click', startDetection);
        stopBtn.addEventListener('click', stopDetection);
        
        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            logArea.innerHTML += `[${timestamp}] ${message}<br>`;
            logArea.scrollTop = logArea.scrollHeight;
        }
        
        async function initializeSystem() {
            try {
                initBtn.disabled = true;
                log('Initializing camera and detector...');
                
                const response = await fetch('/api/initialize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        camera_index: 0,
                        strict_mode: strictMode.checked
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    log('✅ System initialized successfully');
                    startBtn.disabled = false;
                    statusText.textContent = 'Ready to start detection';
                    startStatusUpdates();
                } else {
                    log(`❌ Initialization failed: ${result.error}`);
                    initBtn.disabled = false;
                }
                
            } catch (error) {
                log(`❌ Error: ${error.message}`);
                initBtn.disabled = false;
            }
        }
        
        async function startDetection() {
            try {
                startBtn.disabled = true;
                log('Starting liveness detection...');
                
                const response = await fetch('/api/start_detection', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    log(`✅ Detection started (Session ID: ${result.session_id})`);
                    detectionActive = true;
                    stopBtn.disabled = false;
                    challengePanel.style.display = 'block';
                    resultsPanel.style.display = 'none';
                } else {
                    log(`❌ Failed to start detection: ${result.error}`);
                    startBtn.disabled = false;
                }
                
            } catch (error) {
                log(`❌ Error: ${error.message}`);
                startBtn.disabled = false;
            }
        }
        
        async function stopDetection() {
            try {
                stopBtn.disabled = true;
                log('Stopping detection...');
                
                const response = await fetch('/api/stop_detection', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    log('✅ Detection stopped');
                    detectionActive = false;
                    startBtn.disabled = false;
                    challengePanel.style.display = 'none';
                    showResults(result.result);
                } else {
                    log(`❌ Failed to stop detection: ${result.error}`);
                }
                
            } catch (error) {
                log(`❌ Error: ${error.message}`);
            }
        }
        
        function startStatusUpdates() {
            statusInterval = setInterval(updateStatus, 500);
        }
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                if (status.status === 'active') {
                    updateActiveStatus(status);
                } else if (status.status === 'ready') {
                    updateReadyStatus(status);
                }
                
            } catch (error) {
                console.error('Status update error:', error);
            }
        }
        
        function updateActiveStatus(status) {
            statusText.textContent = `Active - Duration: ${status.session_duration.toFixed(1)}s | Blinks: ${status.total_blinks}`;
            
            if (status.current_challenge) {
                challengeText.textContent = `${status.current_challenge.toUpperCase().replace('_', ' ')}`;
            } else {
                challengeText.textContent = 'Processing...';
            }
            
            const score = status.average_score || 0;
            scoreBar.style.width = `${score * 100}%`;
            scoreText.textContent = `Score: ${score.toFixed(2)}`;
            
            // Check if detection completed automatically
            if (!detectionActive && status.status !== 'active') {
                detectionActive = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                challengePanel.style.display = 'none';
            }
        }
        
        function updateReadyStatus(status) {
            statusText.textContent = `Ready - Sessions completed: ${status.sessions_completed}`;
        }
        
        function showResults(result) {
            resultsPanel.style.display = 'block';
            
            const isLive = result.result.is_live;
            const confidence = result.result.confidence;
            
            let resultHtml = `
                <div style="text-align: center; margin-bottom: 15px;">
                    <h2 style="color: ${isLive ? '#28a745' : '#dc3545'};">
                        ${isLive ? '✅ LIVE PERSON' : '❌ FAKE/SPOOF'}
                    </h2>
                    <p>Confidence: ${(confidence * 100).toFixed(1)}%</p>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${confidence * 100}%;"></div>
                    </div>
                </div>
                
                <h4>Score Breakdown:</h4>
                <ul>
            `;
            
            for (const [metric, score] of Object.entries(result.result.score_breakdown)) {
                resultHtml += `<li>${metric}: ${(score * 100).toFixed(1)}%</li>`;
            }
            
            resultHtml += `
                </ul>
                <h4>Challenges:</h4>
                <p>✅ Passed: ${result.result.challenges_passed.join(', ') || 'None'}</p>
                <p>❌ Failed: ${result.result.challenges_failed.join(', ') || 'None'}</p>
                
                <h4>Session Info:</h4>
                <p>Duration: ${result.duration.toFixed(1)}s</p>
                <p>Frames: ${result.result.frame_analysis.total_frames}</p>
                <p>Blinks: ${result.result.frame_analysis.total_blinks}</p>
            `;
            
            resultInfo.innerHTML = resultHtml;
            
            log(`🔍 Detection Result: ${isLive ? 'LIVE' : 'FAKE'} (${(confidence * 100).toFixed(1)}%)`);
        }
    </script>
</body>
</html>
'''

# Create templates directory and file
import os
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(templates_dir, exist_ok=True)

with open(os.path.join(templates_dir, 'liveness_detection.html'), 'w') as f:
    f.write(template_html)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based Real-time Liveness Detection')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting web application at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    finally:
        logger.info("Cleaning up...")
        liveness_app.cleanup()

if __name__ == "__main__":
    main()