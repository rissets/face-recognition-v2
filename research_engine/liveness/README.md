# Real-time Liveness Detection System 🔍

Advanced real-time liveness detection system to distinguish between real faces and fake/spoofed faces using multiple anti-spoofing techniques.

## Features ✨

### Core Detection Methods
- **Blink Detection**: Advanced Eye Aspect Ratio (EAR) analysis with adaptive thresholds
- **3D Head Movement**: Real-time head pose estimation and movement tracking
- **Texture Analysis**: Detection of printed photos and screen displays
- **Light Reflection Analysis**: Natural eye reflection patterns
- **Challenge System**: Interactive user challenges (blink, turn head, nod)
- **Multi-modal Scoring**: Combines multiple detection methods for robust results

### Anti-Spoofing Techniques
- **Screen Glare Detection**: Identifies reflections from phone/tablet screens
- **Print Artifact Detection**: Recognizes printing patterns and paper textures
- **Uniform Illumination Detection**: Flags unnaturally even lighting
- **Micro-texture Analysis**: Detects missing skin micro-textures
- **Color Histogram Analysis**: Identifies artificial color distributions
- **Frequency Domain Analysis**: Analyzes image frequency characteristics

### Advanced Features
- **Adaptive Learning**: Adjusts detection parameters based on individual characteristics
- **Quality Assessment**: Ensures optimal conditions for accurate detection
- **Real-time Processing**: ~30 FPS performance with live video streams
- **Comprehensive Logging**: Detailed analysis and debugging information
- **Multiple Interfaces**: Command-line, desktop GUI, and web interface
- **Configurable Parameters**: Strict/normal modes, customizable thresholds

## Installation 🚀

### Requirements
```bash
# Core dependencies
pip install opencv-python mediapipe numpy
pip install faiss-cpu scikit-learn
pip install flask flask-cors  # For web interface
```

### Setup
```bash
# Clone/download the liveness detection files
cd research_engine/liveness/

# Install dependencies
pip install -r requirements.txt  # If requirements file exists

# Verify installation
python realtime_liveness_detector.py --help
```

## Usage 📱

### 1. Command Line Interface

#### Basic Usage
```bash
# Start with default settings
python realtime_liveness_detector.py

# Use strict detection mode
python realtime_liveness_detector.py --strict

# Disable interactive challenges
python realtime_liveness_detector.py --no-challenges

# Use different camera
python realtime_liveness_detector.py --camera 1
```

#### Controls
- **'s'**: Start new detection session
- **'r'**: Reset current session  
- **'q'**: Quit application
- **'h'**: Show help

### 2. Desktop Demo Application
```bash
# Run interactive demo
python liveness_demo.py

# Strict mode demo
python liveness_demo.py --strict

# Debug mode
python liveness_demo.py --debug
```

### 3. Web Interface
```bash
# Start web server
python liveness_web_app.py

# Custom host/port
python liveness_web_app.py --host 0.0.0.0 --port 8080

# Access at http://localhost:5000
```

#### Web Interface Features
- Real-time video feed with annotations
- Interactive controls and status display
- Session management and results history
- Challenge progress tracking
- Detailed score breakdowns

## API Usage 🔧

### Python API
```python
from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config

# Create configuration
config = create_detector_config(
    strict_mode=False,
    enable_challenges=True,
    min_blinks=2,
    liveness_threshold=0.7
)

# Initialize detector
detector = RealtimeLivenessDetector(config)

# Start detection session
detector.start_detection()

# Process video frames
import cv2
cap = cv2.VideoCapture(0)

while detector.detection_active:
    ret, frame = cap.read()
    if ret:
        annotated_frame, analysis = detector.process_frame(frame)
        cv2.imshow('Liveness Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Get final result
result = detector.stop_detection()
print(f"Is Live: {result.is_live}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Challenges Passed: {result.challenges_passed}")
```

### Configuration Options
```python
config = {
    # Blink detection
    'ear_threshold': 0.25,
    'blink_frames_threshold': 3,
    'min_blinks_required': 2,
    'blink_timeout': 10.0,
    
    # Head movement
    'head_movement_threshold': 15.0,
    'movement_timeout': 8.0,
    
    # Texture analysis
    'texture_variance_threshold': 100,
    'edge_density_threshold': 0.1,
    
    # Challenge system
    'enable_challenges': True,
    'challenge_timeout': 15.0,
    
    # Scoring
    'liveness_threshold': 0.7,
    'spoof_penalty': -0.3,
    
    # Quality requirements
    'min_face_size': 100,
    'max_face_size': 400,
    'brightness_range': (50, 200)
}
```

## Detection Process 🎯

### 1. Face Detection & Quality Check
- Detects face using MediaPipe Face Detection
- Validates face size, position, and image quality
- Rejects poor quality frames

### 2. Landmark Analysis
- Extracts 468 facial landmarks using MediaPipe Face Mesh
- Analyzes eye regions for blink detection
- Tracks head pose for movement analysis

### 3. Liveness Analysis
- **Blink Detection**: Calculates Eye Aspect Ratio (EAR) with adaptive thresholds
- **Head Movement**: Estimates 3D pose and tracks movement patterns
- **Texture Analysis**: Analyzes image texture for artificial characteristics
- **Reflection Analysis**: Checks for natural eye reflections
- **Depth Analysis**: Evaluates 3D characteristics of facial features

### 4. Anti-Spoofing Detection
- Scans for screen glare patterns
- Detects printing artifacts and paper textures
- Analyzes illumination uniformity
- Checks for missing micro-textures
- Evaluates color distribution anomalies

### 5. Challenge System (Optional)
- **Blink Challenge**: Requires natural blinking
- **Head Turn**: Left and right head movements
- **Nod Challenge**: Up and down head movement
- **Combined Challenges**: Multiple actions in sequence

### 6. Scoring & Decision
- Combines all analysis results with weighted scoring
- Applies penalties for spoofing indicators
- Makes final live/fake determination with confidence score

## Results & Interpretation 📊

### LivenessResult Object
```python
@dataclass
class LivenessResult:
    is_live: bool              # Final decision
    confidence: float          # Confidence score (0.0-1.0)
    score_breakdown: Dict      # Individual metric scores
    challenges_passed: List    # Completed challenges
    challenges_failed: List    # Failed challenges
    frame_analysis: Dict       # Session statistics
```

### Score Breakdown
- **blink_score**: Blink detection performance (0.0-1.0)
- **quality_score**: Overall frame quality (0.0-1.0)
- **spoof_score**: Anti-spoofing confidence (0.0-1.0)
- **challenge_score**: Challenge completion rate (0.0-1.0)

### Confidence Interpretation
- **0.8-1.0**: Very confident LIVE person
- **0.7-0.8**: Confident LIVE person
- **0.5-0.7**: Moderate confidence
- **0.3-0.5**: Low confidence (review recommended)
- **0.0-0.3**: Likely FAKE/SPOOF

## Testing 🧪

### Automated Test Suite
```bash
# Run comprehensive tests
python test_liveness.py

# Verbose testing
python test_liveness.py --verbose

# Custom output directory
python test_liveness.py --output-dir my_test_results
```

### Test Scenarios
- Normal detection with standard parameters
- Strict mode detection
- Fast/slow blink patterns
- Head movement validation
- Challenge system testing
- Poor lighting conditions
- Multiple faces handling
- Edge cases and error conditions

### Manual Testing Tips
1. **Good Conditions**: Well-lit, face centered, clear view
2. **Challenge Response**: Follow on-screen prompts naturally
3. **Natural Behavior**: Blink and move naturally, don't force
4. **Quality Check**: Ensure camera is focused and stable
5. **Distance**: Maintain 1-3 feet from camera

## Troubleshooting 🔧

### Common Issues

#### "No face detected"
- Ensure good lighting
- Position face in center of frame
- Check camera permissions
- Verify camera is working

#### "Poor image quality"
- Improve lighting conditions
- Clean camera lens
- Reduce camera shake
- Adjust distance to camera

#### "Detection timeout"
- Follow challenge prompts
- Blink naturally several times
- Move head slightly during detection
- Check if challenges are enabled

#### Low confidence scores
- Ensure natural behavior
- Improve lighting conditions
- Keep face clearly visible
- Allow sufficient detection time

### Debug Mode
```bash
# Enable detailed logging
python realtime_liveness_detector.py --debug

# Check log files
tail -f liveness_detection.log
```

### Performance Optimization
- Use recommended camera resolution (640x480)
- Ensure adequate CPU resources
- Close unnecessary applications
- Use good quality webcam

## Advanced Configuration ⚙️

### Custom Detection Parameters
```python
# High security configuration
strict_config = create_detector_config(
    strict_mode=True,
    min_blinks=4,
    liveness_threshold=0.85,
    enable_challenges=True
)

# Fast detection configuration  
fast_config = create_detector_config(
    strict_mode=False,
    min_blinks=1,
    liveness_threshold=0.6,
    enable_challenges=False
)
```

### Integration Examples

#### Web Service Integration
```python
from flask import Flask, request, jsonify
from realtime_liveness_detector import RealtimeLivenessDetector

app = Flask(__name__)
detector = RealtimeLivenessDetector()

@app.route('/verify_liveness', methods=['POST'])
def verify_liveness():
    # Process uploaded image or video
    # Return liveness result
    pass
```

#### Security System Integration
```python
class SecuritySystem:
    def __init__(self):
        self.liveness_detector = RealtimeLivenessDetector()
    
    def authenticate_user(self, video_stream):
        # Integrate with face recognition
        # Add liveness verification
        # Return authentication result
        pass
```

## Contributing 🤝

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
python -m pytest test_liveness.py

# Format code
black realtime_liveness_detector.py

# Lint code
flake8 realtime_liveness_detector.py
```

### Adding New Detection Methods
1. Implement detection function in `RealtimeLivenessDetector`
2. Add to `_analyze_liveness` method
3. Update scoring in `_calculate_frame_score`
4. Add tests in `test_liveness.py`
5. Update documentation

## License 📄

This project is part of the Face Recognition V2 system. See main project license for details.

## Support 📞

For issues and questions:
1. Check troubleshooting section
2. Run test suite to identify problems
3. Enable debug logging for detailed analysis
4. Review configuration parameters

## Changelog 📝

### Version 2.0
- Complete rewrite with advanced anti-spoofing
- Added challenge system
- Improved accuracy and performance
- Web interface support
- Comprehensive test suite

### Version 1.0
- Basic blink detection
- Simple texture analysis
- Command-line interface

---

**Note**: This liveness detection system is designed for security applications. Always test thoroughly in your specific environment and use case before deployment in production systems.