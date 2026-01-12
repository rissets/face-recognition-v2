"""
Advanced Passive Liveness Detection System
==========================================
Menggunakan multiple metode untuk deteksi liveness tanpa interaksi pengguna:
1. Texture Analysis - analisis tekstur kulit vs foto/layar
2. Eye Blink Detection - deteksi kedipan mata natural
3. Micro-movement Analysis - analisis gerakan mikro wajah
4. Light Reflection Analysis - analisis pantulan cahaya
5. Spoofing Artifact Detection - deteksi artifak spoofing

Models yang digunakan:
- InsightFace: Face detection & liveness
- MediaPipe: Face mesh & landmarks
- YOLOv8: Object detection untuk deteksi layar/kertas
- Custom attention mechanism untuk multi-modal fusion
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import time
import math

# Import untuk deep learning
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("Warning: InsightFace tidak tersedia")
    INSIGHTFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLO tidak tersedia. Install dengan: pip install ultralytics")
    YOLO_AVAILABLE = False


class AttentionMechanism:
    """Multi-head attention untuk fusion dari berbagai fitur liveness"""
    
    def __init__(self, n_features=5, n_heads=3):
        self.n_features = n_features
        self.n_heads = n_heads
        self.weights = np.ones(n_features) / n_features
        self.history = deque(maxlen=30)
        
    def compute_attention_weights(self, features_dict):
        """
        Compute attention weights berdasarkan reliability masing-masing feature
        
        Args:
            features_dict: Dictionary berisi confidence scores dari setiap metode
        """
        # Extract confidence scores
        confidences = []
        feature_names = ['texture', 'blink', 'movement', 'reflection', 'spoofing']
        
        for name in feature_names:
            if name in features_dict:
                confidences.append(features_dict[name]['confidence'])
            else:
                confidences.append(0.0)
        
        confidences = np.array(confidences)
        
        # Softmax untuk normalisasi
        exp_conf = np.exp(confidences - np.max(confidences))
        attention_weights = exp_conf / exp_conf.sum()
        
        # Smooth weights dengan exponential moving average
        self.weights = 0.7 * self.weights + 0.3 * attention_weights
        
        return self.weights
    
    def fuse_scores(self, scores_dict, features_dict):
        """
        Fuse multiple liveness scores dengan attention weights
        
        Args:
            scores_dict: Dictionary berisi scores dari setiap metode
            features_dict: Dictionary berisi confidence dari setiap metode
        """
        attention_weights = self.compute_attention_weights(features_dict)
        
        # Weighted sum
        feature_names = ['texture', 'blink', 'movement', 'reflection', 'spoofing']
        final_score = 0.0
        
        for i, name in enumerate(feature_names):
            if name in scores_dict:
                final_score += attention_weights[i] * scores_dict[name]
        
        self.history.append(final_score)
        
        # Temporal smoothing
        if len(self.history) > 5:
            final_score = np.mean(list(self.history)[-5:])
        
        return final_score, attention_weights


class TextureAnalyzer:
    """Analisis tekstur untuk membedakan kulit asli vs foto/layar"""
    
    def __init__(self):
        self.history = deque(maxlen=30)
        
    def compute_lbp(self, image, points=8, radius=1):
        """Local Binary Pattern untuk analisis tekstur"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                
                for k in range(points):
                    angle = 2 * np.pi * k / points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if 0 <= x < h and 0 <= y < w:
                        if image[x, y] >= center:
                            code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def analyze_frequency(self, image):
        """Analisis frekuensi untuk deteksi pola moiré dan pixelation"""
        # FFT untuk analisis frekuensi
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency energy (foto/layar memiliki pola berbeda)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Bagi ke ring zones
        mask_low = np.zeros((h, w))
        mask_mid = np.zeros((h, w))
        mask_high = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                if dist < min(h, w) * 0.1:
                    mask_low[i, j] = 1
                elif dist < min(h, w) * 0.3:
                    mask_mid[i, j] = 1
                else:
                    mask_high[i, j] = 1
        
        energy_low = np.sum(magnitude_spectrum * mask_low)
        energy_mid = np.sum(magnitude_spectrum * mask_mid)
        energy_high = np.sum(magnitude_spectrum * mask_high)
        
        total_energy = energy_low + energy_mid + energy_high + 1e-10
        
        # Real face: lebih banyak di mid frequency
        # Foto/layar: lebih banyak di high frequency (artifak)
        mid_ratio = energy_mid / total_energy
        high_ratio = energy_high / total_energy
        
        return mid_ratio, high_ratio
    
    def analyze(self, face_roi):
        """
        Analisis lengkap tekstur wajah
        
        Returns:
            score: 0-1, higher = more likely real
            confidence: seberapa yakin dengan hasil
        """
        if face_roi is None or face_roi.size == 0:
            return 0.5, 0.0
        
        # Convert ke grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Resize untuk konsistensi
        gray = cv2.resize(gray, (128, 128))
        
        # 1. LBP Analysis
        lbp = self.compute_lbp(gray)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-10)
        
        # Entropy - real skin has more varied texture
        lbp_entropy = entropy(lbp_hist + 1e-10)
        texture_score = min(lbp_entropy / 6.0, 1.0)  # Normalize
        
        # 2. Frequency Analysis
        mid_ratio, high_ratio = self.analyze_frequency(gray)
        
        # Real face: mid_ratio high, high_ratio low
        # Foto/layar: mid_ratio low, high_ratio high
        freq_score = mid_ratio * (1 - high_ratio)
        
        # 3. Edge Analysis - foto memiliki edges yang lebih tajam
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Real face: moderate edges (0.05-0.15)
        # Foto: sharp edges (>0.2)
        if edge_density < 0.05:
            edge_score = edge_density / 0.05
        elif edge_density < 0.15:
            edge_score = 1.0
        else:
            edge_score = max(0, 1.0 - (edge_density - 0.15) / 0.15)
        
        # 4. Variance analysis
        variance = np.var(gray)
        variance_score = min(variance / 1000.0, 1.0)
        
        # Combine scores
        final_score = (
            texture_score * 0.35 +
            freq_score * 0.35 +
            edge_score * 0.20 +
            variance_score * 0.10
        )
        
        # Confidence based on consistency
        scores = [texture_score, freq_score, edge_score, variance_score]
        confidence = 1.0 - np.std(scores)
        
        self.history.append(final_score)
        
        # Temporal smoothing
        if len(self.history) > 3:
            final_score = np.mean(list(self.history)[-3:])
        
        return final_score, confidence


class EyeBlinkDetector:
    """Deteksi kedipan mata natural"""
    
    def __init__(self):
        self.EAR_THRESHOLD = 0.21
        self.CONSECUTIVE_FRAMES = 2
        self.blink_counter = 0
        self.frame_counter = 0
        self.eye_state = 'open'
        self.blink_history = deque(maxlen=100)  # Track blinks over time
        self.ear_history = deque(maxlen=30)
        self.last_blink_time = time.time()
        
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Compute vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def analyze_blink_pattern(self):
        """Analisis pola kedipan untuk mendeteksi naturalness"""
        if len(self.blink_history) < 5:
            return 0.5, 0.3
        
        recent_blinks = list(self.blink_history)[-30:]  # Last 30 blinks
        
        # 1. Blink frequency (normal: 15-20 per minute)
        time_span = recent_blinks[-1] - recent_blinks[0] if len(recent_blinks) > 1 else 1
        blinks_per_minute = len(recent_blinks) / (time_span / 60.0 + 1e-6)
        
        # Score based on normal range
        if 10 <= blinks_per_minute <= 25:
            freq_score = 1.0
        elif 5 <= blinks_per_minute <= 35:
            freq_score = 0.7
        else:
            freq_score = 0.3
        
        # 2. Blink interval variability (natural blinks vary)
        if len(recent_blinks) > 2:
            intervals = np.diff(recent_blinks)
            interval_std = np.std(intervals)
            # Natural: some variability (0.5-2.0 seconds std)
            if 0.5 <= interval_std <= 2.0:
                variability_score = 1.0
            elif interval_std < 0.3:  # Too regular (fake)
                variability_score = 0.3
            else:
                variability_score = 0.6
        else:
            variability_score = 0.5
        
        # 3. EAR pattern during blinks
        if len(self.ear_history) > 10:
            ear_array = np.array(list(self.ear_history))
            ear_variance = np.var(ear_array)
            # Natural: moderate variance as eyes blink
            variance_score = min(ear_variance / 0.01, 1.0)
        else:
            variance_score = 0.5
        
        # Combine scores
        final_score = (
            freq_score * 0.4 +
            variability_score * 0.4 +
            variance_score * 0.2
        )
        
        # Confidence
        confidence = 0.8 if len(recent_blinks) >= 3 else 0.3
        
        return final_score, confidence
    
    def detect(self, left_eye, right_eye):
        """
        Detect blinks dan analisis pattern
        
        Args:
            left_eye: array of 6 landmarks for left eye
            right_eye: array of 6 landmarks for right eye
            
        Returns:
            score: liveness score based on blink pattern
            confidence: confidence in the score
        """
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        self.frame_counter += 1
        
        # Detect blink
        if avg_ear < self.EAR_THRESHOLD:
            if self.eye_state == 'open':
                self.eye_state = 'closing'
        else:
            if self.eye_state == 'closing':
                # Blink completed
                current_time = time.time()
                self.blink_counter += 1
                self.blink_history.append(current_time)
                self.last_blink_time = current_time
                self.eye_state = 'open'
        
        # Analyze pattern
        score, confidence = self.analyze_blink_pattern()
        
        # Penalty jika tidak ada blink dalam waktu lama
        time_since_last_blink = time.time() - self.last_blink_time
        if time_since_last_blink > 10.0:  # No blink for 10 seconds
            score *= 0.5
        
        return score, confidence, self.blink_counter


class MicroMovementAnalyzer:
    """Analisis gerakan mikro pada wajah"""
    
    def __init__(self):
        self.landmark_history = deque(maxlen=30)
        self.movement_scores = deque(maxlen=30)
        
    def compute_optical_flow(self, prev_gray, curr_gray, landmarks):
        """Compute optical flow pada region wajah"""
        if prev_gray is None or curr_gray is None:
            return 0.0, 0.0
        
        # Convert landmarks to points for optical flow
        points = np.array(landmarks, dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        try:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, points, None, **lk_params
            )
            
            if new_points is not None and status is not None:
                # Keep only good points
                good_old = points[status == 1]
                good_new = new_points[status == 1]
                
                # Calculate movement
                movements = np.linalg.norm(good_new - good_old, axis=1)
                avg_movement = np.mean(movements)
                std_movement = np.std(movements)
                
                return avg_movement, std_movement
        except Exception as e:
            pass
        
        return 0.0, 0.0
    
    def analyze_subtle_motion(self, landmarks):
        """Analisis gerakan halus antar frame"""
        if len(landmarks) == 0:
            return 0.5, 0.0
        
        self.landmark_history.append(landmarks)
        
        if len(self.landmark_history) < 3:
            return 0.5, 0.3
        
        # Calculate displacement between frames
        recent_landmarks = list(self.landmark_history)[-10:]
        displacements = []
        
        for i in range(1, len(recent_landmarks)):
            prev = np.array(recent_landmarks[i-1])
            curr = np.array(recent_landmarks[i])
            
            # Calculate per-landmark displacement
            disp = np.linalg.norm(curr - prev, axis=1)
            displacements.append(np.mean(disp))
        
        if len(displacements) == 0:
            return 0.5, 0.3
        
        avg_displacement = np.mean(displacements)
        std_displacement = np.std(displacements)
        
        # Real face: small but present movement (0.5-3.0 pixels)
        # Photo: almost no movement (<0.3 pixels)
        # Video replay: might have larger movements (>5 pixels)
        
        if 0.5 <= avg_displacement <= 3.0:
            motion_score = 1.0
        elif avg_displacement < 0.3:
            motion_score = 0.2  # Too static (photo)
        elif avg_displacement > 5.0:
            motion_score = 0.4  # Too much movement (video attack)
        else:
            motion_score = 0.7
        
        # Variability is good (natural micro-movements vary)
        if std_displacement > 0.3:
            variability_score = 1.0
        else:
            variability_score = 0.5
        
        final_score = motion_score * 0.7 + variability_score * 0.3
        confidence = 0.8 if len(displacements) >= 5 else 0.4
        
        self.movement_scores.append(final_score)
        
        # Temporal smoothing
        if len(self.movement_scores) > 3:
            final_score = np.mean(list(self.movement_scores)[-3:])
        
        return final_score, confidence


class LightReflectionAnalyzer:
    """Analisis pantulan cahaya untuk deteksi 3D vs 2D"""
    
    def __init__(self):
        self.reflection_history = deque(maxlen=30)
        
    def detect_specular_highlights(self, face_roi, landmarks):
        """Deteksi specular highlights pada mata dan hidung"""
        if face_roi is None or len(face_roi) == 0:
            return 0.5, 0.0
        
        h, w = face_roi.shape[:2]
        
        # Convert to HSV for better highlight detection
        if len(face_roi.shape) == 3:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
        else:
            v_channel = face_roi
        
        # Detect bright spots (specular highlights)
        _, highlights = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Count highlights
        n_highlights = cv2.countNonZero(highlights)
        highlight_ratio = n_highlights / (h * w)
        
        # Real face: some highlights (0.001-0.01)
        # Photo: fewer highlights
        # Screen: different pattern (moiré, uniform glow)
        
        if 0.001 <= highlight_ratio <= 0.02:
            highlight_score = 1.0
        elif highlight_ratio < 0.0005:
            highlight_score = 0.3  # Too few (photo)
        else:
            highlight_score = 0.5  # Too many (screen)
        
        return highlight_score
    
    def analyze_3d_structure(self, face_roi):
        """Analisis struktur 3D dari shading"""
        if face_roi is None or len(face_roi) == 0:
            return 0.5, 0.0
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Real face: smooth gradients due to 3D curvature
        # Photo: sharper, less natural gradients
        
        # Smoothness of gradients
        grad_variance = np.var(magnitude)
        smoothness_score = 1.0 - min(grad_variance / 10000.0, 1.0)
        
        return smoothness_score
    
    def analyze(self, face_roi, landmarks):
        """
        Analisis lengkap refleksi cahaya
        
        Returns:
            score: liveness score
            confidence: confidence level
        """
        if face_roi is None or face_roi.size == 0:
            return 0.5, 0.0
        
        # 1. Specular highlights
        highlight_score = self.detect_specular_highlights(face_roi, landmarks)
        
        # 2. 3D structure from shading
        structure_score = self.analyze_3d_structure(face_roi)
        
        # Combine
        final_score = highlight_score * 0.6 + structure_score * 0.4
        
        self.reflection_history.append(final_score)
        
        # Temporal smoothing
        if len(self.reflection_history) > 3:
            final_score = np.mean(list(self.reflection_history)[-3:])
        
        confidence = 0.7
        
        return final_score, confidence


class SpoofingArtifactDetector:
    """Deteksi artifak dari serangan spoofing"""
    
    def __init__(self):
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Load YOLOv8 untuk deteksi objek (phone, tablet, paper)
                self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
            except:
                print("Warning: Tidak bisa load YOLO model")
        
        self.artifact_history = deque(maxlen=30)
    
    def detect_screen_edges(self, frame):
        """Deteksi pinggiran layar/kertas"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines untuk deteksi garis lurus (pinggiran layar/kertas)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 1.0  # No suspicious edges
        
        # Count long straight lines
        long_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > frame.shape[1] * 0.3:  # Long lines (>30% of width)
                long_lines += 1
        
        # Many long lines suggest screen edges
        if long_lines > 4:
            return 0.3
        elif long_lines > 2:
            return 0.6
        else:
            return 1.0
    
    def detect_moire_pattern(self, face_roi):
        """Deteksi pola moiré dari layar"""
        if face_roi is None or len(face_roi) == 0:
            return 1.0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # FFT untuk deteksi pola periodik
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Look for strong periodic patterns (moiré)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Mask out DC component
        magnitude[center_h-5:center_h+5, center_w-5:center_w+5] = 0
        
        # Check for strong peaks (periodic pattern)
        max_magnitude = np.max(magnitude)
        mean_magnitude = np.mean(magnitude)
        
        # Ratio of max to mean
        ratio = max_magnitude / (mean_magnitude + 1e-6)
        
        # High ratio indicates strong periodic pattern (moiré)
        if ratio > 100:
            return 0.2
        elif ratio > 50:
            return 0.5
        else:
            return 1.0
    
    def detect_objects(self, frame):
        """Deteksi objek mencurigakan (phone, tablet, paper)"""
        if self.yolo_model is None:
            return 1.0
        
        try:
            results = self.yolo_model(frame, verbose=False)
            
            # Check for detected objects
            suspicious_classes = ['cell phone', 'laptop', 'tv', 'book']
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in suspicious_classes and confidence > 0.5:
                        return 0.2  # Suspicious object detected
            
            return 1.0
        except Exception as e:
            return 1.0
    
    def analyze(self, frame, face_roi):
        """
        Analisis lengkap untuk deteksi artifak spoofing
        
        Returns:
            score: liveness score (1.0 = real, 0.0 = spoof)
            confidence: confidence level
        """
        if frame is None or frame.size == 0:
            return 0.5, 0.0
        
        # 1. Screen edges detection
        edge_score = self.detect_screen_edges(frame)
        
        # 2. Moiré pattern detection
        moire_score = 1.0
        if face_roi is not None and face_roi.size > 0:
            moire_score = self.detect_moire_pattern(face_roi)
        
        # 3. Object detection
        object_score = self.detect_objects(frame)
        
        # Combine scores
        final_score = (
            edge_score * 0.4 +
            moire_score * 0.3 +
            object_score * 0.3
        )
        
        self.artifact_history.append(final_score)
        
        # Temporal smoothing
        if len(self.artifact_history) > 3:
            final_score = np.mean(list(self.artifact_history)[-3:])
        
        confidence = 0.8
        
        return final_score, confidence


class PassiveLivenessDetector:
    """
    Main class untuk Passive Liveness Detection
    Menggabungkan semua metode dengan attention mechanism
    """
    
    def __init__(self):
        print("Initializing Passive Liveness Detector...")
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize InsightFace
        self.insightface_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                self.insightface_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=providers
                )
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                print("✓ InsightFace initialized")
            except Exception as e:
                print(f"Warning: InsightFace initialization failed: {e}")
        
        # Initialize analyzers
        self.texture_analyzer = TextureAnalyzer()
        self.blink_detector = EyeBlinkDetector()
        self.movement_analyzer = MicroMovementAnalyzer()
        self.reflection_analyzer = LightReflectionAnalyzer()
        self.spoofing_detector = SpoofingArtifactDetector()
        
        # Attention mechanism
        self.attention = AttentionMechanism()
        
        # History for temporal consistency
        self.final_scores = deque(maxlen=30)
        self.prev_gray = None
        
        print("✓ All components initialized successfully!")
    
    def extract_eye_landmarks(self, face_landmarks, image_shape):
        """Extract eye landmarks dari MediaPipe"""
        h, w = image_shape[:2]
        
        # Left eye indices (MediaPipe)
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Right eye indices
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        left_eye = []
        right_eye = []
        
        for idx in LEFT_EYE:
            landmark = face_landmarks.landmark[idx]
            left_eye.append([landmark.x * w, landmark.y * h])
        
        for idx in RIGHT_EYE:
            landmark = face_landmarks.landmark[idx]
            right_eye.append([landmark.x * w, landmark.y * h])
        
        return np.array(left_eye), np.array(right_eye)
    
    def extract_face_roi(self, frame, face_landmarks):
        """Extract face region of interest"""
        h, w = frame.shape[:2]
        
        # Get bounding box from landmarks
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    
    def detect(self, frame):
        """
        Main detection method
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            is_live: Boolean indicating if face is live
            final_score: Combined liveness score (0-1)
            details: Dictionary with detailed scores
        """
        if frame is None or frame.size == 0:
            return False, 0.0, {}
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, {"error": "No face detected"}
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract face ROI
        face_roi, bbox = self.extract_face_roi(frame, face_landmarks)
        
        # Extract all landmarks as array
        all_landmarks = []
        h, w = frame.shape[:2]
        for lm in face_landmarks.landmark:
            all_landmarks.append([lm.x * w, lm.y * h])
        all_landmarks = np.array(all_landmarks)
        
        # Extract eye landmarks
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
        
        # === Run all analyzers ===
        scores = {}
        features = {}
        
        # 1. Texture Analysis
        texture_score, texture_conf = self.texture_analyzer.analyze(face_roi)
        scores['texture'] = texture_score
        features['texture'] = {'confidence': texture_conf}
        
        # 2. Eye Blink Detection
        blink_score, blink_conf, blink_count = self.blink_detector.detect(left_eye, right_eye)
        scores['blink'] = blink_score
        features['blink'] = {'confidence': blink_conf, 'count': blink_count}
        
        # 3. Micro-movement Analysis
        movement_score, movement_conf = self.movement_analyzer.analyze_subtle_motion(all_landmarks)
        scores['movement'] = movement_score
        features['movement'] = {'confidence': movement_conf}
        
        # 4. Light Reflection Analysis
        reflection_score, reflection_conf = self.reflection_analyzer.analyze(face_roi, all_landmarks)
        scores['reflection'] = reflection_score
        features['reflection'] = {'confidence': reflection_conf}
        
        # 5. Spoofing Artifact Detection
        spoofing_score, spoofing_conf = self.spoofing_detector.analyze(frame, face_roi)
        scores['spoofing'] = spoofing_score
        features['spoofing'] = {'confidence': spoofing_conf}
        
        # === Fuse with attention mechanism ===
        final_score, attention_weights = self.attention.fuse_scores(scores, features)
        
        # Store for temporal consistency
        self.final_scores.append(final_score)
        
        # Decision threshold
        LIVENESS_THRESHOLD = 0.55
        is_live = final_score > LIVENESS_THRESHOLD
        
        # Prepare detailed results
        details = {
            'final_score': final_score,
            'is_live': is_live,
            'scores': scores,
            'attention_weights': {
                'texture': attention_weights[0],
                'blink': attention_weights[1],
                'movement': attention_weights[2],
                'reflection': attention_weights[3],
                'spoofing': attention_weights[4]
            },
            'bbox': bbox,
            'blink_count': blink_count
        }
        
        # Update previous frame
        self.prev_gray = gray_frame.copy()
        
        return is_live, final_score, details
    
    def visualize_results(self, frame, details):
        """Visualize detection results on frame"""
        if 'bbox' not in details:
            return frame
        
        x_min, y_min, x_max, y_max = details['bbox']
        
        # Choose color based on liveness
        color = (0, 255, 0) if details['is_live'] else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = f"{'LIVE' if details['is_live'] else 'SPOOF'}: {details['final_score']:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw individual scores
        y_offset = y_min + 20
        for name, score in details['scores'].items():
            weight = details['attention_weights'][name]
            text = f"{name}: {score:.2f} (w:{weight:.2f})"
            cv2.putText(frame, text, (x_min, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        
        # Draw blink count
        blink_text = f"Blinks: {details['blink_count']}"
        cv2.putText(frame, blink_text, (x_min, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def main():
    """Demo application"""
    print("=" * 60)
    print("Advanced Passive Liveness Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = PassiveLivenessDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam")
        return
    
    print("\nInstruksi:")
    print("- Hadapkan wajah ke kamera")
    print("- Sistem akan menganalisis secara otomatis")
    print("- Tekan 'q' untuk keluar")
    print("- Tekan 's' untuk screenshot")
    print("\n")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Gagal membaca frame")
            break
        
        frame_count += 1
        
        # Detect liveness
        is_live, score, details = detector.detect(frame)
        
        # Visualize
        output_frame = detector.visualize_results(frame.copy(), details)
        
        # FPS counter
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
            
        # Display FPS
        cv2.putText(output_frame, f"FPS: {fps:.1f}" if frame_count > 30 else "FPS: ...",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Passive Liveness Detection', output_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"liveness_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nProgram selesai!")


if __name__ == "__main__":
    main()
