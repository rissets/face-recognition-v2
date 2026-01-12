"""
STRICT Passive Liveness Detection System
=========================================
Versi yang lebih ketat untuk mendeteksi serangan foto/video dari HP/tablet

IMPROVEMENTS:
1. Screen/Display detection yang lebih agresif
2. Multi-scale texture analysis untuk mendeteksi pixelation
3. Depth estimation dari single image
4. Face quality assessment
5. Stricter thresholds dan multi-stage verification
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist
from scipy.stats import entropy
import time

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ScreenDisplayDetector:
    """Deteksi khusus untuk layar HP/tablet dengan metode agresif"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=30)
        
    def detect_screen_glare(self, frame):
        """Deteksi glare/refleksi khas dari layar"""
        # Convert ke grayscale dan HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Deteksi area yang terlalu terang (glare dari layar)
        _, very_bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_pixels = cv2.countNonZero(very_bright)
        glare_ratio = glare_pixels / (frame.shape[0] * frame.shape[1])
        
        # Layar HP sering punya glare yang tidak natural
        if glare_ratio > 0.05:  # >5% area sangat terang
            return 0.2  # Highly suspicious
        elif glare_ratio > 0.02:
            return 0.5
        return 1.0
    
    def detect_rectangular_boundary(self, frame):
        """Deteksi batas persegi panjang (pinggiran layar HP) - BALANCED"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection dengan parameter ketat
        edges = cv2.Canny(gray, 30, 100)
        
        # Hough lines - BALANCED: butuh lines yang lebih panjang
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,  # Was 80, now 100
                                minLineLength=200, maxLineGap=20)  # Was 150, now 200
        
        if lines is None:
            return 1.0
        
        # Analisis garis-garis
        horizontal_lines = []
        vertical_lines = []
        long_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Count very long lines (layar HP edges)
            if length > frame.shape[1] * 0.4:  # >40% of width
                long_lines += 1
            
            # Horizontal (close to 0 or 180)
            if angle < 10 or angle > 170:
                horizontal_lines.append((line, length))
            # Vertical (close to 90)
            elif 80 < angle < 100:
                vertical_lines.append((line, length))
        
        # BALANCED: Butuh banyak lines yang PANJANG
        # Layar HP: 4+ horizontal AND 4+ vertical lines yang panjang
        if long_lines >= 4 and len(horizontal_lines) >= 4 and len(vertical_lines) >= 4:
            return 0.1  # Very suspicious (likely phone screen)
        elif long_lines >= 3 and len(horizontal_lines) >= 3 and len(vertical_lines) >= 3:
            return 0.3  # Suspicious
        elif len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            return 0.7  # Slightly suspicious
        
        return 1.0
    
    def detect_pixel_grid(self, face_roi):
        """Deteksi pixel grid dari layar (subpixel pattern) - BALANCED"""
        if face_roi is None or face_roi.size == 0:
            return 1.0
        
        # Resize untuk melihat detail pixel
        if face_roi.shape[0] < 200:
            scale = 200 / face_roi.shape[0]
            face_roi = cv2.resize(face_roi, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_LINEAR)
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        
        # FFT untuk deteksi pola periodik
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Mask DC component
        h, w = magnitude.shape
        magnitude[h//2-10:h//2+10, w//2-10:w//2+10] = 0
        
        # Cari peaks yang kuat (indikasi pola periodik)
        # BALANCED: threshold lebih tinggi untuk menghindari false positive
        threshold = np.mean(magnitude) + 4 * np.std(magnitude)  # Was 3, now 4
        peaks = magnitude > threshold
        peak_count = np.sum(peaks)
        
        # BALANCED: threshold lebih tinggi
        # Layar HP memiliki pola periodik yang SANGAT kuat (>150 peaks)
        # Webcam normal biasanya <80 peaks
        if peak_count > 150:
            return 0.1  # Very strong periodic pattern (definitely screen)
        elif peak_count > 100:
            return 0.3  # Strong pattern (likely screen)
        elif peak_count > 80:
            return 0.6  # Moderate pattern (suspicious)
        else:
            return 0.9  # Normal (not a screen)
        
        return 1.0
    
    def detect_color_banding(self, frame):
        """Deteksi color banding khas dari layar"""
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Hitung gradien halus
        grad_b = np.abs(cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3))
        grad_g = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3))
        grad_r = np.abs(cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3))
        
        # Layar digital sering punya banding (gradient yang tidak smooth)
        # Hitung histogram dari gradient
        hist_b, _ = np.histogram(grad_b, bins=50, range=(0, 255))
        hist_g, _ = np.histogram(grad_g, bins=50, range=(0, 255))
        hist_r, _ = np.histogram(grad_r, bins=50, range=(0, 255))
        
        # Normalize
        hist_b = hist_b / (hist_b.sum() + 1e-10)
        hist_g = hist_g / (hist_g.sum() + 1e-10)
        hist_r = hist_r / (hist_r.sum() + 1e-10)
        
        # Entropy - layar memiliki entropy lebih rendah
        ent_b = entropy(hist_b + 1e-10)
        ent_g = entropy(hist_g + 1e-10)
        ent_r = entropy(hist_r + 1e-10)
        
        avg_entropy = (ent_b + ent_g + ent_r) / 3
        
        # Real face: higher entropy (4.5+)
        # Screen: lower entropy (<3.5)
        if avg_entropy < 3.0:
            return 0.2
        elif avg_entropy < 4.0:
            return 0.5
        else:
            return 1.0
    
    def analyze(self, frame, face_roi):
        """Analisis lengkap untuk deteksi layar - BALANCED"""
        if frame is None or frame.size == 0:
            return 0.5, 0.5
        
        scores = []
        
        # 1. Screen glare
        glare_score = self.detect_screen_glare(frame)
        scores.append(glare_score)
        
        # 2. Rectangular boundary
        boundary_score = self.detect_rectangular_boundary(frame)
        scores.append(boundary_score)
        
        # 3. Pixel grid
        if face_roi is not None and face_roi.size > 0:
            grid_score = self.detect_pixel_grid(face_roi)
            scores.append(grid_score)
        
        # 4. Color banding
        banding_score = self.detect_color_banding(frame)
        scores.append(banding_score)
        
        # BALANCED: Butuh MULTIPLE indicators untuk reject
        # Count how many scores are very low
        very_low = sum(1 for s in scores if s < 0.3)
        low = sum(1 for s in scores if s < 0.5)
        
        if very_low >= 3:  # 3+ tests indicate screen
            final_score = min(scores)
        elif very_low >= 2:  # 2 tests very suspicious
            final_score = np.mean(scores) * 0.7  # Penalize
        elif low >= 3:  # 3 tests somewhat suspicious
            final_score = np.mean(scores) * 0.85
        else:
            # Weighted average (grid and boundary more important)
            final_score = (
                glare_score * 0.15 +
                boundary_score * 0.35 +
                grid_score * 0.35 +
                banding_score * 0.15
            )
        
        self.detection_history.append(final_score)
        
        # Temporal consistency - use median (more robust than min)
        if len(self.detection_history) > 5:
            final_score = np.median(list(self.detection_history)[-5:])
        
        confidence = 0.85
        return final_score, confidence


class DepthEstimator:
    """Estimasi depth untuk membedakan 2D vs 3D"""
    
    def __init__(self):
        self.prev_landmarks = None
        
    def estimate_from_shading(self, face_roi):
        """Estimasi depth dari shading (shape from shading)"""
        if face_roi is None or face_roi.size == 0:
            return 0.5, 0.5
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        gray = cv2.resize(gray, (128, 128))
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Laplacian untuk curvature
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        
        # Real 3D face: smooth gradients dengan curvature
        # 2D photo: sharper gradients, flatter
        
        # Smoothness of gradients (variance)
        grad_smoothness = 1.0 / (np.std(grad_mag) + 1)
        
        # Curvature strength
        curvature_strength = np.mean(np.abs(laplacian))
        
        # Real face: high curvature, smooth gradients
        # Photo: low curvature, sharp gradients
        
        depth_score = min(curvature_strength / 10.0, 1.0) * min(grad_smoothness * 10, 1.0)
        
        confidence = 0.7
        return depth_score, confidence
    
    def estimate_from_motion(self, landmarks, prev_landmarks):
        """Estimasi depth dari motion parallax"""
        if landmarks is None or prev_landmarks is None:
            return 0.5, 0.3
        
        if len(landmarks) < 10 or len(prev_landmarks) < 10:
            return 0.5, 0.3
        
        # Compute movement
        movement = landmarks - prev_landmarks
        
        # Dalam 3D face, gerakan berbeda untuk bagian yang berbeda depth
        # Hidung (lebih dekat) bergerak lebih dari telinga (lebih jauh)
        
        # Ambil landmark nose tip dan face outline
        nose_movement = np.linalg.norm(movement[1])  # Simplified
        outline_movement = np.mean([np.linalg.norm(movement[i]) for i in [10, 152, 234, 454]])
        
        # 3D: nose moves more than outline
        # 2D: sama semua
        
        if nose_movement > 0.1:
            ratio = outline_movement / (nose_movement + 1e-6)
            if ratio < 0.7:  # Nose moves more
                depth_score = 1.0
            elif ratio < 0.9:
                depth_score = 0.7
            else:
                depth_score = 0.3  # Too similar (2D)
        else:
            depth_score = 0.5
        
        confidence = 0.6
        return depth_score, confidence


class StrictBlinkDetector:
    """Deteksi kedipan yang lebih ketat"""
    
    def __init__(self):
        self.EAR_THRESHOLD = 0.22  # Slightly higher threshold
        self.blink_counter = 0
        self.frame_counter = 0
        self.eye_state = 'open'
        self.blink_history = deque(maxlen=100)
        self.ear_history = deque(maxlen=30)
        self.last_blink_time = time.time()
        self.no_blink_penalty_start = time.time()
        
    def eye_aspect_ratio(self, eye_landmarks):
        """Calculate EAR"""
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def detect(self, left_eye, right_eye):
        """Deteksi dengan aturan yang lebih strict"""
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
                current_time = time.time()
                self.blink_counter += 1
                self.blink_history.append(current_time)
                self.last_blink_time = current_time
                self.no_blink_penalty_start = current_time
                self.eye_state = 'open'
        
        # STRICT: Require at least 1 blink in first 3 seconds
        time_elapsed = time.time() - self.no_blink_penalty_start
        
        if time_elapsed > 3.0 and self.blink_counter == 0:
            return 0.0, 0.9, self.blink_counter  # No blink = SPOOF
        
        # Require regular blinking
        time_since_last = time.time() - self.last_blink_time
        if time_since_last > 8.0:  # No blink for 8 seconds
            return 0.1, 0.9, self.blink_counter
        
        # Analyze pattern
        if len(self.blink_history) >= 2:
            recent = list(self.blink_history)[-10:]
            time_span = recent[-1] - recent[0] if len(recent) > 1 else 1
            bpm = len(recent) / (time_span / 60.0 + 1e-6)
            
            # Normal: 10-25 bpm
            if 10 <= bpm <= 25:
                score = 1.0
            elif 5 <= bpm <= 35:
                score = 0.7
            else:
                score = 0.3
        else:
            score = 0.4
        
        confidence = 0.9
        return score, confidence, self.blink_counter


class StrictMovementAnalyzer:
    """Analisis gerakan yang lebih ketat"""
    
    def __init__(self):
        self.landmark_history = deque(maxlen=30)
        self.movement_scores = deque(maxlen=30)
        
    def analyze(self, landmarks):
        """Analisis dengan threshold yang lebih strict"""
        if len(landmarks) == 0:
            return 0.2, 0.8  # No landmarks = suspicious
        
        self.landmark_history.append(landmarks)
        
        if len(self.landmark_history) < 5:
            return 0.3, 0.5  # Not enough data
        
        recent = list(self.landmark_history)[-10:]
        displacements = []
        
        for i in range(1, len(recent)):
            prev = np.array(recent[i-1])
            curr = np.array(recent[i])
            disp = np.linalg.norm(curr - prev, axis=1)
            displacements.append(np.mean(disp))
        
        if len(displacements) == 0:
            return 0.2, 0.8
        
        avg_disp = np.mean(displacements)
        std_disp = np.std(displacements)
        
        # STRICT thresholds
        # Real: 0.8-2.5 pixels dengan variability
        # Photo: <0.5 pixels
        # Video: might have movement tapi often >3.0 or too regular
        
        if 0.8 <= avg_disp <= 2.5 and std_disp > 0.4:
            score = 1.0
        elif avg_disp < 0.5:
            score = 0.1  # Too static (photo)
        elif avg_disp > 4.0:
            score = 0.2  # Too much (video replay)
        elif std_disp < 0.2:
            score = 0.3  # Too regular (not natural)
        else:
            score = 0.5
        
        confidence = 0.9
        return score, confidence


class StrictPassiveLivenessDetector:
    """
    Main detector dengan threshold yang sangat ketat untuk menolak foto/video HP
    """
    
    def __init__(self):
        print("Initializing STRICT Passive Liveness Detector...")
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Higher threshold
            min_tracking_confidence=0.6
        )
        
        # Detectors
        self.screen_detector = ScreenDisplayDetector()
        self.depth_estimator = DepthEstimator()
        self.blink_detector = StrictBlinkDetector()
        self.movement_analyzer = StrictMovementAnalyzer()
        
        # YOLO for device detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✓ YOLO loaded for device detection")
            except:
                print("⚠ YOLO not available")
        
        # History
        self.final_scores = deque(maxlen=30)
        self.detection_count = 0
        self.start_time = time.time()
        
        print("✓ STRICT detector initialized!")
    
    def extract_eye_landmarks(self, face_landmarks, image_shape):
        """Extract eye landmarks"""
        h, w = image_shape[:2]
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        left_eye = []
        right_eye = []
        
        for idx in LEFT_EYE:
            lm = face_landmarks.landmark[idx]
            left_eye.append([lm.x * w, lm.y * h])
        
        for idx in RIGHT_EYE:
            lm = face_landmarks.landmark[idx]
            right_eye.append([lm.x * w, lm.y * h])
        
        return np.array(left_eye), np.array(right_eye)
    
    def extract_face_roi(self, frame, face_landmarks):
        """Extract face ROI"""
        h, w = frame.shape[:2]
        
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    
    def detect_device_with_yolo(self, frame):
        """Deteksi HP/tablet dengan YOLO"""
        if self.yolo_model is None:
            return 1.0
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.3)
            
            suspicious_classes = ['cell phone', 'laptop', 'tv', 'monitor', 'tablet']
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_name = self.yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    if cls_name in suspicious_classes and conf > 0.4:
                        print(f"  ⚠ DETECTED: {cls_name} (conf: {conf:.2f})")
                        return 0.0  # REJECT immediately
            
            return 1.0
        except:
            return 1.0
    
    def detect(self, frame):
        """
        Main detection dengan multi-stage verification
        """
        if frame is None or frame.size == 0:
            return False, 0.0, {}
        
        self.detection_count += 1
        
        # === STAGE 1: Device Detection (YOLO) ===
        device_score = self.detect_device_with_yolo(frame)
        if device_score < 0.5:
            return False, 0.0, {
                'error': 'Device detected in frame',
                'final_score': 0.0,
                'is_live': False
            }
        
        # === STAGE 2: Screen Detection ===
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, {"error": "No face detected"}
        
        face_landmarks = results.multi_face_landmarks[0]
        face_roi, bbox = self.extract_face_roi(frame, face_landmarks)
        
        # Screen/display detection - BALANCED threshold
        screen_score, screen_conf = self.screen_detector.analyze(frame, face_roi)
        
        # BALANCED: Only reject if VERY clear evidence of screen
        if screen_score < 0.25:  # Was 0.4, now 0.25 (more strict threshold)
            print(f"  ✗ SCREEN DETECTED (score: {screen_score:.3f})")
            return False, screen_score, {
                'final_score': screen_score,
                'is_live': False,
                'reason': 'Screen/display detected',
                'bbox': bbox
            }
        
        # === STAGE 3: All other tests ===
        
        # Extract landmarks
        all_landmarks = []
        h, w = frame.shape[:2]
        for lm in face_landmarks.landmark:
            all_landmarks.append([lm.x * w, lm.y * h])
        all_landmarks = np.array(all_landmarks)
        
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
        
        # 1. Blink detection
        blink_score, blink_conf, blink_count = self.blink_detector.detect(left_eye, right_eye)
        
        # 2. Movement analysis
        movement_score, movement_conf = self.movement_analyzer.analyze(all_landmarks)
        
        # 3. Depth estimation
        depth_score, depth_conf = self.depth_estimator.estimate_from_shading(face_roi)
        
        # === STAGE 4: Fusion dengan BALANCED rules ===
        
        scores = {
            'screen': screen_score,
            'blink': blink_score,
            'movement': movement_score,
            'depth': depth_score,
            'device': device_score
        }
        
        # BALANCED: Butuh MULTIPLE low scores untuk reject
        very_low_count = sum(1 for s in scores.values() if s < 0.3)
        low_count = sum(1 for s in scores.values() if s < 0.5)
        min_score = min(scores.values())
        
        if very_low_count >= 3:  # 3+ tests fail badly
            final_score = min_score
        elif very_low_count >= 2:  # 2 tests fail badly
            # Weighted average dengan heavy penalty
            weights = [0.35, 0.30, 0.20, 0.10, 0.05]
            final_score = (
                screen_score * weights[0] +
                blink_score * weights[1] +
                movement_score * weights[2] +
                depth_score * weights[3] +
                device_score * weights[4]
            ) * 0.7  # 30% penalty
        else:
            # Normal weighted average
            weights = [0.35, 0.30, 0.20, 0.10, 0.05]  # screen, blink, movement, depth, device
            final_score = (
                screen_score * weights[0] +
                blink_score * weights[1] +
                movement_score * weights[2] +
                depth_score * weights[3] +
                device_score * weights[4]
            )
        
        self.final_scores.append(final_score)
        
        # Temporal verification: use MEDIAN (more robust than min)
        if len(self.final_scores) >= 5:
            final_score = np.median(list(self.final_scores)[-5:])
        
        # BALANCED threshold
        LIVENESS_THRESHOLD = 0.58  # Was 0.65, now 0.58 (slightly lower)
        is_live = final_score > LIVENESS_THRESHOLD
        
        # Additional rule: require at least 1 blink after 3 seconds
        time_elapsed = time.time() - self.start_time
        if time_elapsed > 3.0 and blink_count == 0:
            is_live = False
            final_score = min(final_score, 0.2)
        
        details = {
            'final_score': final_score,
            'is_live': is_live,
            'scores': scores,
            'bbox': bbox,
            'blink_count': blink_count,
            'min_score': min_score,
            'time_elapsed': time_elapsed
        }
        
        return is_live, final_score, details
    
    def visualize_results(self, frame, details):
        """Visualize dengan info lebih detail"""
        if 'bbox' not in details:
            return frame
        
        x_min, y_min, x_max, y_max = details['bbox']
        color = (0, 255, 0) if details['is_live'] else (0, 0, 255)
        
        # Bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Main label
        label = f"{'✓ LIVE' if details['is_live'] else '✗ SPOOF'}: {details['final_score']:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Individual scores
        y_offset = y_min + 25
        for name, score in details.get('scores', {}).items():
            score_color = (0, 255, 0) if score > 0.6 else (0, 165, 255) if score > 0.4 else (0, 0, 255)
            text = f"{name}: {score:.2f}"
            cv2.putText(frame, text, (x_min + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, score_color, 1)
            y_offset += 20
        
        # Blink count
        blink_text = f"Blinks: {details.get('blink_count', 0)}"
        cv2.putText(frame, blink_text, (x_min + 5, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        # Warning if spoof
        if not details['is_live']:
            warning = "REJECTED!"
            cv2.putText(frame, warning, (frame.shape[1]//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def main():
    """Demo with STRICT detection"""
    print("=" * 70)
    print("STRICT PASSIVE LIVENESS DETECTION")
    print("Optimized to REJECT photos/videos from phone/tablet displays")
    print("=" * 70)
    
    detector = StrictPassiveLivenessDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("\n📱 SYSTEM AKAN MENOLAK:")
    print("  - Foto dari layar HP/tablet")
    print("  - Video replay dari device")
    print("  - Wajah tanpa kedipan")
    print("  - Gerakan tidak natural")
    print("\nTekan 'q' untuk keluar\n")
    
    frame_count = 0
    fps = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        is_live, score, details = detector.detect(frame)
        
        # Visualize
        output = detector.visualize_results(frame.copy(), details)
        
        # FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = 30 / elapsed if elapsed > 0 else 0
            start_time = time.time()
        
        # Display FPS
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show
        cv2.imshow('STRICT Liveness Detection', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
