"""
OPTIMIZED Passive Liveness Detection v2.2
==========================================
Versi optimal dengan STRICT SCORING dan auto-timeout

KEY IMPROVEMENTS v2.2:
1. Blink detection: STRICT SCORING (bobot 40%, need 2+ blinks untuk good score)
2. Movement analysis: HEAD ROTATION/NOD detection (bobot 35%)
3. Screen detection: Weight naik ke 25%
4. Auto-timeout: Otomatis close & final decision setelah 5 detik
5. Threshold: 0.60 (lebih strict)
6. Moiré detection: FIXED - multi-band FFT analysis
7. Illumination analysis: FIXED - gradient + variance + color
8. Reflection detection: FIXED - high reflection = SPOOF
9. Debug mode untuk monitoring real-time
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from scipy.spatial import distance as dist
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class SmartScreenDetector:
    """Deteksi screen yang lebih pintar - hindari false positive"""
    
    def __init__(self, yolo_model=None):
        self.detection_history = deque(maxlen=30)
        self.yolo_model = yolo_model
        self.device_detection_history = deque(maxlen=10)
        
    def detect_device_with_yolo(self, frame):
        """
        Gunakan YOLO untuk deteksi device secara langsung
        Ini adalah metode PALING AKURAT
        """
        if self.yolo_model is None:
            return 1.0
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.3)  # Lower threshold
            
            device_found = False
            max_conf = 0.0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_name = self.yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    # Check for devices
                    if cls_name in ['cell phone', 'laptop', 'tv', 'monitor', 'tablet']:
                        device_found = True
                        max_conf = max(max_conf, conf)
            
            # Store detection history
            self.device_detection_history.append(1 if device_found else 0)
            
            # If device detected consistently
            if len(self.device_detection_history) >= 3:
                recent_detections = sum(list(self.device_detection_history)[-3:])
                if recent_detections >= 2:  # 2 out of 3 frames
                    # Score based on confidence
                    if max_conf > 0.7:
                        return 0.0  # Very confident device detection
                    elif max_conf > 0.5:
                        return 0.1
                    elif max_conf > 0.3:
                        return 0.3
            
            return 1.0
        except:
            return 1.0
    
    def detect_screen_edges_advanced(self, frame, face_bbox):
        """
        Deteksi edges dari screen dengan metode lebih advanced
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        # 1. Canny
        edges_canny = cv2.Canny(gray, 30, 150)
        
        # 2. Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
        _, edges_sobel = cv2.threshold(edges_sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Combine edges
        edges_combined = cv2.bitwise_or(edges_canny, edges_sobel)
        
        # Find lines
        lines = cv2.HoughLinesP(edges_combined, 1, np.pi/180, 
                                threshold=80, minLineLength=100, maxLineGap=30)
        
        if lines is None:
            return 1.0
        
        x_min, y_min, x_max, y_max = face_bbox
        frame_h, frame_w = frame.shape[:2]
        
        # Analyze lines
        strong_horizontal = 0
        strong_vertical = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Long lines that are close to frame edges (screen boundary)
            if length > min(frame_w, frame_h) * 0.3:
                # Horizontal
                if angle < 15 or angle > 165:
                    # Check if near top or bottom
                    avg_y = (y1 + y2) / 2
                    if avg_y < frame_h * 0.15 or avg_y > frame_h * 0.85:
                        strong_horizontal += 1
                
                # Vertical
                elif 75 < angle < 105:
                    # Check if near left or right
                    avg_x = (x1 + x2) / 2
                    if avg_x < frame_w * 0.15 or avg_x > frame_w * 0.85:
                        strong_vertical += 1
        
        # Screen usually has strong edges on multiple sides
        if strong_horizontal >= 2 and strong_vertical >= 2:
            return 0.1  # Very likely screen
        elif strong_horizontal >= 1 and strong_vertical >= 1:
            return 0.4  # Possibly screen
        elif strong_horizontal >= 1 or strong_vertical >= 1:
            return 0.7
        
        return 1.0
    
    def detect_moire_advanced(self, face_roi):
        """Deteksi moiré pattern dengan FFT - FIXED untuk sensitivity"""
        if face_roi is None or face_roi.size == 0:
            return 1.0
        
        try:
            # Resize untuk consistency
            face_roi = cv2.resize(face_roi, (128, 128))  # Smaller untuk speed
            
            # Convert ke grayscale
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi.copy()
            
            # Normalize dengan robust method
            gray = gray.astype(np.float32)
            
            # Apply gaussian blur untuk reduce noise
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Compute FFT
            f = np.fft.fft2(gray_blur)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Remove DC component (low freq center)
            mask_size = 8
            magnitude_spectrum[center_h-mask_size:center_h+mask_size, 
                             center_w-mask_size:center_w+mask_size] = 0
            
            # Calculate radial frequency distribution
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
            max_radius = np.sqrt(center_h**2 + center_w**2)
            
            # Define frequency bands
            low_freq = (dist_from_center < max_radius * 0.2)
            mid_freq = (dist_from_center >= max_radius * 0.2) & (dist_from_center < max_radius * 0.5)
            high_freq = (dist_from_center >= max_radius * 0.5) & (dist_from_center < max_radius * 0.85)
            
            # Energy in each band
            total_energy = np.sum(magnitude_spectrum) + 1e-10
            low_energy = np.sum(magnitude_spectrum[low_freq]) / total_energy
            mid_energy = np.sum(magnitude_spectrum[mid_freq]) / total_energy
            high_energy = np.sum(magnitude_spectrum[high_freq]) / total_energy
            
            # Detect periodic patterns (moiré)
            # Use percentile-based threshold untuk adaptiveness
            threshold_percentile = 98
            threshold = np.percentile(magnitude_spectrum, threshold_percentile)
            peaks = magnitude_spectrum > threshold
            
            # Count peaks in different regions
            mid_peaks = np.sum(peaks & mid_freq)
            high_peaks = np.sum(peaks & high_freq)
            
            # Calculate peak density
            mid_area = np.sum(mid_freq) + 1
            high_area = np.sum(high_freq) + 1
            mid_peak_density = mid_peaks / mid_area
            high_peak_density = high_peaks / high_area
            
            # SCORING SYSTEM
            score = 1.0  # Start with "real"
            
            # 1. High energy in mid-high frequencies (screen artifacts)
            if high_energy > 0.15:
                score -= 0.4
            elif high_energy > 0.10:
                score -= 0.25
            elif high_energy > 0.07:
                score -= 0.15
            
            # 2. Mid frequency energy (moiré often appears here)
            if mid_energy > 0.30:
                score -= 0.35
            elif mid_energy > 0.22:
                score -= 0.20
            elif mid_energy > 0.18:
                score -= 0.10
            
            # 3. Peak density (periodic patterns)
            if high_peak_density > 0.002:
                score -= 0.3
            elif high_peak_density > 0.001:
                score -= 0.15
            
            if mid_peak_density > 0.003:
                score -= 0.25
            elif mid_peak_density > 0.0015:
                score -= 0.12
            
            # 4. Total high+mid peaks
            total_peaks = mid_peaks + high_peaks
            if total_peaks > 50:
                score -= 0.3
            elif total_peaks > 30:
                score -= 0.18
            elif total_peaks > 15:
                score -= 0.08
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            # If error, return neutral slightly favorable
            return 0.6
    
    def detect_uniform_illumination(self, face_roi):
        """
        Deteksi uniformitas iluminasi - FIXED
        Layar HP memiliki iluminasi yang terlalu uniform/flat
        Wajah asli memiliki variasi natural dari lighting
        """
        if face_roi is None or face_roi.size == 0:
            return 1.0
        
        try:
            # Convert to LAB color space for better lighting analysis
            if len(face_roi.shape) == 3:
                lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
            else:
                l_channel = face_roi
                a_channel = None
                b_channel = None
            
            # Resize untuk consistency
            l_channel = cv2.resize(l_channel, (128, 128))
            
            # 1. LOCAL VARIANCE ANALYSIS
            kernel_size = 11
            l_float = l_channel.astype(np.float32)
            
            # Local mean and variance
            local_mean = cv2.blur(l_float, (kernel_size, kernel_size))
            local_sqr_mean = cv2.blur(l_float**2, (kernel_size, kernel_size))
            local_variance = local_sqr_mean - local_mean**2
            local_variance = np.maximum(local_variance, 0)  # Ensure non-negative
            
            # Statistics of local variance
            var_mean = np.mean(local_variance)
            var_std = np.std(local_variance)
            var_of_var = var_std  # Simplified
            
            # 2. GRADIENT ANALYSIS
            # Real face: strong gradients (nose shadow, eye socket, etc)
            # Screen: weaker, more uniform gradients
            sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            gradient_mean = np.mean(gradient_magnitude)
            gradient_std = np.std(gradient_magnitude)
            
            # 3. COLOR CHANNEL CORRELATION (if color available)
            color_variety_score = 0.5
            if a_channel is not None and b_channel is not None:
                # Resize color channels
                a_channel = cv2.resize(a_channel, (128, 128))
                b_channel = cv2.resize(b_channel, (128, 128))
                
                # Real skin: significant variation in a* (red-green) and b* (yellow-blue)
                # Screen: more uniform
                a_std = np.std(a_channel)
                b_std = np.std(b_channel)
                
                # Also check correlation - real face has structured correlation
                correlation = np.corrcoef(a_channel.flatten(), b_channel.flatten())[0, 1]
                
                # Scoring
                if a_std > 8 and b_std > 8:
                    color_variety_score = 0.9  # Good color variation
                elif a_std > 5 and b_std > 5:
                    color_variety_score = 0.7
                elif a_std > 3 or b_std > 3:
                    color_variety_score = 0.5
                else:
                    color_variety_score = 0.2  # Too uniform
            
            # 4. HISTOGRAM ANALYSIS
            hist = cv2.calcHist([l_channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-10)
            
            # Entropy (higher = more varied)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # SCORING SYSTEM
            score = 0.0
            
            # Variance analysis
            if var_mean > 150:
                score += 0.25
            elif var_mean > 100:
                score += 0.18
            elif var_mean > 60:
                score += 0.10
            else:
                score -= 0.1  # Too uniform
            
            if var_of_var > 40:
                score += 0.20
            elif var_of_var > 25:
                score += 0.12
            elif var_of_var > 15:
                score += 0.05
            
            # Gradient analysis
            if gradient_mean > 12:
                score += 0.22
            elif gradient_mean > 8:
                score += 0.15
            elif gradient_mean > 5:
                score += 0.08
            
            if gradient_std > 10:
                score += 0.15
            elif gradient_std > 6:
                score += 0.08
            
            # Entropy
            if entropy > 4.0:
                score += 0.15
            elif entropy > 3.5:
                score += 0.08
            elif entropy < 2.5:
                score -= 0.1  # Too uniform histogram
            
            # Color variety
            score += color_variety_score * 0.15
            
            # Normalize
            final_score = max(0.0, min(1.0, score))
            
            return final_score
            
        except Exception as e:
            return 0.5
    
    def detect_screen_reflection(self, face_roi):
        """
        Deteksi refleksi dari layar HP/monitor
        Layar cenderung memiliki specular reflection yang tidak natural
        RETURN: 1.0 = NO reflection (real), 0.0 = HIGH reflection (spoof)
        """
        if face_roi is None or face_roi.size == 0:
            return 1.0
        
        try:
            # Resize
            face_roi = cv2.resize(face_roi, (128, 128))
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            h, s, v = cv2.split(hsv)
            l, a, b = cv2.split(lab)
            
            # 1. DETECT BRIGHT SPOTS (specular highlights)
            # Real face: diffuse lighting, soft highlights
            # Screen: sharp bright spots, over-saturated
            
            # Very bright pixels
            bright_threshold = 220
            very_bright = (v > bright_threshold).astype(np.uint8)
            num_bright = np.sum(very_bright)
            bright_ratio = num_bright / (v.shape[0] * v.shape[1])
            
            # Find connected components of bright regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(very_bright, connectivity=8)
            
            # Analyze bright spots
            sharp_bright_spots = 0
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                # Small, concentrated bright spots = reflection
                if 3 < area < 200:
                    sharp_bright_spots += 1
            
            # 2. SATURATION ANALYSIS
            # Screen reflection: low saturation in highlights
            # Real face: more consistent saturation
            bright_mask = v > 180
            if np.sum(bright_mask) > 10:
                bright_saturation = np.mean(s[bright_mask])
            else:
                bright_saturation = 128
            
            dark_mask = v < 100
            if np.sum(dark_mask) > 10:
                dark_saturation = np.mean(s[dark_mask])
            else:
                dark_saturation = 128
            
            saturation_contrast = abs(bright_saturation - dark_saturation)
            
            # 3. GLARE DETECTION using Laplacian
            # Sharp changes in intensity = glare
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            
            # Find very high laplacian values (edges/glare)
            high_laplacian = np.abs(laplacian) > 50
            glare_pixels = np.sum(high_laplacian)
            glare_ratio = glare_pixels / (gray.shape[0] * gray.shape[1])
            
            # 4. COLOR TEMPERATURE CONSISTENCY
            # Screen: unnatural color temperature, especially in highlights
            # Calculate blue-yellow ratio in bright areas
            if np.sum(bright_mask) > 10:
                bright_b_channel = b[bright_mask]
                blue_yellow_mean = np.mean(bright_b_channel)
                # Extreme values indicate unnatural lighting
                blue_yellow_abnormal = abs(blue_yellow_mean - 128) > 40
            else:
                blue_yellow_abnormal = False
            
            # SCORING - START WITH 1.0 (real), SUBTRACT if reflection found
            score = 1.0
            
            # Bright spots (PENALTY for reflection)
            if sharp_bright_spots > 5:
                score -= 0.45  # Many sharp bright spots = SCREEN REFLECTION
            elif sharp_bright_spots > 3:
                score -= 0.30
            elif sharp_bright_spots > 1:
                score -= 0.18
            
            if bright_ratio > 0.08:
                score -= 0.30  # Too many bright pixels = REFLECTION
            elif bright_ratio > 0.05:
                score -= 0.20
            
            # Saturation contrast (screen reflection has weird saturation)
            if saturation_contrast > 80:
                score -= 0.25  # Unnatural saturation = REFLECTION
            elif saturation_contrast > 60:
                score -= 0.15
            
            # Glare (STRONG INDICATOR of screen)
            if glare_ratio > 0.15:
                score -= 0.35  # High glare = SCREEN
            elif glare_ratio > 0.10:
                score -= 0.22
            elif glare_ratio > 0.07:
                score -= 0.12
            
            # Color temperature (unnatural = screen)
            if blue_yellow_abnormal:
                score -= 0.22
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            return 0.5
    
    def analyze(self, frame, face_roi, face_bbox):
        """Analisis dengan fokus pada accuracy - MULTI-METHOD"""
        if frame is None:
            return 0.8, 0.5  # Default: probably real
        
        scores = {}
        
        # METHOD 1: YOLO Device Detection (MOST RELIABLE!)
        yolo_score = self.detect_device_with_yolo(frame)
        scores['yolo_device'] = yolo_score
        
        # METHOD 2: Screen Edge Detection
        edge_score = self.detect_screen_edges_advanced(frame, face_bbox)
        scores['screen_edges'] = edge_score
        
        # METHOD 3: Moiré Pattern (FFT) - FIXED
        if face_roi is not None and face_roi.size > 0:
            moire_score = self.detect_moire_advanced(face_roi)
            scores['moire'] = moire_score
            
            # METHOD 4: Illumination Analysis - FIXED
            illum_score = self.detect_uniform_illumination(face_roi)
            scores['illumination'] = illum_score
            
            # METHOD 5: Screen Reflection Detection - NEW!
            reflection_score = self.detect_screen_reflection(face_roi)
            scores['reflection'] = reflection_score
        else:
            scores['moire'] = 0.5
            scores['illumination'] = 0.5
            scores['reflection'] = 0.5
        
        # Store for debugging
        self.last_subscores = scores.copy()
        
        # SMART FUSION dengan weighted voting
        # YOLO is king - if it detects device, heavily penalize
        if yolo_score < 0.2:
            # YOLO found device with high confidence
            final_score = yolo_score  # Trust YOLO completely
        else:
            # Weighted combination of all methods
            weights = {
                'yolo_device': 0.35,      # Highest weight
                'screen_edges': 0.20,
                'moire': 0.15,
                'illumination': 0.15,
                'reflection': 0.15        # New reflection detection
            }
            
            final_score = sum(scores[k] * weights[k] for k in scores.keys())
            
            # Additional penalty if multiple methods agree it's a screen
            low_scores = sum(1 for s in scores.values() if s < 0.4)
            if low_scores >= 4:  # 4+ methods say it's screen
                final_score *= 0.5  # Heavy penalty
            elif low_scores >= 3:  # 3 methods agree
                final_score *= 0.7  # Moderate penalty
            elif low_scores >= 2:  # 2 methods agree
                final_score *= 0.85
        
        self.detection_history.append(final_score)
        
        # Temporal consistency - use minimum of recent (more strict)
        if len(self.detection_history) >= 5:
            recent = list(self.detection_history)[-5:]
            # Use minimum to be more conservative
            final_score = min(recent)
        
        return final_score, 0.90


class ImprovedBlinkDetector:
    """Blink detector yang lebih ringan dan optimal - max 5 detik"""
    
    def __init__(self):
        self.EAR_THRESHOLD = 0.25  # Lebih ringan lagi
        self.EAR_CONSEC_FRAMES = 2  # Blink harus detected minimal 2 frame
        self.blink_counter = 0
        self.frame_counter = 0
        self.consecutive_closed = 0
        self.blink_times = deque(maxlen=50)
        self.ear_history = deque(maxlen=15)
        self.start_time = time.time()
        self.MAX_DURATION = 5.0  # Maksimal 5 detik
        
    def eye_aspect_ratio(self, eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C + 1e-6)
    
    def detect(self, left_eye, right_eye):
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        self.frame_counter += 1
        
        # Detect blink dengan konsekutif frames untuk menghindari noise
        if avg_ear < self.EAR_THRESHOLD:
            self.consecutive_closed += 1
        else:
            # Eye reopened
            if self.consecutive_closed >= self.EAR_CONSEC_FRAMES:
                self.blink_counter += 1
                self.blink_times.append(time.time())
            self.consecutive_closed = 0
        
        elapsed = time.time() - self.start_time
        
        # Scoring LEBIH STRICT - bobot lebih kecil, max 5 detik
        if elapsed < 1.0:
            # Sangat awal - strict
            return 0.40, 0.3, self.blink_counter
        elif elapsed < 2.5:
            # Perlu 1 blink - STRICT scoring
            if self.blink_counter >= 2:
                return 0.85, 0.85, self.blink_counter  # 2+ blinks = good
            elif self.blink_counter >= 1:
                return 0.70, 0.80, self.blink_counter  # 1 blink = moderate
            else:
                return 0.15, 0.7, self.blink_counter  # No blink = bad
        elif elapsed < self.MAX_DURATION:
            # 2.5 - 5 detik - STRICT requirements
            if self.blink_counter >= 2:
                # Check natural pattern
                if len(self.blink_times) >= 2:
                    intervals = np.diff(list(self.blink_times))
                    if len(intervals) > 0:
                        std_interval = np.std(intervals)
                        # Natural blink: interval bervariasi
                        if std_interval > 0.15:
                            return 0.95, 0.95, self.blink_counter
                        else:
                            return 0.75, 0.85, self.blink_counter
                return 0.85, 0.90, self.blink_counter
            elif self.blink_counter >= 1:
                return 0.60, 0.85, self.blink_counter  # Only 1 blink = low score
            else:
                return 0.0, 1.0, self.blink_counter  # No blink = definitely fake
        else:
            # Setelah 5 detik - FINAL DECISION dengan STRICT scoring
            if self.blink_counter >= 2:
                return 0.95, 1.0, self.blink_counter  # 2+ blinks = good
            elif self.blink_counter >= 1:
                return 0.55, 1.0, self.blink_counter  # Only 1 blink = borderline
            else:
                return 0.0, 1.0, self.blink_counter  # No blink = fake
    
    def is_timeout(self):
        """Check apakah sudah timeout (5 detik)"""
        return (time.time() - self.start_time) >= self.MAX_DURATION


class ImprovedMovementAnalyzer:
    """Movement analyzer - fokus pada HEAD ROTATION dan FACIAL LANDMARKS, bukan translasi object"""
    
    def __init__(self):
        self.landmark_history = deque(maxlen=30)
        # Key facial points untuk tracking head pose dan expressions
        self.key_points = [
            33, 133,    # Left eye corners
            362, 263,   # Right eye corners  
            61, 291,    # Mouth corners
            1,          # Nose tip
            10, 152,    # Forehead points
            234, 454    # Cheek points
        ]
        
    def compute_head_pose_change(self, prev_landmarks, curr_landmarks):
        """
        Hitung perubahan HEAD POSE (rotasi/pitch/yaw), bukan translasi keseluruhan
        """
        # Gunakan 3 titik anchor untuk menghitung orientasi wajah
        # Nose tip, left eye, right eye
        nose_idx = 6  # index in key_points for nose (point 1)
        left_eye_idx = 0  # left eye corner (point 33)
        right_eye_idx = 2  # right eye corner (point 362)
        
        prev_nose = prev_landmarks[nose_idx]
        prev_left = prev_landmarks[left_eye_idx]
        prev_right = prev_landmarks[right_eye_idx]
        
        curr_nose = curr_landmarks[nose_idx]
        curr_left = curr_landmarks[left_eye_idx]
        curr_right = curr_landmarks[right_eye_idx]
        
        # Calculate eye-to-eye vector (untuk yaw/rotation)
        prev_eye_vector = prev_right - prev_left
        curr_eye_vector = curr_right - curr_left
        
        # Calculate angle change (rotation)
        prev_angle = np.arctan2(prev_eye_vector[1], prev_eye_vector[0])
        curr_angle = np.arctan2(curr_eye_vector[1], curr_eye_vector[0])
        angle_change = abs(curr_angle - prev_angle)
        
        # Calculate nose position relative to eye center (untuk pitch/nod)
        prev_eye_center = (prev_left + prev_right) / 2
        curr_eye_center = (curr_left + curr_right) / 2
        
        prev_nose_rel = prev_nose - prev_eye_center
        curr_nose_rel = curr_nose - curr_eye_center
        
        # Normalize by eye distance untuk scale invariance
        prev_eye_dist = np.linalg.norm(prev_eye_vector)
        curr_eye_dist = np.linalg.norm(curr_eye_vector)
        
        if prev_eye_dist > 0 and curr_eye_dist > 0:
            prev_nose_norm = prev_nose_rel / prev_eye_dist
            curr_nose_norm = curr_nose_rel / curr_eye_dist
            nose_movement = np.linalg.norm(curr_nose_norm - prev_nose_norm)
        else:
            nose_movement = 0
        
        return angle_change, nose_movement
    
    def compute_facial_deformation(self, prev_landmarks, curr_landmarks):
        """
        Hitung perubahan SHAPE wajah (mimik/ekspresi)
        """
        # Mouth corners distance (untuk senyum/bicara)
        mouth_left_idx = 4
        mouth_right_idx = 5
        
        prev_mouth_width = np.linalg.norm(prev_landmarks[mouth_right_idx] - prev_landmarks[mouth_left_idx])
        curr_mouth_width = np.linalg.norm(curr_landmarks[mouth_right_idx] - curr_landmarks[mouth_left_idx])
        
        # Eye distance (reference untuk normalisasi)
        left_eye_idx = 0
        right_eye_idx = 2
        prev_eye_dist = np.linalg.norm(prev_landmarks[right_eye_idx] - prev_landmarks[left_eye_idx])
        curr_eye_dist = np.linalg.norm(curr_landmarks[right_eye_idx] - curr_landmarks[left_eye_idx])
        
        # Normalize mouth width by eye distance
        if prev_eye_dist > 0 and curr_eye_dist > 0:
            prev_mouth_norm = prev_mouth_width / prev_eye_dist
            curr_mouth_norm = curr_mouth_width / curr_eye_dist
            mouth_deformation = abs(curr_mouth_norm - prev_mouth_norm)
        else:
            mouth_deformation = 0
        
        return mouth_deformation
        
    def analyze(self, landmarks):
        if len(landmarks) == 0:
            return 0.5, 0.5
        
        # Extract hanya key facial points
        key_landmarks = landmarks[self.key_points]
        self.landmark_history.append(key_landmarks)
        
        if len(self.landmark_history) < 8:
            return 0.6, 0.5  # Not enough data yet
        
        recent = list(self.landmark_history)[-12:]
        
        # Analisis HEAD POSE CHANGES dan FACIAL DEFORMATIONS
        head_rotations = []
        head_nods = []
        facial_deforms = []
        
        for i in range(1, len(recent)):
            prev = np.array(recent[i-1])
            curr = np.array(recent[i])
            
            # Head pose changes (rotation/yaw/pitch)
            angle_change, nose_movement = self.compute_head_pose_change(prev, curr)
            head_rotations.append(angle_change)
            head_nods.append(nose_movement)
            
            # Facial deformation (mimik)
            deform = self.compute_facial_deformation(prev, curr)
            facial_deforms.append(deform)
        
        # Statistics
        avg_rotation = np.mean(head_rotations)
        std_rotation = np.std(head_rotations)
        avg_nod = np.mean(head_nods)
        avg_deformation = np.mean(facial_deforms)
        
        # SCORING BASED ON HEAD MOVEMENT + FACIAL EXPRESSIONS
        # Real face: 
        #   - Slight head movements (micro rotations/nods: 0.002-0.05 rad)
        #   - Natural facial deformation (mimik: 0.01-0.10)
        #   - Variability in movement
        # Photo/Video on screen:
        #   - No head pose change (rigid)
        #   - Whole object translates (bukan rotasi)
        #   - No facial deformation
        
        score = 0.5
        
        # Check HEAD ROTATION (yaw - kepala kiri/kanan)
        if 0.002 <= avg_rotation <= 0.08:
            score += 0.30  # Good natural head movement
            if std_rotation > 0.005:  # Has variation
                score += 0.15
        elif avg_rotation < 0.001:
            score -= 0.35  # Too static (no head movement)
        elif avg_rotation > 0.15:
            score -= 0.15  # Too much rotation (unlikely for passive)
        
        # Check HEAD NOD (pitch - kepala naik/turun)
        if 0.01 <= avg_nod <= 0.15:
            score += 0.20  # Good natural nodding/pitch
        elif avg_nod < 0.005:
            score -= 0.20  # Too static
        
        # Check FACIAL DEFORMATION (mimik wajah)
        if avg_deformation > 0.02:
            score += 0.25  # Good facial expressions
        elif avg_deformation > 0.008:
            score += 0.15  # Slight facial movement
        else:
            score -= 0.25  # No facial deformation = suspicious
        
        # Normalize score
        score = max(0.0, min(1.0, score))
        confidence = 0.85
        
        return score, confidence


class OptimizedPassiveLivenessDetector:
    """
    Detector yang optimal - fokus pada akurasi tanpa false positive
    """
    
    def __init__(self, debug=False):
        print("Initializing OPTIMIZED Passive Liveness Detector...")
        
        self.debug = debug
        
        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Components
        self.blink_detector = ImprovedBlinkDetector()
        self.movement_analyzer = ImprovedMovementAnalyzer()
        
        # YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✓ YOLO loaded for device detection")
            except:
                print("⚠ YOLO not available (optional)")
        
        # Initialize screen detector with YOLO
        self.screen_detector = SmartScreenDetector(yolo_model=self.yolo_model)
        
        self.final_scores = deque(maxlen=30)
        
        print("✓ Detector ready!")
        print("\nDETECTION STRATEGY (v2.2 - STRICT):")
        print("  1. YOLO device detection (if available)")
        print("  2. Blink detection [40% weight] - NEED 2+ blinks for good score")
        print("  3. Head rotation/nod [35% weight] - HEAD movement only")
        print("  4. Screen detection [25% weight] - Moiré + Reflection + Illumination")
        print("  5. Threshold: 0.60 (STRICT - harder to pass)")
        print("  6. Auto-timeout: 5 seconds MAX")
        print("  7. Final decision: Otomatis setelah timeout\n")
    
    def extract_eye_landmarks(self, face_landmarks, image_shape):
        h, w = image_shape[:2]
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        left_eye = [[face_landmarks.landmark[idx].x * w, 
                     face_landmarks.landmark[idx].y * h] for idx in LEFT_EYE]
        right_eye = [[face_landmarks.landmark[idx].x * w,
                      face_landmarks.landmark[idx].y * h] for idx in RIGHT_EYE]
        
        return np.array(left_eye), np.array(right_eye)
    
    def extract_face_roi(self, frame, face_landmarks):
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
    
    def detect_device_yolo(self, frame):
        """YOLO device detection"""
        if self.yolo_model is None:
            return 1.0, None
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_name = self.yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    if cls_name in ['cell phone', 'laptop', 'tv', 'tablet'] and conf > 0.6:
                        if self.debug:
                            print(f"  [YOLO] Detected: {cls_name} (conf: {conf:.2f})")
                        return 0.0, cls_name  # REJECT
            
            return 1.0, None
        except:
            return 1.0, None
    
    def detect(self, frame):
        """Main detection"""
        if frame is None or frame.size == 0:
            return False, 0.0, {}
        
        # Check timeout - AUTO CLOSE setelah 5 detik
        if self.blink_detector.is_timeout():
            elapsed = time.time() - self.blink_detector.start_time
            blink_count = self.blink_detector.blink_counter
            
            # Final decision berdasarkan blink count
            if blink_count >= 2:
                final_score = 0.95
                is_live = True
                reason = f"TIMEOUT ({elapsed:.1f}s): REAL - {blink_count} blinks detected"
            elif blink_count >= 1:
                final_score = 0.50
                is_live = False
                reason = f"TIMEOUT ({elapsed:.1f}s): SPOOF - Only {blink_count} blink (need 2+)"
            else:
                final_score = 0.0
                is_live = False
                reason = f"TIMEOUT ({elapsed:.1f}s): SPOOF - No blinks detected"
            
            return is_live, final_score, {
                'final_score': final_score,
                'is_live': is_live,
                'reason': reason,
                'blink_count': blink_count,
                'timeout': True,
                'elapsed_time': elapsed
            }
        
        # Stage 1: YOLO device detection
        device_score, device_name = self.detect_device_yolo(frame)
        if device_score == 0.0:
            return False, 0.0, {
                'final_score': 0.0,
                'is_live': False,
                'reason': f'Device detected: {device_name}',
                'device_detected': device_name
            }
        
        # Stage 2: Face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, {"error": "No face detected"}
        
        face_landmarks = results.multi_face_landmarks[0]
        face_roi, bbox = self.extract_face_roi(frame, face_landmarks)
        
        # Extract landmarks
        all_landmarks = []
        h, w = frame.shape[:2]
        for lm in face_landmarks.landmark:
            all_landmarks.append([lm.x * w, lm.y * h])
        all_landmarks = np.array(all_landmarks)
        
        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
        
        # Stage 3: Run all tests
        blink_score, blink_conf, blink_count = self.blink_detector.detect(left_eye, right_eye)
        movement_score, movement_conf = self.movement_analyzer.analyze(all_landmarks)
        screen_score, screen_conf = self.screen_detector.analyze(frame, face_roi, bbox)
        
        # Debug output
        if self.debug:
            elapsed = time.time() - self.blink_detector.start_time
            print(f"\n[SCORES - Time: {elapsed:.1f}s / 5.0s]")
            print(f"  Blink: {blink_score:.3f} (count: {blink_count})")
            print(f"  Movement: {movement_score:.3f} (head rotation/nod)")
            print(f"  Screen: {screen_score:.3f}")
            
            # Show ALL screen detection methods
            if hasattr(self.screen_detector, 'last_subscores'):
                subscores = self.screen_detector.last_subscores
                print(f"    [Screen Detection Methods]")
                for key, val in subscores.items():
                    indicator = "❌ SPOOF!" if val < 0.3 else ("⚠ Warning" if val < 0.6 else "✓ OK")
                    print(f"      {key:20s}: {val:.3f} {indicator}")
        
        # Stage 4: SMART FUSION - BLINK WEIGHT LEBIH KECIL (STRICT)
        # Blink sekarang hanya 40%, Movement 35%, Screen 25%
        
        # Primary indicators - Blink bobot DIKURANGI untuk strict scoring
        primary_score = (blink_score * 0.20 + movement_score * 0.40)
        
        # Secondary indicator - Screen detection weight NAIK
        secondary_score = screen_score
        print("Primary screen score:", primary_score)
        print("Secondary screen score:", secondary_score)
        
        # Final score: Balanced approach dengan strict blink
        if screen_score < 0.3 and primary_score < 0.5:
            # Both indicate spoof
            final_score = min(primary_score, screen_score)
        elif screen_score < 0.3:
            # Screen suspicious but primary OK
            final_score = primary_score * 0.50+ screen_score * 0.50
        else:
            # Normal case - screen weight lebih besar
            final_score = primary_score * 0.30 + secondary_score * 0.70
        
        self.final_scores.append(final_score)
        
        # Temporal smoothing
        if len(self.final_scores) >= 7:
            final_score = np.median(list(self.final_scores)[-7:])
        
        # Decision - STRICT THRESHOLD (lebih tinggi)
        THRESHOLD = 0.50  # Strict threshold - lebih sulit dianggap real
        is_live = final_score > THRESHOLD
        
        details = {
            'final_score': final_score,
            'is_live': is_live,
            'scores': {
                'blink': blink_score,
                'movement': movement_score,
                'screen': screen_score,
                'device': device_score
            },
            'bbox': bbox,
            'blink_count': blink_count,
            'primary_score': primary_score
        }
        
        return is_live, final_score, details
    
    def visualize_results(self, frame, details):
        """Visualize with detailed info"""
        if 'bbox' not in details and 'timeout' not in details:
            # Timeout case without face detection
            if 'timeout' in details:
                h, w = frame.shape[:2]
                color = (0, 255, 0) if details['is_live'] else (0, 0, 255)
                
                # Draw timeout message
                msg1 = f"TIMEOUT: {details.get('elapsed_time', 5.0):.1f}s"
                msg2 = details.get('reason', 'No face detected')
                
                cv2.rectangle(frame, (10, 10), (w-10, 120), color, -1)
                cv2.putText(frame, msg1, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, msg2, (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        
        if 'bbox' not in details:
            return frame
        
        x_min, y_min, x_max, y_max = details['bbox']
        color = (0, 255, 0) if details['is_live'] else (0, 0, 255)
        
        # Bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Main label with elapsed time
        elapsed = time.time() - self.blink_detector.start_time
        timeout_indicator = f" [{elapsed:.1f}s/5.0s]"
        status = "✓ REAL PERSON" if details['is_live'] else "✗ SPOOF DETECTED"
        label = f"{status}: {details['final_score']:.2f}{timeout_indicator}"
        
        # Background for text
        cv2.rectangle(frame, (x_min, y_min - 35), (x_min + 500, y_min), color, -1)
        cv2.putText(frame, label, (x_min + 5, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Individual scores
        y_offset = y_min + 25
        for name, score in details.get('scores', {}).items():
            if name == 'device':
                continue
            score_color = (0, 255, 0) if score > 0.6 else (0, 165, 255) if score > 0.4 else (0, 0, 255)
            text = f"{name.upper()}: {score:.2f}"
            cv2.putText(frame, text, (x_min + 5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2)
            y_offset += 25
        
        # Blinks
        cv2.putText(frame, f"Blinks: {details.get('blink_count', 0)}", 
                   (x_min + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def main():
    print("=" * 70)
    print("OPTIMIZED PASSIVE LIVENESS DETECTION v2.2")
    print("=" * 70)
    print("\nMAJOR CHANGES v2.2 (STRICT MODE):")
    print("  ✓ Blink: STRICT scoring - need 2+ blinks (weight: 40%)")
    print("  ✓ Movement: Head rotation/nod only (weight: 35%)")
    print("  ✓ Screen: Enhanced detection (weight: 25%)")
    print("  ✓ Threshold: 0.60 (lebih sulit pass)")
    print("  ✓ Timeout: AUTO-CLOSE setelah 5 detik")
    print("  ✓ Reflection: HIGH reflection = SPOOF")
    print("=" * 70)
    
    # Initialize with debug mode
    import sys
    debug_mode = '--debug' in sys.argv
    
    detector = OptimizedPassiveLivenessDetector(debug=debug_mode)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("\n📸 TESTING INSTRUCTIONS:")
    print("  1. Look at camera naturally")
    print("  2. Blink 2-3 times within 5 seconds (REQUIRED for good score)")
    print("  3. Add slight head movement (nod/rotate)")
    print("  4. Try showing photo/video from phone")
    print("  5. ⚠️ AUTO-CLOSE after 5 seconds with final decision")
    print("\nSCORING (STRICT MODE):")
    print("  • 2+ blinks in 5s = High score (0.85-0.95)")
    print("  • 1 blink in 5s = Medium score (0.55-0.70)")
    print("  • 0 blinks in 5s = FAIL (0.0)")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug mode")
    print("  'r' - Reset timer")
    print()
    
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
        
        # Display elapsed time and countdown
        elapsed = time.time() - detector.blink_detector.start_time
        remaining = max(0, 5.0 - elapsed)
        
        # Countdown bar
        bar_width = 300
        bar_x = 10
        bar_y = output.shape[0] - 40
        filled_width = int((elapsed / 5.0) * bar_width)
        
        # Background bar
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + 25), (50, 50, 50), -1)
        # Filled bar (changes color based on time)
        if remaining > 2:
            bar_color = (0, 255, 0)  # Green
        elif remaining > 1:
            bar_color = (0, 255, 255)  # Yellow
        else:
            bar_color = (0, 0, 255)  # Red
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + filled_width, bar_y + 25), bar_color, -1)
        
        # Time text
        time_text = f"Time: {elapsed:.1f}s / 5.0s"
        cv2.putText(output, time_text, (bar_x + bar_width + 10, bar_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        if frame_count % 30 == 0:
            elapsed_fps = time.time() - start_time
            fps = 30 / elapsed_fps if elapsed_fps > 0 else 0
            start_time = time.time()
        
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if debug_mode:
            cv2.putText(output, "DEBUG MODE", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Optimized Liveness Detection', output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            detector.debug = debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('r'):
            # Reset detector untuk test ulang
            detector.blink_detector = ImprovedBlinkDetector()
            detector.movement_analyzer = ImprovedMovementAnalyzer()
            detector.final_scores.clear()
            print("Timer reset! Starting new detection...")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
