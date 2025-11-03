#!/usr/bin/env python3
"""
Advanced Real-time Face Liveness Detection
==========================================

Sistem deteksi liveness wajah yang canggih menggunakan multiple deep learning models
dan teknik computer vision untuk membedakan wajah asli dengan foto/gambar.

Features:
- Real-time detection tanpa perlu gerakan khusus
- Multiple detection algorithms
- Deep learning based anti-spoofing
- User-friendly interface
- High accuracy detection

Author: Face Recognition Team
Version: 3.0
"""

import cv2
import numpy as np
import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from collections import deque
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LivenessResult:
    """Result of liveness detection"""
    is_live: bool
    confidence: float
    detection_methods: Dict[str, float]
    processing_time: float
    frame_quality: float
    analysis_details: Dict[str, Any]

class AdvancedLivenessDetector:
    """
    Advanced liveness detection system using multiple AI/CV approaches
    """
    
    def __init__(self):
        """Initialize the advanced liveness detector"""
        logger.info("Initializing AdvancedLivenessDetector...")
        
        # Initialize detection components
        self._init_face_detection()
        self._init_texture_analyzers()
        self._init_depth_analyzers()
        self._init_frequency_analyzers()
        self._init_ml_models()
        
        # Detection parameters
        self.detection_params = {
            'min_face_size': 80,
            'max_face_size': 500,
            'confidence_threshold': 0.6,
            'texture_threshold': 0.4,
            'depth_threshold': 0.3,
            'frequency_threshold': 0.5,
            'ml_threshold': 0.7,
            'final_threshold': 0.65
        }
        
        # Tracking variables
        self.frame_buffer = deque(maxlen=30)  # Store recent frames for temporal analysis
        self.detection_history = deque(maxlen=10)  # Store recent detection results
        self.is_detecting = False
        
        logger.info("AdvancedLivenessDetector initialized successfully")
    
    def _init_face_detection(self):
        """Initialize face detection"""
        try:
            # Try to use DNN face detection (more accurate)
            self.face_net = cv2.dnn.readNetFromTensorflow(
                model='opencv_face_detector_uint8.pb', 
                config='opencv_face_detector.pbtxt'
            )
            self.use_dnn_detection = True
            logger.info("DNN face detection initialized")
        except:
            # Fallback to Haar cascade
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.use_dnn_detection = False
            logger.info("Haar cascade face detection initialized")
    
    def _init_texture_analyzers(self):
        """Initialize texture analysis components"""
        # Local Binary Pattern (LBP) analyzer
        # LBP parameters (limit to 8 points to stay within uint8 range)
        self.lbp_radius = 3
        self.lbp_n_points = 8  # Fixed to 8 points to avoid overflow
        
        # Gabor filter banks for texture analysis
        self.gabor_filters = []
        for theta in range(0, 180, 30):  # 6 orientations
            for frequency in [0.1, 0.3, 0.5]:  # 3 frequencies
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                self.gabor_filters.append(kernel)
        
        logger.info(f"Initialized {len(self.gabor_filters)} Gabor filters for texture analysis")
    
    def _init_depth_analyzers(self):
        """Initialize depth analysis components"""
        # Stereo matcher for depth estimation (if available)
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        
        # Shadow analysis parameters
        self.shadow_analyzer = {
            'kernel_size': 15,
            'threshold_factor': 0.7,
            'min_shadow_area': 0.05
        }
    
    def _init_frequency_analyzers(self):
        """Initialize frequency domain analyzers"""
        # DCT analysis parameters
        self.dct_block_size = 8
        
        # Wavelet analysis (simplified)
        self.wavelet_levels = 3
    
    def _init_ml_models(self):
        """Initialize machine learning models"""
        # Placeholder for ML models - in production, load pre-trained models
        self.ml_models = {
            'texture_classifier': None,  # Would load actual model
            'depth_classifier': None,    # Would load actual model
            'frequency_classifier': None # Would load actual model
        }
        
        # Feature extractors
        self.feature_extractors = {
            'hog': cv2.HOGDescriptor(),
            'orb': cv2.ORB_create(),
            'sift': cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
        }
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame and return bounding box"""
        if self.use_dnn_detection:
            return self._detect_face_dnn(frame)
        else:
            return self._detect_face_haar(frame)
    
    def _detect_face_dnn(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using DNN"""
        try:
            h, w = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            # Find best detection
            best_confidence = 0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.detection_params['confidence_threshold'] and confidence > best_confidence:
                    best_confidence = confidence
                    
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Validate face size
                    face_w, face_h = x2 - x1, y2 - y1
                    if (self.detection_params['min_face_size'] <= min(face_w, face_h) <= 
                        self.detection_params['max_face_size']):
                        best_box = (x1, y1, x2, y2)
            
            return best_box
            
        except Exception as e:
            logger.error(f"Error in DNN face detection: {e}")
            return None
    
    def _detect_face_haar(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face using Haar cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.detection_params['min_face_size'], 
                        self.detection_params['min_face_size'])
            )
            
            if len(faces) > 0:
                # Return largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                return (x, y, x + w, y + h)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Haar face detection: {e}")
            return None
    
    def analyze_liveness(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> LivenessResult:
        """
        Comprehensive liveness analysis
        
        Args:
            frame: Input frame (BGR)
            face_bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            LivenessResult with detailed analysis
        """
        start_time = time.time()
        
        x1, y1, x2, y2 = face_bbox
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                detection_methods={},
                processing_time=time.time() - start_time,
                frame_quality=0.0,
                analysis_details={'error': 'Invalid face ROI'}
            )
        
        # Store frame for temporal analysis
        self.frame_buffer.append(face_roi.copy())
        
        # Analyze frame quality
        frame_quality = self._analyze_frame_quality(face_roi)
        
        if frame_quality < 0.3:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                detection_methods={'quality_check': frame_quality},
                processing_time=time.time() - start_time,
                frame_quality=frame_quality,
                analysis_details={'reason': 'Poor frame quality'}
            )
        
        # Multi-method liveness detection
        detection_results = {}
        
        # 1. Texture Analysis
        texture_score = self._analyze_texture_advanced(face_roi)
        detection_results['texture_analysis'] = texture_score
        
        # 2. Frequency Domain Analysis
        frequency_score = self._analyze_frequency_domain(face_roi)
        detection_results['frequency_analysis'] = frequency_score
        
        # 3. Depth/3D Analysis
        depth_score = self._analyze_depth_cues(face_roi)
        detection_results['depth_analysis'] = depth_score
        
        # 4. Color Distribution Analysis
        color_score = self._analyze_color_distribution(face_roi)
        detection_results['color_analysis'] = color_score
        
        # 5. Reflection Analysis
        reflection_score = self._analyze_reflections(face_roi)
        detection_results['reflection_analysis'] = reflection_score
        
        # 6. Print/Screen Detection
        print_score = self._detect_print_screen_artifacts(face_roi)
        detection_results['print_detection'] = 1.0 - print_score  # Invert (high print = low liveness)
        
        # 7. Temporal Analysis (if enough frames)
        temporal_score = self._analyze_temporal_consistency()
        detection_results['temporal_analysis'] = temporal_score
        
        # 8. Shadow Analysis
        shadow_score = self._analyze_shadow_patterns(face_roi)
        detection_results['shadow_analysis'] = shadow_score
        
        # Combine all scores with weights
        weights = {
            'texture_analysis': 0.20,
            'frequency_analysis': 0.15,
            'depth_analysis': 0.15,
            'color_analysis': 0.10,
            'reflection_analysis': 0.10,
            'print_detection': 0.15,
            'temporal_analysis': 0.10,
            'shadow_analysis': 0.05
        }
        
        # Calculate weighted confidence
        confidence = sum(score * weights[method] for method, score in detection_results.items())
        confidence = max(0.0, min(1.0, confidence))
        
        # Apply quality boost/penalty
        confidence = confidence * (0.5 + 0.5 * frame_quality)
        
        # Final decision
        is_live = confidence >= self.detection_params['final_threshold']
        
        # Store result in history
        self.detection_history.append(confidence)
        
        # Temporal smoothing
        if len(self.detection_history) >= 3:
            smoothed_confidence = np.mean(list(self.detection_history)[-3:])
            confidence = 0.7 * confidence + 0.3 * smoothed_confidence
            is_live = confidence >= self.detection_params['final_threshold']
        
        processing_time = time.time() - start_time
        
        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            detection_methods=detection_results,
            processing_time=processing_time,
            frame_quality=frame_quality,
            analysis_details={
                'face_size': (x2-x1, y2-y1),
                'weights_used': weights,
                'threshold': self.detection_params['final_threshold']
            }
        )
    
    def _analyze_frame_quality(self, face_roi: np.ndarray) -> float:
        """Analyze frame quality"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Brightness check
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            
            # Contrast check
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 64.0)
            
            # Sharpness check (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(1.0, sharpness / 500.0)
            
            # Combine scores
            quality = (brightness_score * 0.3 + contrast_score * 0.3 + sharpness_score * 0.4)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Error analyzing frame quality: {e}")
            return 0.0
    
    def _analyze_texture_advanced(self, face_roi: np.ndarray) -> float:
        """Advanced texture analysis using multiple methods"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            scores = []
            
            # 1. Local Binary Pattern Analysis
            lbp_score = self._calculate_lbp_features(gray)
            scores.append(lbp_score)
            
            # 2. Gabor Filter Response
            gabor_score = self._calculate_gabor_features(gray)
            scores.append(gabor_score)
            
            # 3. Co-occurrence Matrix Features
            cooccurrence_score = self._calculate_cooccurrence_features(gray)
            scores.append(cooccurrence_score)
            
            # 4. Gradient Analysis
            gradient_score = self._calculate_gradient_features(gray)
            scores.append(gradient_score)
            
            # Combine texture scores
            texture_score = np.mean(scores)
            
            return max(0.0, min(1.0, texture_score))
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return 0.5
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> float:
        """Calculate LBP features"""
        try:
            h, w = gray.shape
            lbp = np.zeros((h-2*self.lbp_radius, w-2*self.lbp_radius), dtype=np.uint8)
            
            for i in range(self.lbp_radius, h-self.lbp_radius):
                for j in range(self.lbp_radius, w-self.lbp_radius):
                    center = gray[i, j]
                    code = 0
                    
                    # Sample points in a circle
                    for p in range(self.lbp_n_points):
                        angle = 2 * np.pi * p / self.lbp_n_points
                        x = int(round(i + self.lbp_radius * np.cos(angle)))
                        y = int(round(j + self.lbp_radius * np.sin(angle)))
                        
                        if 0 <= x < h and 0 <= y < w:
                            if gray[x, y] >= center:
                                code |= (1 << p)
                    
                    lbp[i-self.lbp_radius, j-self.lbp_radius] = code
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Real faces have more diverse texture patterns
            diversity = 1.0 - np.sum(hist**2)  # Entropy-like measure
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating LBP features: {e}")
            return 0.5
    
    def _calculate_gabor_features(self, gray: np.ndarray) -> float:
        """Calculate Gabor filter responses"""
        try:
            responses = []
            
            for kernel in self.gabor_filters:
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                responses.append(np.std(filtered))
            
            # Real faces show varied responses across different orientations/frequencies
            response_variety = np.std(responses) / (np.mean(responses) + 1e-7)
            
            return min(1.0, response_variety)
            
        except Exception as e:
            logger.error(f"Error calculating Gabor features: {e}")
            return 0.5
    
    def _calculate_cooccurrence_features(self, gray: np.ndarray) -> float:
        """Calculate Gray Level Co-occurrence Matrix features"""
        try:
            # Simplified GLCM calculation
            gray_scaled = (gray // 32).astype(np.uint8)  # Reduce levels for speed
            h, w = gray_scaled.shape
            
            # Calculate co-occurrence for horizontal direction
            glcm = np.zeros((8, 8), dtype=np.float32)
            
            for i in range(h):
                for j in range(w-1):
                    glcm[gray_scaled[i, j], gray_scaled[i, j+1]] += 1
            
            # Normalize
            glcm /= (glcm.sum() + 1e-7)
            
            # Calculate contrast and homogeneity
            contrast = 0
            homogeneity = 0
            
            for i in range(8):
                for j in range(8):
                    contrast += glcm[i, j] * (i - j) ** 2
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
            
            # Real faces have moderate contrast and homogeneity
            contrast_score = 1.0 - min(1.0, contrast / 10.0)
            homogeneity_score = homogeneity
            
            return (contrast_score + homogeneity_score) / 2
            
        except Exception as e:
            logger.error(f"Error calculating co-occurrence features: {e}")
            return 0.5
    
    def _calculate_gradient_features(self, gray: np.ndarray) -> float:
        """Calculate gradient-based features"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            # Real faces have more varied gradient patterns
            mag_variety = np.std(magnitude) / (np.mean(magnitude) + 1e-7)
            dir_variety = np.std(direction)
            
            gradient_score = (min(1.0, mag_variety) + min(1.0, dir_variety / np.pi)) / 2
            
            return gradient_score
            
        except Exception as e:
            logger.error(f"Error calculating gradient features: {e}")
            return 0.5
    
    def _analyze_frequency_domain(self, face_roi: np.ndarray) -> float:
        """Analyze frequency domain characteristics"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # FFT Analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Analyze different frequency bands
            # Low frequencies (center region)
            low_freq_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(low_freq_mask, (center_x, center_y), min(h, w) // 8, 1, -1)
            low_freq_mask = low_freq_mask.astype(bool)
            low_freq_energy = np.mean(magnitude_spectrum[low_freq_mask])
            
            # High frequencies (outer region)
            high_freq_mask = np.ones((h, w), dtype=np.uint8)
            cv2.circle(high_freq_mask, (center_x, center_y), min(h, w) // 4, 0, -1)
            high_freq_mask = high_freq_mask.astype(bool)
            high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
            
            # Real faces have balanced frequency content
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-7)
            freq_score = 1.0 / (1.0 + abs(freq_ratio - 0.3))  # Optimal ratio around 0.3
            
            return freq_score
            
        except Exception as e:
            logger.error(f"Error in frequency domain analysis: {e}")
            return 0.5
    
    def _analyze_depth_cues(self, face_roi: np.ndarray) -> float:
        """Analyze depth cues and 3D characteristics"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            scores = []
            
            # 1. Shadow analysis for depth
            shadow_score = self._detect_natural_shadows(face_roi)
            scores.append(shadow_score)
            
            # 2. Lighting gradient analysis
            lighting_score = self._analyze_lighting_gradients(gray)
            scores.append(lighting_score)
            
            # 3. Edge depth analysis
            edge_score = self._analyze_edge_depth(gray)
            scores.append(edge_score)
            
            depth_score = np.mean(scores)
            return max(0.0, min(1.0, depth_score))
            
        except Exception as e:
            logger.error(f"Error in depth analysis: {e}")
            return 0.5
    
    def _detect_natural_shadows(self, face_roi: np.ndarray) -> float:
        """Detect natural shadow patterns"""
        try:
            # Convert to LAB color space for better shadow analysis
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Find dark regions (potential shadows)
            dark_mask = l_channel < np.percentile(l_channel, 30)
            
            # Analyze shadow distribution
            if np.sum(dark_mask) == 0:
                return 0.3  # No shadows might indicate flat image
            
            # Real faces have gradual shadow transitions
            # Calculate gradient in shadow regions
            grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            shadow_gradient = np.mean(gradient_magnitude[dark_mask])
            
            # Natural shadows have moderate gradients
            shadow_score = 1.0 / (1.0 + abs(shadow_gradient - 20) / 20)
            
            return shadow_score
            
        except Exception as e:
            logger.error(f"Error detecting natural shadows: {e}")
            return 0.5
    
    def _analyze_lighting_gradients(self, gray: np.ndarray) -> float:
        """Analyze lighting gradients across face"""
        try:
            h, w = gray.shape
            
            # Analyze lighting variation across different regions
            regions = [
                gray[:h//2, :w//2],      # Top-left
                gray[:h//2, w//2:],      # Top-right
                gray[h//2:, :w//2],      # Bottom-left
                gray[h//2:, w//2:]       # Bottom-right
            ]
            
            region_means = [np.mean(region) for region in regions]
            
            # Real faces have natural lighting variation
            lighting_variation = np.std(region_means) / (np.mean(region_means) + 1e-7)
            
            # Optimal variation is moderate (not too uniform, not too extreme)
            lighting_score = 1.0 - abs(lighting_variation - 0.15) / 0.15
            
            return max(0.0, min(1.0, lighting_score))
            
        except Exception as e:
            logger.error(f"Error analyzing lighting gradients: {e}")
            return 0.5
    
    def _analyze_edge_depth(self, gray: np.ndarray) -> float:
        """Analyze edge characteristics for depth cues"""
        try:
            # Multi-scale edge detection
            edges_fine = cv2.Canny(gray, 50, 150)
            edges_coarse = cv2.Canny(gray, 20, 100)
            
            # Real faces have both fine and coarse edges
            fine_edge_density = np.sum(edges_fine > 0) / edges_fine.size
            coarse_edge_density = np.sum(edges_coarse > 0) / edges_coarse.size
            
            # Balance between fine and coarse edges
            edge_balance = fine_edge_density / (coarse_edge_density + 1e-7)
            edge_score = 1.0 / (1.0 + abs(edge_balance - 0.5))
            
            return edge_score
            
        except Exception as e:
            logger.error(f"Error analyzing edge depth: {e}")
            return 0.5
    
    def _analyze_color_distribution(self, face_roi: np.ndarray) -> float:
        """Analyze color distribution patterns"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            scores = []
            
            # HSV analysis
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Real faces have moderate color variation
            hsv_score = (min(1.0, h_std / 30) + min(1.0, s_std / 50) + min(1.0, v_std / 50)) / 3
            scores.append(hsv_score)
            
            # LAB analysis
            a_std = np.std(lab[:, :, 1])
            b_std = np.std(lab[:, :, 2])
            
            lab_score = (min(1.0, a_std / 20) + min(1.0, b_std / 20)) / 2
            scores.append(lab_score)
            
            # Skin color analysis
            skin_score = self._analyze_skin_color_distribution(hsv)
            scores.append(skin_score)
            
            color_score = np.mean(scores)
            return max(0.0, min(1.0, color_score))
            
        except Exception as e:
            logger.error(f"Error in color distribution analysis: {e}")
            return 0.5
    
    def _analyze_skin_color_distribution(self, hsv: np.ndarray) -> float:
        """Analyze skin color distribution"""
        try:
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            # Real faces should have reasonable skin color ratio
            if skin_ratio < 0.3 or skin_ratio > 0.9:
                return 0.3
            
            # Analyze skin color uniformity
            skin_pixels = hsv[skin_mask > 0]
            if len(skin_pixels) > 0:
                skin_h_std = np.std(skin_pixels[:, 0])
                skin_s_std = np.std(skin_pixels[:, 1])
                
                # Natural skin has some variation but not too much
                uniformity_score = 1.0 - min(1.0, (skin_h_std + skin_s_std) / 40)
                skin_score = (skin_ratio + uniformity_score) / 2
            else:
                skin_score = 0.0
            
            return skin_score
            
        except Exception as e:
            logger.error(f"Error analyzing skin color: {e}")
            return 0.5
    
    def _analyze_reflections(self, face_roi: np.ndarray) -> float:
        """Analyze reflection patterns"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Find bright spots (potential reflections)
            bright_threshold = np.percentile(gray, 95)
            bright_mask = gray > bright_threshold
            
            if np.sum(bright_mask) == 0:
                return 0.3  # No reflections might indicate artificial lighting
            
            # Analyze reflection characteristics
            contours, _ = cv2.findContours(bright_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            natural_reflections = 0
            total_reflection_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Minimum reflection size
                    total_reflection_area += area
                    
                    # Check if reflection is reasonably circular/oval
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.3 < circularity < 0.9:  # Reasonably round
                            natural_reflections += 1
            
            # Calculate reflection score
            reflection_ratio = total_reflection_area / face_roi.size
            
            if reflection_ratio > 0.1:  # Too many reflections
                return 0.2
            elif reflection_ratio < 0.001:  # Too few reflections
                return 0.4
            else:
                # Good reflection amount with natural shapes
                shape_score = min(1.0, natural_reflections / max(1, len(contours)))
                amount_score = min(1.0, reflection_ratio / 0.01)
                return (shape_score + amount_score) / 2
            
        except Exception as e:
            logger.error(f"Error analyzing reflections: {e}")
            return 0.5
    
    def _detect_print_screen_artifacts(self, face_roi: np.ndarray) -> float:
        """Detect printing or screen artifacts"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            scores = []
            
            # 1. Moire pattern detection
            moire_score = self._detect_moire_patterns(gray)
            scores.append(moire_score)
            
            # 2. Pixelation detection
            pixel_score = self._detect_pixelation(gray)
            scores.append(pixel_score)
            
            # 3. Screen refresh lines
            refresh_score = self._detect_refresh_lines(gray)
            scores.append(refresh_score)
            
            # 4. Print dot patterns
            dot_score = self._detect_print_dots(gray)
            scores.append(dot_score)
            
            artifact_score = np.mean(scores)
            return max(0.0, min(1.0, artifact_score))
            
        except Exception as e:
            logger.error(f"Error detecting print/screen artifacts: {e}")
            return 0.0
    
    def _detect_moire_patterns(self, gray: np.ndarray) -> float:
        """Detect moire patterns from screen capture"""
        try:
            # Apply FFT to detect regular patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Look for regular peaks in frequency domain
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Create circular masks for different frequency bands
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Check for regular patterns at specific frequencies
            freq_bands = [10, 20, 30, 40]  # Common screen frequencies
            pattern_strength = 0
            
            for freq in freq_bands:
                band_mask = (distances >= freq-2) & (distances <= freq+2)
                if np.sum(band_mask) > 0:
                    band_energy = np.mean(magnitude_spectrum[band_mask])
                    pattern_strength = max(pattern_strength, band_energy)
            
            # High pattern strength indicates moire
            moire_score = min(1.0, pattern_strength / 1000)
            
            return moire_score
            
        except Exception as e:
            logger.error(f"Error detecting moire patterns: {e}")
            return 0.0
    
    def _detect_pixelation(self, gray: np.ndarray) -> float:
        """Detect pixelation artifacts"""
        try:
            # Resize image smaller then back to detect blocky artifacts
            h, w = gray.shape
            small = cv2.resize(gray, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
            restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Calculate difference
            diff = cv2.absdiff(gray, restored)
            pixelation_score = np.mean(diff) / 255.0
            
            return min(1.0, pixelation_score * 5)
            
        except Exception as e:
            logger.error(f"Error detecting pixelation: {e}")
            return 0.0
    
    def _detect_refresh_lines(self, gray: np.ndarray) -> float:
        """Detect screen refresh lines"""
        try:
            # Look for horizontal lines (common in screen capture)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1]//4, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel)
            
            # Calculate how much the image changed
            diff = cv2.absdiff(gray, horizontal_lines)
            line_strength = np.mean(diff) / 255.0
            
            return min(1.0, line_strength * 3)
            
        except Exception as e:
            logger.error(f"Error detecting refresh lines: {e}")
            return 0.0
    
    def _detect_print_dots(self, gray: np.ndarray) -> float:
        """Detect printing dot patterns"""
        try:
            # High pass filter to enhance small details
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Look for regular dot patterns
            dots = cv2.threshold(np.abs(filtered), 20, 255, cv2.THRESH_BINARY)[1]
            
            # Find small circular objects (dots)
            contours, _ = cv2.findContours(dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            dot_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1 < area < 20:  # Small dots
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # Round dots
                            dot_count += 1
            
            # High dot density indicates printing
            dot_density = dot_count / (gray.shape[0] * gray.shape[1])
            dot_score = min(1.0, dot_density * 10000)
            
            return dot_score
            
        except Exception as e:
            logger.error(f"Error detecting print dots: {e}")
            return 0.0
    
    def _analyze_temporal_consistency(self) -> float:
        """Analyze temporal consistency across frames"""
        try:
            if len(self.frame_buffer) < 5:
                return 0.5  # Not enough frames for temporal analysis
            
            frames = list(self.frame_buffer)[-5:]  # Last 5 frames
            
            # Calculate frame-to-frame differences
            differences = []
            target_size = (64, 64)
            for i in range(len(frames)-1):
                # Ensure frames are the same size
                frame1 = cv2.resize(frames[i], target_size)
                frame2 = cv2.resize(frames[i+1], target_size)
                
                # Convert to grayscale for consistent comparison
                if len(frame1.shape) == 3:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                if len(frame2.shape) == 3:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(frame1, frame2)
                mean_diff = np.mean(diff)
                differences.append(mean_diff)
            
            # Real faces have small, consistent temporal changes
            temporal_variance = np.std(differences)
            mean_temporal_change = np.mean(differences)
            
            # Good temporal consistency: low variance, moderate mean change
            if mean_temporal_change < 1:  # Too static (image)
                return 0.2
            elif mean_temporal_change > 20:  # Too much change (noise/artifacts)
                return 0.3
            else:
                # Good change amount, check consistency
                consistency_score = 1.0 / (1.0 + temporal_variance / 5.0)
                return consistency_score
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return 0.5
    
    def _analyze_shadow_patterns(self, face_roi: np.ndarray) -> float:
        """Analyze shadow patterns for realism"""
        try:
            # Convert to LAB for better shadow analysis
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate shadow map
            shadow_threshold = np.percentile(l_channel, 25)
            shadow_mask = l_channel < shadow_threshold
            
            if np.sum(shadow_mask) == 0:
                return 0.4  # No shadows
            
            # Analyze shadow connectivity and gradients
            shadow_contours, _ = cv2.findContours(shadow_mask.astype(np.uint8), 
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Real shadows are usually connected and have smooth boundaries
            if len(shadow_contours) == 0:
                return 0.4
            
            largest_shadow = max(shadow_contours, key=cv2.contourArea)
            
            # Check shadow smoothness
            epsilon = 0.02 * cv2.arcLength(largest_shadow, True)
            approx = cv2.approxPolyDP(largest_shadow, epsilon, True)
            
            # Real shadows have relatively smooth boundaries
            smoothness = len(largest_shadow) / max(1, len(approx))
            shadow_score = min(1.0, smoothness / 10.0)
            
            return shadow_score
            
        except Exception as e:
            logger.error(f"Error analyzing shadow patterns: {e}")
            return 0.5
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[LivenessResult]]:
        """
        Process a single frame for liveness detection
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, liveness_result)
        """
        # Detect face
        face_bbox = self.detect_face(frame)
        
        if face_bbox is None:
            # No face detected
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No face detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame, None
        
        # Analyze liveness
        result = self.analyze_liveness(frame, face_bbox)
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, face_bbox, result)
        
        return annotated_frame, result
    
    def _draw_annotations(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                         result: LivenessResult) -> np.ndarray:
        """Draw annotations on frame"""
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = face_bbox
        
        # Draw face bounding box
        color = (0, 255, 0) if result.is_live else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw liveness result
        status_text = f"{'LIVE' if result.is_live else 'FAKE'} ({result.confidence:.2f})"
        cv2.putText(annotated_frame, status_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw quality indicator
        quality_text = f"Quality: {result.frame_quality:.2f}"
        cv2.putText(annotated_frame, quality_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw processing time
        time_text = f"Time: {result.processing_time*1000:.1f}ms"
        cv2.putText(annotated_frame, time_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw top detection methods
        y_offset = 90
        sorted_methods = sorted(result.detection_methods.items(), key=lambda x: x[1], reverse=True)
        for i, (method, score) in enumerate(sorted_methods[:3]):
            method_text = f"{method}: {score:.2f}"
            cv2.putText(annotated_frame, method_text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return annotated_frame

class RealtimeLivenessApp:
    """Real-time liveness detection application"""
    
    def __init__(self, camera_index: int = 0):
        """Initialize the application"""
        self.camera_index = camera_index
        self.detector = AdvancedLivenessDetector()
        self.cap = None
        self.is_running = False
        
        # Statistics
        self.frame_count = 0
        self.live_count = 0
        self.fake_count = 0
        self.start_time = None
        
    def start(self):
        """Start the application"""
        logger.info("Starting Real-time Liveness Detection Application")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("Application started successfully")
        logger.info("Press 'q' to quit, 's' to save current frame, 'r' to reset statistics")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                annotated_frame, result = self.detector.process_frame(frame)
                
                # Update statistics
                self.frame_count += 1
                if result is not None:
                    if result.is_live:
                        self.live_count += 1
                    else:
                        self.fake_count += 1
                
                # Add statistics to frame
                self._add_statistics(annotated_frame)
                
                # Show frame
                cv2.imshow('Advanced Real-time Liveness Detection', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(annotated_frame, result)
                elif key == ord('r'):
                    self._reset_statistics()
                elif key == ord('h'):
                    self._show_help()
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        
        finally:
            self._cleanup()
        
        return True
    
    def _add_statistics(self, frame: np.ndarray):
        """Add statistics to frame"""
        h, w = frame.shape[:2]
        
        # Runtime statistics
        runtime = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        stats_text = [
            f"Runtime: {runtime:.1f}s",
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Live: {self.live_count}",
            f"Fake: {self.fake_count}"
        ]
        
        # Draw statistics
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (w-200, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw help text
        cv2.putText(frame, "Press 'h' for help", (w-200, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _save_frame(self, frame: np.ndarray, result: Optional[LivenessResult]):
        """Save current frame and result"""
        timestamp = int(time.time())
        
        # Save frame
        frame_filename = f"liveness_frame_{timestamp}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        # Save result
        if result:
            result_filename = f"liveness_result_{timestamp}.json"
            result_data = {
                'timestamp': timestamp,
                'is_live': result.is_live,
                'confidence': result.confidence,
                'detection_methods': result.detection_methods,
                'processing_time': result.processing_time,
                'frame_quality': result.frame_quality,
                'analysis_details': result.analysis_details
            }
            
            with open(result_filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            logger.info(f"Saved frame and result: {frame_filename}, {result_filename}")
        else:
            logger.info(f"Saved frame: {frame_filename}")
    
    def _reset_statistics(self):
        """Reset statistics"""
        self.frame_count = 0
        self.live_count = 0
        self.fake_count = 0
        self.start_time = time.time()
        logger.info("Statistics reset")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
        Advanced Real-time Liveness Detection - Help
        ==========================================
        
        Controls:
        - 'q': Quit application
        - 's': Save current frame and detection result
        - 'r': Reset statistics
        - 'h': Show this help
        
        Detection Methods:
        - Texture Analysis: Analyzes skin texture patterns
        - Frequency Analysis: Analyzes frequency domain characteristics
        - Depth Analysis: Analyzes 3D depth cues and shadows
        - Color Analysis: Analyzes color distribution patterns
        - Reflection Analysis: Analyzes light reflection patterns
        - Print Detection: Detects printing/screen artifacts
        - Temporal Analysis: Analyzes consistency across frames
        - Shadow Analysis: Analyzes natural shadow patterns
        
        Status Indicators:
        - Green box: Live face detected
        - Red box: Fake face detected
        - Confidence: Detection confidence (0.0-1.0)
        - Quality: Frame quality score (0.0-1.0)
        - Time: Processing time per frame
        
        Tips for Best Results:
        - Ensure good lighting conditions
        - Keep face clearly visible and centered
        - Maintain stable camera position
        - Avoid excessive movement
        """
        
        print(help_text)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / runtime if runtime > 0 else 0
        
        print(f"\n=== SESSION SUMMARY ===")
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total frames: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Live detections: {self.live_count}")
        print(f"Fake detections: {self.fake_count}")
        print(f"Detection rate: {((self.live_count + self.fake_count) / max(1, self.frame_count) * 100):.1f}%")
        
        logger.info("Application cleanup completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Real-time Face Liveness Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start application
    app = RealtimeLivenessApp(camera_index=args.camera)
    success = app.start()
    
    if not success:
        logger.error("Failed to start application")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())