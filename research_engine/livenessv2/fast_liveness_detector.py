#!/usr/bin/env python3
"""
Ultra-Fast Real-time Liveness Detection
=======================================

This is an optimized implementation focused on speed and accuracy
using lightweight models and efficient preprocessing.
"""

import cv2
import numpy as np
import time
from collections import deque
import os
import json

class FastLivenessDetector:
    """
    Ultra-fast liveness detector using optimized algorithms
    """
    
    def __init__(self, confidence_threshold=0.7):
        """Initialize the fast detector"""
        self.confidence_threshold = confidence_threshold
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Smoothing parameters
        self.prediction_history = deque(maxlen=10)
        self.fps_history = deque(maxlen=30)
        
        # Texture analysis parameters
        self.texture_threshold = 0.02
        self.color_variance_threshold = 500
        self.edge_density_threshold = 0.15
        
        # Motion detection
        self.prev_gray = None
        self.motion_threshold = 15
        
        print("Fast Liveness Detector initialized")
    
    def analyze_texture_patterns(self, face_roi):
        """
        Analyze texture patterns to detect print/screen artifacts
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            dict: Texture analysis results
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern analysis
        def lbp_simple(image):
            rows, cols = image.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image[i, j]
                    code = 0
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            return lbp
        
        # Calculate LBP
        lbp = lbp_simple(gray)
        lbp_variance = np.var(lbp)
        
        # High-frequency analysis using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        return {
            'lbp_variance': lbp_variance,
            'laplacian_variance': laplacian_variance,
            'gradient_mean': gradient_mean,
            'texture_score': (lbp_variance / 1000 + laplacian_variance / 1000 + gradient_mean / 100) / 3
        }
    
    def analyze_color_distribution(self, face_roi):
        """
        Analyze color distribution for unnatural patterns
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            dict: Color analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # Analyze color variance in each channel
        bgr_variance = [np.var(face_roi[:, :, i]) for i in range(3)]
        hsv_variance = [np.var(hsv[:, :, i]) for i in range(3)]
        lab_variance = [np.var(lab[:, :, i]) for i in range(3)]
        
        # Calculate color uniformity (fake images often have unnatural uniformity)
        color_uniformity = np.mean(bgr_variance)
        
        # Skin color analysis
        # Typical skin color ranges in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_percentage = np.sum(skin_mask > 0) / (face_roi.shape[0] * face_roi.shape[1])
        
        return {
            'bgr_variance': bgr_variance,
            'hsv_variance': hsv_variance,
            'color_uniformity': color_uniformity,
            'skin_percentage': skin_percentage,
            'color_naturalness_score': min(skin_percentage * 2, 1.0)
        }
    
    def analyze_edge_characteristics(self, face_roi):
        """
        Analyze edge characteristics for sharpness artifacts
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            dict: Edge analysis results
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Analyze edge strength distribution
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge strength statistics
        edge_mean = np.mean(edge_strength)
        edge_std = np.std(edge_strength)
        edge_max = np.max(edge_strength)
        
        # Sharpness measure
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'edge_density': edge_density,
            'edge_mean': edge_mean,
            'edge_std': edge_std,
            'sharpness': sharpness,
            'edge_consistency_score': min(edge_density / self.edge_density_threshold, 1.0)
        }
    
    def detect_motion_artifacts(self, current_frame, face_bbox):
        """
        Detect motion artifacts that might indicate video replay attacks
        
        Args:
            current_frame: Current frame
            face_bbox: Face bounding box
            
        Returns:
            dict: Motion analysis results
        """
        x, y, w, h = face_bbox
        face_roi = current_frame[y:y+h, x:x+w]
        current_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        motion_score = 0.0
        frame_diff_mean = 0.0
        
        if self.prev_gray is not None and self.prev_gray.shape == current_gray.shape:
            # Calculate frame difference
            frame_diff = cv2.absdiff(current_gray, self.prev_gray)
            frame_diff_mean = np.mean(frame_diff)
            
            # Motion score based on difference
            motion_score = min(frame_diff_mean / self.motion_threshold, 1.0)
        
        self.prev_gray = current_gray.copy()
        
        return {
            'motion_score': motion_score,
            'frame_diff_mean': frame_diff_mean,
            'has_natural_motion': motion_score > 0.1 and motion_score < 0.8
        }
    
    def calculate_liveness_score(self, face_roi, face_bbox, frame):
        """
        Calculate comprehensive liveness score
        
        Args:
            face_roi: Face region of interest
            face_bbox: Face bounding box
            frame: Full frame
            
        Returns:
            tuple: (liveness_score, details)
        """
        # Run all analyses
        texture_analysis = self.analyze_texture_patterns(face_roi)
        color_analysis = self.analyze_color_distribution(face_roi)
        edge_analysis = self.analyze_edge_characteristics(face_roi)
        motion_analysis = self.detect_motion_artifacts(frame, face_bbox)
        
        # Weight factors for different features
        weights = {
            'texture': 0.3,
            'color': 0.25,
            'edge': 0.25,
            'motion': 0.2
        }
        
        # Calculate weighted score
        texture_score = min(texture_analysis['texture_score'], 1.0)
        color_score = color_analysis['color_naturalness_score']
        edge_score = edge_analysis['edge_consistency_score']
        motion_score = 1.0 if motion_analysis['has_natural_motion'] else 0.5
        
        liveness_score = (
            weights['texture'] * texture_score +
            weights['color'] * color_score +
            weights['edge'] * edge_score +
            weights['motion'] * motion_score
        )
        
        details = {
            'texture': texture_analysis,
            'color': color_analysis,
            'edge': edge_analysis,
            'motion': motion_analysis,
            'individual_scores': {
                'texture': texture_score,
                'color': color_score,
                'edge': edge_score,
                'motion': motion_score
            }
        }
        
        return liveness_score, details
    
    def predict_liveness(self, face_roi, face_bbox, frame):
        """
        Predict liveness with fast algorithms
        
        Args:
            face_roi: Face region of interest
            face_bbox: Face bounding box  
            frame: Full frame
            
        Returns:
            tuple: (prediction, confidence, details)
        """
        try:
            # Calculate liveness score
            liveness_score, details = self.calculate_liveness_score(face_roi, face_bbox, frame)
            
            # Add to history for smoothing
            self.prediction_history.append(liveness_score)
            
            # Apply temporal smoothing
            if len(self.prediction_history) >= 3:
                smoothed_score = np.mean(list(self.prediction_history)[-5:])
            else:
                smoothed_score = liveness_score
            
            # Make final prediction
            if smoothed_score >= self.confidence_threshold:
                prediction = "REAL"
                confidence = smoothed_score
            elif smoothed_score <= (1 - self.confidence_threshold):
                prediction = "FAKE"
                confidence = 1 - smoothed_score
            else:
                prediction = "UNCERTAIN"
                confidence = 0.5
            
            details['smoothed_score'] = smoothed_score
            details['raw_score'] = liveness_score
            
            return prediction, confidence, details
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "ERROR", 0.0, {}
    
    def draw_analysis_overlay(self, frame, faces, predictions):
        """
        Draw analysis overlay with detailed information
        
        Args:
            frame: Input frame
            faces: Detected faces
            predictions: Predictions for each face
        """
        height, width = frame.shape[:2]
        
        # Draw FPS
        current_time = time.time()
        self.fps_history.append(current_time)
        
        if len(self.fps_history) > 1:
            fps = len(self.fps_history) / (current_time - self.fps_history[0])
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detection info for each face
        for i, (x, y, w, h) in enumerate(faces):
            if i < len(predictions):
                pred, conf, details = predictions[i]
                
                # Choose color based on prediction
                if pred == "REAL":
                    color = (0, 255, 0)
                elif pred == "FAKE":
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw prediction info
                text = f"{pred}: {conf:.2f}"
                cv2.putText(frame, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw detailed scores (if available)
                if 'individual_scores' in details:
                    scores = details['individual_scores']
                    score_text = [
                        f"Texture: {scores['texture']:.2f}",
                        f"Color: {scores['color']:.2f}",
                        f"Edge: {scores['edge']:.2f}",
                        f"Motion: {scores['motion']:.2f}"
                    ]
                    
                    for j, score in enumerate(score_text):
                        cv2.putText(frame, score, (x + w + 10, y + 20 + j * 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw instructions
        instructions = [
            "Fast Liveness Detection",
            "- Multi-feature analysis",
            "- Real-time processing", 
            "- Press 'q' to quit",
            "- Press 'd' for debug info"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 120 + (i * 20)
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run_detection(self, camera_index=0, show_debug=False):
        """
        Run fast real-time liveness detection
        
        Args:
            camera_index: Camera device index
            show_debug: Whether to show debug information
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            return
        
        print("Starting Fast Liveness Detection...")
        print("Algorithm: Multi-feature analysis (Texture + Color + Edge + Motion)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(100, 100), maxSize=(400, 400)
                )
                
                # Process each face
                predictions = []
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    pred, conf, details = self.predict_liveness(face_roi, (x, y, w, h), frame)
                    predictions.append((pred, conf, details))
                
                # Draw overlay
                self.draw_analysis_overlay(frame, faces, predictions)
                
                # Display
                cv2.imshow('Fast Liveness Detection', frame)
                
                # Debug window
                if show_debug and len(predictions) > 0:
                    self.show_debug_info(predictions[0][2])
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow('Debug Info')
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"fast_liveness_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def show_debug_info(self, details):
        """Show debug information in separate window"""
        if not details:
            return
            
        # Create debug image
        debug_img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Draw debug information
        y_pos = 30
        line_height = 25
        
        cv2.putText(debug_img, "Debug Information", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height * 2
        
        if 'individual_scores' in details:
            scores = details['individual_scores']
            for name, score in scores.items():
                text = f"{name.capitalize()}: {score:.3f}"
                color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
                cv2.putText(debug_img, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += line_height
        
        if 'raw_score' in details and 'smoothed_score' in details:
            cv2.putText(debug_img, f"Raw Score: {details['raw_score']:.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += line_height
            cv2.putText(debug_img, f"Smoothed: {details['smoothed_score']:.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Debug Info', debug_img)

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 ULTRA-FAST LIVENESS DETECTION SYSTEM 🚀")
    print("=" * 60)
    print("Features:")
    print("✓ Real-time Multi-feature Analysis")
    print("✓ Texture Pattern Detection")
    print("✓ Color Distribution Analysis")
    print("✓ Edge Characteristic Analysis")
    print("✓ Motion Artifact Detection")
    print("✓ Temporal Smoothing")
    print("✓ No Deep Learning Required!")
    print("=" * 60)
    
    # Initialize detector
    detector = FastLivenessDetector(confidence_threshold=0.6)
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()