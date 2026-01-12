#!/usr/bin/env python3
"""
Comprehensive Liveness Detection Demo
====================================

Demo script yang menggabungkan semua model liveness detection:
1. Advanced Liveness Detector - Multi-method analysis
2. Deep Learning Liveness Detector - Feature-based models  
3. State-of-the-Art Anti-Spoofing - Latest neural networks

User dapat memilih model mana yang ingin digunakan atau menjalankan
ensemble dari semua model untuk akurasi maksimal.

Author: Face Recognition Team
Version: 1.0
"""

import cv2
import numpy as np
import time
import logging
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import our liveness detectors
try:
    from advanced_liveness_detector import AdvancedLivenessDetector, LivenessResult
    HAS_ADVANCED = True
except ImportError as e:
    print(f"Warning: Could not import AdvancedLivenessDetector: {e}")
    HAS_ADVANCED = False

try:
    from deep_liveness_detector import DeepLivenessDetector, DeepLivenessResult
    HAS_DEEP = True
except ImportError as e:
    print(f"Warning: Could not import DeepLivenessDetector: {e}")
    HAS_DEEP = False

try:
    from sota_antispoof_detector import StateOfTheArtAntiSpoof, AntiSpoofingResult
    HAS_SOTA = True
except ImportError as e:
    print(f"Warning: Could not import StateOfTheArtAntiSpoof: {e}")
    HAS_SOTA = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveLivenessDetector:
    """
    Comprehensive liveness detector that combines multiple approaches
    """
    
    def __init__(self, models_to_use: List[str] = None, model_dir: str = "models"):
        """
        Initialize comprehensive detector
        
        Args:
            models_to_use: List of models to use ['advanced', 'deep', 'sota']
            model_dir: Directory for model files
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Default to all available models
        if models_to_use is None:
            models_to_use = []
            if HAS_ADVANCED:
                models_to_use.append('advanced')
            if HAS_DEEP:
                models_to_use.append('deep')
            if HAS_SOTA:
                models_to_use.append('sota')
        
        self.models_to_use = models_to_use
        self.detectors = {}
        
        # Initialize detectors
        self._init_detectors()
        
        # Ensemble configuration
        self.ensemble_weights = {
            'advanced': 0.30,  # Multi-method CV approach
            'deep': 0.35,      # Deep learning features
            'sota': 0.35       # State-of-the-art models
        }
        
        # Normalize weights
        total_weight = sum(self.ensemble_weights[k] for k in self.models_to_use)
        if total_weight > 0:
            for k in self.models_to_use:
                self.ensemble_weights[k] /= total_weight
        
        logger.info(f"Comprehensive detector initialized with models: {self.models_to_use}")
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def _init_detectors(self):
        """Initialize all requested detectors"""
        
        if 'advanced' in self.models_to_use and HAS_ADVANCED:
            try:
                self.detectors['advanced'] = AdvancedLivenessDetector()
                logger.info("Advanced Liveness Detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Advanced detector: {e}")
        
        if 'deep' in self.models_to_use and HAS_DEEP:
            try:
                self.detectors['deep'] = DeepLivenessDetector()
                logger.info("Deep Learning Detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Deep detector: {e}")
        
        if 'sota' in self.models_to_use and HAS_SOTA:
            try:
                self.detectors['sota'] = StateOfTheArtAntiSpoof(model_dir=str(self.model_dir))
                logger.info("State-of-the-Art Detector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SOTA detector: {e}")
        
        logger.info(f"Successfully initialized {len(self.detectors)} detectors")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process frame with all available detectors
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated_frame, comprehensive_results)
        """
        start_time = time.time()
        
        # Store individual results
        individual_results = {}
        
        # Process with each detector
        if 'advanced' in self.detectors:
            try:
                adv_frame, adv_result = self.detectors['advanced'].process_frame(frame)
                individual_results['advanced'] = {
                    'result': adv_result,
                    'annotated_frame': adv_frame
                }
            except Exception as e:
                logger.error(f"Error with advanced detector: {e}")
                individual_results['advanced'] = None
        
        if 'deep' in self.detectors:
            try:
                deep_frame, deep_results = self.detectors['deep'].process_frame(frame)
                individual_results['deep'] = {
                    'results': deep_results,
                    'annotated_frame': deep_frame
                }
            except Exception as e:
                logger.error(f"Error with deep detector: {e}")
                individual_results['deep'] = None
        
        if 'sota' in self.detectors:
            try:
                sota_frame, sota_results = self.detectors['sota'].process_frame(frame)
                individual_results['sota'] = {
                    'results': sota_results,
                    'annotated_frame': sota_frame
                }
            except Exception as e:
                logger.error(f"Error with SOTA detector: {e}")
                individual_results['sota'] = None
        
        # Ensemble prediction
        ensemble_result = self._ensemble_predict(individual_results)
        
        # Create comprehensive annotated frame
        annotated_frame = self._create_comprehensive_annotation(frame, individual_results, ensemble_result)
        
        processing_time = time.time() - start_time
        
        comprehensive_results = {
            'individual_results': individual_results,
            'ensemble_result': ensemble_result,
            'processing_time': processing_time,
            'models_used': list(self.detectors.keys())
        }
        
        return annotated_frame, comprehensive_results
    
    def _ensemble_predict(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble prediction from individual results"""
        
        predictions = []
        confidences = []
        model_votes = {}
        
        # Extract predictions from each model
        for model_name, result_data in individual_results.items():
            if result_data is None:
                continue
            
            model_predictions = []
            model_confidences = []
            
            if model_name == 'advanced':
                if result_data['result']:
                    model_predictions.append(result_data['result'].is_live)
                    model_confidences.append(result_data['result'].confidence)
            
            elif model_name == 'deep':
                for result in result_data['results']:
                    model_predictions.append(result.is_live)
                    model_confidences.append(result.confidence)
            
            elif model_name == 'sota':
                for result in result_data['results']:
                    model_predictions.append(result.is_live)
                    model_confidences.append(result.confidence)
            
            # Aggregate model predictions
            if model_predictions:
                model_live_ratio = sum(model_predictions) / len(model_predictions)
                model_avg_confidence = np.mean(model_confidences)
                
                model_votes[model_name] = {
                    'live_ratio': model_live_ratio,
                    'confidence': model_avg_confidence,
                    'num_faces': len(model_predictions)
                }
                
                # Weight by ensemble weights
                weight = self.ensemble_weights.get(model_name, 0)
                predictions.append(model_live_ratio * weight)
                confidences.append(model_avg_confidence * weight)
        
        # Calculate ensemble results
        if predictions:
            ensemble_prediction = sum(predictions)
            ensemble_confidence = sum(confidences)
            ensemble_is_live = ensemble_prediction >= 0.5
            
            # Calculate agreement between models
            model_agreements = [v['live_ratio'] for v in model_votes.values()]
            agreement_std = np.std(model_agreements) if len(model_agreements) > 1 else 0.0
            agreement_score = max(0.0, 1.0 - agreement_std)  # High agreement = low std
            
            # Risk assessment
            if ensemble_confidence >= 0.8 and agreement_score >= 0.8:
                risk_level = 'LOW'
            elif ensemble_confidence >= 0.6 and agreement_score >= 0.6:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
        else:
            ensemble_prediction = 0.0
            ensemble_confidence = 0.0
            ensemble_is_live = False
            agreement_score = 0.0
            risk_level = 'HIGH'
            model_votes = {}
        
        ensemble_result = {
            'is_live': ensemble_is_live,
            'confidence': ensemble_confidence,
            'prediction_score': ensemble_prediction,
            'agreement_score': agreement_score,
            'risk_level': risk_level,
            'model_votes': model_votes,
            'weights_used': self.ensemble_weights
        }
        
        return ensemble_result
    
    def _create_comprehensive_annotation(self, frame: np.ndarray, 
                                       individual_results: Dict[str, Any],
                                       ensemble_result: Dict[str, Any]) -> np.ndarray:
        """Create comprehensive annotation combining all results"""
        
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]
        
        # Draw ensemble result (main display)
        ensemble_color = self._get_risk_color(ensemble_result['risk_level'])
        status_text = f"ENSEMBLE: {'LIVE' if ensemble_result['is_live'] else 'FAKE'}"
        confidence_text = f"Confidence: {ensemble_result['confidence']:.2f}"
        agreement_text = f"Agreement: {ensemble_result['agreement_score']:.2f}"
        risk_text = f"Risk: {ensemble_result['risk_level']}"
        
        # Main status (top center)
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(annotated_frame, status_text, (text_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, ensemble_color, 3)
        
        # Ensemble details (top left panel)
        panel_y = 80
        cv2.rectangle(annotated_frame, (10, panel_y), (400, panel_y + 120), (0, 0, 0, 128), -1)
        
        details = [confidence_text, agreement_text, risk_text]
        for i, detail in enumerate(details):
            cv2.putText(annotated_frame, detail, (20, panel_y + 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Individual model results (right panel)
        model_panel_x = w - 300
        model_panel_y = 80
        cv2.rectangle(annotated_frame, (model_panel_x, model_panel_y), 
                     (w - 10, model_panel_y + 200), (0, 0, 0, 128), -1)
        
        cv2.putText(annotated_frame, "MODEL VOTES:", (model_panel_x + 10, model_panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 50
        for model_name, vote_data in ensemble_result['model_votes'].items():
            model_status = "LIVE" if vote_data['live_ratio'] >= 0.5 else "FAKE"
            model_text = f"{model_name.upper()}: {model_status}"
            model_conf_text = f"  Conf: {vote_data['confidence']:.2f}"
            
            cv2.putText(annotated_frame, model_text, (model_panel_x + 10, model_panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated_frame, model_conf_text, (model_panel_x + 10, model_panel_y + y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            y_offset += 40
        
        # Performance info (bottom)
        active_models_text = f"Active Models: {len(individual_results)}"
        cv2.putText(annotated_frame, active_models_text, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        """Get color based on risk level"""
        if risk_level == 'LOW':
            return (0, 255, 0)  # Green
        elif risk_level == 'MEDIUM':
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red

def main():
    """Main function for comprehensive liveness detection demo"""
    parser = argparse.ArgumentParser(description='Comprehensive Liveness Detection Demo')
    
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--models', nargs='+', 
                       choices=['advanced', 'deep', 'sota'], 
                       help='Models to use (default: all available)')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--save-results', action='store_true', help='Save detection results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--fps-limit', type=int, default=30, help='FPS limit')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check available models
    available_models = []
    if HAS_ADVANCED:
        available_models.append('advanced')
    if HAS_DEEP:
        available_models.append('deep')
    if HAS_SOTA:
        available_models.append('sota')
    
    if not available_models:
        print("ERROR: No liveness detection models are available!")
        print("Please ensure at least one of the detector modules is properly installed.")
        return 1
    
    print(f"Available models: {available_models}")
    
    # Use specified models or all available
    models_to_use = args.models if args.models else available_models
    print(f"Using models: {models_to_use}")
    
    # Initialize detector
    try:
        detector = ComprehensiveLivenessDetector(
            models_to_use=models_to_use,
            model_dir=args.model_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return 1
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return 1
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps_limit)
    
    # Statistics
    frame_count = 0
    detection_count = 0
    live_count = 0
    fake_count = 0
    start_time = time.time()
    
    # FPS control
    fps_target = args.fps_limit
    frame_time = 1.0 / fps_target
    
    print("\n=== COMPREHENSIVE LIVENESS DETECTION ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current results (if --save-results)")
    print("  'r' - Reset statistics")
    print("  'h' - Show help")
    print("  '1' - Show individual model results")
    print("  '2' - Show ensemble result only")
    print("=" * 45)
    
    display_mode = 'ensemble'  # 'ensemble' or 'individual'
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, results = detector.process_frame(frame)
            
            # Update statistics
            ensemble_result = results['ensemble_result']
            if ensemble_result['model_votes']:
                detection_count += 1
                if ensemble_result['is_live']:
                    live_count += 1
                else:
                    fake_count += 1
            
            # Add runtime statistics
            runtime = time.time() - start_time
            fps = frame_count / runtime if runtime > 0 else 0
            
            # Add performance info
            h, w = annotated_frame.shape[:2]
            stats_text = [
                f"FPS: {fps:.1f}",
                f"Frames: {frame_count}",
                f"Detections: {detection_count}",
                f"Live: {live_count} | Fake: {fake_count}",
                f"Processing: {results['processing_time']*1000:.1f}ms"
            ]
            
            # Draw stats background
            stats_bg_h = len(stats_text) * 25 + 20
            cv2.rectangle(annotated_frame, (w-250, 10), (w-10, 10 + stats_bg_h), (0, 0, 0, 128), -1)
            
            for i, stat in enumerate(stats_text):
                cv2.putText(annotated_frame, stat, (w-240, 35 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            window_title = f"Comprehensive Liveness Detection - {' + '.join(models_to_use).upper()}"
            cv2.imshow(window_title, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save_results:
                # Save results
                timestamp = int(time.time())
                frame_filename = f"comprehensive_frame_{timestamp}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                
                results_filename = f"comprehensive_results_{timestamp}.json"
                # Convert results to JSON-serializable format
                json_results = {
                    'timestamp': timestamp,
                    'ensemble_result': ensemble_result,
                    'models_used': results['models_used'],
                    'processing_time': results['processing_time']
                }
                
                with open(results_filename, 'w') as f:
                    json.dump(json_results, f, indent=2, default=str)
                
                print(f"Saved: {frame_filename}, {results_filename}")
            
            elif key == ord('r'):
                # Reset statistics
                frame_count = 0
                detection_count = 0
                live_count = 0
                fake_count = 0
                start_time = time.time()
                print("Statistics reset")
            
            elif key == ord('h'):
                print("""
=== COMPREHENSIVE LIVENESS DETECTION HELP ===

Models:
- ADVANCED: Multi-method computer vision analysis
- DEEP: Deep learning feature extraction
- SOTA: State-of-the-art neural networks

Ensemble Method:
- Combines predictions from all active models
- Weighted voting based on model confidence
- Agreement score measures model consensus
- Risk level based on confidence and agreement

Risk Levels:
- LOW (Green): High confidence, good agreement
- MEDIUM (Yellow): Moderate confidence/agreement  
- HIGH (Red): Low confidence or poor agreement

Controls:
- 'q': Quit application
- 's': Save current frame and results
- 'r': Reset statistics  
- 'h': Show this help

Tips for Best Results:
- Ensure good lighting conditions
- Keep face centered and clearly visible
- Maintain stable camera position
- Avoid excessive movement
- Test with different lighting conditions
- Try holding up photos/screens to test spoofing detection
                """)
            
            # FPS limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary
        runtime = time.time() - start_time
        avg_fps = frame_count / runtime if runtime > 0 else 0
        
        print(f"\n=== COMPREHENSIVE LIVENESS DETECTION SUMMARY ===")
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total detections: {detection_count}")
        print(f"Live detections: {live_count}")
        print(f"Fake detections: {fake_count}")
        if detection_count > 0:
            print(f"Live detection rate: {(live_count/detection_count)*100:.1f}%")
        
        print(f"Models used: {', '.join(models_to_use)}")
        print("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit(main())