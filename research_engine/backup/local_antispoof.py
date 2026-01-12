#!/usr/bin/env python3
"""
Anti-Spoofing Realtime System dengan Model Lokal
Version 4.0 - Menggunakan model lokal anti_spoofing_model.h5
Solusi terpercaya untuk deteksi real vs fake faces
"""

import cv2
import numpy as np
import logging
import time
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_antispoof.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalAntiSpoofDetector:
    def __init__(self):
        """Initialize Local Anti-Spoofing Detector"""
        logger.info("🚀 Inisialisasi Local Anti-Spoofing System...")
        
        # Paths
        self.base_path = Path(__file__).parent / "antispoofing"
        self.model_path = self.base_path / "models" / "anti_spoofing_model.h5"
        self.cascade_path = self.base_path / "data" / "haarcascade_frontalface_default.xml"
        
        # Load anti-spoofing model
        try:
            if self.model_path.exists():
                self.model = keras.models.load_model(str(self.model_path))
                logger.info("✓ Local anti-spoofing model loaded")
                
                # Auto-detect correct input size from model
                self.input_size = self._detect_input_size()
                logger.info(f"✓ Detected model input size: {self.input_size}")
                
                # Display model info
                self._log_model_info()
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        # Load face cascade
        try:
            if self.cascade_path.exists():
                self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
            else:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            logger.info("✓ Face cascade loaded")
        except Exception as e:
            logger.error(f"❌ Error loading cascade: {e}")
            raise
        
        # Model settings (input_size sudah di-set oleh _detect_input_size di atas)
        self.frame_skip = 3  # Process every 3rd frame
        self.confidence_threshold = 0.5  # Threshold untuk real/fake decision
        
        # Results smoothing
        self.result_history = []
        self.history_size = 7
        self.stable_result = None
        self.stable_confidence = 0.0
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.real_count = 0
        self.fake_count = 0
        self.processing_times = []
        
        logger.info("✓ Local Anti-Spoofing Detector ready!")

    def _detect_input_size(self):
        """Auto-detect correct input size from model architecture"""
        try:
            # Get model input shape
            input_shape = self.model.input_shape
            logger.info(f"Model input shape: {input_shape}")
            
            if len(input_shape) == 4:  # (batch, height, width, channels)
                height, width = input_shape[1], input_shape[2]
                if height and width:
                    return (int(width), int(height))
            
            # Fallback: try common sizes and see what works
            common_sizes = [(96, 96), (112, 112), (128, 128), (160, 160), (224, 224)]
            
            for size in common_sizes:
                try:
                    # Create test input
                    test_input = np.random.random((1, size[1], size[0], 3)).astype(np.float32)
                    # Try prediction
                    _ = self.model.predict(test_input, verbose=0)
                    logger.info(f"✓ Working input size found: {size}")
                    return size
                except Exception as e:
                    logger.debug(f"Size {size} failed: {str(e)[:100]}")
                    continue
            
            # If nothing works, calculate from expected dense layer input
            try:
                # Get first dense layer expected input
                for layer in self.model.layers:
                    if 'dense' in layer.name.lower():
                        expected_features = layer.input_shape[-1]
                        # Calculate size: expected_features = height * width * channels
                        # Assuming 3 channels (RGB)
                        pixels = expected_features // 3
                        size = int(np.sqrt(pixels))
                        calculated_size = (size, size)
                        logger.info(f"✓ Calculated input size from dense layer: {calculated_size}")
                        return calculated_size
            except Exception as e:
                logger.warning(f"Could not calculate from dense layer: {e}")
            
            # Final fallback
            logger.warning("Using default size (96, 96)")
            return (96, 96)
            
        except Exception as e:
            logger.error(f"Error detecting input size: {e}")
            return (96, 96)

    def _log_model_info(self):
        """Log detailed model information for debugging"""
        try:
            logger.info("=== MODEL INFORMATION ===")
            logger.info(f"Input shape: {self.model.input_shape}")
            logger.info(f"Output shape: {self.model.output_shape}")
            
            # Log layers info
            logger.info("Model layers:")
            for i, layer in enumerate(self.model.layers):
                logger.info(f"  {i}: {layer.name} - {layer.__class__.__name__}")
                if hasattr(layer, 'input_shape'):
                    logger.info(f"      Input: {layer.input_shape}")
                if hasattr(layer, 'output_shape'):
                    logger.info(f"      Output: {layer.output_shape}")
            
            logger.info("========================")
        except Exception as e:
            logger.error(f"Error logging model info: {e}")

    def preprocess_face(self, face_region):
        """Preprocess face for model input with enhanced error handling"""
        try:
            # Validate input
            if face_region is None or face_region.size == 0:
                logger.warning("Empty face region provided")
                return None
            
            # Ensure 3 channels (RGB)
            if len(face_region.shape) == 2:
                face_region = cv2.cvtColor(face_region, cv2.COLOR_GRAY2RGB)
            elif len(face_region.shape) == 3 and face_region.shape[2] == 4:
                face_region = cv2.cvtColor(face_region, cv2.COLOR_BGRA2RGB)
            elif len(face_region.shape) == 3 and face_region.shape[2] == 3:
                face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            face_resized = cv2.resize(face_region, self.input_size, interpolation=cv2.INTER_LINEAR)
            
            # Validate resize result
            expected_shape = (self.input_size[1], self.input_size[0], 3)
            if face_resized.shape != expected_shape:
                logger.error(f"Unexpected shape after resize: {face_resized.shape}, expected: {expected_shape}")
                return None
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Final validation
            expected_batch_shape = (1, self.input_size[1], self.input_size[0], 3)
            if face_batch.shape != expected_batch_shape:
                logger.error(f"Final shape mismatch: {face_batch.shape}, expected: {expected_batch_shape}")
                return None
            
            return face_batch
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def predict_spoofing(self, face_region):
        """Predict if face is real or spoofed using local model with enhanced error handling"""
        try:
            start_time = time.time()
            
            # Preprocess face
            processed_face = self.preprocess_face(face_region)
            if processed_face is None:
                logger.debug("Preprocessing failed, skipping prediction")
                return None, 0.0, 0.0
            
            # Validate input shape matches model expectation (ignoring batch dimension)
            expected_shape = self.model.input_shape
            if len(processed_face.shape) != len(expected_shape):
                logger.error(f"Input shape dimension mismatch: {processed_face.shape} vs expected {expected_shape}")
                return None, 0.0, 0.0
            
            # Check dimensions except batch dimension (first dimension)
            for i in range(1, len(expected_shape)):
                if expected_shape[i] is not None and processed_face.shape[i] != expected_shape[i]:
                    logger.error(f"Input shape mismatch at dimension {i}: {processed_face.shape[i]} vs expected {expected_shape[i]}")
                    return None, 0.0, 0.0
            
            # Make prediction with error handling
            try:
                prediction = self.model.predict(processed_face, verbose=0)
            except Exception as pred_error:
                logger.error(f"Model prediction failed: {pred_error}")
                # Try to reshape input if it's a shape issue
                try:
                    reshaped_input = processed_face.reshape(expected_shape)
                    prediction = self.model.predict(reshaped_input, verbose=0)
                    logger.info("✓ Prediction succeeded after reshaping")
                except Exception as reshape_error:
                    logger.error(f"Reshape attempt failed: {reshape_error}")
                    return None, 0.0, 0.0
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                logger.error("Empty prediction result")
                return None, 0.0, 0.0
            
            # Parse prediction based on output shape
            if len(prediction.shape) == 2 and prediction.shape[1] == 2:
                # Binary classification: [fake_prob, real_prob]
                fake_prob = float(prediction[0][0])
                real_prob = float(prediction[0][1])
                is_real = real_prob > fake_prob
                confidence = max(fake_prob, real_prob)
            elif len(prediction.shape) == 2 and prediction.shape[1] == 1:
                # Single output: probability of being real
                real_prob = float(prediction[0][0])
                is_real = real_prob > self.confidence_threshold
                confidence = real_prob if is_real else (1.0 - real_prob)
            else:
                # Handle other output formats
                flat_pred = prediction.flatten()
                if len(flat_pred) == 1:
                    real_prob = float(flat_pred[0])
                    is_real = real_prob > self.confidence_threshold
                    confidence = real_prob if is_real else (1.0 - real_prob)
                elif len(flat_pred) == 2:
                    fake_prob, real_prob = float(flat_pred[0]), float(flat_pred[1])
                    is_real = real_prob > fake_prob
                    confidence = max(fake_prob, real_prob)
                else:
                    logger.error(f"Unexpected prediction shape: {prediction.shape}")
                    return None, 0.0, 0.0
            
            # Validate confidence value
            confidence = max(0.0, min(1.0, confidence))
            
            return is_real, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None, 0.0, 0.0

    def detect_faces(self, frame):
        """Detect faces using Haar Cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def smooth_results(self, new_result):
        """Smooth results using weighted history"""
        if new_result is None:
            return self.stable_result, self.stable_confidence
        
        is_real, confidence = new_result
        
        # Add to history
        self.result_history.append((is_real, confidence))
        
        # Keep history size limited
        if len(self.result_history) > self.history_size:
            self.result_history.pop(0)
        
        # Need at least 3 results for stability
        if len(self.result_history) < 3:
            self.stable_result = is_real
            self.stable_confidence = confidence
            return self.stable_result, self.stable_confidence
        
        # Calculate weighted average (recent results have more weight)
        weights = [i+1 for i in range(len(self.result_history))]
        total_weight = sum(weights)
        
        # Weighted confidence
        weighted_confidence = sum(conf * w for (_, conf), w in zip(self.result_history, weights)) / total_weight
        
        # Majority vote with confidence weighting
        real_score = sum(w for (is_r, _), w in zip(self.result_history, weights) if is_r)
        fake_score = sum(w for (is_r, _), w in zip(self.result_history, weights) if not is_r)
        
        # Decision with slight bias towards real (to reduce false positives)
        stable_is_real = real_score > fake_score * 0.9
        
        self.stable_result = stable_is_real
        self.stable_confidence = weighted_confidence
        
        return self.stable_result, self.stable_confidence

    def run_detection(self):
        """Run real-time anti-spoofing detection"""
        logger.info("🎬 Starting Local Anti-Spoofing Detection...")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("✓ Camera initialized")
        
        print("\n" + "="*85)
        print("    🎯 LOCAL ANTI-SPOOFING REALTIME SYSTEM v4.0")
        print("="*85)
        print("Features:")
        print("✓ Local TensorFlow Model (anti_spoofing_model.h5)")
        print("✓ No Internet Required")
        print("✓ Fast & Reliable Detection")
        print("✓ Advanced Result Smoothing")
        print("✓ Real-time Performance")
        print("\nControls:")
        print("q=quit | s=screenshot | d=debug | r=reset | t=threshold")
        print("="*85)
        
        show_debug = True
        
        # Performance tracking
        fps_counter = 0
        fps_start = time.time()
        display_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.frame_count += 1
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end = time.time()
                    display_fps = 30 / (fps_end - fps_start)
                    fps_start = fps_end
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process faces
                should_process = len(faces) > 0 and (self.frame_count % self.frame_skip == 0)
                current_result = None
                
                if should_process:
                    # Process the largest face
                    if len(faces) > 0:
                        # Sort by area
                        faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in faces]
                        faces_with_area.sort(key=lambda f: f[4], reverse=True)
                        x, y, w, h, _ = faces_with_area[0]
                        
                        # Extract face with padding
                        padding = 20
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        
                        face_region = frame[y1:y2, x1:x2]
                        
                        if face_region.size > 0:
                            # Predict with local model
                            is_real, confidence, proc_time = self.predict_spoofing(face_region)
                            
                            if is_real is not None:
                                current_result = (is_real, confidence)
                                self.detection_count += 1
                                
                                # Update stats
                                if is_real:
                                    self.real_count += 1
                                else:
                                    self.fake_count += 1
                                
                                logger.info(f"Frame {self.frame_count}: {'REAL' if is_real else 'FAKE'} "
                                          f"- Conf: {confidence:.3f}, Time: {proc_time:.3f}s")
                
                # Get smoothed result
                stable_is_real, stable_confidence = self.smooth_results(current_result)
                
                # Draw results
                for (x, y, w, h) in faces:
                    if stable_is_real is not None:
                        if stable_is_real:
                            color = (0, 255, 0)  # Green for REAL
                            status_text = f"REAL ({stable_confidence:.2f})"
                        else:
                            color = (0, 0, 255)  # Red for FAKE
                            status_text = f"FAKE ({stable_confidence:.2f})"
                    else:
                        color = (128, 128, 128)  # Gray for processing
                        status_text = "PROCESSING..."
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw status text
                    cv2.putText(frame, status_text, (x, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Debug information
                    if show_debug and current_result:
                        debug_y = y + h + 25
                        debug_info = [
                            f"Raw: {'R' if current_result[0] else 'F'} ({current_result[1]:.2f})",
                            f"History: {len(self.result_history)}",
                            f"Processed: {self.detection_count}"
                        ]
                        
                        for info in debug_info:
                            cv2.putText(frame, info, (x, debug_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            debug_y += 18
                
                # Status information
                status_y = 30
                status_lines = [
                    f"FPS: {display_fps:.1f} | Frames: {self.frame_count} | Faces: {len(faces)}",
                    f"Detections: {self.detection_count} | Real: {self.real_count} | Fake: {self.fake_count}",
                    f"Threshold: {self.confidence_threshold:.2f} | Debug: {'ON' if show_debug else 'OFF'}"
                ]
                
                for line in status_lines:
                    cv2.putText(frame, line, (10, status_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    status_y += 20
                
                # Performance indicator
                if self.processing_times:
                    avg_time = np.mean(self.processing_times[-10:])
                    perf_color = (0, 255, 0) if avg_time < 0.05 else (0, 255, 255) if avg_time < 0.1 else (0, 0, 255)
                    cv2.putText(frame, f"Avg: {avg_time:.3f}s", (frame.shape[1]-150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
                
                # Show frame
                cv2.imshow('Local Anti-Spoofing System v4.0', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"local_antispoof_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Screenshot: {filename}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    logger.info(f"🔍 Debug: {'ON' if show_debug else 'OFF'}")
                elif key == ord('r'):
                    # Reset
                    self.frame_count = 0
                    self.detection_count = 0
                    self.real_count = 0
                    self.fake_count = 0
                    self.result_history.clear()  
                    self.processing_times.clear()
                    self.stable_result = None
                    self.stable_confidence = 0.0
                    logger.info("🔄 Stats reset")
                elif key == ord('t'):
                    # Adjust threshold
                    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
                    current_idx = thresholds.index(self.confidence_threshold) if self.confidence_threshold in thresholds else 2
                    new_idx = (current_idx + 1) % len(thresholds)
                    self.confidence_threshold = thresholds[new_idx]
                    logger.info(f"🎯 Threshold: {self.confidence_threshold}")
        
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final summary
            logger.info("=== LOCAL ANTISPOOFING SESSION SUMMARY ===")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info(f"Real detections: {self.real_count}")
            logger.info(f"Fake detections: {self.fake_count}")
            
            if self.detection_count > 0:
                real_percentage = (self.real_count / self.detection_count) * 100
                fake_percentage = (self.fake_count / self.detection_count) * 100
                logger.info(f"Real percentage: {real_percentage:.1f}%")
                logger.info(f"Fake percentage: {fake_percentage:.1f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                logger.info(f"Average processing time: {avg_time:.3f}s")
                logger.info(f"Theoretical max FPS: {1/avg_time:.1f}")

if __name__ == "__main__":
    try:
        detector = LocalAntiSpoofDetector()
        detector.run_detection()
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        print(f"\n❌ Error: {e}")
        print("📋 Check log: local_antispoof.log")