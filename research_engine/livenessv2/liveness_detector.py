#!/usr/bin/env python3
"""
Advanced Real-time Face Liveness Detection System
================================================

This module implements a state-of-the-art face liveness detection system using
deep learning techniques to distinguish between real faces and spoof attacks
(photos, videos, 3D masks, etc.).

Key Features:
- Real-time webcam detection
- Deep CNN-based classification
- Robust preprocessing pipeline
- Multi-scale face detection
- Confidence scoring
- User-friendly interface

Author: Developed based on multiple research papers and implementations
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Dropout, Flatten, Dense, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class LivenessNet:
    """
    Enhanced CNN Architecture for Face Liveness Detection
    
    This network is designed specifically for distinguishing between
    real faces and various types of spoof attacks including:
    - Printed photos
    - Digital displays (phones, tablets, monitors)
    - Video replays
    - 3D masks (basic detection)
    """
    
    @staticmethod
    def build(width, height, depth, classes):
        """
        Build the CNN architecture for liveness detection
        
        Args:
            width (int): Input image width
            height (int): Input image height
            depth (int): Number of channels (3 for RGB)
            classes (int): Number of output classes (2: real/fake)
            
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1  # For channels_last format
        
        # First Convolutional Block
        # Small kernel size to capture fine-grained details
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second Convolutional Block
        # Increased filters to learn more complex patterns
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third Convolutional Block
        # Higher-level feature extraction
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Output layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model

class FaceProcessor:
    """
    Advanced face detection and preprocessing pipeline
    """
    
    def __init__(self):
        """Initialize face cascade classifier"""
        # Use DNN-based face detection for better accuracy
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load DNN face detector if available
        try:
            self.net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            self.use_dnn = True
        except:
            self.use_dnn = False
            print("DNN face detector not found, using Haar cascades")
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.5):
        """
        Detect faces using DNN-based detector
        
        Args:
            frame: Input image frame
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            list: List of face bounding boxes
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1-x, y1-y))
        
        return faces
    
    def detect_faces_haar(self, frame):
        """
        Detect faces using Haar cascade classifier
        
        Args:
            frame: Input image frame
            
        Returns:
            list: List of face bounding boxes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def detect_faces(self, frame):
        """
        Detect faces using the best available method
        
        Args:
            frame: Input image frame
            
        Returns:
            list: List of face bounding boxes
        """
        if self.use_dnn:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def preprocess_face(self, face_roi, target_size=(128, 128)):
        """
        Preprocess face ROI for model input
        
        Args:
            face_roi: Face region of interest
            target_size: Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed face array
        """
        # Resize to target size
        face = cv2.resize(face_roi, target_size)
        
        # Apply Gaussian blur to reduce noise
        face = cv2.GaussianBlur(face, (3, 3), 0)
        
        # Normalize pixel values
        face = face.astype("float32") / 255.0
        
        # Convert to array format expected by Keras
        face = img_to_array(face)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face

class LivenessDetector:
    """
    Main class for real-time face liveness detection
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize the liveness detector
        
        Args:
            model_path (str): Path to pre-trained model
            confidence_threshold (float): Minimum confidence for classification
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.face_processor = FaceProcessor()
        self.model = None
        self.is_running = False
        
        # Smoothing parameters
        self.prediction_history = deque(maxlen=10)
        self.smoothing_factor = 0.4
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. You'll need to train a model first.")
    
    def load_model(self, model_path):
        """
        Load pre-trained model
        
        Args:
            model_path (str): Path to model file
        """
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def create_model(self, input_shape=(128, 128, 3), num_classes=2):
        """
        Create a new model for training
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of classes
        """
        self.model = LivenessNet.build(
            width=input_shape[1],
            height=input_shape[0], 
            depth=input_shape[2],
            classes=num_classes
        )
        
        # Compile model
        opt = Adam(learning_rate=0.001, decay=0.001/25)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"]
        )
        
        print("New model created successfully")
        return self.model
    
    def predict_liveness(self, face_roi):
        """
        Predict if a face is real or fake
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None:
            return "No Model", 0.0
        
        # Preprocess face
        processed_face = self.face_processor.preprocess_face(face_roi)
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_face, verbose=0)[0]
            
            # Get confidence scores
            real_confidence = prediction[0] if len(prediction) > 1 else 1 - prediction[0]
            fake_confidence = prediction[1] if len(prediction) > 1 else prediction[0]
            
            # Apply smoothing
            self.prediction_history.append([real_confidence, fake_confidence])
            
            if len(self.prediction_history) > 1:
                # Exponential moving average
                smoothed = np.mean(self.prediction_history, axis=0)
                real_confidence, fake_confidence = smoothed
            
            # Determine final prediction
            if real_confidence > fake_confidence and real_confidence > self.confidence_threshold:
                return "REAL", real_confidence
            elif fake_confidence > self.confidence_threshold:
                return "FAKE", fake_confidence
            else:
                return "UNCERTAIN", max(real_confidence, fake_confidence)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "ERROR", 0.0
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time
        
        if elapsed_time >= 1.0:
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_detection_info(self, frame, faces, predictions):
        """
        Draw detection information on frame
        
        Args:
            frame: Input frame
            faces: List of detected faces
            predictions: List of predictions for each face
        """
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Instructions:",
            "- Look directly at camera",
            "- Keep face well lit",
            "- Stay within detection area",
            "- Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 120 + (i * 25)
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detection box area
        frame_height, frame_width = frame.shape[:2]
        box_width, box_height = 400, 300
        box_x = (frame_width - box_width) // 2
        box_y = (frame_height - box_height) // 2
        
        cv2.rectangle(frame, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     (255, 255, 0), 2)
        
        cv2.putText(frame, "Keep face in this area", 
                   (box_x, box_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw face detections and predictions
        for i, (x, y, w, h) in enumerate(faces):
            if i < len(predictions):
                label, confidence = predictions[i]
                
                # Choose color based on prediction
                if label == "REAL":
                    color = (0, 255, 0)  # Green
                elif label == "FAKE":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 165, 255)  # Orange
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw prediction label and confidence
                text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
                cv2.putText(frame, text, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def run_detection(self, camera_index=0, window_name="Liveness Detection"):
        """
        Run real-time liveness detection
        
        Args:
            camera_index (int): Camera device index
            window_name (str): OpenCV window name
        """
        if self.model is None:
            print("Error: No model loaded. Cannot run detection.")
            return
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Starting liveness detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        self.is_running = True
        self.fps_start_time = time.time()
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.face_processor.detect_faces(frame)
                
                # Make predictions for each face
                predictions = []
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict liveness
                    prediction, confidence = self.predict_liveness(face_roi)
                    predictions.append((prediction, confidence))
                
                # Draw detection information
                self.draw_detection_info(frame, faces, predictions)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"liveness_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped")

def create_demo_data():
    """
    Create demo data for testing (you would replace this with real data collection)
    """
    print("Demo data creation would go here...")
    print("In a real implementation, you would:")
    print("1. Collect real face images from webcam")
    print("2. Collect fake face images (photos, screens)")
    print("3. Organize data into train/validation sets") 
    print("4. Train the model with proper data augmentation")

def train_model(data_path, model_save_path="liveness_model.h5"):
    """
    Train liveness detection model
    
    Args:
        data_path (str): Path to training data directory
        model_save_path (str): Path to save trained model
    """
    print(f"Training model with data from: {data_path}")
    
    # This is a template - you would implement actual training here
    detector = LivenessDetector()
    model = detector.create_model()
    
    # Training would go here with real data
    print("Training implementation would go here...")
    print("Steps would include:")
    print("1. Load and preprocess training data")
    print("2. Split into train/validation sets")
    print("3. Apply data augmentation")
    print("4. Train model with early stopping")
    print("5. Evaluate model performance")
    print("6. Save best model")
    
    # model.save(model_save_path)
    print(f"Model would be saved to: {model_save_path}")

def main():
    """
    Main function to run the liveness detection system
    """
    print("=" * 60)
    print("Advanced Real-time Face Liveness Detection System")
    print("=" * 60)
    
    # Configuration
    model_path = "liveness_model.h5"  # Path to pre-trained model
    
    try:
        # Initialize detector
        detector = LivenessDetector(model_path=model_path)
        
        # If no model exists, create a demo model for testing
        if detector.model is None:
            print("Creating demo model for testing...")
            detector.create_model()
            print("Demo model created. For best results, train with real data.")
        
        # Run real-time detection
        detector.run_detection()
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Working webcam")
        print("2. Required dependencies installed")
        print("3. Proper model file (if using pre-trained)")

if __name__ == "__main__":
    main()