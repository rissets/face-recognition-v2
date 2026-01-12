#!/usr/bin/env python3
"""
Simple Data Collection Tool for Liveness Detection
=================================================

This tool helps collect training data for liveness detection by capturing
real and fake face images from webcam.
"""

import cv2
import os
import time
import numpy as np
from pathlib import Path
import json

class LivenessDataCollector:
    """
    Tool for collecting liveness detection training data
    """
    
    def __init__(self, output_dir="liveness_training_data"):
        """
        Initialize data collector
        
        Args:
            output_dir (str): Output directory for collected data
        """
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create directory structure
        self.create_directories()
        
        # Collection statistics
        self.stats = {
            'real_images': 0,
            'fake_images': 0,
            'session_start': time.time()
        }
        
        print(f"Data collector initialized")
        print(f"Output directory: {self.output_dir}")
    
    def create_directories(self):
        """Create necessary directories"""
        subdirs = [
            "train/real", "train/fake",
            "validation/real", "validation/fake", 
            "test/real", "test/fake"
        ]
        
        for subdir in subdirs:
            Path(f"{self.output_dir}/{subdir}").mkdir(parents=True, exist_ok=True)
        
        print("Directory structure created")
    
    def collect_real_faces(self, target_count=500):
        """
        Collect real face images
        
        Args:
            target_count (int): Target number of images to collect
        """
        print(f"\n📸 COLLECTING REAL FACES (Target: {target_count} images)")
        print("Instructions:")
        print("- Look directly at the camera")
        print("- Try different expressions and slight head movements")
        print("- Ensure good lighting")
        print("- Press SPACE to capture, Q to quit")
        print("- Images will be saved automatically when face is detected")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collected = 0
        last_capture_time = 0
        capture_interval = 0.5  # Minimum interval between captures
        
        try:
            while collected < target_count:
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
                
                # Draw detection box and info
                current_time = time.time()
                
                for (x, y, w, h) in faces:
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Auto-capture if enough time has passed
                    if current_time - last_capture_time > capture_interval:
                        # Extract and save face
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Quality check
                        if self.is_good_quality_face(face_roi):
                            # Save to train/validation/test based on count
                            subset = self.get_data_subset(collected, target_count)
                            filename = f"real_{int(current_time)}_{collected:04d}.jpg"
                            filepath = os.path.join(self.output_dir, subset, "real", filename)
                            
                            # Resize and save
                            face_resized = cv2.resize(face_roi, (128, 128))
                            cv2.imwrite(filepath, face_resized)
                            
                            collected += 1
                            last_capture_time = current_time
                            self.stats['real_images'] += 1
                            
                            print(f"✅ Captured real face {collected}/{target_count} -> {filepath}")
                
                # Draw UI
                self.draw_collection_ui(frame, "REAL FACES", collected, target_count)
                
                # Display
                cv2.imshow('Real Face Collection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Manual capture
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if self.is_good_quality_face(face_roi):
                            subset = self.get_data_subset(collected, target_count)
                            filename = f"real_manual_{int(current_time)}_{collected:04d}.jpg"
                            filepath = os.path.join(self.output_dir, subset, "real", filename)
                            
                            face_resized = cv2.resize(face_roi, (128, 128))
                            cv2.imwrite(filepath, face_resized)
                            
                            collected += 1
                            self.stats['real_images'] += 1
                            
                            print(f"📷 Manual capture {collected}/{target_count} -> {filepath}")
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Real face collection completed: {collected} images")
    
    def collect_fake_faces(self, target_count=500):
        """
        Collect fake face images (photos, screens, etc.)
        
        Args:
            target_count (int): Target number of images to collect
        """
        print(f"\n📱 COLLECTING FAKE FACES (Target: {target_count} images)")
        print("Instructions:")
        print("- Show photos on phone/tablet screen")
        print("- Use printed photos")
        print("- Display photos on computer monitor")
        print("- Try different angles and distances")
        print("- Press SPACE to capture, Q to quit")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        collected = 0
        last_capture_time = 0
        capture_interval = 0.5
        
        try:
            while collected < target_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(80, 80), maxSize=(500, 500)
                )
                
                # Draw detection and capture
                current_time = time.time()
                
                for (x, y, w, h) in faces:
                    # Draw face rectangle (red for fake)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Auto-capture
                    if current_time - last_capture_time > capture_interval:
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Save fake face
                        subset = self.get_data_subset(collected, target_count)
                        filename = f"fake_{int(current_time)}_{collected:04d}.jpg"
                        filepath = os.path.join(self.output_dir, subset, "fake", filename)
                        
                        face_resized = cv2.resize(face_roi, (128, 128))
                        cv2.imwrite(filepath, face_resized)
                        
                        collected += 1
                        last_capture_time = current_time
                        self.stats['fake_images'] += 1
                        
                        print(f"🔴 Captured fake face {collected}/{target_count} -> {filepath}")
                
                # Draw UI
                self.draw_collection_ui(frame, "FAKE FACES", collected, target_count, color=(0, 0, 255))
                
                # Display
                cv2.imshow('Fake Face Collection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Manual capture
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = frame[y:y+h, x:x+w]
                        
                        subset = self.get_data_subset(collected, target_count)
                        filename = f"fake_manual_{int(current_time)}_{collected:04d}.jpg"
                        filepath = os.path.join(self.output_dir, subset, "fake", filename)
                        
                        face_resized = cv2.resize(face_roi, (128, 128))
                        cv2.imwrite(filepath, face_resized)
                        
                        collected += 1
                        self.stats['fake_images'] += 1
                        
                        print(f"📷 Manual capture {collected}/{target_count} -> {filepath}")
        
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n✅ Fake face collection completed: {collected} images")
    
    def is_good_quality_face(self, face_roi):
        """
        Check if face image is good quality
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            bool: True if good quality
        """
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            return False
        
        # Check sharpness using Laplacian variance
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check brightness
        brightness = np.mean(gray)
        
        return laplacian_var > 50 and 30 < brightness < 220
    
    def get_data_subset(self, current_count, total_count):
        """
        Determine which subset (train/validation/test) to save to
        
        Args:
            current_count (int): Current number of collected images
            total_count (int): Total target count
            
        Returns:
            str: Subset name
        """
        # Split: 70% train, 20% validation, 10% test
        train_split = int(total_count * 0.7)
        val_split = int(total_count * 0.9)
        
        if current_count < train_split:
            return "train"
        elif current_count < val_split:
            return "validation"
        else:
            return "test"
    
    def draw_collection_ui(self, frame, mode, collected, target, color=(0, 255, 0)):
        """
        Draw collection UI on frame
        
        Args:
            frame: Input frame
            mode (str): Collection mode
            collected (int): Number collected
            target (int): Target count
            color (tuple): UI color
        """
        height, width = frame.shape[:2]
        
        # Progress bar
        progress = collected / target
        bar_width = 400
        bar_height = 20
        bar_x = (width - bar_width) // 2
        bar_y = 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), color, -1)
        
        # Text
        progress_text = f"{mode}: {collected}/{target} ({progress*100:.1f}%)"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, progress_text, (text_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Instructions
        instructions = [
            "SPACE: Manual capture",
            "Q: Quit collection",
            "Auto-capture when face detected"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 80 + (i * 25)
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def save_collection_stats(self):
        """Save collection statistics"""
        self.stats['session_duration'] = time.time() - self.stats['session_start']
        self.stats['total_images'] = self.stats['real_images'] + self.stats['fake_images']
        
        stats_path = os.path.join(self.output_dir, 'collection_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Collection statistics saved to: {stats_path}")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            'dataset_name': 'Liveness Detection Training Data',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classes': ['real', 'fake'],
            'image_size': '128x128',
            'format': 'JPEG',
            'structure': {
                'train': 'Training data (70%)',
                'validation': 'Validation data (20%)', 
                'test': 'Test data (10%)'
            },
            'collection_stats': self.stats
        }
        
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Dataset info saved to: {info_path}")
    
    def run_collection_session(self, real_target=500, fake_target=500):
        """
        Run complete data collection session
        
        Args:
            real_target (int): Target number of real face images
            fake_target (int): Target number of fake face images
        """
        print("=" * 70)
        print("🎯 LIVENESS DETECTION DATA COLLECTION SESSION")
        print("=" * 70)
        print(f"Target: {real_target} real faces + {fake_target} fake faces")
        print(f"Output: {self.output_dir}")
        print("=" * 70)
        
        try:
            # Collect real faces
            self.collect_real_faces(real_target)
            
            print("\n" + "="*50)
            print("Phase 1 completed! Now collecting fake faces...")
            print("Prepare photos, phone screens, or printed images")
            input("Press Enter when ready...")
            
            # Collect fake faces
            self.collect_fake_faces(fake_target)
            
            # Save statistics and info
            self.save_collection_stats()
            self.create_dataset_info()
            
            # Final summary
            print("\n" + "="*70)
            print("🎉 DATA COLLECTION COMPLETED!")
            print("="*70)
            print(f"Real faces collected: {self.stats['real_images']}")
            print(f"Fake faces collected: {self.stats['fake_images']}")
            print(f"Total images: {self.stats['total_images']}")
            print(f"Session duration: {self.stats['session_duration']:.1f} seconds")
            print(f"Data saved in: {self.output_dir}")
            print("="*70)
            print("Your data is ready for training!")
            print("Next step: Run the training script")
            
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    print("=" * 70)
    print("📸 LIVENESS DETECTION DATA COLLECTION TOOL")
    print("=" * 70)
    print("This tool will help you collect training data for liveness detection")
    print("You'll collect both REAL faces and FAKE faces (photos/screens)")
    print("=" * 70)
    
    # Configuration
    output_dir = "liveness_training_data"
    real_target = 500
    fake_target = 500
    
    print(f"Configuration:")
    print(f"- Output directory: {output_dir}")
    print(f"- Real face target: {real_target}")
    print(f"- Fake face target: {fake_target}")
    print(f"- Total target: {real_target + fake_target}")
    print()
    
    # Confirm to proceed
    response = input("Ready to start collection? (y/n): ")
    if response.lower() != 'y':
        print("Collection cancelled")
        return
    
    # Initialize collector
    collector = LivenessDataCollector(output_dir)
    
    # Run collection
    collector.run_collection_session(real_target, fake_target)

if __name__ == "__main__":
    main()