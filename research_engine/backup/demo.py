#!/usr/bin/env python3
"""
Demo script untuk testing Face Authentication System
"""

import cv2
import numpy as np
from face_auth_system import SecureFaceAuth
import time

def test_camera():
    """Test kamera dan tampilkan preview"""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Tidak dapat membuka kamera!")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Kamera berhasil dibuka!")
    print("Tekan 'q' untuk keluar dari preview")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def demo_liveness_detection():
    """Demo liveness detection with debug info"""
    print("\n" + "="*50)
    print("DEMO: LIVENESS DETECTION (DEBUG MODE)")
    print("="*50)
    print("Instruksi:")
    print("1. Lihat ke kamera")
    print("2. Berkedip beberapa kali")
    print("3. Sistem akan mendeteksi kedipan Anda")
    print("4. Tekan 'q' untuk keluar, 'd' untuk toggle debug landmarks")
    
    auth_system = SecureFaceAuth()
    show_debug = True
    
    try:
        auth_system.initialize_camera()
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 60:  # 60 detik demo
            ret, frame = auth_system.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            frame_count += 1
            
            # Deteksi liveness
            is_face_detected, ear = auth_system.liveness_detector.detect_blink(frame)
            
            # Get debug info
            debug_info = auth_system.liveness_detector.get_debug_info()
            
            # Draw debug landmarks if enabled
            if show_debug and is_face_detected:
                # Get face landmarks for visualization
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = auth_system.liveness_detector.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        auth_system.liveness_detector.draw_eye_landmarks(display_frame, face_landmarks)
            
            # Status display with more info
            y_offset = 30
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(display_frame, f"Blinks: {debug_info['total_blinks']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(display_frame, f"EAR: {ear:.4f} (Threshold: {debug_info['ear_threshold']})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(display_frame, f"Blink Counter: {debug_info['blink_counter']}/{debug_info['consecutive_frames']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(display_frame, f"Face Detected: {'YES' if is_face_detected else 'NO'}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if is_face_detected else (0, 0, 255), 2)
            
            y_offset += 25
            if auth_system.liveness_detector.is_live():
                cv2.putText(display_frame, "LIVE DETECTED!", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Blink to prove liveness", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw EAR threshold line (visual indicator)
            if ear > 0:
                bar_x = display_frame.shape[1] - 150
                bar_y = 50
                bar_height = 200
                
                # Background bar
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_height), (50, 50, 50), -1)
                
                # Current EAR level
                ear_normalized = min(1.0, ear / 0.5)  # Normalize to 0-1 range
                ear_level = int(ear_normalized * bar_height)
                cv2.rectangle(display_frame, (bar_x, bar_y + bar_height - ear_level), 
                             (bar_x + 20, bar_y + bar_height), (0, 255, 0), -1)
                
                # Threshold line
                threshold_level = int((debug_info['ear_threshold'] / 0.5) * bar_height)
                cv2.line(display_frame, (bar_x - 5, bar_y + bar_height - threshold_level), 
                        (bar_x + 25, bar_y + bar_height - threshold_level), (0, 0, 255), 2)
                
                cv2.putText(display_frame, "EAR", (bar_x - 10, bar_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(display_frame, "Press 'q' to quit, 'd' to toggle debug", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Liveness Detection Demo (Debug)', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug landmarks: {'ON' if show_debug else 'OFF'}")
        
        print(f"\nDemo selesai!")
        print(f"Total kedipan terdeteksi: {debug_info['total_blinks']}")
        print(f"Total frame processed: {frame_count}")
        print(f"Debug info: {debug_info}")
        
    finally:
        auth_system.close()

def demo_face_detection():
    """Demo face detection dengan bounding box"""
    print("\n" + "="*50)
    print("DEMO: FACE DETECTION")
    print("="*50)
    print("Sistem akan mendeteksi wajah dan menampilkan bounding box")
    print("Tekan 'q' untuk keluar")
    
    auth_system = SecureFaceAuth()
    
    try:
        auth_system.initialize_camera()
        
        while True:
            ret, frame = auth_system.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Extract embedding dan bbox
            embedding, bbox = auth_system.embedding_system.extract_embedding(frame)
            
            if embedding is not None:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Deteksi obstacle
                obstacles = auth_system.obstacle_detector.detect_obstacles(frame, bbox)
                
                if obstacles:
                    cv2.putText(display_frame, f"Obstacles: {', '.join(obstacles)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "No obstacles detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Detection Demo', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        auth_system.close()

def main():
    """Main demo menu"""
    import logging
    
    # Setup logging level
    print("Set logging level:")
    print("1. ERROR (minimal)")
    print("2. INFO (normal)")
    print("3. DEBUG (verbose)")
    
    log_choice = input("Pilih level logging (1-3, default: 2): ").strip()
    
    if log_choice == '1':
        logging.getLogger().setLevel(logging.ERROR)
    elif log_choice == '3':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    while True:
        print("\n" + "="*50)
        print("FACE AUTHENTICATION SYSTEM - DEMO")
        print("="*50)
        print("1. Test Camera")
        print("2. Demo Liveness Detection (Debug Mode)")
        print("3. Demo Face Detection")
        print("4. Run Full System")
        print("5. Exit")
        print("="*50)
        
        choice = input("Pilih demo (1-5): ").strip()
        
        if choice == '1':
            test_camera()
            
        elif choice == '2':
            demo_liveness_detection()
            
        elif choice == '3':
            demo_face_detection()
            
        elif choice == '4':
            print("Memulai sistem penuh...")
            from face_auth_system import main as run_full_system
            run_full_system()
            
        elif choice == '5':
            print("Terima kasih!")
            break
            
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()