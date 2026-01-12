#!/usr/bin/env python3
"""
Real-time Liveness Detection Demo
================================

Simple demo application to test the real-time liveness detection system.
"""

import cv2
import time
import logging
from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LivenessDemo:
    """Demo application for liveness detection"""
    
    def __init__(self, camera_index=0, strict_mode=False):
        """Initialize demo"""
        self.camera_index = camera_index
        self.strict_mode = strict_mode
        
        # Create detector configuration
        self.config = create_detector_config(
            strict_mode=strict_mode,
            enable_challenges=True,
            min_blinks=3 if strict_mode else 2,
            liveness_threshold=0.8 if strict_mode else 0.7
        )
        
        # Initialize detector
        self.detector = RealtimeLivenessDetector(self.config)
        
        # Demo state
        self.detection_sessions = []
        self.current_session_start = None
        
    def run(self):
        """Run the demo"""
        logger.info("Starting Liveness Detection Demo")
        logger.info(f"Strict mode: {self.strict_mode}")
        logger.info("Controls:")
        logger.info("  's' - Start new detection session")
        logger.info("  'r' - Reset current session")
        logger.info("  'q' - Quit")
        logger.info("  'h' - Show help")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._start_new_session()
                elif key == ord('r'):
                    self._reset_session()
                elif key == ord('h'):
                    self._show_help()
                
                # Process frame
                annotated_frame, analysis = self.detector.process_frame(frame)
                
                # Add demo info
                self._add_demo_info(annotated_frame, analysis)
                
                # Show frame
                cv2.imshow('Real-time Liveness Detection Demo', annotated_frame)
                
                # Check for session completion
                if not self.detector.detection_active and self.detector.final_result is not None:
                    self._handle_session_complete()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._show_session_summary()
            logger.info("Demo completed")
    
    def _start_new_session(self):
        """Start a new detection session"""
        if self.detector.detection_active:
            logger.warning("Detection already active. Reset first.")
            return
        
        self.detector.start_detection()
        self.current_session_start = time.time()
        logger.info("=== NEW DETECTION SESSION STARTED ===")
    
    def _reset_session(self):
        """Reset current session"""
        if self.detector.detection_active:
            self.detector.stop_detection()
        
        self.detector._reset_session()
        self.current_session_start = None
        logger.info("Session reset")
    
    def _handle_session_complete(self):
        """Handle completed detection session"""
        result = self.detector.final_result
        session_duration = time.time() - (self.current_session_start or time.time())
        
        # Store session result
        session_data = {
            'timestamp': time.time(),
            'duration': session_duration,
            'result': result,
            'strict_mode': self.strict_mode
        }
        self.detection_sessions.append(session_data)
        
        # Print detailed results
        print(f"\n{'='*50}")
        print(f"LIVENESS DETECTION COMPLETED")
        print(f"{'='*50}")
        print(f"Result: {'✅ LIVE PERSON' if result.is_live else '❌ FAKE/SPOOF'}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"\nScore Breakdown:")
        for metric, score in result.score_breakdown.items():
            print(f"  {metric}: {score:.3f}")
        
        print(f"\nChallenges:")
        print(f"  Passed: {result.challenges_passed}")
        print(f"  Failed: {result.challenges_failed}")
        
        print(f"\nFrame Analysis:")
        for key, value in result.frame_analysis.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        print(f"{'='*50}\n")
        
        # Reset for next session
        self.detector.final_result = None
        self.current_session_start = None
    
    def _add_demo_info(self, frame, analysis):
        """Add demo information to frame"""
        try:
            h, w = frame.shape[:2]
            
            # Add title
            cv2.putText(frame, "Real-time Liveness Detection Demo", 
                       (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add mode info
            mode_text = f"Mode: {'Strict' if self.strict_mode else 'Normal'}"
            cv2.putText(frame, mode_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add session count
            session_text = f"Sessions completed: {len(self.detection_sessions)}"
            cv2.putText(frame, session_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add status
            if self.detector.detection_active:
                status = self.detector.get_current_status()
                duration = status['session_duration']
                status_text = f"Active: {duration:.1f}s | Blinks: {status['total_blinks']}"
                cv2.putText(frame, status_text, (10, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add current challenge
                if status['current_challenge']:
                    challenge_text = f"Challenge: {status['current_challenge'].upper()}"
                    cv2.putText(frame, challenge_text, (10, h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Press 's' to start detection", (10, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add controls hint
            cv2.putText(frame, "Press 'h' for help", (w - 150, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        except Exception as e:
            logger.error(f"Error adding demo info: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("LIVENESS DETECTION DEMO - HELP")
        print("="*50)
        print("Controls:")
        print("  's' - Start new detection session")
        print("  'r' - Reset current session")
        print("  'q' - Quit demo")
        print("  'h' - Show this help")
        print("\nInstructions:")
        print("1. Position your face in the camera view")
        print("2. Press 's' to start detection")
        print("3. Follow the challenges that appear:")
        print("   - Blink naturally several times")
        print("   - Turn head left when prompted")
        print("   - Turn head right when prompted")
        print("   - Nod up and down when prompted")
        print("4. Wait for the result")
        print("\nTips:")
        print("- Ensure good lighting")
        print("- Keep your face clearly visible")
        print("- Move naturally, don't force movements")
        print("- Be patient, detection takes 5-15 seconds")
        print("="*50 + "\n")
    
    def _show_session_summary(self):
        """Show summary of all sessions"""
        if not self.detection_sessions:
            print("No detection sessions completed.")
            return
        
        print(f"\n{'='*60}")
        print(f"DEMO SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total sessions: {len(self.detection_sessions)}")
        
        live_count = sum(1 for s in self.detection_sessions if s['result'].is_live)
        fake_count = len(self.detection_sessions) - live_count
        
        print(f"Live detections: {live_count}")
        print(f"Fake detections: {fake_count}")
        
        if self.detection_sessions:
            avg_confidence = sum(s['result'].confidence for s in self.detection_sessions) / len(self.detection_sessions)
            avg_duration = sum(s['duration'] for s in self.detection_sessions) / len(self.detection_sessions)
            
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Average duration: {avg_duration:.1f}s")
        
        print(f"{'='*60}\n")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Liveness Detection Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--strict', action='store_true', help='Enable strict detection mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run demo
    demo = LivenessDemo(
        camera_index=args.camera,
        strict_mode=args.strict
    )
    
    demo.run()

if __name__ == "__main__":
    main()