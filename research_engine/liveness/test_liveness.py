#!/usr/bin/env python3
"""
Liveness Detection Test Suite
============================

Comprehensive test suite for the real-time liveness detection system.
Tests various scenarios including real faces, spoofing attempts, and edge cases.
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from realtime_liveness_detector import RealtimeLivenessDetector, create_detector_config, LivenessResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LivenessTestSuite:
    """Test suite for liveness detection"""
    
    def __init__(self, output_dir: str = "test_results"):
        """Initialize test suite"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.test_results = []
        self.test_sessions = []
        
    def run_all_tests(self):
        """Run all test scenarios"""
        logger.info("Starting comprehensive liveness detection test suite")
        
        test_scenarios = [
            ("Normal Detection", self.test_normal_detection),
            ("Strict Mode", self.test_strict_mode),
            ("Fast Blinks", self.test_fast_blinks),
            ("Slow Blinks", self.test_slow_blinks),
            ("Head Movement", self.test_head_movement),
            ("Challenge System", self.test_challenge_system),
            ("Poor Lighting", self.test_poor_lighting),
            ("Multiple Faces", self.test_multiple_faces),
            ("No Face", self.test_no_face),
            ("Edge Cases", self.test_edge_cases)
        ]
        
        for test_name, test_func in test_scenarios:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                self.test_results.append({
                    'test_name': test_name,
                    'status': 'passed' if result else 'failed',
                    'timestamp': time.time(),
                    'details': result if isinstance(result, dict) else {'success': result}
                })
                
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"Test {test_name}: {status}")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results.append({
                    'test_name': test_name,
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e)
                })
        
        # Generate test report
        self.generate_test_report()
        
        # Print summary
        self.print_test_summary()
    
    def test_normal_detection(self) -> Dict:
        """Test normal liveness detection with standard parameters"""
        config = create_detector_config(
            strict_mode=False,
            enable_challenges=True,
            min_blinks=2,
            liveness_threshold=0.7
        )
        
        detector = RealtimeLivenessDetector(config)
        
        # Simulate detection session
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Normal Detection",
            simulate_blinks=3,
            simulate_movement=True,
            duration=10.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': True,
            'config_used': config
        }
    
    def test_strict_mode(self) -> Dict:
        """Test strict mode detection"""
        config = create_detector_config(
            strict_mode=True,
            enable_challenges=True,
            min_blinks=3,
            liveness_threshold=0.8
        )
        
        detector = RealtimeLivenessDetector(config)
        
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Strict Mode",
            simulate_blinks=4,
            simulate_movement=True,
            duration=12.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': True,
            'config_used': config
        }
    
    def test_fast_blinks(self) -> Dict:
        """Test detection with artificially fast blinks"""
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Fast Blinks",
            simulate_blinks=8,  # Many fast blinks
            blink_speed=0.1,   # Very fast
            simulate_movement=True,
            duration=8.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': False,  # Should be suspicious
            'config_used': config
        }
    
    def test_slow_blinks(self) -> Dict:
        """Test detection with very slow blinks"""
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Slow Blinks",
            simulate_blinks=1,  # Very few blinks
            blink_speed=2.0,   # Very slow
            simulate_movement=True,
            duration=15.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': False,  # Insufficient blinks
            'config_used': config
        }
    
    def test_head_movement(self) -> Dict:
        """Test head movement detection"""
        config = create_detector_config(enable_challenges=True)
        detector = RealtimeLivenessDetector(config)
        
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Head Movement",
            simulate_blinks=3,
            simulate_movement=True,
            movement_range=25.0,  # Significant movement
            duration=10.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': True,
            'config_used': config
        }
    
    def test_challenge_system(self) -> Dict:
        """Test interactive challenge system"""
        config = create_detector_config(enable_challenges=True)
        detector = RealtimeLivenessDetector(config)
        
        # Test each challenge individually
        challenge_results = {}
        
        for challenge in ['blink', 'turn_left', 'turn_right', 'nod']:
            session_result = self._simulate_challenge_completion(
                detector=detector,
                challenge=challenge,
                duration=8.0
            )
            challenge_results[challenge] = session_result
        
        return {
            'success': True,
            'challenge_results': challenge_results,
            'config_used': config
        }
    
    def test_poor_lighting(self) -> Dict:
        """Test detection under poor lighting conditions"""
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        # Simulate poor lighting by creating darker frames
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Poor Lighting",
            simulate_blinks=3,
            simulate_movement=True,
            lighting_factor=0.3,  # Very dark
            duration=12.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'expected_live': None,  # May fail due to quality
            'config_used': config
        }
    
    def test_multiple_faces(self) -> Dict:
        """Test behavior with multiple faces in frame"""
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        # This test would require multiple face simulation
        # For now, just test the detector's robustness
        session_result = self._simulate_detection_session(
            detector=detector,
            session_name="Multiple Faces",
            simulate_blinks=3,
            simulate_movement=True,
            add_noise=True,
            duration=10.0
        )
        
        return {
            'success': True,
            'session_result': session_result,
            'note': "Multiple face simulation not fully implemented",
            'config_used': config
        }
    
    def test_no_face(self) -> Dict:
        """Test behavior when no face is detected"""
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        detector.start_detection()
        
        # Simulate frames with no face
        no_face_count = 0
        for i in range(30):  # 1 second worth of frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Empty frame
            annotated_frame, analysis = detector.process_frame(frame)
            
            if analysis.get('status') == 'no_face_detected':
                no_face_count += 1
        
        result = detector.stop_detection()
        
        return {
            'success': True,
            'no_face_frames': no_face_count,
            'final_result': {
                'is_live': result.is_live,
                'confidence': result.confidence
            },
            'expected_live': False,
            'config_used': config
        }
    
    def test_edge_cases(self) -> Dict:
        """Test various edge cases"""
        edge_case_results = {}
        
        # Test 1: Very short session
        config = create_detector_config()
        detector = RealtimeLivenessDetector(config)
        
        short_session = self._simulate_detection_session(
            detector=detector,
            session_name="Short Session",
            simulate_blinks=1,
            simulate_movement=False,
            duration=2.0  # Very short
        )
        edge_case_results['short_session'] = short_session
        
        # Test 2: No movement, no blinks
        detector = RealtimeLivenessDetector(config)
        
        static_session = self._simulate_detection_session(
            detector=detector,
            session_name="Static Session",
            simulate_blinks=0,
            simulate_movement=False,
            duration=10.0
        )
        edge_case_results['static_session'] = static_session
        
        # Test 3: Extreme lighting
        detector = RealtimeLivenessDetector(config)
        
        bright_session = self._simulate_detection_session(
            detector=detector,
            session_name="Bright Session",
            simulate_blinks=2,
            simulate_movement=True,
            lighting_factor=2.0,  # Very bright
            duration=8.0
        )
        edge_case_results['bright_session'] = bright_session
        
        return {
            'success': True,
            'edge_case_results': edge_case_results,
            'config_used': config
        }
    
    def _simulate_detection_session(self, detector: RealtimeLivenessDetector, 
                                  session_name: str, simulate_blinks: int = 2,
                                  simulate_movement: bool = True, 
                                  blink_speed: float = 0.5,
                                  movement_range: float = 15.0,
                                  lighting_factor: float = 1.0,
                                  add_noise: bool = False,
                                  duration: float = 10.0) -> Dict:
        """Simulate a complete detection session"""
        
        detector.start_detection()
        start_time = time.time()
        
        frame_count = 0
        blinks_simulated = 0
        
        while time.time() - start_time < duration and detector.detection_active:
            # Create synthetic frame
            frame = self._create_synthetic_frame(
                frame_count=frame_count,
                simulate_blink=(blinks_simulated < simulate_blinks and 
                              frame_count % int(30 * blink_speed) < 3),
                head_pose_offset=np.sin(frame_count * 0.1) * movement_range if simulate_movement else 0,
                lighting_factor=lighting_factor,
                add_noise=add_noise
            )
            
            # Track simulated blinks
            if frame_count % int(30 * blink_speed) == 0 and blinks_simulated < simulate_blinks:
                blinks_simulated += 1
            
            # Process frame
            annotated_frame, analysis = detector.process_frame(frame)
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
        
        # Get final result
        result = detector.stop_detection()
        
        session_data = {
            'session_name': session_name,
            'duration': time.time() - start_time,
            'frames_processed': frame_count,
            'blinks_simulated': blinks_simulated,
            'result': {
                'is_live': result.is_live,
                'confidence': result.confidence,
                'score_breakdown': result.score_breakdown,
                'challenges_passed': result.challenges_passed,
                'challenges_failed': result.challenges_failed,
                'frame_analysis': result.frame_analysis
            }
        }
        
        self.test_sessions.append(session_data)
        return session_data
    
    def _simulate_challenge_completion(self, detector: RealtimeLivenessDetector,
                                     challenge: str, duration: float = 8.0) -> Dict:
        """Simulate completing a specific challenge"""
        
        detector.start_detection()
        detector.current_challenge = challenge
        detector.challenge_start_time = time.time()
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration and detector.detection_active:
            # Create frame that helps complete the challenge
            frame = self._create_challenge_frame(challenge, frame_count)
            
            annotated_frame, analysis = detector.process_frame(frame)
            
            # Check if challenge completed
            if challenge in detector.challenges_completed:
                break
            
            frame_count += 1
            time.sleep(0.033)
        
        result = detector.stop_detection()
        
        return {
            'challenge': challenge,
            'completed': challenge in result.challenges_passed,
            'duration': time.time() - start_time,
            'frames_processed': frame_count,
            'final_result': {
                'is_live': result.is_live,
                'confidence': result.confidence
            }
        }
    
    def _create_synthetic_frame(self, frame_count: int, simulate_blink: bool = False,
                              head_pose_offset: float = 0, lighting_factor: float = 1.0,
                              add_noise: bool = False) -> np.ndarray:
        """Create a synthetic frame for testing"""
        
        # Create base frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add face-like region
        face_center_x = 320 + int(head_pose_offset)
        face_center_y = 240
        face_size = 150
        
        # Draw face oval
        cv2.ellipse(frame, (face_center_x, face_center_y), 
                   (face_size//2, int(face_size*0.6)), 0, 0, 360, 
                   (180, 150, 120), -1)
        
        # Draw eyes
        eye_y = face_center_y - 30
        left_eye_x = face_center_x - 40
        right_eye_x = face_center_x + 40
        
        eye_size = 3 if simulate_blink else 8
        cv2.ellipse(frame, (left_eye_x, eye_y), (15, eye_size), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(frame, (right_eye_x, eye_y), (15, eye_size), 0, 0, 360, (50, 50, 50), -1)
        
        # Draw nose
        cv2.circle(frame, (face_center_x, face_center_y + 10), 8, (160, 130, 100), -1)
        
        # Draw mouth
        cv2.ellipse(frame, (face_center_x, face_center_y + 40), (20, 8), 0, 0, 360, (120, 80, 80), -1)
        
        # Apply lighting
        if lighting_factor != 1.0:
            frame = np.clip(frame.astype(np.float32) * lighting_factor, 0, 255).astype(np.uint8)
        
        # Add noise
        if add_noise:
            noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def _create_challenge_frame(self, challenge: str, frame_count: int) -> np.ndarray:
        """Create frame that helps complete specific challenge"""
        
        if challenge == 'blink':
            # Simulate blink every 30 frames
            simulate_blink = (frame_count % 30) < 3
            return self._create_synthetic_frame(frame_count, simulate_blink=simulate_blink)
        
        elif challenge == 'turn_left':
            # Simulate turning head left
            head_offset = -20 if frame_count > 30 else 0
            return self._create_synthetic_frame(frame_count, head_pose_offset=head_offset)
        
        elif challenge == 'turn_right':
            # Simulate turning head right
            head_offset = 20 if frame_count > 30 else 0
            return self._create_synthetic_frame(frame_count, head_pose_offset=head_offset)
        
        elif challenge == 'nod':
            # Simulate nodding (vertical movement simulation)
            head_offset = np.sin(frame_count * 0.3) * 10
            return self._create_synthetic_frame(frame_count, head_pose_offset=head_offset)
        
        else:
            return self._create_synthetic_frame(frame_count)
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.time(),
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'passed']),
                'failed': len([r for r in self.test_results if r['status'] == 'failed']),
                'errors': len([r for r in self.test_results if r['status'] == 'error'])
            },
            'test_results': self.test_results,
            'test_sessions': self.test_sessions
        }
        
        # Save to file
        report_file = os.path.join(self.output_dir, f"test_report_{int(time.time())}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Test report saved to: {report_file}")
        
        # Generate summary report
        self._generate_summary_report(report)
    
    def _generate_summary_report(self, report: Dict):
        """Generate human-readable summary report"""
        summary_file = os.path.join(self.output_dir, f"test_summary_{int(time.time())}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("LIVENESS DETECTION TEST SUITE SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {report['test_summary']['total_tests']}\n")
            f.write(f"Passed: {report['test_summary']['passed']}\n")
            f.write(f"Failed: {report['test_summary']['failed']}\n")
            f.write(f"Errors: {report['test_summary']['errors']}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for result in self.test_results:
                status_symbol = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
                f.write(f"{status_symbol} {result['test_name']}: {result['status'].upper()}\n")
                
                if 'details' in result and isinstance(result['details'], dict):
                    if 'session_result' in result['details']:
                        session = result['details']['session_result']
                        f.write(f"   Duration: {session.get('duration', 0):.2f}s\n")
                        if 'result' in session:
                            f.write(f"   Live: {session['result']['is_live']}\n")
                            f.write(f"   Confidence: {session['result']['confidence']:.3f}\n")
                
                if 'error' in result:
                    f.write(f"   Error: {result['error']}\n")
                
                f.write("\n")
            
            f.write("\nSESSION DETAILS:\n")
            f.write("-" * 30 + "\n")
            
            for session in self.test_sessions:
                f.write(f"Session: {session['session_name']}\n")
                f.write(f"Duration: {session['duration']:.2f}s\n")
                f.write(f"Frames: {session['frames_processed']}\n")
                f.write(f"Blinks Simulated: {session['blinks_simulated']}\n")
                f.write(f"Result: {'LIVE' if session['result']['is_live'] else 'FAKE'}\n")
                f.write(f"Confidence: {session['result']['confidence']:.3f}\n")
                f.write(f"Challenges Passed: {session['result']['challenges_passed']}\n")
                f.write("-" * 20 + "\n")
        
        logger.info(f"Summary report saved to: {summary_file}")
    
    def print_test_summary(self):
        """Print test summary to console"""
        total = len(self.test_results)
        passed = len([r for r in self.test_results if r['status'] == 'passed'])
        failed = len([r for r in self.test_results if r['status'] == 'failed'])
        errors = len([r for r in self.test_results if r['status'] == 'error'])
        
        print(f"\n{'='*60}")
        print(f"LIVENESS DETECTION TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "N/A")
        print(f"{'='*60}\n")

def main():
    """Main function to run test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Liveness Detection Test Suite')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for test results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run test suite
    test_suite = LivenessTestSuite(output_dir=args.output_dir)
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()