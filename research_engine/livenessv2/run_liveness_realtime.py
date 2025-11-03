#!/usr/bin/env python3
"""Run real-time liveness detection (wrapper)

Usage:
    python run_liveness_realtime.py [--detector cnn|fast|advanced] [--model PATH] [--camera IDX] [--test-load]

Options:
    --detector    Which detector to run. Defaults to 'cnn' (uses LivenessDetector).
    --model       Path to model file (for CNN/advanced). Defaults to 'models/best_model.h5'.
    --camera      Camera index (default 0).
    --test-load   Only attempt to load the model and print status, then exit.

This helper chooses between the provided detectors in this folder and runs
the real-time loop. It also provides a safe --test-load mode which verifies
that the trained model can be loaded successfully (helpful for CI/local checks)
without opening a GUI camera window.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def main():
    parser = argparse.ArgumentParser(description="Run liveness realtime detector")
    parser.add_argument('--detector', choices=['cnn', 'fast', 'advanced'], default='cnn')
    parser.add_argument('--model', default=os.path.join(ROOT, 'models', 'best_model.h5'))
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--test-load', action='store_true', help='Only test loading the model and exit')
    args = parser.parse_args()

    if args.detector == 'fast':
        try:
            from fast_liveness_detector import FastLivenessDetector
        except Exception as e:
            print(f"Error importing FastLivenessDetector: {e}")
            return

        detector = FastLivenessDetector(confidence_threshold=args.confidence)

        if args.test_load:
            print("Fast detector does not require external model. Ready to run.")
            return

        detector.run_detection(camera_index=args.camera)

    elif args.detector == 'advanced':
        try:
            from advanced_liveness_detector import RealTimeLivenessDetector
        except Exception as e:
            print(f"Error importing RealTimeLivenessDetector: {e}")
            return

        detector = RealTimeLivenessDetector(model_path=args.model, use_advanced_model=True)
        # Update confidence threshold
        detector.confidence_threshold = args.confidence

        if args.test_load:
            if detector.model is not None:
                print(f"Advanced model ready. Model parameters: {detector.model.count_params():,}")
            else:
                print("Advanced model not loaded or missing")
            return

        detector.run_detection(camera_index=args.camera)

    else:
        # Default: cnn-based LivenessDetector
        try:
            from liveness_detector import LivenessDetector
        except Exception as e:
            print(f"Error importing LivenessDetector: {e}")
            return

        detector = LivenessDetector(model_path=args.model, confidence_threshold=args.confidence)

        if args.test_load:
            if detector.model is not None:
                try:
                    print(f"Model loaded: {args.model}")
                    # Print a compact summary
                    try:
                        print(f"Model name: {detector.model.name}")
                        print(f"Total params: {detector.model.count_params():,}")
                    except Exception:
                        print("Loaded model (summary unavailable)")
                except Exception as e:
                    print(f"Error inspecting model: {e}")
            else:
                print(f"Model not found or failed to load: {args.model}")
            return

        detector.run_detection(camera_index=args.camera)

if __name__ == '__main__':
    main()
