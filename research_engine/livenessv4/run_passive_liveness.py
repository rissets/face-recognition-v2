#!/usr/bin/env python3
"""
CLI demo for passive liveness detection (v4).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2

from .config import PassiveLivenessConfig
from .pipeline import PassiveLivenessDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive liveness detection demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--headless", action="store_true", help="Print JSON lines without UI")
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="Optional YOLO weight path for artefact detection",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for the attention fusion network",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PassiveLivenessConfig()
    if args.yolo_weights:
        config.yolo_model_path = args.yolo_weights

    detector = PassiveLivenessDetector(config=config, fusion_checkpoint=args.fusion_checkpoint)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    running = True

    def shutdown_handler(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("Passive liveness detector started. Press 'q' to exit.")
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        result = detector.process_frame(frame)
        if args.headless:
            print(result, flush=True)
        else:
            _render_overlay(frame, result)
            cv2.imshow("Passive Liveness v4", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        time.sleep(0.01)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def _render_overlay(frame, result):
    verdict = result.get("verdict", "unknown")
    probability = result.get("smoothed_probability", 0.0)
    features = result.get("features", {})
    status_text = f"{verdict.upper()} ({probability:.2f})"
    color = (0, 200, 0) if verdict == "live" else (0, 0, 200) if verdict == "spoof" else (0, 200, 200)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    for idx, (name, score) in enumerate(features.items()):
        text = f"{name}: {score:.2f}"
        cv2.putText(frame, text, (20, 80 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    bbox = result.get("observation", {}).get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


if __name__ == "__main__":
    sys.exit(main())

