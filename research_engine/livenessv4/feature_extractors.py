"""
Feature extraction utilities for passive liveness detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    mp = None
    _MP_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis  # type: ignore

    _INSIGHTFACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FaceAnalysis = None
    _INSIGHTFACE_AVAILABLE = False

try:
    import torch
except Exception as exc:  # pragma: no cover - torch is hard requirement
    raise RuntimeError("PyTorch is required for livenessv4. Install torch first.") from exc


@dataclass
class FaceObservation:
    """Container for a single face observation."""

    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray
    mesh_landmarks: Optional[np.ndarray] = None
    insight_landmarks_3d: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    pose: Optional[np.ndarray] = None
    timestamp: float = 0.0


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Compute the eye aspect ratio given 6 mediapipe landmarks."""
    p2_minus_p6 = np.linalg.norm(landmarks[1] - landmarks[5])
    p3_minus_p5 = np.linalg.norm(landmarks[2] - landmarks[4])
    p1_minus_p4 = np.linalg.norm(landmarks[0] - landmarks[3])
    if p1_minus_p4 == 0:
        return 0.0
    return float((p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4))


class FaceTracker:
    """Detect and track faces using Mediapipe Face Mesh and InsightFace (optional)."""

    def __init__(
        self,
        max_faces: int = 1,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        insightface_model: str = "buffalo_l",
        use_gpu: bool = False,
    ) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError("mediapipe is required for FaceTracker but is not installed.")

        mesh = mp.solutions.face_mesh
        self._face_mesh = mesh.FaceMesh(
            max_num_faces=max_faces,
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=True,
        )
        self._max_faces = max_faces
        self._insightface = None
        if _INSIGHTFACE_AVAILABLE:
            self._insightface = FaceAnalysis(name=insightface_model, providers=["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"])
            self._insightface.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))

    def process(self, frame: np.ndarray) -> List[FaceObservation]:
        """Return face observations for the provided frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return []

        observations: List[FaceObservation] = []
        h, w, _ = frame.shape
        timestamp = time.time()
        insight_faces = []
        if self._insightface is not None:
            try:
                insight_faces = self._insightface.get(frame)
            except Exception:
                insight_faces = []

        for face_landmarks in results.multi_face_landmarks[: self._max_faces]:
            coords = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
            min_xy = coords[:, :2].min(axis=0)
            max_xy = coords[:, :2].max(axis=0)
            x1, y1 = np.clip(min_xy, 0, [w, h]).astype(int)
            x2, y2 = np.clip(max_xy, 0, [w, h]).astype(int)
            observation = FaceObservation(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                landmarks=coords[:, :2],
                mesh_landmarks=coords,
                timestamp=timestamp,
            )
            if insight_faces:
                matched_face = self._match_insight_face(observation.bbox, insight_faces)
                if matched_face is not None:
                    try:
                        if hasattr(matched_face, "embedding") and matched_face.embedding is not None:
                            observation.embedding = matched_face.embedding.copy()
                        elif hasattr(matched_face, "normed_embedding") and matched_face.normed_embedding is not None:
                            observation.embedding = matched_face.normed_embedding.copy()
                        if hasattr(matched_face, "det_score") and matched_face.det_score is not None:
                            observation.quality_score = float(matched_face.det_score)
                        elif observation.embedding is not None:
                            observation.quality_score = float(np.linalg.norm(observation.embedding))
                        if hasattr(matched_face, "pose") and matched_face.pose is not None:
                            observation.pose = np.array(matched_face.pose, dtype=np.float32)
                        if hasattr(matched_face, "landmark_3d_68") and matched_face.landmark_3d_68 is not None:
                            observation.insight_landmarks_3d = np.array(matched_face.landmark_3d_68, dtype=np.float32)
                    except Exception:
                        pass
            observation.pose = self._estimate_pose_from_mesh(coords)
            observations.append(observation)
        return observations

    def close(self) -> None:
        """Release mediapipe resources."""
        self._face_mesh.close()

    def _match_insight_face(self, bbox: Tuple[int, int, int, int], faces: List) -> Optional:
        """Match mediapipe face bbox with closest insightface detection."""
        if not faces:
            return None
        x1, y1, x2, y2 = bbox
        best_face = None
        best_score = -np.inf
        for face in faces:
            try:
                box = face.bbox.astype(int) if hasattr(face, "bbox") else None
            except Exception:
                box = None
            if box is None:
                continue
            fx1, fy1, fx2, fy2 = box
            inter_x1 = max(x1, fx1)
            inter_y1 = max(y1, fy1)
            inter_x2 = min(x2, fx2)
            inter_y2 = min(y2, fy2)
            inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
            area_a = (x2 - x1) * (y2 - y1) + 1e-6
            area_b = (fx2 - fx1) * (fy2 - fy1) + 1e-6
            iou = inter_area / (area_a + area_b - inter_area)
            if iou > best_score:
                best_score = iou
                best_face = face
        return best_face

    @staticmethod
    def _estimate_pose_from_mesh(mesh: np.ndarray) -> Optional[np.ndarray]:
        """Approximate head pose (yaw, pitch, roll in degrees) from mesh geometry."""
        if mesh.shape[0] < 468:
            return None
        try:
            left_eye = mesh[33]
            right_eye = mesh[263]
            left_cheek = mesh[234]
            right_cheek = mesh[454]
            nose_tip = mesh[1]
            chin = mesh[152]
            forehead = mesh[10]

            eye_vector = right_eye - left_eye
            cheek_vector = right_cheek - left_cheek
            face_vertical = chin - forehead

            yaw = float(np.degrees(np.arctan2(cheek_vector[2], cheek_vector[0] + 1e-6)))
            pitch = float(np.degrees(np.arctan2(face_vertical[2], face_vertical[1] + 1e-6)))
            roll = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0] + 1e-6)))

            # Normalise to manageable range
            yaw = np.clip(yaw, -45.0, 45.0)
            pitch = np.clip(pitch, -45.0, 45.0)
            roll = np.clip(roll, -45.0, 45.0)
            return np.array([yaw, pitch, roll], dtype=np.float32)
        except Exception:
            return None


class TextureAnalyzer:
    """Compute texture-based anti-spoofing measures using LBP and variance metrics."""

    def __init__(self, patch_size: int = 128) -> None:
        self._patch_size = patch_size

    def compute_score(self, frame: np.ndarray, observation: FaceObservation) -> float:
        x1, y1, x2, y2 = observation.bbox
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._patch_size, self._patch_size), interpolation=cv2.INTER_LINEAR)
        lbp_hist = self._local_binary_pattern(resized)
        variance_score = float(np.var(resized) / 255.0)
        entropy = self._shannon_entropy(resized)
        texture_score = float(0.5 * lbp_hist.mean() + 0.3 * variance_score + 0.2 * entropy)
        return float(np.clip(texture_score, 0.0, 1.5))

    def _local_binary_pattern(self, image: np.ndarray) -> np.ndarray:
        """Compute LBP histogram."""
        lbp = np.zeros_like(image, dtype=np.uint8)
        padded = np.pad(image, pad_width=1, mode="edge")
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                shifted = padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
                lbp = (lbp << 1) | (shifted >= image)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        return hist

    @staticmethod
    def _shannon_entropy(image: np.ndarray) -> float:
        hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-8)))


class BlinkDetector:
    """Estimate blink frequency using eye aspect ratio dynamics."""

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self, window_size: int = 32, ear_threshold: float = 0.21, min_frames: int = 2) -> None:
        self.window_size = window_size
        self.ear_threshold = ear_threshold
        self.min_frames = min_frames
        self.ear_window: Deque[float] = deque(maxlen=window_size)
        self.last_state = False
        self.blink_count = 0
        self.last_blink_ts = 0.0
        self.start_time = 0.0

    def update(self, observation: FaceObservation) -> float:
        if observation.mesh_landmarks is None:
            return 0.0
        left_eye = observation.mesh_landmarks[self.LEFT_EYE_IDX, :2]
        right_eye = observation.mesh_landmarks[self.RIGHT_EYE_IDX, :2]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_window.append(ear)
        if self.start_time == 0.0:
            self.start_time = observation.timestamp

        closed = ear < self.ear_threshold
        if closed and not self.last_state:
            self.last_state = True
            self.last_blink_ts = observation.timestamp
        elif not closed and self.last_state:
            duration = observation.timestamp - self.last_blink_ts
            if len(self.ear_window) >= self.min_frames and 0.05 < duration < 0.6:
                self.blink_count += 1
            self.last_state = False
        if not self.ear_window:
            return 0.0
        stability = np.std(np.array(self.ear_window))
        elapsed = max(observation.timestamp - self.start_time, 1.0)
        blink_rate = self.blink_count / elapsed
        score = float(np.clip(blink_rate * (1.0 - stability), 0.0, 2.0))
        return score


class MicroMovementAnalyzer:
    """Capture subtle facial motion via landmark displacement."""

    def __init__(self, window: int = 12) -> None:
        self.window = window
        self.landmark_history: Deque[np.ndarray] = deque(maxlen=window)
        self.time_history: Deque[float] = deque(maxlen=window)

    def compute_score(self, frame: np.ndarray, observation: FaceObservation) -> float:
        if observation.landmarks is None or observation.landmarks.size == 0:
            self.reset()
            return 0.0
        landmarks = observation.landmarks.astype(np.float32)
        normalized = self._normalize_landmarks(landmarks, observation.bbox)
        timestamp = observation.timestamp

        self.landmark_history.append(normalized)
        self.time_history.append(timestamp)

        if len(self.landmark_history) < 2:
            return 0.0

        displacements = []
        times = list(self.time_history)
        for idx in range(1, len(self.landmark_history)):
            prev = self.landmark_history[idx - 1]
            curr = self.landmark_history[idx]
            delta = np.linalg.norm(curr - prev, axis=1)
            displacements.append(np.median(delta))
        displacements = np.array(displacements, dtype=np.float32)

        dt = times[-1] - times[0]
        if dt <= 0.0:
            velocity = displacements[-1]
        else:
            velocity = float(np.sum(displacements) / dt)

        recent_motion = float(np.median(displacements[-3:])) if displacements.size >= 3 else float(displacements[-1])
        score = 0.6 * recent_motion + 0.4 * velocity
        return float(np.clip(score * 4.0, 0.0, 2.0))

    @staticmethod
    def _normalize_landmarks(landmarks: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        normalized = np.empty_like(landmarks)
        normalized[:, 0] = (landmarks[:, 0] - x1) / width
        normalized[:, 1] = (landmarks[:, 1] - y1) / height
        return normalized

    def reset(self) -> None:
        self.landmark_history.clear()
        self.time_history.clear()


class LightReflectionAnalyzer:
    """Assess specular highlights to differentiate 2D vs 3D surfaces."""

    def __init__(self, hist_bins: int = 32) -> None:
        self.hist_bins = hist_bins

    def compute_score(self, frame: np.ndarray, observation: FaceObservation) -> float:
        x1, y1, x2, y2 = observation.bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        specular_thresh = np.percentile(brightness, 90)
        specular_mask = brightness >= specular_thresh
        total_pixels = roi.shape[0] * roi.shape[1] + 1e-6
        specular_pixels = brightness[specular_mask]
        if specular_pixels.size == 0:
            return 0.05

        hist, _ = np.histogram(specular_pixels, bins=self.hist_bins, range=(0, 255), density=True)
        spread = float(np.std(hist))
        coverage = float(specular_pixels.size / total_pixels)
        mean_specular = float(np.mean(specular_pixels))

        mid_mask = (brightness > np.percentile(brightness, 60)) & (~specular_mask)
        mid_pixels = brightness[mid_mask]
        mean_mid = float(np.mean(mid_pixels)) if mid_pixels.size else 0.0
        contrast = (mean_specular - mean_mid) / 255.0
        low_sat_ratio = float(np.mean(saturation < 35))

        bright_mask = gray >= np.percentile(gray, 75)
        bright_ratio = float(np.mean(bright_mask))
        edges = cv2.Canny(gray, 40, 120)
        edge_strength = float(np.mean(edges > 0))

        # Rectangularity score for large bright regions (e.g., screens)
        rect_score = 0.0
        try:
            contours, _ = cv2.findContours(bright_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(largest))
                if area > 0:
                    rect = cv2.minAreaRect(largest)
                    w_rect, h_rect = rect[1]
                    rect_area = max(w_rect * h_rect, 1e-6)
                    rect_score = float(np.clip(area / rect_area, 0.0, 1.0)) * float(np.clip(rect_area / total_pixels, 0.0, 1.0))
        except Exception:
            rect_score = 0.0

        score = (
            coverage * 1.3
            + spread * 0.35
            + contrast * 1.5
            + low_sat_ratio * 0.7
            + bright_ratio * 0.9
            + rect_score * 1.2
            + edge_strength * 0.3
        )
        return float(np.clip(score, 0.0, 2.2))


class HeadMovementAnalyzer:
    """Measure head pose dynamics using InsightFace pose estimates."""

    def __init__(self, window: int = 15) -> None:
        self.window = window
        self.pose_history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=window)

    def update(self, observation: FaceObservation) -> float:
        if observation.pose is None:
            return 0.0
        pose = np.array(observation.pose, dtype=np.float32)
        self.pose_history.append((observation.timestamp, pose))
        if len(self.pose_history) < 2:
            return 0.0
        times = np.array([t for t, _ in self.pose_history], dtype=np.float32)
        poses = np.stack([p for _, p in self.pose_history])  # shape (n,3)
        dt = times[-1] - times[0]
        if dt <= 0.0:
            return 0.0
        diffs = np.diff(poses, axis=0)
        magnitudes = np.linalg.norm(diffs, axis=1)
        velocity = float(np.sum(magnitudes) / dt)
        avg_delta = float(np.mean(magnitudes))
        score = 0.6 * velocity / 6.0 + 0.4 * avg_delta / 5.0
        return float(np.clip(score, 0.0, 2.0))

    def reset(self) -> None:
        self.pose_history.clear()


class ArtifactDetector:
    """Detect spoofing artefacts like screens or masks using YOLO models."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        custom_labels: Optional[Dict[str, float]] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.custom_labels = custom_labels or {}
        self.weights_path = weights_path
        self.model = self._load_model()
        self.last_immediate_spoof = False

    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore

            if self.weights_path:
                return YOLO(self.weights_path)
            return YOLO("yolov8n.pt")
        except Exception:
            pass
        try:
            return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        except Exception:
            return None

    def compute_score(self, frame: np.ndarray, observation: FaceObservation) -> float:
        self.last_immediate_spoof = False
        if self.model is None:
            return 0.0
        detections = self._run_detection(frame)
        if not detections:
            return 0.0
        fx1, fy1, fx2, fy2 = observation.bbox
        face_area = max((fx2 - fx1) * (fy2 - fy1), 1)
        frame_h, frame_w = frame.shape[:2]
        face_center = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)

        penalty_score = 0.0
        for cls_name, confidence, bbox in detections:
            weight = self.custom_labels.get(cls_name, 0.0)
            if weight <= 0.0:
                continue
            immediate_spoof = False
            bonus = 0.0
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                det_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                overlap = self._iou((fx1, fy1, fx2, fy2), bbox)
                area_ratio = ((bx2 - bx1) * (by2 - by1)) / face_area
                area_ratio = float(np.clip(area_ratio, 0.0, 8.0))
                dist = np.linalg.norm(
                    [
                        (det_center[0] - face_center[0]) / frame_w,
                        (det_center[1] - face_center[1]) / frame_h,
                    ]
                )
                if cls_name in {"cell phone", "screen", "tablet"}:
                    if overlap > 0.1 or area_ratio > 0.25 or dist < 0.35:
                        immediate_spoof = True
                    bonus = area_ratio * 1.2 + (1.0 - dist) * 0.8 + overlap * 1.5
                elif cls_name in {"laptop", "tv"}:
                    bonus = area_ratio * 0.8 + overlap * 1.2
                else:
                    bonus = overlap * 0.8
            penalty = weight * confidence * (1.0 + bonus)
            if immediate_spoof:
                self.last_immediate_spoof = True
                return 3.0
            penalty_score = max(penalty_score, penalty)
        return float(np.clip(penalty_score, 0.0, 3.0))

    def _run_detection(self, image: np.ndarray) -> List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]]:
        results: List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]] = []
        if hasattr(self.model, "predict"):
            outcome = self.model.predict(image, imgsz=416, verbose=False)
            for prediction in outcome:
                boxes = getattr(prediction, "boxes", None)
                if boxes is None:
                    continue
                names = getattr(self.model, "names", {})
                for box in boxes:
                    conf = float(box.conf.item())
                    if conf < self.confidence_threshold:
                        continue
                    cls_idx = int(box.cls.item())
                    cls_name = names.get(cls_idx, str(cls_idx))
                    xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                    results.append((cls_name, conf, tuple(xyxy.tolist())))
        elif hasattr(self.model, "__call__"):
            outcome = self.model(image)
            names = self.model.names if hasattr(self.model, "names") else {}
            for pred in outcome.xyxy[0]:  # type: ignore[attr-defined]
                conf = float(pred[4].item())
                if conf < self.confidence_threshold:
                    continue
                cls_idx = int(pred[5].item())
                cls_name = names.get(cls_idx, str(cls_idx))
                bbox = tuple(pred[:4].int().tolist())
                results.append((cls_name, conf, bbox))
        return results

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(inter_x2 - inter_x1, 0)
        inter_h = max(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom


class DepthVariationAnalyzer:
    """Estimate 3D facial relief to detect flat spoof surfaces."""

    def compute_score(self, observation: FaceObservation) -> float:
        if observation.mesh_landmarks is None:
            return 0.0
        z = observation.mesh_landmarks[:, 2]
        z_centered = z - np.median(z)
        percentile_range = float(np.percentile(z_centered, 95) - np.percentile(z_centered, 5))
        local_std = float(np.std(z_centered))

        ridge_indices = [1, 9, 152, 234, 454, 10, 197, 50, 280]
        ridge_vals = z_centered[ridge_indices] if len(z_centered) > max(ridge_indices) else z_centered
        ridge_span = float(np.max(ridge_vals) - np.min(ridge_vals))

        insight_depth = 0.0
        if observation.insight_landmarks_3d is not None:
            insight_z = observation.insight_landmarks_3d[:, 2]
            insight_centered = insight_z - np.median(insight_z)
            insight_range = float(np.percentile(insight_centered, 95) - np.percentile(insight_centered, 5))
            insight_std = float(np.std(insight_centered))
            insight_depth = 0.6 * insight_range + 0.4 * insight_std

        depth_signal = 0.45 * percentile_range + 0.3 * ridge_span + 0.15 * local_std + 0.4 * insight_depth
        adjusted = max(depth_signal - 0.0015, 0.0)
        return float(np.clip(adjusted / 0.009, 0.0, 2.0))
