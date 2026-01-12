"""
Main passive liveness pipeline (v4).
"""

from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch

from .attention_fusion import load_fusion_network, tensor_from_features
from .config import PassiveLivenessConfig
from .feature_extractors import (
    ArtifactDetector,
    BlinkDetector,
    DepthVariationAnalyzer,
    FaceObservation,
    FaceTracker,
    LightReflectionAnalyzer,
    MicroMovementAnalyzer,
    TextureAnalyzer,
    HeadMovementAnalyzer,
)


class PassiveLivenessDetector:
    """Orchestrates multi-cue passive liveness detection."""

    FEATURE_ORDER = [
        "texture",
        "blink",
        "movement",
        "head_movement",
        "reflection",
        "artifact",
        "depth",
    ]

    def __init__(
        self,
        config: Optional[PassiveLivenessConfig] = None,
        fusion_checkpoint: Optional[str] = None,
    ) -> None:
        self.config = config or PassiveLivenessConfig()
        self.device = torch.device(
            "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.face_tracker = FaceTracker(
            max_faces=self.config.mediapipe_max_num_faces,
            static_image_mode=self.config.mediapipe_static_image_mode,
            min_detection_confidence=self.config.mediapipe_min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe_min_tracking_confidence,
            insightface_model=self.config.insightface_model,
            use_gpu=self.config.use_gpu,
        )
        self.texture_analyzer = TextureAnalyzer(self.config.texture_patch_size)
        self.blink_detector = BlinkDetector(
            window_size=self.config.blink_window_size,
            ear_threshold=self.config.blink_ear_threshold,
            min_frames=self.config.blink_min_frames,
        )
        self.movement_analyzer = MicroMovementAnalyzer(
            window=self.config.movement_history_window
        )
        self.head_movement_analyzer = HeadMovementAnalyzer()
        self.light_analyzer = LightReflectionAnalyzer(self.config.reflection_hist_bins)
        yolo_weights = self.config.resolve_yolo_weights()
        self.artifact_detector = ArtifactDetector(
            confidence_threshold=self.config.artifact_detection_threshold,
            custom_labels=self.config.custom_labels,
            weights_path=str(yolo_weights) if yolo_weights else None,
        )
        self.depth_analyzer = DepthVariationAnalyzer()

        self.fusion = None
        if fusion_checkpoint:
            checkpoint_path = Path(fusion_checkpoint)
            self.fusion = load_fusion_network(
                feature_names=self.FEATURE_ORDER,
                checkpoint_path=checkpoint_path,
                hidden_dim=self.config.attention_hidden_dim,
                heads=self.config.attention_heads,
                dropout=self.config.attention_dropout,
                use_gpu=self.config.use_gpu,
            )
            if self.fusion is None:
                print("[PassiveLiveness] Attention fusion disabled. Using heuristic fusion only.")

        self.smooth_score: Optional[float] = None
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.config.log_history_size)
        self.last_inference_ts: float = 0.0
        self.live_streak: int = 0
        self.spoof_streak: int = 0

    def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
        """Compute liveness prediction for the given frame."""
        start_ts = time.time()
        observations = self.face_tracker.process(frame)
        if not observations:
            return {
                "status": "no_face",
                "probability": 0.0,
                "smoothed_probability": self.smooth_score or 0.0,
                "verdict": "no_face",
                "features": {},
                "latency_ms": (time.time() - start_ts) * 1000.0,
            }
        observation = observations[0]
        feature_map = self._compute_features(frame, observation)

        weighted_features = self._apply_feature_weights(feature_map)
        attention_probability: Optional[float] = None
        if self.fusion is not None:
            tensor = tensor_from_features(
                weighted_features,
                self.FEATURE_ORDER,
                device=self.device,
            )
            with torch.no_grad():
                attention_probability = float(self.fusion.predict_proba(tensor).cpu().item())

        prior_probability = self._compute_prior_score(feature_map, observation)
        spoof_risk = self._compute_spoof_risk(feature_map)
        immediate_spoof = self.artifact_detector.last_immediate_spoof
        if attention_probability is not None:
            probability = 0.7 * prior_probability + 0.3 * attention_probability
        else:
            probability = prior_probability

        self.smooth_score = (
            probability
            if self.smooth_score is None
            else self.config.smoothing_alpha * probability
            + (1.0 - self.config.smoothing_alpha) * self.smooth_score
        )

        verdict = self._decide(
            smoothed_score=self.smooth_score,
            spoof_risk=spoof_risk,
            immediate_spoof=immediate_spoof,
            feature_map=feature_map,
        )
        latency = (time.time() - start_ts) * 1000.0

        result = {
            "status": "ok",
            "probability": probability,
            "prior_probability": prior_probability,
            "smoothed_probability": self.smooth_score,
            "verdict": verdict,
             "spoof_risk": spoof_risk,
            "immediate_spoof_flag": immediate_spoof,
            "features": feature_map,
            "weighted_features": weighted_features,
            "latency_ms": latency,
            "observation": {
                "bbox": observation.bbox,
                "quality_score": observation.quality_score,
                "timestamp": observation.timestamp,
            },
        }
        if attention_probability is not None:
            result["attention_probability"] = attention_probability

        if self.config.enable_logging:
            history_entry = {
                "timestamp": observation.timestamp,
                "probability": probability,
                "smoothed_probability": self.smooth_score,
                "prior_probability": prior_probability,
                "spoof_risk": spoof_risk,
                "immediate_spoof": immediate_spoof,
                **weighted_features,
            }
            if attention_probability is not None:
                history_entry["attention_probability"] = attention_probability
            self.history.append(history_entry)
        self.last_inference_ts = observation.timestamp
        return result

    def _compute_features(
        self,
        frame: np.ndarray,
        observation: FaceObservation,
    ) -> Dict[str, float]:
        """Compute raw feature scores with exception isolation."""
        features: Dict[str, float] = {}
        try:
            features["texture"] = self.texture_analyzer.compute_score(frame, observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Texture analyzer failed: {exc}")
            features["texture"] = 0.0

        try:
            features["blink"] = self.blink_detector.update(observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Blink detector failed: {exc}")
            features["blink"] = 0.0

        try:
            features["movement"] = self.movement_analyzer.compute_score(frame, observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Movement analyzer failed: {exc}")
            features["movement"] = 0.0

        try:
            features["head_movement"] = self.head_movement_analyzer.update(observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Head movement analyzer failed: {exc}")
            features["head_movement"] = 0.0

        try:
            features["reflection"] = self.light_analyzer.compute_score(frame, observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Reflection analyzer failed: {exc}")
            features["reflection"] = 0.0

        try:
            features["artifact"] = self.artifact_detector.compute_score(frame, observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Artifact detector failed: {exc}")
            features["artifact"] = 0.0

        quality_norm = 0.0
        if observation.quality_score > 0.0:
            quality_norm = float(np.clip(observation.quality_score / 35.0, 0.0, 1.5))
        features["quality"] = quality_norm

        try:
            features["depth"] = self.depth_analyzer.compute_score(observation)
        except Exception as exc:
            print(f"[PassiveLiveness] Depth analyzer failed: {exc}")
            features["depth"] = 0.0

        return features

    def _apply_feature_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale features using configuration weights."""
        return {
            "texture": features.get("texture", 0.0) * self.config.texture_score_weight,
            "blink": features.get("blink", 0.0) * self.config.blink_score_weight,
            "movement": features.get("movement", 0.0) * self.config.movement_score_weight,
            "head_movement": features.get("head_movement", 0.0) * self.config.head_movement_score_weight,
            "reflection": features.get("reflection", 0.0) * self.config.reflection_score_weight,
            "artifact": features.get("artifact", 0.0) * self.config.artifact_score_weight,
            "depth": features.get("depth", 0.0) * self.config.depth_score_weight,
        }

    def _compute_prior_score(
        self,
        features: Dict[str, float],
        observation: FaceObservation,
    ) -> float:
        """Heuristic fusion balancing pro-live cues against spoof penalties."""
        texture_raw = float(np.clip(features.get("texture", 0.0), 0.0, 1.6))
        blink_raw = float(np.clip(features.get("blink", 0.0), 0.0, 2.0))
        movement_raw = float(np.clip(features.get("movement", 0.0), 0.0, 2.0))
        reflection_raw = float(np.clip(features.get("reflection", 0.0), 0.0, 1.6))
        artifact_raw = float(np.clip(features.get("artifact", 0.0), 0.0, 3.0))
        quality_raw = float(np.clip(features.get("quality", 0.0), 0.0, 1.6))
        depth_raw = float(np.clip(features.get("depth", 0.0), 0.0, 2.0))
        head_movement_raw = float(np.clip(features.get("head_movement", 0.0), 0.0, 2.0))

        texture_live = np.clip((texture_raw - 0.25) / 0.8, 0.0, 1.2)
        blink_live = np.clip((blink_raw - 0.05) / 0.35, 0.0, 1.2)
        movement_live = np.clip((movement_raw - 0.04) / 0.3, 0.0, 1.2)
        quality_live = np.clip((quality_raw - 0.1) / 0.5, 0.0, 1.2)
        depth_live = np.clip((depth_raw - 0.25) / 0.6, 0.0, 1.2)
        head_live = np.clip((head_movement_raw - 0.05) / 0.35, 0.0, 1.2)

        blink_low = np.clip((0.08 - blink_raw) / 0.08, 0.0, 1.0)
        stillness = np.clip((0.15 - movement_raw) / 0.15, 0.0, 1.0)
        head_still = np.clip((0.12 - head_movement_raw) / 0.12, 0.0, 1.0)
        reflection_high = np.clip((reflection_raw - 0.35) / 0.5, 0.0, 1.0)
        artifact_norm = np.clip(artifact_raw / 0.8, 0.0, 1.0)
        depth_flat = np.clip((0.4 - depth_raw) / 0.4, 0.0, 1.0)

        live_strength = (
            0.35 * texture_live
            + 0.45 * blink_live
            + 0.4 * movement_live
            + 0.2 * quality_live
            + 0.6 * depth_live
            + 0.45 * head_live
        )
        spoof_strength = (
            1.4 * artifact_norm
            + 0.7 * reflection_high
            + 0.9 * stillness
            + 0.45 * blink_low
            + 0.9 * depth_flat
            + 0.6 * head_still
        )
        presence_bonus = 0.15 if observation is not None else 0.0

        margin = live_strength + presence_bonus - spoof_strength
        margin = float(np.clip(margin, -2.5, 2.5))
        probability = 1.0 / (1.0 + math.exp(-2.2 * margin))
        return float(np.clip(probability, 0.0, 1.0))

    def _compute_spoof_risk(self, features: Dict[str, float]) -> float:
        artifact = float(np.clip(features.get("artifact", 0.0), 0.0, 3.0))
        reflection = float(np.clip(features.get("reflection", 0.0), 0.0, 1.6))
        movement = float(np.clip(features.get("movement", 0.0), 0.0, 2.0))
        blink = float(np.clip(features.get("blink", 0.0), 0.0, 2.0))
        depth = float(np.clip(features.get("depth", 0.0), 0.0, 2.0))
        texture = float(np.clip(features.get("texture", 0.0), 0.0, 1.6))
        head = float(np.clip(features.get("head_movement", 0.0), 0.0, 2.0))

        artifact_penalty = artifact / 1.5
        flatness_penalty = np.clip((0.45 - depth) / 0.45, 0.0, 1.0)
        stillness_penalty = np.clip((0.18 - movement) / 0.18, 0.0, 1.0)
        blink_penalty = np.clip((0.1 - blink) / 0.1, 0.0, 1.0)
        reflection_penalty = np.clip((reflection - 0.35) / 0.5, 0.0, 1.0)
        texture_penalty = np.clip((0.35 - texture) / 0.35, 0.0, 1.0)
        head_penalty = np.clip((0.12 - head) / 0.12, 0.0, 1.0)

        spoof_risk = (
            0.45 * artifact_penalty
            + 0.25 * flatness_penalty
            + 0.25 * stillness_penalty
            + 0.15 * blink_penalty
            + 0.15 * reflection_penalty
            + 0.15 * texture_penalty
            + 0.2 * head_penalty
        )
        return float(np.clip(spoof_risk, 0.0, 1.0))

    def _decide(
        self,
        smoothed_score: Optional[float],
        spoof_risk: float,
        immediate_spoof: bool,
        feature_map: Dict[str, float],
    ) -> str:
        """Convert score and risk cues into categorical verdict with hysteresis."""
        if smoothed_score is None:
            smoothed_score = 0.0

        if immediate_spoof or spoof_risk >= self.config.spoof_risk_threshold:
            self.spoof_streak += 1
            self.live_streak = max(self.live_streak - 1, 0)
        else:
            self.spoof_streak = max(self.spoof_streak - 1, 0)

        if (
            smoothed_score >= self.config.positive_threshold
            and spoof_risk < self.config.spoof_risk_threshold * 0.8
            and feature_map.get("depth", 0.0) > 0.35
            and feature_map.get("head_movement", 0.0) > 0.15
        ):
            self.live_streak += 1
            self.spoof_streak = max(self.spoof_streak - 1, 0)
        else:
            self.live_streak = max(self.live_streak - 1, 0)

        if immediate_spoof or self.spoof_streak >= self.config.spoof_confirmation_frames:
            return "spoof"
        if self.live_streak >= self.config.live_confirmation_frames:
            return "live"

        if smoothed_score <= self.config.positive_threshold - self.config.uncertain_margin:
            return "spoof" if spoof_risk > 0.4 else "uncertain"
        if smoothed_score >= self.config.positive_threshold:
            return "live" if spoof_risk < 0.4 else "uncertain"
        return "uncertain"

    def reset(self) -> None:
        """Clear internal state and history."""
        self.smooth_score = None
        self.history.clear()
        self.movement_analyzer.reset()
        self.head_movement_analyzer.reset()
        self.blink_detector.ear_window.clear()
        self.blink_detector.last_state = False
        self.blink_detector.blink_count = 0
        self.blink_detector.start_time = 0.0
        self.live_streak = 0
        self.spoof_streak = 0

    def close(self) -> None:
        """Release resources."""
        self.face_tracker.close()

    def export_history(self) -> Dict[str, object]:
        """Return diagnostic history."""
        return {
            "config": asdict(self.config),
            "history": list(self.history),
            "last_inference_ts": self.last_inference_ts,
        }
