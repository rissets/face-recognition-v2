"""
Configuration objects for passive liveness detection (v4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PassiveLivenessConfig:
    """Aggregated configuration for the passive liveness pipeline."""

    use_gpu: bool = False
    mediapipe_static_image_mode: bool = False
    mediapipe_max_num_faces: int = 1
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5

    blink_window_size: int = 32
    blink_ear_threshold: float = 0.21
    blink_min_frames: int = 2
    blink_score_weight: float = 1.0

    texture_patch_size: int = 128
    texture_score_weight: float = 1.2

    movement_history_window: int = 12
    movement_score_weight: float = 1.1
    head_movement_score_weight: float = 1.0

    reflection_hist_bins: int = 32
    reflection_score_weight: float = 0.8

    artifact_detection_threshold: float = 0.25
    artifact_score_weight: float = 1.5

    depth_score_weight: float = 1.3

    smoothing_alpha: float = 0.4

    attention_hidden_dim: int = 128
    attention_heads: int = 4
    attention_dropout: float = 0.1

    positive_threshold: float = 0.5
    uncertain_margin: float = 0.15
    spoof_risk_threshold: float = 0.55
    live_confirmation_frames: int = 4
    spoof_confirmation_frames: int = 3

    enable_logging: bool = True
    log_history_size: int = 120

    insightface_model: str = "buffalo_l"
    yolo_model_path: Optional[Path] = None

    custom_labels: Dict[str, float] = field(
        default_factory=lambda: {
            "cell phone": 1.2,
            "screen": 1.1,
            "laptop": 0.8,
            "tv": 0.8,
            "mask": 0.7,
            "tablet": 1.0,
        }
    )

    def resolve_yolo_weights(self, base_dir: Optional[Path] = None) -> Optional[Path]:
        """Return the path to the YOLO weights if available."""
        if self.yolo_model_path is not None:
            return Path(self.yolo_model_path)
        if base_dir is None:
            return None
        candidate = base_dir / "models" / "yolo" / "yolov8n.pt"
        return candidate if candidate.exists() else None
