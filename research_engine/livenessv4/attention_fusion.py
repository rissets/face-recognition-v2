"""
Attention-based fusion network for passive liveness features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


class AttentionFusionNetwork(nn.Module):
    """Fuse heterogeneous features using a lightweight transformer block."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.value_proj = nn.Linear(1, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_features, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch, num_features).

        Returns:
            Tensor of shape (batch,) containing logits.
        """
        if features.dim() != 2:
            raise ValueError("features tensor must be of shape (batch, num_features)")
        if features.shape[1] != self.num_features:
            raise ValueError(f"expected num_features={self.num_features}, got {features.shape[1]}")

        tokens = features.unsqueeze(-1)
        tokens = self.value_proj(tokens) + self.position_embedding[:, : tokens.shape[1], :]

        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm1(tokens + self.dropout(attn_out))

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_out))

        pooled = tokens.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability for convenience."""
        logits = self.forward(features)
        return torch.sigmoid(logits)


def tensor_from_features(
    feature_map: Dict[str, float],
    feature_order: Iterable[str],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert feature dict to tensor respecting a fixed order."""
    vector: List[float] = [float(feature_map.get(name, 0.0)) for name in feature_order]
    tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def load_fusion_network(
    feature_names: Iterable[str],
    checkpoint_path: Optional[Path],
    hidden_dim: int,
    heads: int,
    dropout: float,
    use_gpu: bool,
) -> Optional[AttentionFusionNetwork]:
    """Instantiate the fusion network and load weights if available."""
    if not checkpoint_path:
        return None

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[AttentionFusion] Checkpoint not found at {ckpt_path}. Falling back to heuristic fusion.")
        return None

    feature_names = list(feature_names)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = AttentionFusionNetwork(
        num_features=len(feature_names),
        hidden_dim=hidden_dim,
        num_heads=heads,
        dropout=dropout,
    )
    model.to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[AttentionFusion] Loaded with missing={missing} unexpected={unexpected}")

    model.eval()
    return model
