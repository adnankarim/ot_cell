"""
training/feature_utils.py
--------------------------
Feature extraction helpers for OT pairing in CellFlux.

Centralises the choice of feature space so OTMatcher stays agnostic
to how features are produced. Add new modes here as needed (e.g.
backbone embeddings for Phase 3 latent flow).
"""

import torch


def get_ot_features(x: torch.Tensor, mode: str = "pooled_image") -> torch.Tensor:
    """
    Extract a 2-D feature matrix from a batch of images for use in OT cost computation.

    Args:
        x:    Image batch of shape [B, C, H, W] on any device.
        mode: Feature extraction strategy.
              - ``'pooled_image'`` (default): global average pool over spatial dims → [B, C].
                Fast and memory-friendly; works well as a first approximation.
              - ``'flatten'``:  flatten all dims → [B, C*H*W].
                Exact pixel-level cost; slow for large images, mostly for debugging.

    Returns:
        Feature tensor of shape [B, D] on the same device as ``x``.
    """
    if mode == "pooled_image":
        return x.mean(dim=(2, 3)).float()          # [B, C]
    elif mode == "flatten":
        return x.reshape(x.size(0), -1).float()    # [B, C*H*W]
    else:
        raise ValueError(
            f"Unknown OT feature mode: '{mode}'. "
            "Choose from: 'pooled_image', 'flatten'."
        )
