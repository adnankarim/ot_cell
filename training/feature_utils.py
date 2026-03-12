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
              - ``'pooled_image'``: global average pool → [B, C].
                Fast and memory-friendly.
              - ``'pooled_mean_std'`` (recommended): per-channel mean + std → [B, 2C].
                Captures both brightness and spread; stronger signal for OT.
              - ``'flatten'``: full flattened image → [B, C*H*W].
                Exact pixel cost; very slow for large images, mainly for debugging.

    Returns:
        Feature tensor of shape [B, D] on the same device as ``x``.
    """
    if mode == "pooled_image":
        return x.mean(dim=(2, 3)).float()                      # [B, C]

    elif mode == "pooled_mean_std":
        mean = x.mean(dim=(2, 3))                              # [B, C]
        std = x.std(dim=(2, 3), unbiased=False)                # [B, C]
        return torch.cat([mean, std], dim=1).float()           # [B, 2C]

    elif mode == "flatten":
        return x.reshape(x.size(0), -1).float()                # [B, C*H*W]

    else:
        raise ValueError(
            f"Unknown OT feature mode: '{mode}'. "
            "Choose from: 'pooled_image', 'pooled_mean_std', 'flatten'."
        )
