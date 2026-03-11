"""
OT Matcher for CellFlux — Milestone 1 (Phase 1: OT Pairing)

Replaces random control/treated pairing in the training loop with a
Sinkhorn Optimal Transport assignment. Requires `pip install pot`.

Usage (in train_loop.py):
    perm = ot_matcher.get_indices(x_ctrl, x_trt)
    x_trt_paired = x_trt[perm]
"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    import ot
except ImportError:
    raise ImportError(
        "The `pot` library is required for OT pairing. Install it with: pip install pot"
    )


class OTMatcher:
    """
    Computes an Optimal Transport assignment between a batch of control images
    and a batch of treated images using the Sinkhorn algorithm.

    Args:
        epsilon (float): Sinkhorn entropic regularisation strength. Larger values
            give softer (more uniform) plans; smaller values give harder assignments.
        max_iter (int): Maximum Sinkhorn iterations.
        cost (str): Cost function to use for the transport plan.
            - 'l2'     : squared L2 distance in pixel-flattened feature space.
            - 'cosine' : cosine dissimilarity (1 - cosine_similarity).
    """

    def __init__(self, epsilon: float = 0.05, max_iter: int = 100, cost: str = "l2"):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.cost = cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten image tensor to 2-D feature matrix [B, D]."""
        return x.reshape(x.size(0), -1).float()

    def _cost_matrix(
        self, x_feat: torch.Tensor, y_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise cost matrix C[i, j] between rows of x_feat and y_feat.

        Returns:
            C: [Nx, Ny] torch.Tensor on CPU.
        """
        if self.cost == "cosine":
            x_n = F.normalize(x_feat, dim=-1)
            y_n = F.normalize(y_feat, dim=-1)
            return 1.0 - (x_n @ y_n.T)
        else:  # l2 (squared)
            x2 = (x_feat ** 2).sum(1, keepdim=True)          # [Nx, 1]
            y2 = (y_feat ** 2).sum(1, keepdim=True).T         # [1, Ny]
            return (x2 + y2 - 2 * (x_feat @ y_feat.T)).clamp(min=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_indices(self, ctrl: torch.Tensor, pert: torch.Tensor) -> torch.LongTensor:
        """
        Compute the hard OT assignment from ctrl images to pert images.

        Args:
            ctrl: Control image batch  [B, C, H, W]  (any device)
            pert: Treated image batch  [B, C, H, W]  (any device)

        Returns:
            perm: LongTensor of shape [B] — use `pert[perm]` to obtain
                  OT-paired treated images aligned to each control image.
        """
        with torch.no_grad():
            ctrl_feat = self._features(ctrl).cpu()
            pert_feat = self._features(pert).cpu()

            C = self._cost_matrix(ctrl_feat, pert_feat).numpy()

            # Uniform marginals
            a = ot.unif(ctrl_feat.shape[0])
            b = ot.unif(pert_feat.shape[0])

            # Sinkhorn transport plan  [Nc, Nt]
            P = ot.sinkhorn(
                a, b, C,
                reg=self.epsilon,
                numItermax=self.max_iter,
                warn=False,
            )

            # Hard assignment: for each control row, pick the pert column with max weight
            perm = P.argmax(axis=1)

        return torch.from_numpy(perm.astype(np.int64)).long()
