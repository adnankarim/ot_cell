"""
OT Matcher for CellFlux — Phase 1: OT Pairing

Replaces random control/treated pairing in the training loop with a
Sinkhorn Optimal Transport assignment. Requires ``pip install pot``.

Design:
- `get_indices` accepts **pre-computed feature tensors** [B, D].
  The caller (train_loop.py) is responsible for extracting features via
  `training.feature_utils.get_ot_features`, keeping this class agnostic
  to the chosen feature space.
- Supports hard pairing (argmax of transport plan) or returning the full
  soft plan for downstream use.
"""

from __future__ import annotations

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
    Computes an Optimal Transport assignment between feature representations
    of control and treated image batches via the Sinkhorn algorithm.

    Args:
        epsilon (float): Sinkhorn regularisation strength.
            Larger → softer (more uniform) plan; smaller → harder assignments.
        max_iter (int): Maximum Sinkhorn iterations.
        cost (str): Pairwise cost function.
            ``'l2'`` — squared Euclidean distance.
            ``'cosine'`` — cosine dissimilarity (1 − cosine similarity).
        hard_pairing (bool): If True (default), return a hard permutation via
            argmax. If False, return the full soft transport plan.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 100,
        cost: str = "l2",
        hard_pairing: bool = True,
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.cost = cost
        self.hard_pairing = hard_pairing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_matrix(
        self, x_feat: torch.Tensor, y_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise cost matrix C[i, j] between rows of x_feat and y_feat.

        Args:
            x_feat: [Nx, D] float tensor on CPU.
            y_feat: [Ny, D] float tensor on CPU.

        Returns:
            C: [Nx, Ny] float tensor on CPU.
        """
        if self.cost == "cosine":
            x_n = F.normalize(x_feat, dim=-1)
            y_n = F.normalize(y_feat, dim=-1)
            return 1.0 - (x_n @ y_n.T)
        else:  # l2 (squared Euclidean)
            x2 = (x_feat ** 2).sum(1, keepdim=True)          # [Nx, 1]
            y2 = (y_feat ** 2).sum(1, keepdim=True).T         # [1, Ny]
            return (x2 + y2 - 2 * (x_feat @ y_feat.T)).clamp(min=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_indices(
        self,
        ctrl_feat: torch.Tensor,
        pert_feat: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Compute the OT-optimal assignment from control features to
        treated features.

        The caller is responsible for extracting features from raw images
        (e.g. via ``training.feature_utils.get_ot_features``).

        Args:
            ctrl_feat: Control feature matrix  [B, D]  (any device)
            pert_feat: Treated feature matrix  [B, D]  (any device)

        Returns:
            perm: LongTensor of shape [B] — use ``pert_images[perm]`` to
                  obtain OT-paired treated images aligned to each control.
        """
        with torch.no_grad():
            x = ctrl_feat.float().cpu()
            y = pert_feat.float().cpu()

            C = self._cost_matrix(x, y).numpy()

            # Uniform marginals
            a = ot.unif(x.shape[0])
            b = ot.unif(y.shape[0])

            # Sinkhorn transport plan  [Nc, Nt]
            P = ot.sinkhorn(
                a, b, C,
                reg=self.epsilon,
                numItermax=self.max_iter,
                warn=False,
            )

            if self.hard_pairing:
                perm = P.argmax(axis=1)
                return torch.from_numpy(perm.astype(np.int64)).long()
            else:
                return torch.from_numpy(P.astype(np.float32))
