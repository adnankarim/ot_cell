"""
OT Matcher for CellFlux — corrected version

Key fixes over previous version:
- Hungarian matching enforces true one-to-one assignment (argmax can assign
  the same treated cell to multiple controls, which is wrong)
- Exposes get_cost_matrix() for external use / debugging
- hard_method: 'hungarian' (default) or 'argmax' (faster, for ablation)
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

from scipy.optimize import linear_sum_assignment


class OTMatcher:
    """
    Computes an Optimal Transport-style assignment between control and treated
    feature representations.

    Args:
        epsilon (float): Sinkhorn regularisation strength (used for soft plan).
        max_iter (int): Maximum Sinkhorn iterations.
        cost (str): ``'l2'`` or ``'cosine'``.
        hard_pairing (bool): If True, return a hard permutation vector.
        hard_method (str):
            ``'hungarian'`` — true one-to-one assignment via scipy linear_sum_assignment.
            ``'argmax'`` — row-wise argmax of the Sinkhorn plan (may duplicate matches).
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 100,
        cost: str = "l2",
        hard_pairing: bool = True,
        hard_method: str = "hungarian",
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.cost = cost
        self.hard_pairing = hard_pairing
        self.hard_method = hard_method

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_matrix(
        self, x_feat: torch.Tensor, y_feat: torch.Tensor
    ) -> torch.Tensor:
        if self.cost == "cosine":
            x_n = F.normalize(x_feat, dim=-1)
            y_n = F.normalize(y_feat, dim=-1)
            return 1.0 - (x_n @ y_n.T)
        elif self.cost == "l2":
            x2 = (x_feat ** 2).sum(1, keepdim=True)
            y2 = (y_feat ** 2).sum(1, keepdim=True).T
            return (x2 + y2 - 2 * (x_feat @ y_feat.T)).clamp(min=0)
        else:
            raise ValueError(f"Unknown cost type: {self.cost}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cost_matrix(
        self,
        ctrl_feat: torch.Tensor,
        pert_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw [Nc, Nt] pairwise cost matrix (on CPU)."""
        with torch.no_grad():
            return self._cost_matrix(
                ctrl_feat.float().cpu(),
                pert_feat.float().cpu(),
            )

    def get_indices(
        self,
        ctrl_feat: torch.Tensor,
        pert_feat: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Return permutation ``perm`` such that ``pert[perm[i]]`` is the best
        OT match for ``ctrl[i]``.

        Both inputs should be pre-computed feature tensors ``[B, D]``.
        Use ``training.feature_utils.get_ot_features`` to extract them.

        Returns:
            perm: LongTensor ``[B]`` (hard matching) or
                  FloatTensor ``[B, B]`` (soft Sinkhorn plan when
                  ``hard_pairing=False``).
        """
        with torch.no_grad():
            C = self.get_cost_matrix(ctrl_feat, pert_feat).numpy()

            if self.hard_pairing:
                if self.hard_method == "hungarian":
                    # True one-to-one assignment — O(n³) but exact
                    row_ind, col_ind = linear_sum_assignment(C)
                    perm = np.full(C.shape[0], -1, dtype=np.int64)
                    perm[row_ind] = col_ind
                    if (perm < 0).any():
                        raise RuntimeError(
                            "Hungarian matching produced incomplete assignment. "
                            "Ensure ctrl and pert group sizes match."
                        )
                    return torch.from_numpy(perm).long()

                elif self.hard_method == "argmax":
                    # Row-wise argmax of Sinkhorn plan (fast, not one-to-one)
                    a = ot.unif(C.shape[0])
                    b = ot.unif(C.shape[1])
                    P = ot.sinkhorn(
                        a, b, C,
                        reg=self.epsilon,
                        numItermax=self.max_iter,
                        warn=False,
                    )
                    return torch.from_numpy(P.argmax(axis=1).astype(np.int64)).long()

                else:
                    raise ValueError(f"Unknown hard_method: {self.hard_method}")

            else:
                # Return full soft transport plan
                a = ot.unif(C.shape[0])
                b = ot.unif(C.shape[1])
                P = ot.sinkhorn(
                    a, b, C,
                    reg=self.epsilon,
                    numItermax=self.max_iter,
                    warn=False,
                )
                return torch.from_numpy(P.astype(np.float32))
