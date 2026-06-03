"""Frequency-aware token squeezing and the reusable CompressionPlan (Sec. 3.4-3.5).

The squeeze is realized as **high-frequency exemption + low-frequency window
pooling**, which is a grid-consistent, reusable concretization of Eq. (10):

  * Tokens whose high-frequency saliency P^high >= tau_b (the (1-rho) quantile)
    are EXEMPT: kept at full resolution and original position.
  * The remaining low-frequency tokens are pooled within non-overlapping s x s
    grid windows; each window contributes one merged token whose position is the
    receptive-field centre (mean (h, w) of its members).

Crucially the *plan* (which tokens are exempt, how low tokens group, and the
output positions) is a pure function of (grid, P^high, s, rho). Building it once
and applying it to the base sequence **and to every DeepStack injection feature
set** is exactly the "compute once, propagate consistently" mechanism of Sec. 3.5
that keeps multi-level residual injections aligned after compression.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .topology import grid_positions


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Mean of ``src`` rows grouped by ``index``. src [M, C] -> [num_groups, C]."""
    c = src.shape[1]
    out = src.new_zeros(num_groups, c)
    out.index_add_(0, index, src)
    counts = torch.bincount(index, minlength=num_groups).clamp(min=1).to(src.dtype)
    return out / counts.unsqueeze(-1)


@dataclass
class CompressionPlan:
    """Reusable, feature-agnostic compression plan for one sample (the Phi_b)."""

    keep_idx: torch.Tensor      # [n_keep] original indices kept at full res
    low_idx: torch.Tensor       # [n_low] original indices to be pooled
    group_inv: torch.Tensor     # [n_low] group id (0..n_groups-1) per low token
    num_groups: int             # number of pooled tokens
    out_positions: torch.Tensor # [n_keep + n_groups, 2] (h, w) of the output tokens
    n_in: int                   # original token count

    @property
    def n_keep(self) -> int:
        return int(self.keep_idx.numel())

    @property
    def n_out(self) -> int:
        return self.n_keep + self.num_groups

    @property
    def compression_ratio(self) -> float:
        return self.n_in / max(self.n_out, 1)

    def apply(self, feat: torch.Tensor) -> torch.Tensor:
        """Propagate the plan to any [N_in, D] feature -> [N_out, D].

        Output ordering is always [exempt tokens ..., pooled tokens ...], matching
        ``out_positions``. Exempt rows are copied verbatim; pooled rows are the
        mean of their members. Applying the same plan to the base stream and to
        each DeepStack injection guarantees positional consistency across levels.
        """
        if feat.shape[0] != self.n_in:
            raise ValueError(f"feature has {feat.shape[0]} tokens, plan expects {self.n_in}")
        kept = feat.index_select(0, self.keep_idx)
        if self.num_groups == 0:
            return kept
        pooled = _scatter_mean(feat.index_select(0, self.low_idx), self.group_inv, self.num_groups)
        return torch.cat([kept, pooled], dim=0)


def build_plan(
    p_high: torch.Tensor,
    height: int,
    width: int,
    stride: int,
    keep_ratio: float,
) -> CompressionPlan:
    """Construct the per-sample compression plan from high-frequency saliency.

    Args:
        p_high: [N] high-frequency saliency per token (N = height * width).
        height, width: physical grid dims.
        stride: s, the low-frequency pooling window size.
        keep_ratio: rho, fraction of tokens exempt at full resolution.
    """
    n = height * width
    if p_high.numel() != n:
        raise ValueError(f"p_high has {p_high.numel()} entries, expected {n}")
    device = p_high.device

    if keep_ratio >= 1.0:
        keep = torch.ones(n, dtype=torch.bool, device=device)
    elif keep_ratio <= 0.0:
        keep = torch.zeros(n, dtype=torch.bool, device=device)
    else:
        # (1 - rho) quantile threshold; ">=" keeps roughly the top-rho fraction.
        tau = torch.quantile(p_high.float(), 1.0 - keep_ratio)
        keep = p_high >= tau

    idx = torch.arange(n, device=device)
    pos = grid_positions(height, width, device=device)      # [N, 2] float
    keep_idx = idx[keep]
    low_idx = idx[~keep]
    keep_pos = pos[keep]

    if low_idx.numel() > 0:
        n_win_w = math.ceil(width / stride)
        hh = torch.div(low_idx, width, rounding_mode="floor")
        ww = low_idx % width
        win_id = torch.div(hh, stride, rounding_mode="floor") * n_win_w + torch.div(
            ww, stride, rounding_mode="floor"
        )
        _, group_inv = torch.unique(win_id, return_inverse=True)
        num_groups = int(group_inv.max().item()) + 1
        pooled_pos = _scatter_mean(pos[~keep], group_inv, num_groups)
        out_positions = torch.cat([keep_pos, pooled_pos], dim=0)
    else:
        group_inv = low_idx.new_empty(0)
        num_groups = 0
        out_positions = keep_pos

    return CompressionPlan(
        keep_idx=keep_idx,
        low_idx=low_idx,
        group_inv=group_inv,
        num_groups=num_groups,
        out_positions=out_positions,
        n_in=n,
    )


def apply_plan(feat: torch.Tensor, plan: CompressionPlan) -> torch.Tensor:
    """Functional alias for ``plan.apply(feat)`` (DeepStack propagation, Eq. 11)."""
    return plan.apply(feat)
