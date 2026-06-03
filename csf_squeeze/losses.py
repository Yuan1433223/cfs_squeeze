"""Shannon-entropy exploration regularizer and its warmup/decay schedule (Sec. 3.6)."""
from __future__ import annotations

import math

import torch

_LOG2 = math.log(2.0)


def entropy_loss(pi: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L_ent = H_max - H(mean_pi), with H_max = log2(K).

    Args:
        pi: [N, M, K] routing probabilities (or any [..., K]).
    Returns:
        scalar loss; 0 when the batch-mean routing is uniform, H_max when collapsed.
    """
    k = pi.shape[-1]
    pbar = pi.float().reshape(-1, k).mean(dim=0)            # [K], batch-mean activation
    pbar = pbar.clamp_min(eps)
    entropy = -(pbar * (pbar.log() / _LOG2)).sum()          # log base 2
    h_max = math.log2(k)
    return h_max - entropy


def lambda_at(step: int, total_steps: int, peak: float, warmup_frac: float) -> float:
    """Linear warmup to ``peak`` over the first ``warmup_frac`` of training, then
    linear decay to 0 (Sec. 3.6 / Eq. 13). Returns the lambda for the given step."""
    if total_steps <= 0:
        return 0.0
    frac = step / total_steps
    if frac >= 1.0:
        return 0.0
    warm = max(warmup_frac, 1e-8)
    if frac <= warm:
        return peak * (frac / warm)
    # decay from peak (at frac=warm) down to 0 (at frac=1)
    return peak * (1.0 - (frac - warm) / (1.0 - warm))
