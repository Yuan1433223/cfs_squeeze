"""Head-wise stateless routing (Sec. 3.3).

A single bias-free linear layer W_g produces, for each token, an [M, K] logit
tensor. A per-head softmax over the K bases gives the routing distribution pi,
which (a) modulates the four complementary components into an enhanced feature
and (b) yields the per-token high-frequency saliency that drives the squeeze.

Parameter cost is exactly D*K*M (constant w.r.t. any FFN; no expert duplication).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadwiseRouter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_bases: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_bases = num_bases
        # No bias: avoid injecting an input-agnostic constant shift (Sec. 3.3).
        self.w_g = nn.Linear(hidden_size, num_bases * num_heads, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [N, D] -> routing probs pi [N, M, K] (softmax over K, fp32)."""
        n = feat.shape[0]
        logits = self.w_g(feat).view(n, self.num_heads, self.num_bases)
        pi = F.softmax(logits.float(), dim=-1).to(feat.dtype)
        return pi

    @property
    def num_params(self) -> int:
        return self.hidden_size * self.num_bases * self.num_heads


def modulate(operators: torch.Tensor, pi: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Per-head convex combination of the four components.

    Args:
        operators: [K, N, D] stacked components.
        pi: [N, M, K] routing probabilities.
        num_heads: M (D = M * d).
    Returns:
        [N, D] enhanced feature.
    """
    k, n, d = operators.shape
    head_dim = d // num_heads
    ops = operators.view(k, n, num_heads, head_dim)         # [K, N, M, d]
    w = pi.permute(2, 0, 1).unsqueeze(-1)                   # [K, N, M, 1]
    mod = (w * ops).sum(dim=0)                               # [N, M, d]
    return mod.reshape(n, d)


def high_freq_saliency(pi: torch.Tensor, high_index: int = 3) -> torch.Tensor:
    """P^high_i = mean over heads of pi[i, :, high_index].  [N]."""
    return pi[:, :, high_index].mean(dim=1)
