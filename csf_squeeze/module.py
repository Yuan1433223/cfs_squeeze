"""CSF-Squeeze module: router modulation + frequency-aware squeeze (Secs. 3.2-3.6).

Forward on one sample:
    1. compute the four complementary operators on the 2D grid;
    2. route head-wise -> pi -> enhanced (modulated) feature + high-freq saliency;
    3. build the CompressionPlan and apply it to the modulated stream;
    4. expose the plan so the caller propagates it to DeepStack injections.

The module is device/dtype agnostic. Token selection is a forward-only decision
(as in standard top-k MoE); the router learns through the differentiable
modulation path and the entropy regularizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .config import CSFSqueezeConfig
from .operators import compute_operators
from .router import HeadwiseRouter, modulate, high_freq_saliency
from .squeeze import CompressionPlan, build_plan
from .losses import entropy_loss


@dataclass
class CSFOutput:
    compressed: torch.Tensor              # [N_out, D] compressed (modulated) base stream
    positions: torch.Tensor               # [N_out, 2] (h, w) of output tokens
    plan: CompressionPlan                 # reusable plan for DeepStack propagation
    entropy_loss: torch.Tensor            # scalar exploration regularizer
    pi: torch.Tensor                      # [N, M, K] routing probs (pre-compression)
    p_high: torch.Tensor                  # [N] high-frequency saliency


class CSFSqueeze(nn.Module):
    def __init__(self, config: CSFSqueezeConfig):
        super().__init__()
        self.config = config
        self.router = HeadwiseRouter(
            config.hidden_size, config.num_heads, config.num_bases
        )
        self.high_index = config.num_bases - 1   # last basis is the high-frequency one
        # Experiment controls (do not affect the default frequency-aware path):
        #   selection_mode: "freq" (default) | "random" -> the content-homogeneous
        #     control arm exempts the SAME rho fraction but chosen content-agnostically,
        #     giving matched token counts to isolate frequency-awareness (Sec. 4.6).
        #   enabled: when False the patched forward bypasses compression (dense arm).
        self.selection_mode = "freq"
        self.enabled = True
        self.last_n_in = 0      # token counts of the most recent forward (for logging)
        self.last_n_out = 0

    def _saliency(self, pi: torch.Tensor, n: int, device, dtype) -> torch.Tensor:
        if self.selection_mode == "random":
            g = torch.Generator(device="cpu").manual_seed(n)  # deterministic per resolution
            return torch.rand(n, generator=g).to(device=device, dtype=dtype)
        return high_freq_saliency(pi, self.high_index)        # "freq"

    def forward(
        self,
        feat: torch.Tensor,
        height: int,
        width: int,
        keep_ratio: Optional[float] = None,
        stride: Optional[int] = None,
    ) -> CSFOutput:
        """Process one sample. ``feat`` is [N, D] with N = height * width."""
        cfg = self.config
        rho = cfg.keep_ratio if keep_ratio is None else keep_ratio
        s = cfg.downsample_stride if stride is None else stride

        operators = compute_operators(feat, height, width, cfg.low_freq_kernel)  # [K, N, D]
        pi = self.router(feat)                                                   # [N, M, K]
        modulated = modulate(operators, pi, cfg.num_heads)                       # [N, D]
        p_high = self._saliency(pi, feat.shape[0], feat.device, feat.dtype)      # [N]

        plan = build_plan(p_high, height, width, s, rho)
        compressed = plan.apply(modulated)
        self.last_n_in, self.last_n_out = plan.n_in, plan.n_out

        return CSFOutput(
            compressed=compressed,
            positions=plan.out_positions,
            plan=plan,
            entropy_loss=entropy_loss(pi),
            pi=pi,
            p_high=p_high,
        )

    def forward_batch(
        self, samples: List[tuple], keep_ratio: Optional[float] = None, stride: Optional[int] = None
    ) -> List[CSFOutput]:
        """Process a ragged batch. ``samples`` is a list of (feat [N,D], H, W)."""
        return [self.forward(f, h, w, keep_ratio, stride) for (f, h, w) in samples]
