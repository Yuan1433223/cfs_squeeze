"""The four non-parametric complementary spatial-frequency operators (Sec. 3.2).

Given the 2D feature grid H_2D of one sample, we produce four complementary
components along two independent axes:

    k=0  global  : global average pooling, broadcast back over the grid
    k=1  local   : spatial residual  H - global
    k=2  low      : 3x3 box-blur (AvgPool2d, stride 1)  -> low-frequency
    k=3  high     : spectral residual  H - low          -> high-frequency

By construction  global + local == H  and  low + high == H  exactly, so the four
components span {H, G(H), Blur(H)} and can represent the identity (Appendix A).
The k=3 (high) component's per-token energy drives the frequency-aware squeeze.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .topology import seq_to_grid, grid_to_seq


def compute_operators(feat: torch.Tensor, height: int, width: int, kernel: int = 3) -> torch.Tensor:
    """Compute the four operators for one sample.

    Args:
        feat: [N, D] sample features (N = height * width).
        height, width: physical grid dims.
        kernel: box-blur kernel size for the low-frequency basis (odd).
    Returns:
        [K=4, N, D] stacked components in seq layout, same dtype/device as feat.
    """
    if kernel % 2 == 0:
        raise ValueError("low_freq_kernel must be odd for symmetric padding")
    grid = seq_to_grid(feat, height, width)                 # [D, H, W]
    d = grid.shape[0]

    glob = grid.mean(dim=(1, 2), keepdim=True).expand_as(grid)  # [D, H, W] broadcast
    local = grid - glob

    # Box-blur in fp32 for numerical stability, then cast back.
    pad = kernel // 2
    low = F.avg_pool2d(
        grid.unsqueeze(0).float(), kernel_size=kernel, stride=1, padding=pad
    ).squeeze(0).to(grid.dtype)
    high = grid - low

    stacked = torch.stack(
        [grid_to_seq(glob), grid_to_seq(local), grid_to_seq(low), grid_to_seq(high)],
        dim=0,
    )                                                        # [4, N, D]
    return stacked
