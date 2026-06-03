"""Sample-wise topological reshaping between the ragged 1D token stream and the
2D spatial grid (Sec. 3.1, the operator T_b and its inverse).

Qwen3-VL emits variable-length token sequences (native dynamic resolution), so
these operators are defined per sample given its physical grid (H, W) with
N = H * W. Everything is device/dtype agnostic.
"""
from __future__ import annotations

import torch


def seq_to_grid(feat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """T_b: [N, D] -> [D, H, W] (channel-first, ready for 2D conv/pool)."""
    n, d = feat.shape
    if height * width != n:
        raise ValueError(f"H*W ({height}*{width}) != N ({n})")
    return feat.view(height, width, d).permute(2, 0, 1).contiguous()


def grid_to_seq(grid: torch.Tensor) -> torch.Tensor:
    """T_b^{-1}: [D, H, W] -> [N, D]."""
    d, h, w = grid.shape
    return grid.permute(1, 2, 0).reshape(h * w, d).contiguous()


def grid_positions(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Original (h, w) coordinates for every token, in row-major order. [N, 2]."""
    idx = torch.arange(height * width, device=device)
    hh = torch.div(idx, width, rounding_mode="floor")
    ww = idx % width
    return torch.stack([hh, ww], dim=-1).to(dtype)
