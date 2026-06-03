"""Qwen3-VL grid / M-RoPE arithmetic for the CSF-Squeeze integration.

Constants and formulas verified against the official Qwen3-VL repository
(github.com/QwenLM/Qwen3-VL, main):
  * qwen-vl-utils/src/qwen_vl_utils/vision_process.py
        SPATIAL_MERGE_SIZE = 2 ; image patch size = 14 ; smart_resize factor =
        14 * 2 = 28, so resized H, W are multiples of 28.
  * qwen-vl-finetune/qwenvl/data/rope2d.py (get_rope_index_3)
        llm_grid_h, llm_grid_w = grid_thw.h // 2, grid_thw.w // 2
        each visual token's M-RoPE position is (t=0, h=row, w=col) on that
        post-merge grid, plus a constant per-image text offset.

Consequence for CSF-Squeeze: the module operates on the POST-MERGE grid
(H = grid_thw.h // 2, W = grid_thw.w // 2), and the (h, w) tracked by a
CompressionPlan ARE the M-RoPE h/w indices. Exempt tokens keep integer (row,
col); pooled tokens take the fractional receptive-field centroid, which is a
faithful continuous position for the rotary embedding.
"""
from __future__ import annotations

from typing import Tuple

import torch

SPATIAL_MERGE_SIZE = 2
IMAGE_PATCH_SIZE = 14
SMART_RESIZE_FACTOR = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28


def llm_grid_from_thw(t: int, h: int, w: int, merge_size: int = SPATIAL_MERGE_SIZE) -> Tuple[int, int, int]:
    """Map ``image_grid_thw`` (patch units) to the post-merge LLM grid.

    Returns (llm_t, llm_h, llm_w); the visual token count is their product.
    """
    if h % merge_size or w % merge_size:
        raise ValueError(
            f"grid h={h}, w={w} not divisible by merge_size={merge_size}; "
            "smart_resize should guarantee this."
        )
    return t, h // merge_size, w // merge_size


def visual_mrope_positions(out_positions: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Build the 3 x N_out M-RoPE position ids for a compressed visual block.

    Args:
        out_positions: [N_out, 2] (h, w) from a CompressionPlan (may be fractional
            for pooled tokens).
        offset: the running text position before this image (the ``text_len +
            st_idx`` of the reference implementation); added to all three axes.
    Returns:
        [3, N_out] tensor of (t, h, w) positions, t == offset for an image
        (t_index is 0 before the offset), h/w == row/col + offset.
    """
    n = out_positions.shape[0]
    t_row = torch.full((n,), float(offset), dtype=out_positions.dtype, device=out_positions.device)
    h_row = out_positions[:, 0] + offset
    w_row = out_positions[:, 1] + offset
    return torch.stack([t_row, h_row, w_row], dim=0)
