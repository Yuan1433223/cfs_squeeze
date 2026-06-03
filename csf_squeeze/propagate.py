"""DeepStack-consistent multi-level propagation (Sec. 3.5).

DeepStack injects ViT features from several depths into different LLM layers, at
the *same* token positions, via residual connections. Compressing only the base
sequence would misalign those injections. We therefore apply the single
per-sample CompressionPlan to the base stream and to every injection feature set.
"""
from __future__ import annotations

from typing import List

import torch

from .squeeze import CompressionPlan


def propagate_to_deepstack(
    deepstack_features: List[torch.Tensor], plan: CompressionPlan
) -> List[torch.Tensor]:
    """Apply one plan to each DeepStack injection feature set.

    Args:
        deepstack_features: list of J tensors, each [N_in, D], one per injection
            layer (e.g. layers [8, 16, 24]); all share the base token positions.
        plan: the CompressionPlan built once from the base stream.
    Returns:
        list of J tensors, each [N_out, D], positionally aligned with the
        compressed base sequence.
    """
    return [plan.apply(level) for level in deepstack_features]
