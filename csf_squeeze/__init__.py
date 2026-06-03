"""CSF-Squeeze: DeepStack-aware Frequency-Differentiated Visual Token Compression.

Device-agnostic, GPU-standard PyTorch implementation. Validated on CPU via the
test suite in ``tests/`` (see ``run_cpu_tests.py``). The same code runs unchanged
on GPU; numerically sensitive paths (softmax / entropy / quantile) are computed
in fp32 regardless of the autocast dtype.

Public API:
    CSFSqueezeConfig   - hyper-parameter container
    CSFSqueeze         - the nn.Module (router modulation + frequency-aware squeeze)
    CompressionPlan    - reusable per-sample compression plan (the Phi_b of Sec. 3.5)
    apply_plan         - propagate a plan to any [N, D] feature (DeepStack consistency)
    entropy_loss, lambda_at - exploration regularizer and its warmup/decay schedule
"""
from .config import CSFSqueezeConfig
from .squeeze import CompressionPlan, build_plan, apply_plan
from .losses import entropy_loss, lambda_at
from .module import CSFSqueeze, CSFOutput

__all__ = [
    "CSFSqueezeConfig",
    "CSFSqueeze",
    "CSFOutput",
    "CompressionPlan",
    "build_plan",
    "apply_plan",
    "entropy_loss",
    "lambda_at",
]
