"""Configuration for CSF-Squeeze.

All architectural dimensions (``hidden_size`` D, ``num_heads`` M, the DeepStack
injection layer indices J) must be read from the backbone config at integration
time -- never hard-coded. For Qwen3-VL-8B these happen to be D=4096, M=32 with
DeepStack layers [8, 16, 24]; the 4B variant is smaller. See ``from_hf_config``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CSFSqueezeConfig:
    # --- backbone-derived (read from model config, do NOT hard-code) ---
    hidden_size: int                      # D, projector/LLM hidden dim
    num_heads: int                        # M, number of attention heads (D = M * d)
    deepstack_layers: List[int] = field(default_factory=lambda: [8, 16, 24])

    # --- method hyper-parameters ---
    num_bases: int = 4                    # K, the four complementary operators
    low_freq_kernel: int = 3              # box-blur kernel for the low-frequency basis
    keep_ratio: float = 0.5               # rho, high-frequency exemption ratio (kept full-res)
    downsample_stride: int = 2            # s, low-frequency window pooling stride

    # --- entropy exploration regularizer schedule ---
    entropy_peak: float = 1.5             # peak lambda
    entropy_warmup_frac: float = 0.10     # fraction of total steps spent ramping up

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if not (0.0 <= self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in [0, 1], got {self.keep_ratio}")
        if self.downsample_stride < 1:
            raise ValueError(f"downsample_stride must be >= 1, got {self.downsample_stride}")
        if self.num_bases != 4:
            raise ValueError("CSF-Squeeze is defined for exactly K=4 complementary bases.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_deepstack_levels(self) -> int:
        return len(self.deepstack_layers)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> "CSFSqueezeConfig":
        """Build from a HuggingFace Qwen3-VL config, reading dims at runtime.

        Robust to the text config living under ``.text_config`` and to the
        DeepStack layer indices being named ``deepstack_visual_indexes``.
        """
        text = getattr(hf_config, "text_config", hf_config)
        hidden = getattr(text, "hidden_size")
        heads = getattr(text, "num_attention_heads")
        ds_layers = (
            getattr(hf_config, "deepstack_visual_indexes", None)
            or getattr(text, "deepstack_visual_indexes", None)
            or [8, 16, 24]
        )
        kwargs = dict(hidden_size=hidden, num_heads=heads, deepstack_layers=list(ds_layers))
        kwargs.update(overrides)
        return cls(**kwargs)
