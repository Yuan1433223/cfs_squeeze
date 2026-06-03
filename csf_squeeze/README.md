# CSF-Squeeze (code)

Reference implementation of **DeepStack-aware Frequency-Differentiated Visual
Token Compression** (paper: `../paper-zh.txt`). Device-agnostic, GPU-standard
PyTorch; mechanism validated on CPU.

## Layout
```
csf_squeeze/
  config.py            # CSFSqueezeConfig; dims read from HF config, never hard-coded
  topology.py          # T_b / T_b^-1 : seq <-> 2D grid (per sample, ragged)   [Sec 3.1]
  operators.py         # 4 non-parametric spatial-frequency operators           [Sec 3.2]
  router.py            # bias-free head-wise router W_g, modulation, P^high      [Sec 3.3]
  squeeze.py           # CompressionPlan: high-freq exempt + low-freq pooling    [Sec 3.4]
  propagate.py         # apply one plan to all DeepStack injections              [Sec 3.5]
  losses.py            # entropy exploration regularizer + lambda schedule       [Sec 3.6]
  grid_utils.py        # Qwen3-VL grid/M-RoPE arithmetic (verified vs official src)
  module.py            # CSFSqueeze nn.Module integrating the above
  integrate_qwen3vl.py # stable seam + wiring template for a real Qwen3-VL model
tests/test_csf_squeeze.py  # 11 CPU mechanism checks
tests/test_grid_and_rope.py # 4 checks vs the real Qwen3-VL rope construction
run_cpu_tests.py           # standalone runner (no pytest needed)
scripts/run_kill_experiment.py  # GPU kill-or-confirm experiment scaffold
```

## Run the CPU tests
```bash
./.venv/Scripts/python.exe run_cpu_tests.py        # 15 passed
```
Covers: topology roundtrip; identity lemma `0.5·Σ ops == H`; router shape & exact
`D·K·M` param count (no bias); uniform routing → `0.5·H`; squeeze ratio & exact
high-freq exemption; position remap (receptive-field centroid); **DeepStack
multi-level consistency**; entropy loss (uniform→0, collapsed→`log2 K`); lambda
schedule; gradient flow to `W_g` and the projector; ragged end-to-end + backward.
Plus, validated against the official Qwen3-VL source: post-merge LLM grid
(`grid_thw.h//2, .w//2`) and **exact match of CSF positions to `rope2d.py`'s
M-RoPE indices** when uncompressed; pooled tokens carry fractional centroids.

## Provenance
`grid_utils.py` constants/formulas are verified against
github.com/QwenLM/Qwen3-VL (main): `qwen-vl-utils/.../vision_process.py`
(`SPATIAL_MERGE_SIZE=2`, patch 14, smart_resize factor 28) and
`qwen-vl-finetune/qwenvl/data/rope2d.py` (`get_rope_index_3`).

## GPU experiment (rented single GPU)
Needs `transformers accelerate datasets pillow` and a CUDA device. `scripts/
run_kill_experiment.py` documents the three arms (dense / uniform / csf) sharing
DeepStack-consistent propagation, differing only in selection policy, to isolate
the effect of frequency-awareness at matched compression. `compress_visual_stream`
in `integrate_qwen3vl.py` is the stable seam to wire against the model.

## Notes
- Numerically sensitive paths (softmax, entropy, quantile) run in fp32 under any
  autocast dtype.
- Token selection is a forward-only decision (as in top-k MoE); the router learns
  via the differentiable modulation path and the entropy regularizer.
