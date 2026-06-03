"""Integration seam for Qwen3-VL (GPU / transformers required at call time).

This module is import-safe without transformers: heavy imports are deferred. It
exposes the *clean seam* of CSF-Squeeze and a documented scaffold for wiring it
into a real Qwen3-VL model.

The seam (fully covered by the CPU test suite):
    compress_visual_stream(merged_feats, grids, deepstack_feats, module)
    -> compressed base + per-level DeepStack injections + new (h, w) positions,
       all positionally consistent (Sec. 3.5).

What still needs wiring against the concrete HF model (marked TODO below):
    * obtain per-image (H, W) grid from image_grid_thw / rope metadata;
    * capture the DeepStack injection features (the model routes ViT features at
      ``config.deepstack_visual_indexes`` to LLM layers via the multi-level
      merger) so the same plan can be applied to them;
    * rebuild position_ids / M-RoPE indices from the returned positions.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .config import CSFSqueezeConfig
from .module import CSFSqueeze, CSFOutput
from .propagate import propagate_to_deepstack
from .grid_utils import llm_grid_from_thw, visual_mrope_positions


def grids_from_image_grid_thw(image_grid_thw) -> List[Tuple[int, int]]:
    """Map a batch of ``image_grid_thw`` rows (patch units) to post-merge (H, W).

    The module operates on the post-merge LLM grid (grid.h // 2, grid.w // 2);
    see ``grid_utils`` for provenance against the official Qwen3-VL source.
    """
    out = []
    for row in image_grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        _, lh, lw = llm_grid_from_thw(t, h, w)
        out.append((lh, lw))
    return out


def compress_visual_stream(
    merged_feats: List[torch.Tensor],          # list over images: [N_i, D]
    grids: List[Tuple[int, int]],              # list over images: (H_i, W_i)
    module: CSFSqueeze,
    deepstack_feats: Optional[List[List[torch.Tensor]]] = None,  # [image][level] -> [N_i, D]
):
    """Apply CSF-Squeeze to each image's visual stream and propagate to DeepStack.

    Returns a dict with, per image: compressed base [N_out, D], compressed
    DeepStack levels (list of [N_out, D]), output positions [N_out, 2], the plan,
    and the summed entropy loss across images (for the training objective).
    """
    out_base, out_deepstack, out_positions, out_mrope, plans = [], [], [], [], []
    entropy = merged_feats[0].new_zeros(())
    for img_idx, (feat, (h, w)) in enumerate(zip(merged_feats, grids)):
        res: CSFOutput = module(feat, h, w)
        out_base.append(res.compressed)
        out_positions.append(res.positions)
        # 3 x N_out M-RoPE (t, h, w) ids for the compressed block; caller adds the
        # running text offset for the absolute position of this image.
        out_mrope.append(visual_mrope_positions(res.positions, offset=0))
        plans.append(res.plan)
        entropy = entropy + res.entropy_loss
        if deepstack_feats is not None:
            out_deepstack.append(propagate_to_deepstack(deepstack_feats[img_idx], res.plan))
        else:
            out_deepstack.append(None)
    return {
        "base": out_base,
        "deepstack": out_deepstack,
        "positions": out_positions,
        "mrope_position_ids": out_mrope,
        "plans": plans,
        "entropy_loss": entropy / max(len(merged_feats), 1),
    }


def build_module_from_hf(hf_config, **overrides) -> CSFSqueeze:
    """Construct a CSFSqueeze whose dims are read from the live model config."""
    cfg = CSFSqueezeConfig.from_hf_config(hf_config, **overrides)
    return CSFSqueeze(cfg)


# ---------------------------------------------------------------------------
# Realistic patch for a live Qwen3-VL model, written against the actual forward
# in transformers/models/qwen3_vl/modeling_qwen3_vl.py (verified, main branch).
#
# Real control flow we hook (Qwen3VLModel.forward):
#   image_outputs = self.get_image_features(pixel_values, image_grid_thw)
#       -> pooler_output: per-image list of merged tokens [N_i, D]
#          (split by split_sizes = grid_thw.prod(-1) // merge**2)        [L1067]
#       -> deepstack_features: list of J tensors, each [sum_i N_i, D]    [L731]
#   image_embeds = cat(pooler_output)                                     [L1200]
#   image_mask = (input_ids == config.image_token_id)                    [L1094]
#   inputs_embeds.masked_scatter(image_mask, image_embeds)               [L1204]
#   visual_pos_masks = image_mask[...,0]                                  [L1235]
#   language_model(..., visual_pos_masks=, deepstack_visual_embeds=)     [L1253]
#   _deepstack_process: hidden[visual_pos] += deepstack_embeds[layer]    [L856]
#
# CSF-Squeeze inserts compression between feature production and the scatter.
# Because compression changes the visual token count (N_i -> N_out_i), the
# placeholder span in input_ids must shrink accordingly, so we rebuild the
# per-row sequence, M-RoPE position ids (mirroring get_rope_index_3), the
# attention mask, visual_pos_masks, and the (compressed) deepstack embeds.
#
# NOTE: This is a faithful template; it MUST be validated on a GPU against the
# installed transformers version (private forward signatures drift across
# releases). Videos are omitted for the document-centric scope.
# ---------------------------------------------------------------------------


def _compress_row(
    input_ids_row: torch.Tensor,        # [L] unpadded token ids for one sample
    inputs_embeds_row: torch.Tensor,    # [L, D] text+placeholder embeddings
    per_image: list,                    # list of (compressed_embeds [N_out,D], out_positions [N_out,2])
    image_token_id: int,
):
    """Rebuild one sequence with compressed visual blocks + M-RoPE positions.

    Mirrors get_rope_index_3: walk text/image segments, assign arange positions to
    text (all 3 axes equal) and (t=0, h, w) to visual tokens, with a running
    offset st_idx = previous_max + 1.
    Returns (new_embeds [L',D], position_ids [3,L'], visual_mask [L']).
    """
    device = inputs_embeds_row.device
    is_img = input_ids_row == image_token_id
    out_embeds, pos_segments, vis_flags = [], [], []
    st_idx = 0
    i = 0
    img_ptr = 0
    L = input_ids_row.shape[0]
    while i < L:
        if not bool(is_img[i]):
            # text run [i, j)
            j = i
            while j < L and not bool(is_img[j]):
                j += 1
            text_len = j - i
            out_embeds.append(inputs_embeds_row[i:j])
            rng = torch.arange(text_len, device=device) + st_idx
            pos_segments.append(rng.view(1, -1).expand(3, -1))
            vis_flags.append(torch.zeros(text_len, dtype=torch.bool, device=device))
            st_idx = int(rng.max().item()) + 1 if text_len > 0 else st_idx
            i = j
        else:
            # one image placeholder run -> replace with compressed block
            j = i
            while j < L and bool(is_img[j]):
                j += 1
            comp_embeds, out_pos = per_image[img_ptr]
            img_ptr += 1
            n_out = comp_embeds.shape[0]
            out_embeds.append(comp_embeds.to(inputs_embeds_row.dtype))
            t_row = torch.zeros(n_out, device=device)
            h_row = out_pos[:, 0].to(device)
            w_row = out_pos[:, 1].to(device)
            pos_segments.append(torch.stack([t_row, h_row, w_row], dim=0) + st_idx)
            vis_flags.append(torch.ones(n_out, dtype=torch.bool, device=device))
            seg_max = float(torch.stack([t_row, h_row, w_row]).max().item()) + st_idx
            st_idx = int(seg_max) + 1
            i = j
    new_embeds = torch.cat(out_embeds, dim=0)
    position_ids = torch.cat(pos_segments, dim=1)            # [3, L']
    visual_mask = torch.cat(vis_flags, dim=0)                # [L']
    return new_embeds, position_ids, visual_mask


def make_csf_model_forward(model, module: CSFSqueeze):  # pragma: no cover - needs real model
    """Return a CSF-aware replacement for Qwen3VLModel.forward (images only).

    ``model`` here is the inner Qwen3VLModel. Decode steps (no pixel_values) and
    text-only batches fall through to the original forward unchanged.
    """
    import torch as _torch

    orig_forward = model.forward
    image_token_id = model.config.image_token_id
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
    except Exception:  # version drift: fall back to a plain namespace-like return
        Qwen3VLModelOutputWithPast = None

    def csf_forward(
        input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, pixel_values=None, image_grid_thw=None, **kwargs,
    ):
        # Decode steps / text-only, or compression disabled (dense arm): original behaviour.
        if pixel_values is None or not getattr(module, "enabled", True):
            return orig_forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        # 1) uncompressed visual features (real API).
        image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        per_image_embeds = list(image_outputs.pooler_output)          # list of [N_i, D]
        deepstack = image_outputs.deepstack_features                   # list_J of [sum N_i, D]

        # 2) per-image compression + DeepStack-consistent propagation.
        grids = grids_from_image_grid_thw(image_grid_thw)             # post-merge (H_i, W_i)
        split = [g[0] * g[1] for g in grids]
        ds_per_image = [list(_torch.split(level, split, dim=0)) for level in deepstack]
        ds_by_image = [[ds_per_image[j][i] for j in range(len(deepstack))] for i in range(len(grids))]
        stream = compress_visual_stream(per_image_embeds, grids, module, ds_by_image)
        comp_embeds, comp_positions, comp_deepstack = stream["base"], stream["positions"], stream["deepstack"]

        # 3) rebuild each row: shrink placeholder spans, build M-RoPE positions.
        rows_embeds, rows_pos, rows_vmask = [], [], []
        for b in range(input_ids.shape[0]):
            per_image = list(zip(comp_embeds, comp_positions))        # single-image-per-row assumption
            e, p, vm = _compress_row(input_ids[b], inputs_embeds[b], per_image, image_token_id)
            rows_embeds.append(e); rows_pos.append(p); rows_vmask.append(vm)

        # 4) right-pad ragged rows back into a batch.
        Lmax = max(e.shape[0] for e in rows_embeds)
        Bsz, D = len(rows_embeds), inputs_embeds.shape[-1]
        new_embeds = inputs_embeds.new_zeros(Bsz, Lmax, D)
        new_mask = inputs_embeds.new_zeros(Bsz, Lmax, dtype=_torch.long)
        new_pos = inputs_embeds.new_zeros(3, Bsz, Lmax, dtype=_torch.long)
        new_vmask = inputs_embeds.new_zeros(Bsz, Lmax, dtype=_torch.bool)
        deltas = []
        for b, (e, p, vm) in enumerate(zip(rows_embeds, rows_pos, rows_vmask)):
            l = e.shape[0]
            new_embeds[b, :l] = e
            new_mask[b, :l] = 1
            new_pos[:, b, :l] = p.round().long()
            new_vmask[b, :l] = vm
            deltas.append(int(p.max().item()) + 1 - l)
        # rope_deltas drives position advancement during decode.
        model.rope_deltas = _torch.tensor(deltas, device=new_embeds.device).unsqueeze(1)

        # 5) DeepStack injection embeds, per level, aligned to visual positions.
        J = len(deepstack)
        if getattr(module, "deepstack_mode", "consistent") == "naive":
            # Ablation: DeepStack-unaware compressor. Inject the first N_out original
            # features per image (no plan applied) -> positional misalignment with the
            # compressed base stream. Demonstrates the necessity of consistent propagation.
            deepstack_visual_embeds = []
            for j in range(J):
                per_img = [ds_by_image[i][j][: comp_embeds[i].shape[0]] for i in range(len(grids))]
                deepstack_visual_embeds.append(_torch.cat(per_img, dim=0))
        else:
            deepstack_visual_embeds = [
                _torch.cat([comp_deepstack[i][j] for i in range(len(grids))], dim=0) for j in range(J)
            ]

        outputs = model.language_model(
            input_ids=None, position_ids=new_pos, attention_mask=new_mask,
            past_key_values=past_key_values, inputs_embeds=new_embeds,
            visual_pos_masks=new_vmask, deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
        )
        if Qwen3VLModelOutputWithPast is not None:
            return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=model.rope_deltas)
        return outputs

    return csf_forward


def patch_qwen3vl(model, module: CSFSqueeze):  # pragma: no cover - needs real model
    """Install CSF-Squeeze into a live Qwen3-VL model (images, GPU-validated path).

    Replaces the inner ``Qwen3VLModel.forward``. Only ``module.router`` (and any
    LoRA on the projector) are trainable; DeepStack injections are compressed by
    the same per-image plan to preserve multi-level alignment (Sec. 3.5).

    M1 validation gate: smoke-test on one image against the *installed*
    transformers version before running experiments (private signatures drift).
    """
    inner = model.model if hasattr(model, "model") else model
    inner.forward = make_csf_model_forward(inner, module)
    return model
