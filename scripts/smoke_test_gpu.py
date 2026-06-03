"""M1 smoke test on a real Qwen3-VL model (REQUIRES GPU + transformers + model).

Two stages, increasing risk:
  Stage A (seam, low risk): run the real vision tower, compress its features with
    compress_visual_stream, and verify token reduction + DeepStack alignment on
    REAL features. This exercises only the CPU-validated seam.
  Stage B (full patch, higher risk): patch the model and run one generate() to
    validate the end-to-end forward against the INSTALLED transformers version.

Run on the ModelScope DSW instance (24G):
    python scripts/smoke_test_gpu.py --rho 0.25 --stride 2
First failures are expected in Stage B if private signatures drift; the printed
diagnostics tell us exactly what to adjust.
"""
from __future__ import annotations

import argparse


def make_demo_image(size=(896, 1280)):
    """A synthetic document-like image (dense text rows on white) — no assets needed."""
    from PIL import Image, ImageDraw
    w, h = size[1], size[0]
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(40, h - 40, 36):                      # many thin text-like rows (high-freq)
        d.text((40, y), "The quick brown fox 0123456789 jumps over the lazy dog.", fill=(0, 0, 0))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--stage", choices=["A", "B", "all"], default="all")
    args = p.parse_args()

    import torch
    from modelscope import snapshot_download
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import transformers

    from csf_squeeze.integrate_qwen3vl import (
        build_module_from_hf, grids_from_image_grid_thw, compress_visual_stream, patch_qwen3vl,
    )

    print(f"[env] transformers={transformers.__version__}  cuda={torch.cuda.is_available()}")
    assert hasattr(transformers.models, "qwen3_vl"), \
        "Installed transformers has no qwen3_vl; run `pip install -U transformers`."

    model_dir = snapshot_download(args.model)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    inner = model.model
    cfg = model.config
    print(f"[model] {type(model).__name__}  image_token_id={cfg.image_token_id}  "
          f"deepstack={getattr(cfg, 'deepstack_visual_indexes', '?')}  "
          f"merge={inner.visual.spatial_merge_size}")

    # Build inputs from one synthetic dense-text image.
    image = make_demo_image()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Read the text in the image."}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
    thw = inputs["image_grid_thw"]
    n_visual = int((thw.prod(-1) // inner.visual.spatial_merge_size ** 2).sum())
    print(f"[input] image_grid_thw={thw.tolist()}  visual_tokens={n_visual}")

    module = build_module_from_hf(cfg, keep_ratio=args.rho, downsample_stride=args.stride).to(
        "cuda", dtype=torch.bfloat16
    )

    # ---- Stage A: compress REAL vision features at the seam ----
    if args.stage in ("A", "all"):
        with torch.no_grad():
            feats = inner.get_image_features(inputs["pixel_values"], thw, return_dict=True)
            grids = grids_from_image_grid_thw(thw)
            split = [g[0] * g[1] for g in grids]
            ds_by_image = [[lvl.split(split, 0)[i] for lvl in feats.deepstack_features]
                           for i in range(len(grids))]
            stream = compress_visual_stream(list(feats.pooler_output), grids, module, ds_by_image)
        n_out = sum(b.shape[0] for b in stream["base"])
        for i, (b, levels) in enumerate(zip(stream["base"], stream["deepstack"])):
            assert all(l.shape == b.shape for l in levels), "DeepStack misalignment on real features!"
        print(f"[Stage A PASS] real-feature compression {n_visual} -> {n_out} tokens "
              f"({n_visual / max(n_out,1):.2f}x), DeepStack levels aligned.")

    # ---- Stage B: full patched forward via generate ----
    if args.stage in ("B", "all"):
        patch_qwen3vl(model, module)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            print(f"[Stage B PASS] patched generate produced {out.shape[1]} tokens:")
            print("   ", proc.batch_decode(out, skip_special_tokens=True)[0][-200:])
        except Exception as e:
            print(f"[Stage B NEEDS FIX] patched forward raised: {type(e).__name__}: {e}")
            print("    -> share this traceback; we adjust make_csf_model_forward to the installed API.")
            raise


if __name__ == "__main__":
    main()
