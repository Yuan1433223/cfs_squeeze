"""Training script for the learned CSF-Squeeze router (REQUIRES GPU + transformers + peft).

Trains:
    1. The head-wise router W_g (full update, ~0.5M params).
    2. LoRA adapters on the LLM attention modules (rank 64 by default).

Frozen:
    1. Vision tower (no compute waste).
    2. LLM base weights (only the LoRA deltas update).
    3. The four DSP operators (non-parametric).

Loss = standard CE on next-token prediction + lambda(t) * H_max - H(pi)
where lambda follows a linear warm-up + cosine decay schedule (Sec. 3.6 / 4.1).

The forward path during training enables compression with selection_mode='freq'
so the router gets a real task signal -- gradient reaches W_g via the
differentiable modulation path (Sigma pi_k X_hat_k) and the entropy regularizer.

Run on the ModelScope DSW (24G):
    python scripts/train_csf.py --steps 2000 --batch-accum 4 --rho 0.35
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Heavy imports are deferred into main() so the script stays inspectable on the
# CPU dev box (no transformers / peft required at parse time).


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--out", default="checkpoints/csf-trained")
    # data
    p.add_argument("--train-samples", type=int, default=10000,
                   help="number of (image, question, answer) triplets to draw")
    p.add_argument("--data-mix", default="docvqa:0.7,pope:0.3",
                   help="benchmark:fraction list; supported: docvqa, infovqa, pope")
    # training
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-accum", type=int, default=4, help="gradient accumulation; effective batch = batch-accum")
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-router", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # compression
    p.add_argument("--rho", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=2)
    # entropy schedule
    p.add_argument("--ent-peak", type=float, default=1.5)
    p.add_argument("--ent-warmup-frac", type=float, default=0.10)
    # ablation switches (Sec 4.6-3, 4.6-4)
    p.add_argument("--no-ent", action="store_true", help="disable L_ent (ablation)")
    p.add_argument("--axis", choices=["both", "spatial", "frequency"], default="both",
                   help="ablate routing axes: keep all 4 / freeze freq pair / freeze spatial pair")
    # housekeeping
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-new-tokens-skip", type=int, default=64,
                   help="prompts whose answer span exceeds this are skipped")
    p.add_argument("--seed", type=int, default=0)
    return p


# --------------------------------------------------------------------------- #
# Data sampling                                                                #
# --------------------------------------------------------------------------- #
def build_data_iter(spec: str, total: int):
    """Yield (image, prompt, answer_str) tuples drawn round-robin from the mix."""
    from _docvqa_eval import load_train_subset
    parts = []
    for tok in spec.split(","):
        name, frac = tok.split(":")
        parts.append((name.strip(), float(frac)))
    pools = {}
    for name, frac in parts:
        n = max(2, int(total * frac))
        pools[name] = load_train_subset(name, n)   # NOTE: lightweight, fetches only what is needed
        print(f"[data] {name}: {len(pools[name])} samples")
    keys = list(pools.keys())
    n_each = {k: 0 for k in keys}
    while True:
        for k in keys:
            pool = pools[k]
            r = pool[n_each[k] % len(pool)]
            n_each[k] += 1
            ans = _gold_answer_text(r)
            if ans is None:
                continue
            yield r["image"], _format_prompt(r), ans


def _gold_answer_text(r):
    """Pick a single string gold answer from the diverse field formats."""
    task = r.get("task", "anls")
    if task == "yesno":
        return r.get("answer") or None
    if task == "anls":
        from _docvqa_eval import _as_answer_list
        ans_list = _as_answer_list(r.get("answers"))
        return ans_list[0] if ans_list else None
    return None    # mc not used for training (option strings are not free-form answers)


def _format_prompt(r):
    """The same SHORT_ANSWER instruction we use at eval -- keeps train/test aligned."""
    from _docvqa_eval import SHORT_ANSWER
    task = r.get("task", "anls")
    if task == "yesno":
        return r["question"] + " Answer yes or no."
    return r["question"] + SHORT_ANSWER


# --------------------------------------------------------------------------- #
# Forward / loss for one training sample                                       #
# --------------------------------------------------------------------------- #
def _encode_with_label(proc, image, question, answer):
    """Build inputs whose labels mask everything before the answer span."""
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}]}]
    prefix = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prefix + answer + (proc.tokenizer.eos_token or "")
    enc = proc(text=[full], images=[image], return_tensors="pt")
    pre_ids = proc(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
    pre_len = pre_ids.shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :pre_len] = -100                 # mask question/system prompt
    enc["labels"] = labels
    return enc


def _ablate_axis_(module, axis):
    """Freeze rows of W_g that route to disabled bases (Sec 4.6-4)."""
    if axis == "both":
        return
    K, M, D = module.config.num_bases, module.config.num_heads, module.config.hidden_size
    # W_g: [D, K*M] -> view as [D, M, K]
    with __import__("torch").no_grad():
        W = module.router.w_g.weight.view(K * M, D).view(M, K, D)
        if axis == "spatial":
            W[:, 2:4, :] = 0          # zero low/high freq routing -> only global/local
        elif axis == "frequency":
            W[:, 0:2, :] = 0          # zero global/local -> only low/high
    # also freeze those slices via parameter mask hook
    def _hook(grad):
        g = grad.view(M, K, D)
        if axis == "spatial":
            g[:, 2:4, :] = 0
        elif axis == "frequency":
            g[:, 0:2, :] = 0
        return g.view(K * M, D)
    module.router.w_g.weight.register_hook(_hook)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args = build_argparser().parse_args()

    import torch
    from transformers import AutoProcessor
    from modelscope import snapshot_download
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls
    from peft import LoraConfig, get_peft_model

    from csf_squeeze.integrate_qwen3vl import build_module_from_hf, patch_qwen3vl
    from csf_squeeze.losses import lambda_at

    torch.manual_seed(args.seed)
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")

    # ----- model -----
    model_dir = snapshot_download(args.model)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = ModelCls.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    # freeze the vision tower entirely
    for p in model.model.visual.parameters():
        p.requires_grad = False
    # LoRA on the LLM attention modules
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----- CSF-Squeeze module + patch -----
    cfg = model.config if not hasattr(model, "base_model") else model.base_model.model.config
    module = build_module_from_hf(cfg, keep_ratio=args.rho, downsample_stride=args.stride).to(
        "cuda", dtype=torch.bfloat16
    )
    module.enabled = True
    module.selection_mode = "freq"             # train the learned router signal
    _ablate_axis_(module, args.axis)
    inner = (model.base_model.model.model if hasattr(model, "base_model")
             else model.model)
    patch_qwen3vl(model.base_model.model if hasattr(model, "base_model") else model, module)

    # ----- optim -----
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    router_params = list(module.parameters())
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": router_params, "lr": args.lr_router},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps)

    print(f"[train] LoRA={sum(p.numel() for p in lora_params)/1e6:.2f}M  "
          f"router={sum(p.numel() for p in router_params)/1e6:.2f}M  "
          f"mode={args.axis} ent={'OFF' if args.no_ent else 'ON'}")

    # ----- progress bar -----
    try:
        from tqdm import tqdm
        bar = tqdm(total=args.steps, dynamic_ncols=True)
    except Exception:
        bar = None

    # ----- training loop -----
    data_iter = build_data_iter(args.data_mix, args.train_samples)
    model.train()
    optim.zero_grad()
    losses_ce, losses_ent = [], []
    t0 = time.perf_counter()
    step = 0
    DEBUG_FIRST = 2          # verbose for the first 2 steps
    while step < args.steps:
        verbose = step < DEBUG_FIRST
        if verbose: print(f"[dbg s{step}] next sample"); sys.stdout.flush()
        try:
            image, question, answer = next(data_iter)
        except StopIteration:
            break
        if verbose: print(f"[dbg s{step}] got sample, encode..."); sys.stdout.flush()
        try:
            batch = _encode_with_label(proc, image, question, answer)
        except Exception as e:
            print(f"[skip] encoding failed: {type(e).__name__}: {e}")
            continue
        ans_len = int((batch["labels"] != -100).sum().item())
        if ans_len == 0 or ans_len > args.max_new_tokens_skip:
            if verbose: print(f"[dbg s{step}] skip: ans_len={ans_len}"); sys.stdout.flush()
            continue
        if verbose:
            print(f"[dbg s{step}] enc shape ids={tuple(batch['input_ids'].shape)} "
                  f"pix={tuple(batch.get('pixel_values', torch.empty(0)).shape)}")
            sys.stdout.flush()
        batch = {k: v.to("cuda") for k, v in batch.items()}
        if verbose: print(f"[dbg s{step}] -> cuda; calling forward..."); sys.stdout.flush()

        out = model(**batch)
        if verbose: print(f"[dbg s{step}] forward done, loss={float(out.loss):.4f}; backward..."); sys.stdout.flush()
        loss_ce = out.loss
        # collect entropy across the latest forward via the patched module's stash:
        # we reuse module.entropy_running set in patch_qwen3vl when available.
        ent = getattr(module, "_last_entropy_loss", None)
        if ent is None:
            # fallback: zero if patch didn't stash it (still trains via CE+modulation).
            ent = torch.zeros((), device=loss_ce.device, dtype=loss_ce.dtype)
        if args.no_ent:
            lam = 0.0
        else:
            lam = lambda_at(step, args.steps, args.ent_peak, args.ent_warmup_frac)
        loss = loss_ce + lam * ent
        (loss / args.batch_accum).backward()
        if verbose: print(f"[dbg s{step}] backward done"); sys.stdout.flush()

        losses_ce.append(float(loss_ce.detach()))
        losses_ent.append(float(ent.detach()) if isinstance(ent, torch.Tensor) else 0.0)

        if (step + 1) % args.batch_accum == 0:
            torch.nn.utils.clip_grad_norm_([*lora_params, *router_params], 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()

        step += 1
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(
                f"ce={sum(losses_ce[-10:])/min(10,len(losses_ce)):.3f} "
                f"ent={sum(losses_ent[-10:])/min(10,len(losses_ent)):.3f} "
                f"lam={lam:.2f}"
            )
        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"[step {step}/{args.steps}] ce={losses_ce[-1]:.3f} ent={losses_ent[-1]:.3f} "
                  f"lam={lam:.2f} ({elapsed:.0f}s)")
        if step % args.save_every == 0:
            _save_checkpoint(model, module, args.out, step)

    if bar is not None:
        bar.close()
    _save_checkpoint(model, module, args.out, step, final=True)
    print(f"[done] saved to {args.out}")


def _save_checkpoint(model, module, out, step, final=False):
    """Save LoRA adapter + router weights only -- never the full base model."""
    import torch
    tag = "final" if final else f"step-{step}"
    path = os.path.join(out, tag)
    os.makedirs(path, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)            # peft saves adapter only
    torch.save(module.state_dict(), os.path.join(path, "csf_router.pt"))
    with open(os.path.join(path, "csf_meta.json"), "w") as f:
        json.dump({"step": step, "rho": module.config.keep_ratio,
                   "stride": module.config.downsample_stride}, f, indent=2)
    print(f"  [ckpt] {path}")


if __name__ == "__main__":
    main()
