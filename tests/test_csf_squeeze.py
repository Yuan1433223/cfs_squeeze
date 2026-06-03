"""CPU mechanism-verification tests for CSF-Squeeze.

Each ``check_*`` function is a self-contained assertion-based test. They run
under pytest (auto-discovered) and standalone via ``run_cpu_tests.py`` -- no
pytest dependency required. Everything runs on CPU; the code under test is
device-agnostic, so passing here certifies the *mechanism* (the headline GPU
accuracy numbers must still be produced on GPU).
"""
import math

import torch

from csf_squeeze import CSFSqueezeConfig, CSFSqueeze, build_plan, apply_plan, entropy_loss, lambda_at
from csf_squeeze.topology import seq_to_grid, grid_to_seq, grid_positions
from csf_squeeze.operators import compute_operators
from csf_squeeze.router import HeadwiseRouter, modulate, high_freq_saliency
from csf_squeeze.propagate import propagate_to_deepstack

torch.manual_seed(0)
TOL = 1e-5


def check_topology_roundtrip():
    for (h, w) in [(4, 6), (7, 3), (1, 9), (5, 5)]:
        x = torch.randn(h * w, 8)
        back = grid_to_seq(seq_to_grid(x, h, w))
        assert torch.allclose(x, back, atol=TOL), f"roundtrip failed for {(h, w)}"


def check_operators_identity_lemma():
    """global+local == H, low+high == H, and 0.5*(sum of 4) == H (Appendix A)."""
    h, w, d = 6, 5, 16
    x = torch.randn(h * w, d)
    ops = compute_operators(x, h, w, kernel=3)               # [4, N, D]
    glob, local, low, high = ops
    assert torch.allclose(glob + local, x, atol=TOL), "global+local != H"
    assert torch.allclose(low + high, x, atol=TOL), "low+high != H"
    recon = 0.5 * ops.sum(dim=0)
    assert torch.allclose(recon, x, atol=TOL), "0.5*sum(operators) != H (identity lemma)"


def check_router_shapes_and_params():
    d, m, k = 32, 4, 4
    r = HeadwiseRouter(d, m, k)
    # no bias parameter
    assert r.w_g.bias is None, "router must be bias-free"
    # exact parameter count D*K*M
    n_params = sum(p.numel() for p in r.parameters())
    assert n_params == d * k * m, f"param count {n_params} != D*K*M {d * k * m}"
    x = torch.randn(20, d)
    pi = r(x)
    assert pi.shape == (20, m, k), f"pi shape {pi.shape}"
    sums = pi.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=TOL), "pi not normalized over K"


def check_modulation_uniform_is_half_identity():
    """Under uniform routing pi=1/K, modulation == 0.5*H (Appendix A scale)."""
    h, w, d, m = 4, 4, 16, 4
    x = torch.randn(h * w, d)
    ops = compute_operators(x, h, w)
    pi = torch.full((h * w, m, 4), 0.25)
    mod = modulate(ops, pi, m)
    assert torch.allclose(mod, 0.5 * x, atol=TOL), "uniform routing should give 0.5*H"


def check_squeeze_ratio_and_exemption():
    h, w, d = 8, 8, 16
    n = h * w
    x = torch.randn(n, d)
    p_high = torch.rand(n)
    rho, s = 0.25, 2
    plan = build_plan(p_high, h, w, s, rho)
    # exempt count is about rho * N
    assert abs(plan.n_keep - round(rho * n)) <= 2, f"n_keep={plan.n_keep} far from {rho*n}"
    # output strictly shorter than input under compression
    assert plan.n_out < n, "no compression happened"
    out = plan.apply(x)
    assert out.shape == (plan.n_out, d)
    # exempt tokens preserved exactly, in order, at the front
    assert torch.allclose(out[: plan.n_keep], x.index_select(0, plan.keep_idx), atol=TOL), \
        "exempt tokens not preserved verbatim"
    # rho=1.0 -> identity (no pooling)
    full = build_plan(p_high, h, w, s, 1.0)
    assert full.n_out == n and torch.allclose(full.apply(x), x, atol=TOL), "rho=1 not identity"


def check_position_remap():
    h, w = 8, 8
    p_high = torch.rand(h * w)
    plan = build_plan(p_high, h, w, stride=2, keep_ratio=0.25)
    pos = plan.out_positions
    # all positions inside the original coordinate frame
    assert pos[:, 0].min() >= 0 and pos[:, 0].max() <= h - 1
    assert pos[:, 1].min() >= 0 and pos[:, 1].max() <= w - 1
    # exempt positions equal original (h, w)
    orig = grid_positions(h, w)
    assert torch.allclose(pos[: plan.n_keep], orig.index_select(0, plan.keep_idx), atol=TOL)
    # a pooled token's position equals the mean of its members
    if plan.num_groups > 0:
        members = plan.low_idx[plan.group_inv == 0]
        expected = orig.index_select(0, members).mean(dim=0)
        assert torch.allclose(pos[plan.n_keep], expected, atol=TOL), "pooled pos != member centroid"


def check_deepstack_consistency():
    """Same plan on base + J injections: identical layout, exact exemption, correct pooling."""
    h, w, d, j = 6, 6, 16, 3
    n = h * w
    base = torch.randn(n, d)
    injections = [torch.randn(n, d) for _ in range(j)]
    p_high = torch.rand(n)
    plan = build_plan(p_high, h, w, stride=2, keep_ratio=0.5)

    out_base = plan.apply(base)
    out_inj = propagate_to_deepstack(injections, plan)
    # every level has identical length / layout
    for lvl in out_inj:
        assert lvl.shape == out_base.shape, "injection level shape mismatch -> misalignment"
    # exempt rows of each injection equal that injection's original exempt rows
    for orig, comp in zip(injections, out_inj):
        assert torch.allclose(comp[: plan.n_keep], orig.index_select(0, plan.keep_idx), atol=TOL)
    # pooled rows equal the per-level mean of members (consistency of the merge)
    if plan.num_groups > 0:
        g0 = plan.low_idx[plan.group_inv == 0]
        for orig, comp in zip(injections, out_inj):
            assert torch.allclose(comp[plan.n_keep].view(-1), orig.index_select(0, g0).mean(0), atol=TOL)


def check_entropy_loss():
    k = 4
    uniform = torch.full((50, 4, k), 1.0 / k)
    assert entropy_loss(uniform).abs().item() < 1e-5, "uniform should give ~0 loss"
    collapsed = torch.zeros(50, 4, k)
    collapsed[..., 0] = 1.0
    assert abs(entropy_loss(collapsed).item() - math.log2(k)) < 1e-4, "collapsed should give H_max"
    # gradient flows to logits
    logits = torch.randn(10, 4, k, requires_grad=True)
    pi = torch.softmax(logits, dim=-1)
    entropy_loss(pi).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0, "no gradient to logits"


def check_lambda_schedule():
    total, peak, warm = 100, 1.5, 0.1
    assert lambda_at(0, total, peak, warm) == 0.0
    assert abs(lambda_at(10, total, peak, warm) - peak) < 1e-6, "peak at end of warmup"
    assert lambda_at(100, total, peak, warm) == 0.0, "zero at end"
    mid = lambda_at(55, total, peak, warm)
    assert 0.0 < mid < peak, "should be decaying in the middle"


def check_module_grad_flow():
    """End-to-end on one sample: forward + backward reaches W_g and a dummy projector."""
    h, w, d, m = 6, 6, 32, 4
    cfg = CSFSqueezeConfig(hidden_size=d, num_heads=m, keep_ratio=0.5, downsample_stride=2)
    mod = CSFSqueeze(cfg)
    proj = torch.nn.Linear(d, d)                              # stand-in for the LoRA projector
    raw = torch.randn(h * w, d)
    out = mod(proj(raw), h, w)
    assert out.compressed.shape[0] < h * w, "sequence did not shrink"
    loss = out.compressed.pow(2).mean() + out.entropy_loss
    loss.backward()
    assert mod.router.w_g.weight.grad is not None and mod.router.w_g.weight.grad.abs().sum() > 0, \
        "no gradient to router W_g"
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0, \
        "no gradient to projector (modulation path not differentiable)"


def check_end2end_with_deepstack():
    """Full pipeline incl. DeepStack propagation, ragged batch, backward."""
    cfg = CSFSqueezeConfig(hidden_size=32, num_heads=4, keep_ratio=0.5, downsample_stride=2,
                           deepstack_layers=[8, 16, 24])
    mod = CSFSqueeze(cfg)
    samples = [(torch.randn(h * w, 32), h, w) for (h, w) in [(6, 6), (4, 9), (8, 5)]]
    outs = mod.forward_batch(samples)
    total = 0.0
    for (feat, h, w), out in zip(samples, outs):
        # simulate J DeepStack injections at the same positions
        injections = [torch.randn(h * w, 32) for _ in cfg.deepstack_layers]
        comp_inj = propagate_to_deepstack(injections, out.plan)
        for lvl in comp_inj:
            assert lvl.shape[0] == out.compressed.shape[0], "DeepStack misalignment after squeeze"
        total = total + out.compressed.sum() + out.entropy_loss
    total.backward()
    assert mod.router.w_g.weight.grad is not None, "no gradient after batch end-to-end"


ALL_CHECKS = [
    check_topology_roundtrip,
    check_operators_identity_lemma,
    check_router_shapes_and_params,
    check_modulation_uniform_is_half_identity,
    check_squeeze_ratio_and_exemption,
    check_position_remap,
    check_deepstack_consistency,
    check_entropy_loss,
    check_lambda_schedule,
    check_module_grad_flow,
    check_end2end_with_deepstack,
]


# pytest entry points
def test_topology_roundtrip():            check_topology_roundtrip()
def test_operators_identity_lemma():      check_operators_identity_lemma()
def test_router_shapes_and_params():      check_router_shapes_and_params()
def test_modulation_uniform():            check_modulation_uniform_is_half_identity()
def test_squeeze_ratio_and_exemption():   check_squeeze_ratio_and_exemption()
def test_position_remap():                check_position_remap()
def test_deepstack_consistency():         check_deepstack_consistency()
def test_entropy_loss():                  check_entropy_loss()
def test_lambda_schedule():               check_lambda_schedule()
def test_module_grad_flow():              check_module_grad_flow()
def test_end2end_with_deepstack():        check_end2end_with_deepstack()
