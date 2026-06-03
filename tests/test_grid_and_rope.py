"""Validate CSF-Squeeze grid/position handling against the REAL Qwen3-VL rope code.

Mirrors the visual-token branch of get_rope_index_3 (qwen-vl-finetune/qwenvl/
data/rope2d.py) and checks that, with no compression, CSF-Squeeze produces
exactly the same (h, w) M-RoPE indices. Runs on CPU.
"""
import torch

from csf_squeeze import CSFSqueezeConfig, CSFSqueeze
from csf_squeeze.grid_utils import llm_grid_from_thw, visual_mrope_positions, SPATIAL_MERGE_SIZE

TOL = 1e-5


def _reference_rope_hw(llm_h: int, llm_w: int):
    """The h_index / w_index construction from get_rope_index_3 (offset 0)."""
    h_index = torch.arange(llm_h).view(-1, 1).expand(-1, llm_w).flatten()
    w_index = torch.arange(llm_w).view(1, -1).expand(llm_h, -1).flatten()
    return h_index.float(), w_index.float()


def check_llm_grid_arithmetic():
    # grid_thw in patch units; smart_resize makes h, w multiples of 2.
    for (t, h, w) in [(1, 8, 12), (1, 4, 4), (1, 16, 10)]:
        lt, lh, lw = llm_grid_from_thw(t, h, w)
        assert (lt, lh, lw) == (t, h // SPATIAL_MERGE_SIZE, w // SPATIAL_MERGE_SIZE)


def check_positions_match_reference_no_compression():
    """rho=1.0 (no squeeze): CSF positions must equal the reference rope indices."""
    t, h_patch, w_patch = 1, 8, 12
    _, H, W = llm_grid_from_thw(t, h_patch, w_patch)         # post-merge grid
    cfg = CSFSqueezeConfig(hidden_size=32, num_heads=4, keep_ratio=1.0, downsample_stride=2)
    mod = CSFSqueeze(cfg)
    feat = torch.randn(H * W, 32)
    out = mod(feat, H, W)
    assert out.compressed.shape[0] == H * W, "rho=1 should not compress"
    ref_h, ref_w = _reference_rope_hw(H, W)
    assert torch.allclose(out.positions[:, 0], ref_h, atol=TOL), "h indices differ from rope2d"
    assert torch.allclose(out.positions[:, 1], ref_w, atol=TOL), "w indices differ from rope2d"


def check_mrope_position_ids_shape_and_offset():
    t, h_patch, w_patch = 1, 4, 6
    _, H, W = llm_grid_from_thw(t, h_patch, w_patch)
    cfg = CSFSqueezeConfig(hidden_size=16, num_heads=4, keep_ratio=0.5, downsample_stride=2)
    out = CSFSqueeze(cfg)(torch.randn(H * W, 16), H, W)
    offset = 7
    pos = visual_mrope_positions(out.positions, offset=offset)
    assert pos.shape == (3, out.compressed.shape[0]), "position_ids must be [3, N_out]"
    assert torch.allclose(pos[0], torch.full_like(pos[0], float(offset))), "t-axis must be constant offset"
    # h/w stay within [offset, offset + grid) range
    assert pos[1].min() >= offset - TOL and pos[1].max() <= offset + H - 1 + TOL
    assert pos[2].min() >= offset - TOL and pos[2].max() <= offset + W - 1 + TOL


def check_pooled_positions_are_fractional_centroids():
    """Under compression, pooled tokens carry sub-grid (fractional) positions."""
    t, h_patch, w_patch = 1, 8, 8
    _, H, W = llm_grid_from_thw(t, h_patch, w_patch)
    cfg = CSFSqueezeConfig(hidden_size=16, num_heads=4, keep_ratio=0.25, downsample_stride=2)
    out = CSFSqueeze(cfg)(torch.randn(H * W, 16), H, W)
    if out.plan.num_groups > 0:
        pooled = out.positions[out.plan.n_keep:]
        # at least one pooled position is non-integer (a true centroid)
        frac = (pooled - pooled.round()).abs().max()
        assert frac > 0.0, "pooled positions should be sub-grid centroids"


def check_compress_row_rebuild():
    """The patch's per-row sequence rebuild (pure tensor logic, no model)."""
    from csf_squeeze.integrate_qwen3vl import _compress_row
    IMG = 999
    ids = torch.tensor([1, 2, 3, IMG, IMG, IMG, IMG, 4, 5])
    emb = torch.randn(9, 8)
    comp = torch.randn(2, 8)
    pos = torch.tensor([[0.0, 0.0], [1.5, 0.5]])
    new_e, new_p, vm = _compress_row(ids, emb, [(comp, pos)], IMG)
    assert new_e.shape[0] == 3 + 2 + 2, "rebuilt length wrong"
    assert vm.tolist() == [False, False, False, True, True, False, False], "visual mask wrong"
    # text run before image keeps its embeddings verbatim
    assert torch.allclose(new_e[:3], emb[:3], atol=TOL)
    # positions: text monotonic, visual offset applied, text after continues past image max
    assert new_p[0, :3].tolist() == [0.0, 1.0, 2.0]
    assert new_p[0, -1].item() > new_p[0, 4].item(), "positions not monotonic across image"


ALL_CHECKS = [
    check_llm_grid_arithmetic,
    check_positions_match_reference_no_compression,
    check_mrope_position_ids_shape_and_offset,
    check_pooled_positions_are_fractional_centroids,
    check_compress_row_rebuild,
]


def test_llm_grid_arithmetic():                       check_llm_grid_arithmetic()
def test_positions_match_reference_no_compression():  check_positions_match_reference_no_compression()
def test_mrope_position_ids_shape_and_offset():       check_mrope_position_ids_shape_and_offset()
def test_pooled_positions_are_fractional_centroids(): check_pooled_positions_are_fractional_centroids()
def test_compress_row_rebuild():                      check_compress_row_rebuild()
