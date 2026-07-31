import pytest
import torch

from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2

_ARCH = get_rocm_arch()
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"sparse conv fp32 needs mfma_f32_16x16x4f32 (CDNA4 gfx95x), got {_ARCH}",
)


def _coords_conv_reference(coords_xyz, batch, features, weight):
    n = coords_xyz.shape[0]
    kv = 27
    dev = coords_xyz.device
    c = coords_xyz.to(torch.int64)
    cmin = c.min(0).values
    s = c - cmin
    sx, sy, sz = (c.max(0).values - cmin + 1).tolist()
    nb = int(batch.max().item()) + 1
    pres = torch.full((nb * sx * sy * sz,), -1, dtype=torch.int64, device=dev)
    b64 = batch.to(torch.int64)
    pres[((b64 * sx + s[:, 0]) * sy + s[:, 1]) * sz + s[:, 2]] = torch.arange(n, device=dev)
    sw = weight.reshape(weight.shape[0], weight.shape[1], kv).permute(2, 1, 0).float()
    out = torch.zeros(n, weight.shape[0], dtype=torch.float32, device=dev)
    ff = features.float()
    for k in range(kv):
        dx, dy, dz = k % 3 - 1, (k // 3) % 3 - 1, k // 9 - 1
        nx, ny, nz = s[:, 0] + dx, s[:, 1] + dy, s[:, 2] + dz
        inr = (nx >= 0) & (nx < sx) & (ny >= 0) & (ny < sy) & (nz >= 0) & (nz < sz)
        gi = (((b64 * sx + nx) * sy + ny) * sz + nz).clamp(0, nb * sx * sy * sz - 1)
        inp = torch.where(inr, pres[gi], torch.full_like(gi, -1))
        sel = inr & (inp >= 0)
        out[torch.arange(n, device=dev)[sel]] += ff[inp[sel]] @ sw[k]
    return out


def _generic_reference(coords, features, weight, kv_order):
    n = coords.shape[0]
    k = weight.shape[2]
    r = k // 2
    dev = coords.device
    c = coords.to(torch.int64)
    cmin = c.min(0).values
    s = c - cmin
    sx, sy, sz = (c.max(0).values - cmin + 1).tolist()
    pres = torch.full((sx * sy * sz,), -1, dtype=torch.int64, device=dev)
    pres[(s[:, 0] * sy + s[:, 1]) * sz + s[:, 2]] = torch.arange(n, device=dev)
    out = torch.zeros(n, weight.shape[0], dtype=torch.float32, device=dev)
    ff = features.float()
    idx = torch.arange(n, device=dev)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                w = weight[:, :, dz + r, dy + r, dx + r] if kv_order == "zyx" else weight[:, :, dx + r, dy + r, dz + r]
                nx, ny, nz = s[:, 0] + dx, s[:, 1] + dy, s[:, 2] + dz
                inr = (nx >= 0) & (nx < sx) & (ny >= 0) & (ny < sy) & (nz >= 0) & (nz < sz)
                gi = ((nx * sy + ny) * sz + nz).clamp(0, sx * sy * sz - 1)
                inp = torch.where(inr, pres[gi], torch.full_like(gi, -1))
                sel = inr & (inp >= 0)
                out[idx[sel]] += ff[inp[sel]] @ w.t().float()
    return out


@_skip_non_cdna4
@pytest.mark.parametrize(
    "k,c_in,c_out,box,n_active",
    [
        (3, 32, 32, 20, 432),
        (5, 6, 32, 20, 432),
        (5, 32, 64, 12, 200),
        (3, 6, 32, 12, 200),
        (3, 32, 40, 12, 200),
        (3, 10, 24, 12, 200),
        (3, 1, 1, 12, 100),
    ],
)
@pytest.mark.parametrize("kv_order", ["zyx", "spconv"])
def test_sparse_conv_f32_generic_shapes(k, c_in, c_out, box, n_active, kv_order):
    from kernels.conv.sparse_conv3d_implicit import clear_lut_cache, sparse_conv3d_from_coords_f32

    torch.manual_seed(n_active + c_in + c_out + k)
    device = "cuda"
    coords = _rand_voxels(box, n_active, n_active, device)
    features = torch.randn(n_active, c_in, dtype=torch.float32, device=device) * 0.1
    weight = torch.randn(c_out, c_in, k, k, k, dtype=torch.float32, device=device) * 0.1

    clear_lut_cache()
    out = sparse_conv3d_from_coords_f32(coords, features, weight, kv_order=kv_order)
    ref = _generic_reference(coords, features, weight, kv_order)
    torch.cuda.synchronize()
    assert out.shape == (n_active, c_out)
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


def _rand_voxels(box, n, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    lin = torch.randperm(box**3, generator=g, device=device)[:n]
    return torch.stack([lin // (box * box), (lin // box) % box, lin % box], 1).to(torch.int32)


@_skip_non_cdna4
@pytest.mark.parametrize(
    "box,n_active,c_in,c_out",
    [
        (20, 432, 32, 32),
        (12, 56, 128, 128),
        (6, 9, 512, 512),
    ],
)
def test_sparse_conv_f32_from_coords(box, n_active, c_in, c_out):
    from kernels.conv.sparse_conv3d_implicit import sparse_conv3d_from_coords_f32

    torch.manual_seed(n_active)
    device = "cuda"
    coords = _rand_voxels(box, n_active, n_active, device)
    features = torch.randn(n_active, c_in, dtype=torch.float32, device=device) * 0.1
    weight = torch.randn(c_out, c_in, 3, 3, 3, dtype=torch.float32, device=device) * 0.1

    out = sparse_conv3d_from_coords_f32(coords, features, weight)
    ref = _coords_conv_reference(coords, torch.zeros(n_active, dtype=torch.int32, device=device), features, weight)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@_skip_non_cdna4
@pytest.mark.parametrize("n_batch,per_batch,box,c_in,c_out", [(2, 200, 16, 32, 32), (4, 8, 6, 64, 64)])
def test_sparse_conv_f32_from_coords_batched(n_batch, per_batch, box, c_in, c_out):
    from kernels.conv.sparse_conv3d_implicit import sparse_conv3d_from_coords_f32

    torch.manual_seed(n_batch * 100 + per_batch)
    device = "cuda"
    xyz = torch.cat([_rand_voxels(box, per_batch, b * 77 + per_batch, device) for b in range(n_batch)])
    batch = torch.cat([torch.full((per_batch,), b, dtype=torch.int32, device=device) for b in range(n_batch)])
    coords4 = torch.cat([batch.unsqueeze(1), xyz], dim=1)
    n = xyz.shape[0]
    features = torch.randn(n, c_in, dtype=torch.float32, device=device) * 0.1
    weight = torch.randn(c_out, c_in, 3, 3, 3, dtype=torch.float32, device=device) * 0.1

    out = sparse_conv3d_from_coords_f32(coords4, features, weight)
    ref = _coords_conv_reference(xyz, batch, features, weight)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)

    out_ss = sparse_conv3d_from_coords_f32(coords4, features, weight, spatial_shape=(box, box, box), n_batch=n_batch)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_ss, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@_skip_non_cdna4
@pytest.mark.parametrize("box,n_active,c_in,c_out", [(20, 432, 32, 32), (6, 9, 512, 512)])
def test_sparse_conv_f32_spatial_shape_matches_auto(box, n_active, c_in, c_out):
    from kernels.conv.sparse_conv3d_implicit import sparse_conv3d_from_coords_f32

    torch.manual_seed(n_active)
    device = "cuda"
    coords = _rand_voxels(box, n_active, n_active, device)
    features = torch.randn(n_active, c_in, dtype=torch.float32, device=device) * 0.1
    weight = torch.randn(c_out, c_in, 3, 3, 3, dtype=torch.float32, device=device) * 0.1

    auto = sparse_conv3d_from_coords_f32(coords, features, weight)
    given = sparse_conv3d_from_coords_f32(coords, features, weight, spatial_shape=(box, box, box))
    torch.cuda.synchronize()
    torch.testing.assert_close(auto, given, rtol=1e-5, atol=1e-5)


@_skip_non_cdna4
@pytest.mark.parametrize("box,n_active,k", [(17, 100, 3), (17, 100, 5), (40, 1000, 3), (8, 7, 3)])
def test_zdelta_map_matches_dense(box, n_active, k):
    from kernels.conv.sparse_conv3d_implicit import build_lut_dense, build_lut_zdelta

    coords = _rand_voxels(box, n_active, n_active, "cuda")
    dense = build_lut_dense(coords, 16, spatial_shape=(box, box, box), kernel_size=k)
    zdelta = build_lut_zdelta(coords, 16, spatial_shape=(box, box, box), kernel_size=k)
    torch.cuda.synchronize()
    for name, a, b in zip(("lut", "mask", "akv", "count"), dense[:4], zdelta[:4]):
        assert torch.equal(a, b), f"{name} differs between dense and zdelta mappers"


@_skip_non_cdna4
def test_zdelta_handles_grid_too_large_for_dense():
    from kernels.conv.sparse_conv3d_implicit import ZDELTA_CELL_THRESHOLD, build_lut_auto

    ss = (2095, 2095, 695)
    assert ss[0] * ss[1] * ss[2] > ZDELTA_CELL_THRESHOLD
    torch.manual_seed(0)
    n = 4096
    xyz = torch.stack([(torch.rand(n, device="cuda") * ss[i]).to(torch.int32) for i in range(3)], dim=1).contiguous()
    xyz = torch.unique(xyz, dim=0)

    lut, mask, _akv, _cnt, num_tiles, num_act = build_lut_auto(xyz, 16, spatial_shape=ss, kernel_size=3)
    torch.cuda.synchronize()
    assert num_act == xyz.shape[0]
    assert lut.shape == (num_tiles, 27, 16)
    centre = (lut[:, 13, :] >= 0).sum().item()
    assert centre == num_act, f"centre tap found {centre} of {num_act} self-matches"


@_skip_non_cdna4
@pytest.mark.parametrize(
    "ss,expect32",
    [((64, 64, 64), True), ((1440, 1440, 40), True), ((2095, 2095, 695), False)],
)
def test_zdelta_key_width_selection(ss, expect32):
    from kernels.conv.sparse_conv3d_implicit import build_lut_zdelta

    r = 1
    fits = (ss[0] + 2 * r) * (ss[1] + 2 * r) * (ss[2] + 2 * r) < (1 << 31)
    assert fits is expect32

    torch.manual_seed(1)
    n = 2000
    xyz = torch.stack([(torch.rand(n, device="cuda") * ss[i]).to(torch.int32) for i in range(3)], dim=1).contiguous()
    xyz = torch.unique(xyz, dim=0)

    lut, _mask, _akv, _cnt, _nt, num_act = build_lut_zdelta(xyz, 16, spatial_shape=ss, kernel_size=3)
    torch.cuda.synchronize()
    assert (lut[:, 13, :] >= 0).sum().item() == num_act


@_skip_non_cdna4
def test_weight_cache_invalidates_on_mutation():
    from kernels.conv.sparse_conv3d_implicit import (
        clear_lut_cache,
        clear_weight_cache,
        sparse_conv3d_from_coords_f32,
    )

    torch.manual_seed(4)
    box, n, c = 20, 300, 32
    coords = _rand_voxels(box, n, n, "cuda")
    features = torch.randn(n, c, dtype=torch.float32, device="cuda") * 0.1
    weight = torch.randn(c, c, 3, 3, 3, dtype=torch.float32, device="cuda") * 0.1

    clear_lut_cache()
    clear_weight_cache()
    first = sparse_conv3d_from_coords_f32(coords, features, weight, spatial_shape=(box, box, box)).clone()
    with torch.no_grad():
        weight.mul_(2.0)
    second = sparse_conv3d_from_coords_f32(coords, features, weight, spatial_shape=(box, box, box))
    torch.cuda.synchronize()
    torch.testing.assert_close(second, first * 2.0, rtol=BF16_RTOL, atol=BF16_ATOL)

    other = torch.randn(c, c, 3, 3, 3, dtype=torch.float32, device="cuda") * 0.1
    out_other = sparse_conv3d_from_coords_f32(coords, features, other, spatial_shape=(box, box, box))
    torch.cuda.synchronize()
    assert (out_other - second).abs().max().item() > 1e-3


@_skip_non_cdna4
def test_weight_cache_does_not_leak():
    import gc

    from kernels.conv.sparse_conv3d_implicit import (
        _WPACK_CACHE,
        clear_weight_cache,
        sparse_conv3d_from_coords_f32,
    )

    torch.manual_seed(5)
    box, n, c = 16, 200, 32
    coords = _rand_voxels(box, n, n, "cuda")
    features = torch.randn(n, c, dtype=torch.float32, device="cuda") * 0.1

    clear_weight_cache()
    for _ in range(20):
        w = torch.randn(c, c, 3, 3, 3, dtype=torch.float32, device="cuda") * 0.1
        sparse_conv3d_from_coords_f32(coords, features, w, spatial_shape=(box, box, box))
        del w
    gc.collect()
    assert len(_WPACK_CACHE) <= 2, f"cache retained {len(_WPACK_CACHE)} entries for dead weights"


@_skip_non_cdna4
@pytest.mark.parametrize("box,n_active,c", [(24, 500, 128), (30, 1000, 256), (17, 200, 512), (12, 100, 128)])
def test_compacted_path_matches_indexed(box, n_active, c):
    import os

    from kernels.conv.sparse_conv3d_implicit import clear_lut_cache, sparse_conv3d_from_coords_f32

    torch.manual_seed(n_active)
    coords = _rand_voxels(box, n_active, n_active, "cuda")
    feat = torch.randn(n_active, c, dtype=torch.float32, device="cuda") * 0.1
    w = torch.randn(c, c, 3, 3, 3, dtype=torch.float32, device="cuda") * 0.1
    ss = (box, box, box)

    prev = os.environ.get("FLYDSL_SPCONV_COMPACT")
    try:
        os.environ["FLYDSL_SPCONV_COMPACT"] = "0"
        clear_lut_cache()
        ref = sparse_conv3d_from_coords_f32(coords, feat, w, spatial_shape=ss)
        torch.cuda.synchronize()
        ref = ref.clone()

        os.environ["FLYDSL_SPCONV_COMPACT"] = "1"
        os.environ["FLYDSL_SPCONV_COMPACT_MIN_N"] = "0"
        clear_lut_cache()
        got = sparse_conv3d_from_coords_f32(coords, feat, w, spatial_shape=ss)
        torch.cuda.synchronize()
    finally:
        os.environ.pop("FLYDSL_SPCONV_COMPACT_MIN_N", None)
        if prev is None:
            os.environ.pop("FLYDSL_SPCONV_COMPACT", None)
        else:
            os.environ["FLYDSL_SPCONV_COMPACT"] = prev
        clear_lut_cache()

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@_skip_non_cdna4
@pytest.mark.parametrize("box,n_active,c", [(24, 500, 128), (20, 400, 256), (17, 200, 512)])
def test_bf16_compacted_matches_fp32_reference(box, n_active, c):
    from kernels.common.tensor_shim import _run_compiled
    from kernels.conv.sparse_conv3d_implicit import (
        _compile_bf16_compacted,
        _pick_ct_per_block,
        _pick_k_unroll,
        _prepare_weight_bf16,
        build_compacted_pairs,
        build_lut_auto,
    )

    torch.manual_seed(n_active)
    coords = _rand_voxels(box, n_active, n_active, "cuda")
    feat = torch.randn(n_active, c, dtype=torch.float32, device="cuda").relu()
    w = torch.randn(c, c, 3, 3, 3, dtype=torch.float32, device="cuda") * (2.0 / c) ** 0.5

    lut, _mask, _a, _cn, nt, na = build_lut_auto(coords, 16, spatial_shape=(box, box, box), kernel_size=3)
    in_rows, out_rows, tile_kv, n_tiles_c = build_compacted_pairs(lut, nt, na, 16, 3)
    if n_tiles_c == 0:
        pytest.skip("no non-centre pairs at this shape")

    packed = _prepare_weight_bf16(w, "zyx", 16)[0]
    fn = _compile_bf16_compacted(c, c, _pick_ct_per_block(c), _pick_k_unroll(c))
    got = torch.zeros(na, c, dtype=torch.float32, device="cuda")
    _run_compiled(
        fn,
        feat.to(torch.bfloat16).contiguous(),
        packed,
        got,
        in_rows,
        out_rows,
        tile_kv,
        n_tiles_c,
        torch.cuda.current_stream(),
    )

    sw = w.reshape(c, c, -1).permute(2, 1, 0).contiguous()
    ref = torch.zeros(na, c, dtype=torch.float32, device="cuda")
    kv_per_slot = tile_kv.repeat_interleave(16)
    sel = out_rows >= 0
    ins, outs, kvs = in_rows[sel].long(), out_rows[sel].long(), kv_per_slot[sel].long()
    for kv in torch.unique(kvs).tolist():
        m = kvs == kv
        ref.index_add_(0, outs[m], feat[ins[m]] @ sw[kv])
    torch.cuda.synchronize()

    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 1e-2, f"bf16 relative error {rel:.2e} is too large for a rounding-only difference"


@_skip_non_cdna4
def test_compacted_pairs_cover_every_pair():
    from kernels.conv.sparse_conv3d_implicit import build_compacted_pairs, build_lut_auto

    torch.manual_seed(9)
    box, n = 20, 400
    coords = _rand_voxels(box, n, n, "cuda")
    lut, _mask, _a, _c, nt, na = build_lut_auto(coords, 16, spatial_shape=(box, box, box), kernel_size=3)
    in_rows, out_rows, tile_kv, n_tiles_c = build_compacted_pairs(lut, nt, na, 16, 3)
    torch.cuda.synchronize()

    lv = lut.reshape(nt, 27, 16)
    orow = torch.arange(nt * 16, device="cuda").reshape(nt, 1, 16)
    valid = (lv >= 0) & (orow < na)
    assert int((out_rows >= 0).sum().item()) == int(valid.sum().item())

    assert int(valid[:, 13].sum().item()) == na

    kv_per_slot = tile_kv.repeat_interleave(16)
    sel = out_rows >= 0
    o = out_rows[sel].long()
    assert torch.equal(lv[o // 16, kv_per_slot[sel].long(), o % 16], in_rows[sel])
