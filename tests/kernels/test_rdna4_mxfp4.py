# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""RDNA4 MXFP4 numerical accuracy across scale modes, tails and graph replay."""

import pytest
import torch

from kernels.gemm.rdna4_mxfp4 import gemm, prepare_weights

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.fixture(autouse=True)
def require_rdna4():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm RDNA4")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] not in ("gfx1200", "gfx1201"):
        pytest.skip("requires gfx1200/gfx1201")


def make_case(shape, device="cuda", seed=19):
    m, n, k = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(m, k, generator=generator, device=device) * 3).to(torch.float8_e4m3fn)
    packed = torch.randint(0, 256, (n, k // 2), device=device, generator=generator, dtype=torch.uint8)
    scales = torch.randint(124, 132, (n, k // 32), device=device, generator=generator, dtype=torch.uint8)
    x_scale = torch.linspace(0.5, 1.5, m, device=device, dtype=torch.float32)
    return x, packed, x_scale, scales


def reference(x, packed, x_scale, scales):
    # Independent OCP MXFP4 decoding; no FlyDSL packing or conversion helpers.
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    values = magnitudes[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    weights = values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return (x.float() @ weights.T) * x_scale[:, None]


def check(actual, expected):
    assert torch.isfinite(actual).all()
    relative_rms = ((actual.float() - expected).square().mean() / expected.square().mean().clamp_min(1e-35)).sqrt()
    assert float(relative_rms) < 0.006


@pytest.mark.parametrize("shape", [(1, 70, 32), (7, 128, 96), (33, 70, 160), (73, 513, 1120), (257, 128, 512)])
@pytest.mark.parametrize("mode", ["e4m3", "e5m2", "grouped"])
@pytest.mark.parametrize("strategy", ["register", "hybrid", "auto"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_formats_tails_and_strategies(shape, mode, strategy, dtype):
    if strategy == "hybrid" and mode == "grouped":
        pytest.skip("exact grouped scaling uses register")
    x, packed, xs, scales = make_case(shape)
    prepared = prepare_weights(packed, scales, mode=mode)
    cfg = None if strategy == "auto" else {"engine": strategy}
    if strategy == "hybrid":
        cfg.update(tile_k=32, operand_prefetch=1)
    storage = torch.full((shape[0] * shape[1] + 128,), 127, device=x.device, dtype=dtype)
    out = storage[64:-64].view(shape[:2])
    y = gemm(x, prepared, xs, out=out, config=cfg)
    assert y.dtype == dtype and y.shape == shape[:2]
    check(y, reference(x, packed, xs, scales))
    assert bool((storage[:64] == 127).all()) and bool((storage[-64:] == 127).all())


@pytest.mark.parametrize("mode", ["e4m3", "e5m2", "grouped"])
@pytest.mark.parametrize("shape", [(73, 2117, 1120), (96, 2053, 8192), (2049, 63, 19040)])
def test_accuracy_after_changed_input_graph_replay(mode, shape):
    x, packed, xs, scales = make_case(shape)
    prepared = prepare_weights(packed, scales, mode=mode)
    y = gemm(x, prepared, xs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(x, prepared, xs, out=y)
    x.view(torch.uint8).bitwise_xor_(128)
    xs.mul_(0.5)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(y, reference(x, packed, xs, scales))


@pytest.mark.parametrize("mode,gap", [("e4m3", 14), ("e5m2", 28)])
def test_folded_subnormal_bits_are_lossless(mode, gap):
    x = torch.zeros(16, 64, device="cuda")
    x[:, :32] = 1
    x = x.to(torch.float8_e4m3fn)
    codes = torch.arange(16, device="cuda", dtype=torch.uint8)[:, None]
    packed = (codes | codes << 4).expand(16, 32).contiguous()
    scales = torch.full((16, 2), 100, device="cuda", dtype=torch.uint8)
    scales[:, 1] += gap
    xs = torch.ones(16, device="cuda")
    y = gemm(x, prepare_weights(packed, scales, mode=mode), xs)
    torch.testing.assert_close(y, reference(x, packed, xs, scales).to(y.dtype), atol=0, rtol=0)


def test_unsafe_scale_spans_use_exact_grouped_fallback():
    x, packed, xs, scales = make_case((33, 70, 160))
    scales[:, 0] = 100
    scales[:, 1] = 129
    prepared = prepare_weights(packed, scales)
    check(gemm(x, prepared, xs), reference(x, packed, xs, scales))


def test_bf16_epilogue_preserves_subnormal_output():
    k = 17408
    x = torch.full((16, k), 448.0, device="cuda").to(torch.float8_e4m3fn)
    packed = torch.full((16, k // 2), 0x77, device="cuda", dtype=torch.uint8)
    scales = torch.zeros(16, k // 32, device="cuda", dtype=torch.uint8)
    xs = torch.full((16,), 2.0**-30, device="cuda")
    y = gemm(x, prepare_weights(packed, scales), xs)
    bits = int(torch.tensor(448 * 6 * k * 2.0**-157, dtype=torch.bfloat16).view(torch.int16))
    assert bits != 0
    assert bool((y.view(torch.int16) == bits).all())


@pytest.mark.parametrize("shape", [(17, 17, 4096), (192, 2048, 8192), (512, 1024, 8192)])
def test_accuracy_on_narrow_and_prefill_shapes(shape):
    x, packed, xs, scales = make_case(shape)
    actual = gemm(x, prepare_weights(packed, scales), xs)
    check(actual, reference(x, packed, xs, scales))
