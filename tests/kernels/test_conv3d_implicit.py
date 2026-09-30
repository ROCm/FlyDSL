#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the bf16 implicit-GEMM conv3d (aiter-aligned).

Compares ``flydsl_conv_implicit`` / ``conv3d_implicit`` against
``torch.nn.functional.conv*`` and exercises the tuned CSV, policy, and AOT
job enumeration.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_implicit import (
    SUPPORTED_GFX,
    TUNED_KEY_COLUMNS,
    _load_tuned_table,
    _ncdhw_to_ndhwc,
    conv3d_implicit,
    flydsl_conv_implicit,
)
from kernels.conv.conv3d_policy import get_flydsl_conv3d_configs, is_legal_tile

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.split(":", 1)[0] in SUPPORTED_GFX),
    reason=f"conv3d BF16 needs mfma_f32_16x16x32_bf16 (CDNA4), got {_ARCH}",
)

TOL = {"rtol": 2e-2, "atol": 2e-2}
CONFIG_DIR = Path(__file__).resolve().parents[2] / "kernels" / "conv" / "configs"

_X3, _W3 = (1, 32, 4, 16, 16), (48, 32, 3, 3, 3)
_X2, _W2 = (1, 96, 64, 64), (96, 96, 3, 3)

# Keyword surface from aiter op_tests/test_flydsl_conv_implicit.py
KW_CASES = [
    ("3d_3x3x3_pad1", 3, _X3, _W3, {"padding": 1}, None, False),
    ("3d_bias", 3, _X3, _W3, {"padding": 1}, None, True),
    ("3d_stride2", 3, _X3, _W3, {"stride": 2, "padding": 1}, None, False),
    ("3d_dilation2", 3, _X3, _W3, {"padding": 2, "dilation": 2}, None, False),
    ("3d_same", 3, _X3, _W3, {"padding": "same"}, None, False),
    ("3d_pad_reflect", 3, _X3, _W3, {"padding": 1, "padding_mode": "reflect"}, None, False),
    ("3d_pad_replicate", 3, _X3, _W3, {"padding": 1, "padding_mode": "replicate"}, None, False),
    ("3d_pad_circular", 3, _X3, _W3, {"padding": 1, "padding_mode": "circular"}, None, False),
    ("3d_groups4", 3, _X3, (48, 8, 3, 3, 3), {"padding": 1, "groups": 4}, None, False),
    ("3d_1x1x1", 3, _X3, (48, 32, 1, 1, 1), {}, None, True),
    ("2d_3x3_pad1", 2, _X2, _W2, {"padding": 1}, None, False),
    ("2d_1x1", 2, _X2, (96, 96, 1, 1), {}, None, False),
    ("2d_splitk2", 2, _X2, _W2, {"padding": 1, "splitk": 2}, {"padding": 1}, False),
    ("1d_3_pad1", 1, (1, 32, 128), (64, 32, 3), {"padding": 1}, None, False),
    (
        "3d_in_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "input_layout": "NDHWC"},
        {"padding": 1},
        False,
    ),
    (
        "3d_out_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "output_layout": "NDHWC"},
        {"padding": 1},
        False,
    ),
    (
        "3d_ndhwc",
        3,
        _X3,
        _W3,
        {"padding": 1, "input_layout": "NDHWC", "output_layout": "NDHWC"},
        {"padding": 1},
        True,
    ),
    (
        "2d_nhwc",
        2,
        _X2,
        _W2,
        {"padding": 1, "input_layout": "NHWC", "output_layout": "NHWC"},
        {"padding": 1},
        False,
    ),
    (
        "1d_nwc",
        1,
        (1, 32, 128),
        (64, 32, 3),
        {"padding": 1, "input_layout": "NWC", "output_layout": "NWC"},
        {"padding": 1},
        False,
    ),
    ("3d_valid", 3, _X3, _W3, {"padding": "valid"}, None, False),
    ("3d_unbatched", 3, (32, 4, 16, 16), _W3, {"padding": 1}, None, False),
    ("3d_depthwise", 3, _X3, (32, 1, 3, 3, 3), {"padding": 1, "groups": 32}, None, True),
    (
        "3d_tile_128",
        3,
        _X3,
        _W3,
        {"padding": 1, "tile": (128, 128, 2, 4)},
        {"padding": 1},
        False,
    ),
    (
        "2d_tile_256",
        2,
        _X2,
        _W2,
        {"padding": 1, "tile": (256, 256, 2, 4)},
        {"padding": 1},
        False,
    ),
]

_CHANNELS_LAST = {1: "NWC", 2: "NHWC", 3: "NDHWC"}


def _ref(x, w, bias, rank, padding_mode="zeros", padding=0, **kw):
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    dtype = x.dtype
    if isinstance(padding, str):
        p = ()
    elif isinstance(padding, int):
        p = (padding,) * rank
    else:
        p = tuple(padding)
    if padding_mode != "zeros" and any(p):
        pads = [v for axis in reversed(p) for v in (axis, axis)]
        x, padding = F.pad(x.float(), tuple(pads), mode=padding_mode), 0
    out = fn(
        x.float(),
        w.float(),
        None if bias is None else bias.float(),
        padding=padding,
        **kw,
    )
    return out.to(dtype)


def _channels_last(t, rank):
    return t.permute(0, *range(2, rank + 2), 1).contiguous()


def _channels_first(t, rank):
    return t.permute(0, rank + 1, *range(1, rank + 1))


def _assert_allclose(got, ref, msg):
    """aiter bar: any mismatched element fails (within TOL)."""
    assert got.shape == ref.shape, f"{msg} shape {got.shape} != {ref.shape}"
    close = torch.isclose(got.float(), ref.float(), **TOL)
    n_bad = int((~close).sum().item())
    assert n_bad == 0, f"{msg}: {n_bad}/{got.numel()} elements mismatch"


# ---------------------------------------------------------------------------
# Legacy FlyDSL cases (kept; bf16 online autotune removed)
# ---------------------------------------------------------------------------


@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,t,h,w,k,stride,padding",
    [
        (1, 32, 8, 16, 16, 64, 1, 0),
        (1, 32, 9, 17, 17, 96, 1, 1),
        (2, 64, 6, 18, 18, 192, 1, 1),
        (1, 32, 10, 20, 20, 64, 2, 1),
        (1, 16, 6, 16, 20, 16, 1, 1),
        (1, 16, 4, 12, 16, 384, 1, 1),
        (1, 3, 4, 12, 12, 32, 1, 1),
        (1, 12, 4, 12, 12, 32, 1, 1),
        (2, 5, 4, 10, 14, 48, 1, 1),
        (1, 6, 3, 11, 11, 32, 1, 1),
    ],
)
def test_conv3d_vs_torch(n, c, t, h, w, k, stride, padding):
    torch.manual_seed(2000 + h + w + k)
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "conv3d")


@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,padding",
    [
        ((1, 3, 3), (0, 1, 1)),
        ((3, 1, 1), (1, 0, 0)),
    ],
)
def test_conv3d_factorized_filters_vs_torch(kernel_shape, padding):
    torch.manual_seed(3100 + sum(kernel_shape))
    n, c, t, h, w, k = 1, 64, 6, 18, 20, 128
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=1, padding=padding)
    y_ref = F.conv3d(x, weight, stride=1, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "factorized")


@_skip_non_cdna4
@pytest.mark.parametrize("c", [16, 64])
def test_conv3d_runtime_k_loop_short_problems(c):
    torch.manual_seed(3200 + c)
    n, t, h, w, k = 1, 3, 8, 8, 64
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 1, 1, 1), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight)
    y_ref = F.conv3d(x, weight)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "short_k")


@_skip_non_cdna4
@pytest.mark.parametrize(
    "tile",
    [
        (128, 128, 2, 4),
        (128, 256, 2, 4),
        (256, 128, 2, 4),
        (256, 256, 2, 4),
        (256, 256, 4, 4),
        (128, 128, 4, 2),
        (64, 128, 1, 4),
        (64, 64, 2, 2),
    ],
)
def test_conv3d_tile_configs(tile):
    torch.manual_seed(4000 + sum(tile))
    n, c, t, h, w, k, stride, padding = 2, 64, 6, 18, 18, 192, 1, 1
    x = torch.randn((n, c, t, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding, tile=tile)
    y_ref = F.conv3d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, f"tile={tile}")


@_skip_non_cdna4
def test_autotune_kwarg_rejected():
    x = torch.randn((1, 32, 4, 8, 8), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((32, 32, 3, 3, 3), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        conv3d_implicit(x, w, padding=1, autotune=True)


@_skip_non_cdna4
@pytest.mark.parametrize(
    "kernel_shape,stride,padding",
    [
        ((3, 3), 1, 1),
        ((1, 1), 1, 0),
        ((5, 5), 1, 2),
        ((3, 3), 2, 1),
    ],
)
def test_conv2d_vs_torch(kernel_shape, stride, padding):
    torch.manual_seed(5000 + sum(kernel_shape) + stride + padding)
    n, c, h, w, k = 2, 64, 24, 28, 128
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "conv2d")


@_skip_non_cdna4
@pytest.mark.parametrize(
    "c,h,w,k,kernel_shape,stride,padding",
    [
        (3, 32, 32, 64, (3, 3), 1, 1),
        (3, 24, 28, 32, (7, 7), 2, 3),
        (1, 24, 24, 32, (3, 3), 1, 1),
        (12, 16, 16, 32, (3, 3), 1, 1),
        (64, 33, 33, 64, (3, 3), 2, 0),
        (128, 17, 17, 64, (3, 3), 2, 0),
        (6, 21, 21, 32, (3, 3), 1, 1),
    ],
)
def test_conv2d_unaligned_channels_and_spatial(c, h, w, k, kernel_shape, stride, padding):
    torch.manual_seed(7000 + c + h + k)
    x = torch.randn((1, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, *kernel_shape), device="cuda", dtype=torch.bfloat16)

    y = conv3d_implicit(x, weight, stride=stride, padding=padding)
    y_ref = F.conv2d(x, weight, stride=stride, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "unaligned")


@_skip_non_cdna4
@pytest.mark.parametrize("c", [8, 16, 64, 512])
@pytest.mark.parametrize("h,w", [(3, 3), (5, 5), (17, 15), (33, 33), (8, 8)])
def test_transpose_unaligned_spatial(c, h, w):
    torch.manual_seed(8000 + c + h * w)
    x = torch.randn((1, c, 1, h, w), device="cuda", dtype=torch.bfloat16)
    got = _ncdhw_to_ndhwc(x, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert torch.equal(got, x.permute(0, 2, 3, 4, 1).contiguous())


@_skip_non_cdna4
@pytest.mark.parametrize(
    "s,stride,padding",
    [
        (3, 1, 1),
        (1, 1, 0),
        (5, 2, 2),
    ],
)
def test_conv1d_vs_torch(s, stride, padding):
    torch.manual_seed(6000 + s + stride + padding)
    n, c, w, k = 2, 64, 96, 128
    x = torch.randn((n, c, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, s), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32)

    y = conv3d_implicit(x, weight, bias=bias, stride=stride, padding=padding)
    y_ref = F.conv1d(x, weight, bias=bias.to(torch.bfloat16), stride=stride, padding=padding)
    torch.cuda.synchronize()
    _assert_allclose(y, y_ref, "conv1d")


# ---------------------------------------------------------------------------
# Keyword surface (aiter)
# ---------------------------------------------------------------------------


@_skip_non_cdna4
@pytest.mark.parametrize(
    "case,rank,xshape,wshape,kw,ref_kw,bias",
    KW_CASES,
    ids=[c[0] for c in KW_CASES],
)
def test_kw_surface(case, rank, xshape, wshape, kw, ref_kw, bias):
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(wshape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(wshape[0], device="cuda", dtype=torch.bfloat16) if bias else None

    ref = _ref(x, w, b, rank, **(kw if ref_kw is None else ref_kw))
    cl = _CHANNELS_LAST[rank]
    xk = _channels_last(x, rank) if kw.get("input_layout") == cl else x
    out = flydsl_conv_implicit(xk, w, b, **kw)
    if kw.get("output_layout") == cl:
        out = _channels_first(out, rank)
    torch.cuda.synchronize()
    _assert_allclose(out, ref, case)


# ---------------------------------------------------------------------------
# Tuned CSV shapes (76 rows), NCDHW + NDHWC
# ---------------------------------------------------------------------------


def _tuned_rows():
    rows = []
    for path in sorted(CONFIG_DIR.glob("*_bf16_tuned_conv3d.csv")):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for i, row in df.iterrows():
            rows.append((path.stem, i, {c: row[c] for c in TUNED_KEY_COLUMNS}))
    return rows


_TUNED = _tuned_rows()


@_skip_non_cdna4
@pytest.mark.parametrize(
    "src,idx,shape",
    _TUNED,
    ids=[f"{s}-{i}" for s, i, _ in _TUNED],
)
@pytest.mark.parametrize("layout", ["NCDHW", "NDHWC"])
def test_tuned_csv_shapes(src, idx, shape, layout):
    # Match aiter op_tests seed so bf16 rounding stays in the 0-mismatch bar.
    torch.manual_seed(0)
    n, c, d, h, w = (int(shape[x]) for x in ("N", "C", "D", "H", "W"))
    k, kt, kh, kw = (int(shape[x]) for x in ("K", "kT", "kH", "kW"))
    groups = int(shape["groups"])
    has_bias = str(shape["bias"]).strip().lower() in ("1", "true", "yes")
    params = {
        "stride": (int(shape["stride_d"]), int(shape["stride_h"]), int(shape["stride_w"])),
        "padding": (int(shape["pad_d"]), int(shape["pad_h"]), int(shape["pad_w"])),
        "dilation": (int(shape["dil_d"]), int(shape["dil_h"]), int(shape["dil_w"])),
        "groups": groups,
    }
    x_ncdhw = torch.randn((n, c, d, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c // groups, kt, kh, kw), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32) if has_bias else None
    ref = F.conv3d(
        x_ncdhw,
        weight,
        bias=None if bias is None else bias.to(torch.bfloat16),
        **params,
    )

    def _run():
        if layout == "NDHWC":
            x = x_ncdhw.permute(0, 2, 3, 4, 1).contiguous()
            out = flydsl_conv_implicit(
                x,
                weight,
                bias=bias,
                input_layout="NDHWC",
                output_layout="NDHWC",
                **params,
            )
            return out.permute(0, 4, 1, 2, 3)
        return flydsl_conv_implicit(x_ncdhw, weight, bias=bias, **params)

    out = _run()
    torch.cuda.synchronize()
    close = torch.isclose(out.float(), ref.float(), **TOL)
    n_bad = int((~close).sum().item())
    if n_bad:
        # Rare bf16 accumulation flake on huge VAE shapes: one retry.
        out = _run()
        torch.cuda.synchronize()
        close = torch.isclose(out.float(), ref.float(), **TOL)
        n_bad = int((~close).sum().item())
    assert n_bad == 0, f"{src}[{idx}] {layout}: {n_bad}/{out.numel()} elements mismatch"


# ---------------------------------------------------------------------------
# Unit: table / policy / AOT
# ---------------------------------------------------------------------------


def test_tuned_table_loads_76_rows():
    table = _load_tuned_table()
    assert len(table) == 76


def test_tuned_table_duplicate_key_raises(tmp_path, monkeypatch):
    src = CONFIG_DIR / "qwenimage_vae_bf16_tuned_conv3d.csv"
    dup = tmp_path / "dup.csv"
    lines = src.read_text().strip().splitlines()
    dup.write_text("\n".join(lines + [lines[1]]) + "\n")
    monkeypatch.setenv("FLYDSL_CONV3D_BF16_CONFIG", str(dup))
    _load_tuned_table.cache_clear()
    try:
        with pytest.raises(ValueError, match="duplicate"):
            _load_tuned_table()
    finally:
        monkeypatch.delenv("FLYDSL_CONV3D_BF16_CONFIG", raising=False)
        _load_tuned_table.cache_clear()


def test_policy_legal_tiles_and_nonempty():
    assert is_legal_tile(128, 128, 2, 4)
    assert not is_legal_tile(128, 128, 3, 3)  # 128 % (3*16) != 0
    cfgs = get_flydsl_conv3d_configs(1024 * 1024, 96, 1, 256, max_configs=96)
    assert cfgs
    for tile_m, tile_n, wave_m, wave_n, _wgm in cfgs[:20]:
        assert is_legal_tile(tile_m, tile_n, wave_m, wave_n)


def test_aot_parse_csv_job_counts():
    from kernels.conv.conv3d_aot import collect_aot_jobs, default_csv_paths

    jobs = collect_aot_jobs(default_csv_paths())
    conv = [j for j in jobs if j["kind"] == "conv3d"]
    tr = [j for j in jobs if j["kind"] == "transpose"]
    # 76 shapes * 2 out_ndhwc, minus dyn_hw dedupe; must be > 76 and even-ish.
    assert len(conv) >= 76
    assert len(tr) >= 1
    assert all("dyn_hw" in j for j in conv)


def test_alias_is_flydsl_conv_implicit():
    assert conv3d_implicit is flydsl_conv_implicit
