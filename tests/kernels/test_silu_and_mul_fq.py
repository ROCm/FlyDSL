#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused gate-activation-and-mul reduction kernel test (split-K MoE stage1 post-process).

`build_silu_and_mul_fq_module` applies the deferred activation (silu / swiglu / gelu_tanh)
on the raw gate/up partials produced by the split-K stage1 GEMM:

    out[row, :] = act(gate[row, :]) * up[row, :]

This exercises the full parameter space of the kernel: all three activations, every
`quant_mode` (bf16 `none`, `fp8`, `fp4`), bias on/off, and both input layouts
(`gui_layout` gate-up-separated vs block-interleaved). For `none` the activation math is
verified directly against a torch reference; for `fp8`/`fp4` the packed output plus the
kernel's shuffled sorted-scale buffer are decoded back to fp32 and compared with
quant-appropriate tolerance.
"""

import argparse
import os
import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.moe.silu_and_mul_fq import BLOCK_THREADS, build_silu_and_mul_fq_module  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

try:
    from tests.kernels.utils import fp4_utils  # noqa: E402

    _HAVE_FP4_UTILS = True
except Exception:  # triton not installed, etc.
    fp4_utils = None
    _HAVE_FP4_UTILS = False

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = get_rocm_arch()
# GFX950 (MI350) uses OCP standard float8_e4m3fn; older archs use the fnuz variant.
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

# Default swiglu clamp bound used by the kernel when swiglu_limit == 0.
_SWIGLU_LIMIT = 7.0
# MXFP4 / MXFP8 block size (fixed by spec).
_QUANT_BLOCK = 32


def _derive_vec(inter_dim: int) -> int:
    """Replicate the kernel's VEC derivation (silu_and_mul_fq.py)."""
    elems_per_thread = (inter_dim + BLOCK_THREADS - 1) // BLOCK_THREADS
    vec = max(elems_per_thread, 2)
    if vec % 2 != 0:
        vec += 1
    return vec


def _torch_ref(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
    """Host reference for act(gate) * up in fp32, matching the kernel formulas."""
    g = gate.float()
    u = up.float()
    if act == "silu":
        return F.silu(g) * u
    if act == "gelu_tanh":
        return F.gelu(g, approximate="tanh") * u
    if act == "swiglu":
        # gate: upper-clamped; linear: clamped to [-lim, lim]; then
        # gate * sigmoid(1.702 * gate) * (linear + 1).
        gate_c = torch.clamp(g, max=_SWIGLU_LIMIT)
        lin_c = torch.clamp(u, min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
        return gate_c * torch.sigmoid(1.702 * gate_c) * (lin_c + 1.0)
    raise ValueError(f"unknown act {act!r}")


def _pack_gui_layout(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up into the kernel's block-interleaved layout (block=16):

        [gate_0:16, up_0:16, gate_16:32, up_16:32, ...]

    Input gate/up are [rows, inter_dim]; output is [rows, 2*inter_dim].
    """
    rows, inter_dim = gate.shape
    assert inter_dim % 16 == 0, "gui_layout requires inter_dim divisible by 16"
    nblk = inter_dim // 16
    g = gate.view(rows, nblk, 16)
    u = up.view(rows, nblk, 16)
    inter = torch.stack((g, u), dim=2)  # [rows, nblk, 2, 16]
    return inter.reshape(rows, 2 * inter_dim).contiguous()


def _gather_sorted_scales(out_scale_sorted: torch.Tensor, rows: int, scale_cols: int) -> torch.Tensor:
    """Gather the e8m0 scale byte for every (row, block) out of the kernel's shuffled
    sorted-scale buffer, replicating the kernel's s_byte_off formula:

        d0=r>>5, d1=(r>>4)&1, d2=r&15, d3=c>>3, d4=(c>>2)&1, d5=c&3
        off = d0*(scale_cols*32) + d3*256 + d5*64 + d2*4 + d4*2 + d1

    Returns a [rows, scale_cols] uint8 tensor.
    """
    flat = out_scale_sorted.view(torch.uint8).reshape(-1)
    r = torch.arange(rows, device=out_scale_sorted.device).view(rows, 1)
    c = torch.arange(scale_cols, device=out_scale_sorted.device).view(1, scale_cols)
    d0 = r >> 5
    d1 = (r >> 4) & 1
    d2 = r & 15
    d3 = c >> 3
    d4 = (c >> 2) & 1
    d5 = c & 3
    off = d0 * (scale_cols * 32) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1
    return flat[off.reshape(-1).long()].reshape(rows, scale_cols)


def _dequant(out_buf: torch.Tensor, scales_e8m0: torch.Tensor, inter_dim: int, mode: str) -> torch.Tensor:
    """Decode the kernel's packed fp4/fp8 output + e8m0 block scales back to fp32."""
    scale_f32 = fp4_utils.e8m0_to_f32(scales_e8m0.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(_QUANT_BLOCK, dim=1)[:, :inter_dim]
    if mode == "fp4":
        codes = fp4_utils.mxfp4_to_f32(out_buf.view(torch.uint8))[:, :inter_dim]
    else:
        codes = fp4_utils.fp8_e4m3_to_f32(out_buf.view(torch.uint8))[:, :inter_dim]
    return codes * scale_expanded


def run_silu_and_mul_fq_test(
    token_num: int,
    topk: int,
    inter_dim: int,
    act: str = "silu",
    quant_mode: str = "none",
    enable_bias: bool = False,
    gui_layout: bool = False,
    num_experts: int = 8,
    num_iters: int = 20,
    num_warmup: int = 5,
):
    """Compile + launch the reduction kernel and check correctness for the given mode."""
    device = torch.device("cuda")
    torch.manual_seed(0)

    vec = _derive_vec(inter_dim)
    if gui_layout and vec > 16:
        pytest.skip(f"gui_layout requires VEC<=16, got VEC={vec} for inter_dim={inter_dim}")

    print(
        f"=== silu_and_mul_fq: token_num={token_num}, topk={topk}, inter_dim={inter_dim}, "
        f"act={act}, quant={quant_mode}, bias={enable_bias}, gui={gui_layout} ==="
    )

    launch = build_silu_and_mul_fq_module(
        inter_dim,
        topk,
        quant_mode=quant_mode,
        gui_layout=gui_layout,
        act=act,
        enable_bias=enable_bias,
    )

    rows = token_num * topk
    gate = torch.randn((rows, inter_dim), device=device, dtype=torch.bfloat16)
    up = torch.randn((rows, inter_dim), device=device, dtype=torch.bfloat16)
    if gui_layout:
        x = _pack_gui_layout(gate, up)
    else:
        x = torch.cat((gate, up), dim=1).contiguous()

    # 1:1 sorted-id map: block/sorted-row r <-> input/output row r.
    # Kernel derives in_row = token_id*topk + slot_id, so pack token=r//topk, slot=r%topk.
    tok = torch.arange(rows, device=device, dtype=torch.int32) // topk
    slot = torch.arange(rows, device=device, dtype=torch.int32) % topk
    sorted_ids = (tok | (slot << 24)).contiguous()
    num_valid_ids = torch.tensor([rows], device=device, dtype=torch.int32)

    # Bias path: topk_ids maps each in_row -> expert; bias is [experts, 2*inter_dim].
    if enable_bias:
        expert_of_row = torch.arange(rows, device=device, dtype=torch.int32) % num_experts
        topk_ids = expert_of_row.contiguous()
        bias = torch.randn((num_experts, 2 * inter_dim), device=device, dtype=torch.float32)
        gate_bias = bias[expert_of_row.long(), :inter_dim]
        up_bias = bias[expert_of_row.long(), inter_dim:]
        ref = _torch_ref(gate.float() + gate_bias, up.float() + up_bias, act)
    else:
        topk_ids = torch.empty((0,), device=device, dtype=torch.int32)
        bias = torch.empty((0,), device=device, dtype=torch.float32)
        ref = _torch_ref(gate, up, act)

    scale_cols = inter_dim // _QUANT_BLOCK
    if quant_mode == "none":
        out_buf = torch.empty((rows, inter_dim), device=device, dtype=torch.bfloat16)
        out_scale_sorted = torch.empty((0,), device=device, dtype=torch.uint8)
    else:
        out_cols = inter_dim // 2 if quant_mode == "fp4" else inter_dim
        out_buf = torch.empty((rows, out_cols), device=device, dtype=torch.uint8)
        # Shuffled sorted-scale buffer: rows padded to 256, scale-cols to 8.
        rows_pad = (rows + 255) // 256 * 256
        cols_pad = (scale_cols + 7) // 8 * 8
        out_scale_sorted = torch.zeros((rows_pad * cols_pad,), device=device, dtype=torch.uint8)

    stream = torch.cuda.current_stream()

    def _launch(o, xin):
        launch(
            xin,
            o,
            out_scale_sorted,
            sorted_ids,
            num_valid_ids,
            topk_ids,
            bias,
            token_num,
            rows,
            stream,
        )

    _, us = run_perftest(
        _launch,
        out_buf,
        x,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    torch.cuda.synchronize()

    if quant_mode == "none":
        assert verify_output(out_buf.float(), ref, rtol=1e-2, atol=1e-2, msg=f"[silu_and_mul_fq {act}]")
    else:
        scales = _gather_sorted_scales(out_scale_sorted, rows, scale_cols)
        deq = _dequant(out_buf, scales, inter_dim, quant_mode)
        # Quant paths: per-element error is large (fp4 has 7 magnitudes), so lean on
        # verify_output's cosine-similarity (logits_diff) fallback with a looser threshold.
        ld = 5e-3 if quant_mode == "fp8" else 5e-2
        rt = 0.1 if quant_mode == "fp8" else 0.25
        assert verify_output(
            deq, ref, rtol=rt, atol=rt, logits_diff_threshold=ld, msg=f"[silu_and_mul_fq {act} {quant_mode}]"
        )

    if quant_mode == "none":
        elem_bytes = x.element_size()
        bytes_moved = (rows * 2 * inter_dim + rows * inter_dim) * elem_bytes
        bw_gb_s = bytes_moved / 1e9 / (us / 1e6)
        print(f"[FlyDSL {act}] {us:.1f} us, Bandwidth: {bw_gb_s:.2f} GB/s")


_QUANT_PARAMS = [
    pytest.param("none", id="none"),
    pytest.param("fp8", id="fp8", marks=pytest.mark.skipif("gfx95" not in ARCH, reason="fp8 requires gfx950+")),
    pytest.param("fp4", id="fp4", marks=pytest.mark.skipif("gfx95" not in ARCH, reason="fp4 requires gfx950+")),
]


@pytest.mark.skipif(not _HAVE_FP4_UTILS, reason="fp4_utils not available (triton not installed)")
@pytest.mark.parametrize("gui_layout", [False, True], ids=["sep", "gui"])
@pytest.mark.parametrize("enable_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("quant_mode", _QUANT_PARAMS)
@pytest.mark.parametrize("act", ["silu", "swiglu", "gelu_tanh"])
@pytest.mark.parametrize(
    "token_num, topk, inter_dim",
    [
        pytest.param(1, 8, 512, id="decode-S"),
        pytest.param(5, 8, 512, id="decode-M"),
        pytest.param(65, 8, 1024, id="decode-L"),
        pytest.param(128, 6, 2048, id="prefill"),
    ],
)
def test_silu_and_mul_fq(
    token_num: int,
    topk: int,
    inter_dim: int,
    act: str,
    quant_mode: str,
    enable_bias: bool,
    gui_layout: bool,
):
    """Reduction-kernel correctness across act / quant / bias / layout."""
    run_silu_and_mul_fq_test(
        token_num=token_num,
        topk=topk,
        inter_dim=inter_dim,
        act=act,
        quant_mode=quant_mode,
        enable_bias=enable_bias,
        gui_layout=gui_layout,
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Fused gate-activation-and-mul reduction kernel — correctness & perf test.\n"
            "\n"
            "Applies act(gate)*up on split-K stage1 gate/up partials."
        ),
    )
    parser.add_argument("--token_num", "-t", type=int, default=128)
    parser.add_argument("--topk", "-k", type=int, default=8)
    parser.add_argument("--inter_dim", "-d", type=int, default=2048)
    parser.add_argument("--act", type=str, default="gelu_tanh", choices=["silu", "swiglu", "gelu_tanh"])
    parser.add_argument("--quant_mode", type=str, default="none", choices=["none", "fp8", "fp4"])
    parser.add_argument("--enable_bias", action="store_true")
    parser.add_argument("--gui_layout", action="store_true")
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=5)

    args = parser.parse_args()
    run_silu_and_mul_fq_test(
        token_num=args.token_num,
        topk=args.topk,
        inter_dim=args.inter_dim,
        act=args.act,
        quant_mode=args.quant_mode,
        enable_bias=args.enable_bias,
        gui_layout=args.gui_layout,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
    )
