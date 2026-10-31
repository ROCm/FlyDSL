#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1100 dense flash-attention correctness and regression tests.

These tests focus on catching the regressions called out in
https://github.com/ROCm/FlyDSL/pull/1216 during the 2026-10-28 review.
They purposely run only on gfx1100-class GPUs (W7900/W7900D).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Callable, Literal, NamedTuple

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - the suite only runs with the deps.
    np = None
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping gfx1100 tests.", allow_module_level=True)

import kernels.attention.flash_attn_interface as flash_attn_interface  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx1100 import build_flash_attn_func_module, hoist_q  # noqa: E402
from kernels.attention.flash_attn_gfx1100_autotune import pick_tile  # noqa: E402
from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # noqa: E402

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx110"):
    pytest.skip(f"gfx1100 flash attention tests require gfx110*, got {_ARCH}", allow_module_level=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers


def _as_dtype(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast ``value`` to ``dtype`` on the same device, keeping the original layout."""

    return value.to(dtype=dtype, copy=True)


def _bottom_right_causal_mask(seq_q: int, seq_kv: int, device: torch.device) -> torch.Tensor:
    """Return a [seq_q, seq_kv] boolean mask implementing bottom-right causal alignment."""

    offset = seq_kv - seq_q
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_kv, device=device).unsqueeze(0)
    allowed_upto = offset + q_idx
    return k_idx <= allowed_upto


def _softmax_u64(scores: torch.Tensor, causal_mask: torch.Tensor | None) -> torch.Tensor:
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    scores = scores - scores.max(dim=-1, keepdim=True).values
    return torch.softmax(scores, dim=-1)


def _ref_attn_fp64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Compute the flash-attention reference in float64 with bottom-right alignment."""

    # Shapes: [B, Sq, H, D] and [B, Skv, H, D].
    q_64 = q.to(torch.float64)
    k_64 = k.to(torch.float64)
    v_64 = v.to(torch.float64)
    batch, seq_q, num_heads, head_dim = q_64.shape
    seq_kv = k_64.shape[1]

    q_flat = q_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_q, head_dim)
    k_flat = k_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_kv, head_dim)
    v_flat = v_64.permute(0, 2, 1, 3).reshape(batch * num_heads, seq_kv, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    if causal:
        mask = _bottom_right_causal_mask(seq_q, seq_kv, q_64.device).unsqueeze(0)
    else:
        mask = None

    scores = torch.matmul(q_flat, k_flat.transpose(1, 2)) * scale
    probs = _softmax_u64(scores, mask)
    out = torch.matmul(probs, v_flat)
    return out.view(batch, num_heads, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()


def _ulp_err(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> float:
    """Return the maximum per-row difference (in ulp of ``dtype`` at |ref| max)."""

    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: out={tuple(out.shape)}, ref={tuple(ref.shape)}")

    out64 = out.to(torch.float64)
    ref64 = ref.to(torch.float64)
    rows = out64.reshape(-1, out64.shape[-1])
    ref_rows = ref64.reshape_as(rows)
    abs_diff = (rows - ref_rows).abs()
    ref_abs = ref_rows.abs()
    row_max_abs = ref_abs.max(dim=-1).values

    dtype_info = torch.finfo(dtype)
    # Treat fully masked rows as having 0 ulp error.
    safe_row_max = torch.where(
        row_max_abs == 0,
        torch.full_like(row_max_abs, dtype_info.tiny, dtype=torch.float64),
        row_max_abs,
    )
    row_max_dtype = safe_row_max.to(dtype)
    next_up = torch.nextafter(row_max_dtype, torch.full_like(row_max_dtype, float("inf")))
    ulp = (next_up - row_max_dtype).to(torch.float64)
    ulp = torch.where(row_max_abs == 0, torch.full_like(ulp, dtype_info.tiny, dtype=torch.float64), ulp)
    row_max_diff = abs_diff.max(dim=-1).values
    ulp_ratio = row_max_diff / ulp
    return float(ulp_ratio.max().item())


def _seed_all(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentionInput(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _make_inputs(
    batch: int,
    num_heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    num_kv_heads: int | None = None,
    v_kind: Literal["random", "ones"] = "random",
    layout: Literal["bshd"] = "bshd",
    device: torch.device | None = None,
) -> AttentionInput:
    if layout != "bshd":
        raise NotImplementedError(f"layout {layout} not supported yet")

    if device is None:
        device = torch.device("cuda")

    if num_kv_heads is None:
        num_kv_heads = num_heads

    torch_dtype = dtype
    q = torch.randn(batch, seq_q, num_heads, head_dim, dtype=torch.float64, device=device)
    k = torch.randn(batch, seq_kv, num_kv_heads, head_dim, dtype=torch.float64, device=device)
    if v_kind == "ones":
        v = torch.ones(batch, seq_kv, num_kv_heads, head_dim, dtype=torch.float64, device=device)
    else:
        v = torch.randn(batch, seq_kv, num_kv_heads, head_dim, dtype=torch.float64, device=device)
    q = _as_dtype(q, torch_dtype)
    k = _as_dtype(k, torch_dtype)
    v = _as_dtype(v, torch_dtype)
    return AttentionInput(q=q, k=k, v=v)


def _cpu_placeholder(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    base_shape = [1] * (len(shape) - 1) + [shape[-1]]
    base = torch.empty(base_shape, dtype=dtype, device="cpu")
    return base.expand(shape)


def _factor_batch_heads(bh: int) -> tuple[int, int]:
    for heads in (64, 32, 16, 8, 4, 2, 1):
        if bh % heads == 0:
            return bh // heads, heads
    return bh, 1


# ──────────────────────────────────────────────────────────────────────────────
# Isolated subprocess harness


class IsolatedResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def _run_isolated(case: str) -> IsolatedResult:
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    proc = subprocess.run(
        [sys.executable, "-m", "tests.kernels.test_flash_attn_gfx1100", "--case", case],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return IsolatedResult(proc.returncode, proc.stdout, proc.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Case implementations


@dataclasses.dataclass(frozen=True)
class DecodeCase:
    dtype: torch.dtype
    head_dim: int
    batch: int
    num_heads: int
    seq_q: int
    seq_kv: int
    causal: bool


def _case_decode_matrix() -> dict[str, float]:
    device = torch.device("cuda")
    worst: dict[str, float | int | str | bool] | None = None
    max_ulp = 0.0
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 128, 256):
            for batch in (1, 2):
                for num_heads in (1, 4):
                    for seq_kv in (64, 4096):
                        for causal in (False, True):
                            for seq_q in (1, 2, 7, 16):
                                _seed_all(20261029)
                                case = DecodeCase(
                                    dtype=dtype,
                                    head_dim=head_dim,
                                    batch=batch,
                                    num_heads=num_heads,
                                    seq_q=seq_q,
                                    seq_kv=seq_kv,
                                    causal=causal,
                                )
                                inputs = _make_inputs(
                                    batch=case.batch,
                                    num_heads=case.num_heads,
                                    seq_q=case.seq_q,
                                    seq_kv=case.seq_kv,
                                    head_dim=case.head_dim,
                                    dtype=case.dtype,
                                    device=device,
                                )
                                out = flydsl_flash_attn_func(
                                    inputs.q,
                                    inputs.k,
                                    inputs.v,
                                    causal=case.causal,
                                )
                                torch.cuda.synchronize()
                                ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=case.causal)
                                ulp = _ulp_err(out, ref.to(out.dtype), dtype=case.dtype)
                                if ulp > max_ulp:
                                    max_ulp = ulp
                                    worst = {
                                        "dtype": str(case.dtype),
                                        "head_dim": case.head_dim,
                                        "batch": case.batch,
                                        "num_heads": case.num_heads,
                                        "seq_q": case.seq_q,
                                        "seq_kv": case.seq_kv,
                                        "causal": case.causal,
                                        "max_ulp": ulp,
                                    }
    return {"max_ulp": max_ulp, "worst": worst or {}}


def _case_v_equals_one(dtype: torch.dtype) -> dict[str, float]:
    _seed_all(20261029)
    device = torch.device("cuda")
    inputs = _make_inputs(
        batch=2,
        num_heads=4,
        seq_q=16,
        seq_kv=4096,
        head_dim=128,
        dtype=dtype,
        v_kind="ones",
        device=device,
    )
    out = flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=True)
    torch.cuda.synchronize()
    ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=True)
    diff = (out - inputs.v[: out.shape[0], : out.shape[1]]).abs().max().item()
    max_ulp = _ulp_err(out, ref.to(out.dtype), dtype=dtype)
    return {"dtype": str(dtype), "max_abs_diff": diff, "max_ulp": max_ulp}


def _case_guard_buffer() -> dict[str, object]:
    _seed_all(20261029)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    num_heads = 4
    seq_q = 512
    seq_kv = 512
    head_dim = 128
    inputs = _make_inputs(batch, num_heads, seq_q, seq_kv, head_dim, dtype, device=device)

    # The kernel expects to write ``seq_q`` rows, but we deliberately expose only half of them.
    expected_elems = batch * seq_q * num_heads * head_dim
    exposed_seq_q = seq_q // 2
    exposed_elems = batch * exposed_seq_q * num_heads * head_dim
    guard_elems = 4096
    sentinel_value = -1.2345
    sentinel = torch.full(
        (guard_elems + expected_elems + guard_elems,),
        sentinel_value,
        dtype=dtype,
        device=device,
    )
    out_start = guard_elems
    out_view = sentinel[out_start : out_start + exposed_elems].view(batch, exposed_seq_q, num_heads, head_dim)

    raised: str | None = None
    error_msg: str | None = None
    out_tensor: torch.Tensor | None = None
    expected_shape = (batch, seq_q, num_heads, head_dim)
    view_shape = (batch, exposed_seq_q, num_heads, head_dim)

    try:
        out_tensor = flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, out=out_view, causal=True)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - record the failure mode for the report
        raised = type(exc).__name__
        error_msg = str(exc)
        torch.cuda.synchronize()

    post_start = out_start + exposed_elems
    post_end = out_start + expected_elems
    touched_left = bool(torch.any(sentinel[:out_start] != sentinel_value))
    touched_middle = bool(torch.any(sentinel[post_start:post_end] != sentinel_value))
    touched_right = bool(torch.any(sentinel[post_end:] != sentinel_value))
    guard_clean = not (touched_left or touched_middle or touched_right)

    max_ulp: float | None
    out_shape: tuple[int, ...] | None
    if out_tensor is not None:
        out_shape = tuple(int(dim) for dim in out_tensor.shape)
        ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=True)
        if out_tensor.shape == ref.shape:
            max_ulp = _ulp_err(out_tensor, ref.to(out_tensor.dtype), dtype=dtype)
        else:
            max_ulp = None
            error_msg = (error_msg or "") + (
                f"; output shape {out_shape} does not match reference {tuple(int(d) for d in ref.shape)}"
            ).lstrip("; ")
    else:
        out_shape = None
        max_ulp = None

    return {
        "raised": raised,
        "error": error_msg,
        "guard_clean": guard_clean,
        "touched_before": touched_left,
        "touched_after_view": touched_middle,
        "touched_tail_guard": touched_right,
        "max_ulp": max_ulp,
        "expected_shape": expected_shape,
        "view_shape": view_shape,
        "out_shape": out_shape,
    }


def _case_dense_matrix() -> dict[str, float | dict[str, object]]:
    device = torch.device("cuda")
    worst: dict[str, object] | None = None
    max_ulp = 0.0
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 128, 256):
            for causal in (False, True):
                for batch in (1, 2):
                    for num_heads in (1, 4):
                        for seq_q in (128, 256):
                            seq_kv = seq_q
                            _seed_all(20261029)
                            inputs = _make_inputs(
                                batch=batch,
                                num_heads=num_heads,
                                seq_q=seq_q,
                                seq_kv=seq_kv,
                                head_dim=head_dim,
                                dtype=dtype,
                                device=device,
                            )
                            out = flydsl_flash_attn_func(
                                inputs.q,
                                inputs.k,
                                inputs.v,
                                causal=causal,
                            )
                            torch.cuda.synchronize()
                            ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=causal)
                            ulp = _ulp_err(out, ref.to(out.dtype), dtype=dtype)
                            if ulp > max_ulp:
                                max_ulp = ulp
                                worst = {
                                    "dtype": str(dtype),
                                    "head_dim": head_dim,
                                    "batch": batch,
                                    "num_heads": num_heads,
                                    "seq_q": seq_q,
                                    "seq_kv": seq_kv,
                                    "causal": causal,
                                    "max_ulp": float(ulp),
                                }
    return {"max_ulp": max_ulp, "worst": worst or {}}


def _case_unaligned_lengths() -> dict[str, float | dict[str, int | float | str]]:
    device = torch.device("cuda")
    _seed_all(20261029)
    seq_q = 1000
    seq_kv = seq_q
    worst: dict[str, int | float | str] | None = None
    max_ulp = 0.0
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 128):
            inputs = _make_inputs(
                batch=2,
                num_heads=4,
                seq_q=seq_q,
                seq_kv=seq_kv,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            for causal in (False, True):
                out = flydsl_flash_attn_func(
                    inputs.q,
                    inputs.k,
                    inputs.v,
                    causal=causal,
                )
                torch.cuda.synchronize()
                ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=causal)
                ulp = _ulp_err(out, ref.to(out.dtype), dtype=dtype)
                if ulp > max_ulp:
                    max_ulp = ulp
                    worst = {
                        "dtype": str(dtype),
                        "head_dim": head_dim,
                        "causal": causal,
                        "max_ulp": float(ulp),
                    }
    return {"max_ulp": max_ulp, "worst": worst or {}}


def _case_cross_attention_bottom_right() -> dict[str, float | dict[str, object] | bool]:
    device = torch.device("cuda")
    max_ulp = 0.0
    worst: dict[str, object] | None = None
    fully_masked_zero = True
    for dtype in (torch.float16, torch.bfloat16):
        _seed_all(20261029)
        seq_q = 16
        seq_kv = 8
        inputs = _make_inputs(
            batch=2,
            num_heads=4,
            seq_q=seq_q,
            seq_kv=seq_kv,
            head_dim=128,
            dtype=dtype,
            device=device,
        )
        out = flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=True)
        torch.cuda.synchronize()
        ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=True)
        ulp = _ulp_err(out, ref.to(out.dtype), dtype=dtype)
        if ulp > max_ulp:
            max_ulp = ulp
            worst = {
                "dtype": str(dtype),
                "max_ulp": float(ulp),
            }
        mask = _bottom_right_causal_mask(seq_q, seq_kv, device=device)
        fully_masked_rows = torch.where(mask.sum(dim=-1) == 0)[0]
        if fully_masked_rows.numel():
            zeros = torch.zeros_like(out[:, fully_masked_rows])
            if not torch.allclose(out[:, fully_masked_rows], zeros):
                fully_masked_zero = False
    return {"max_ulp": max_ulp, "worst": worst or {}, "fully_masked_zero": fully_masked_zero}


def _case_deterministic() -> dict[str, float | bool]:
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=2,
        num_heads=8,
        seq_q=32,
        seq_kv=32,
        head_dim=128,
        dtype=torch.float16,
        device=device,
    )
    out1 = flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False)
    torch.cuda.synchronize()
    out2 = flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False)
    torch.cuda.synchronize()
    ref = _ref_attn_fp64(inputs.q, inputs.k, inputs.v, causal=False)
    return {
        "deterministic": bool(torch.equal(out1, out2)),
        "max_ulp": _ulp_err(out1, ref.to(out1.dtype), dtype=out1.dtype),
    }


_CASE_REGISTRY: dict[str, Callable[[], dict[str, object]]] = {
    "decode_matrix": _case_decode_matrix,
    "v_equals_one_f16": lambda: _case_v_equals_one(torch.float16),
    "v_equals_one_bf16": lambda: _case_v_equals_one(torch.bfloat16),
    "guard_buffer": _case_guard_buffer,
    "dense_matrix": _case_dense_matrix,
    "unaligned_lengths": _case_unaligned_lengths,
    "cross_attention": _case_cross_attention_bottom_right,
    "deterministic": _case_deterministic,
}


def _dispatch_case(case: str) -> dict[str, object]:
    if case not in _CASE_REGISTRY:
        raise SystemExit(f"unknown case: {case}")
    return _CASE_REGISTRY[case]()


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="run a single flash-attn gfx1100 case")
    parser.add_argument("--case", required=True)
    args = parser.parse_args(argv)
    result = _dispatch_case(args.case)
    print(json.dumps(result))


# ──────────────────────────────────────────────────────────────────────────────
# PyTest entry points


def test_decode_matrix():
    res = _run_isolated("decode_matrix")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["max_ulp"] <= 1.0, payload


@pytest.mark.parametrize("case", ["v_equals_one_f16", "v_equals_one_bf16"])
def test_v_equals_one(case: str):
    res = _run_isolated(case)
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["max_abs_diff"] == pytest.approx(0.0, abs=0.0)
    assert payload["max_ulp"] <= 1.0, payload


def test_guard_buffer_clean():
    res = _run_isolated("guard_buffer")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload.get("raised") == "ValueError", payload
    assert payload["guard_clean"], payload


def test_out_device_mismatch():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=2,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    out_cpu = torch.empty_like(inputs.q, device="cpu")
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, out=out_cpu, causal=False)


def test_int32_slice_guard(monkeypatch: pytest.MonkeyPatch):
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=1,
        seq_q=1,
        seq_kv=1,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    out = torch.empty_like(inputs.q)
    monkeypatch.setattr(flash_attn_interface, "_FP8_MAX_FLAT_ELEMS", 8)
    with pytest.raises(ValueError, match="per-batch slice must contain fewer than"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, out=out, causal=False)


def test_gfx1100_rejects_gqa():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=8,
        num_kv_heads=2,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="GQA/MQA"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, num_kv_heads=2)


def test_gfx1100_rejects_cross_seq_above_16():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=96,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="seq_q=32"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False)


def test_gfx1100_rejects_unsupported_head_dim():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=32,
        head_dim=96,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="head_dim 96"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False)


def test_gfx1100_rejects_bias():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    bias = torch.randn(
        inputs.q.shape[1],
        inputs.k.shape[1],
        dtype=inputs.q.dtype,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="attention bias"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, bias=bias)


def test_gfx1100_rejects_alibi():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    alibi = torch.randn(inputs.q.shape[0], inputs.q.shape[2], dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="ALiBi"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, alibi_slopes=alibi)


def test_gfx1100_rejects_sink():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    sink = torch.randn(inputs.q.shape[2], dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="sink"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, sink=sink)


def test_gfx1100_rejects_return_lse():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=32,
        seq_kv=32,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="return_lse"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, return_lse=True)


def test_gfx1100_rejects_split_k():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=512,
        seq_kv=512,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(NotImplementedError, match="split-K"):
        flydsl_flash_attn_func(inputs.q, inputs.k, inputs.v, causal=False, num_kv_splits=2)


def test_pick_tile_respects_device_cu_count():
    tile_96 = pick_tile(64, 16, True, 360, num_cu=96)
    tile_84 = pick_tile(64, 16, True, 360, num_cu=84)
    assert tile_96 != tile_84


def test_select_tile_autotunes_when_enabled(monkeypatch):
    import kernels.attention.flash_attn_gfx1100_autotune as autotune

    original_tiles = autotune.feasible_tiles(64)[:2]
    assert len(original_tiles) >= 2, "expected multiple feasible tiles for head_dim=64"

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setattr(autotune, "feasible_tiles", lambda head_dim: list(original_tiles))

    launch_records: list[tuple[int, int, int, int, bool]] = []

    def fake_build(tile):
        def _launcher(q, k, v, out):
            launch_records.append(tile)

        return _launcher

    def fake_bench(fn, warmup, rep):
        fn()
        # Return a monotonically decreasing value so later candidates win.
        return 1.0 / float(len(launch_records))

    monkeypatch.setattr(autotune, "do_bench", fake_bench)

    chosen = autotune.select_tile(
        batch=1,
        num_heads=4,
        seq_q=128,
        seq_kv=128,
        head_dim=64,
        dtype_str="f16",
        causal=False,
        num_cu=96,
        build_kernel=fake_build,
    )

    assert launch_records, "autotune path should invoke the builder"
    assert len(launch_records) == len(original_tiles)
    assert chosen == original_tiles[len(original_tiles) - 1]


class _TileKey(NamedTuple):
    head_dim: int
    tile: tuple[int, int, int, int, bool]
    hoist_q: bool
    causal: bool
    decode: bool
    prefetch: bool
    kv_aligned: bool
    multi_block: bool


_D64_DECODE = (1, 1, 64, 8, True)
_D256_DECODE = (4, 1, 32, 8, False)
_D256_PREFILL = (8, 1, 32, 8, False)
_D64_DECODE_SPILL = "the single-wave D64 decode kernel already spilled 1-6 VGPRs before the f16 RNE change"
_D256_TAIL_SPILL = "unchanged since the head_dim 256 decode kernel was added"
_D256_HOIST = "hoisting Q is faster than spill-free re-reads on this shape class (see hoist_q)"

# Kernels allowed to spill: (max vgpr_spill_count, max private_segment_fixed_size)
# as measured on W7900D, and why. Every other variant must be spill-free.
_SPILL_EXCEPTIONS: dict[_TileKey, tuple[int, int, str]] = {
    _TileKey(64, _D64_DECODE, True, True, True, False, True, False): (1, 8, _D64_DECODE_SPILL),
    _TileKey(64, _D64_DECODE, True, False, True, False, True, True): (3, 16, _D64_DECODE_SPILL),
    _TileKey(64, _D64_DECODE, True, False, True, False, False, True): (7, 32, _D64_DECODE_SPILL),
    _TileKey(256, _D256_DECODE, False, False, True, False, False, True): (34, 140, _D256_TAIL_SPILL),
    _TileKey(256, _D256_DECODE, True, False, True, False, True, True): (75, 304, _D256_HOIST),
    _TileKey(256, _D256_DECODE, True, True, True, False, True, True): (125, 488, _D256_HOIST),
    _TileKey(256, _D256_DECODE, True, False, True, False, False, True): (75, 304, _D256_HOIST),
    _TileKey(256, _D256_DECODE, True, True, True, False, False, True): (80, 324, _D256_HOIST),
    _TileKey(256, _D256_PREFILL, True, False, False, False, True, True): (71, 288, _D256_HOIST),
    _TileKey(256, _D256_PREFILL, True, True, False, False, True, True): (79, 320, _D256_HOIST),
}


def _gfx1100_tile_samples() -> dict[_TileKey, tuple[int, int, int]]:
    """One (seq_q, seq_kv, bh) sample per distinct compiled kernel variant."""
    prefill_seqs = [32, 63, 64, 96, 128, 256, 512, 1024, 4096, 8192, 16384, 24576, 32768]
    decode_seq_kv = [64, 4096, 4097, 8192, 8193, 32768]
    bh_base = [1, 4, 16, 32]
    shapes = [(seq, seq, bh) for seq in prefill_seqs for bh in bh_base]
    shapes += [(seq_q, seq_kv, bh) for seq_q in (1, 16) for seq_kv in decode_seq_kv for bh in bh_base + [384]]
    samples: dict[_TileKey, tuple[int, int, int]] = {}
    for head_dim in (64, 128, 256):
        for seq_q, seq_kv, bh in shapes:
            for causal in (False, True):
                tile = pick_tile(head_dim, seq_q, causal, bh, num_cu=96)
                block_n = tile[2]
                key = _TileKey(
                    head_dim=head_dim,
                    tile=tile,
                    hoist_q=hoist_q(head_dim, seq_q, seq_kv, bh),
                    causal=causal,
                    decode=seq_q <= 16,
                    prefetch=seq_q <= 16 and head_dim == 128 and bh <= 128,
                    kv_aligned=seq_kv % block_n == 0,
                    multi_block=seq_kv > 2 * block_n,
                )
                samples.setdefault(key, (seq_q, seq_kv, bh))
    return samples


def _compile_tile_ir(head_dim: int, seq_q: int, seq_kv: int, causal: bool, bh: int) -> str:
    batch, num_heads = _factor_batch_heads(bh)
    launcher = build_flash_attn_func_module(
        batch=batch,
        num_heads=num_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=head_dim,
        causal=causal,
        dtype_str="f16",
        num_cu=96,
    )
    shape_q = (batch, seq_q, num_heads, head_dim)
    shape_kv = (batch, seq_kv, num_heads, head_dim)
    q = _cpu_placeholder(shape_q, torch.float16)
    k = _cpu_placeholder(shape_kv, torch.float16)
    v = _cpu_placeholder(shape_kv, torch.float16)
    out = _cpu_placeholder(shape_q, torch.float16)
    call_cache = getattr(launcher, "_call_state_cache", None)
    if call_cache is not None:
        call_cache.clear()
    mem_cache = getattr(launcher, "_mem_cache", None)
    if mem_cache is not None:
        mem_cache.clear()
    prev_last = getattr(launcher, "_last_compiled", None)
    launcher._last_compiled = None  # type: ignore[attr-defined]
    prev_cache_flag = os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE")
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    try:
        launcher._compile_only(q, k, v, out)
    finally:
        if prev_cache_flag is None:
            os.environ.pop("FLYDSL_RUNTIME_ENABLE_CACHE", None)
        else:
            os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = prev_cache_flag
    compiled = getattr(launcher, "_last_compiled", None)
    if compiled is None and mem_cache:
        try:
            compiled = next(iter(mem_cache.items()))
        except StopIteration:
            compiled = None
    if compiled is None:
        launcher._last_compiled = prev_last  # type: ignore[attr-defined]
        raise AssertionError("compile_only should record the compiled artifact")
    launcher._last_compiled = compiled  # type: ignore[attr-defined]
    return compiled[1].ir


def _spill_metadata(ir_text: str) -> tuple[int, int]:
    spills = [int(val) for val in re.findall(r"vgpr_spill_count\s*=\s*(\d+)", ir_text)]
    scratch = [int(val) for val in re.findall(r"private_segment_fixed_size\s*=\s*(\d+)", ir_text)]
    assert spills, "no vgpr_spill_count metadata found"
    assert scratch, "no private_segment_fixed_size metadata found"
    return max(spills), max(scratch)


def test_gfx1100_tiles_avoid_spills():
    samples = _gfx1100_tile_samples()
    assert samples, "expected at least one tile sample"
    failures = []
    for key, (seq_q, seq_kv, bh) in samples.items():
        spills, scratch = _spill_metadata(_compile_tile_ir(key.head_dim, seq_q, seq_kv, key.causal, bh))
        max_spills, max_scratch, _ = _SPILL_EXCEPTIONS.get(key, (0, 0, ""))
        if spills > max_spills or scratch > max_scratch:
            failures.append(f"{key} (seq_q={seq_q}, seq_kv={seq_kv}, bh={bh}): spills={spills}, scratch={scratch}")
    assert not failures, "unexpected VGPR spills:\n" + "\n".join(failures)


def test_gfx1100_spill_exceptions_are_reachable():
    samples = _gfx1100_tile_samples()
    stale = [key for key in _SPILL_EXCEPTIONS if key not in samples]
    assert not stale, f"spill exceptions no longer selected by pick_tile/hoist_q: {stale}"


def _run_flash_attn_with_override(
    inputs: AttentionInput,
    *,
    causal: bool,
    override: dict[str, bool] | None,
) -> torch.Tensor:
    q = inputs.q
    device = q.device
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    launcher = build_flash_attn_func_module(
        batch=q.shape[0],
        num_heads=q.shape[2],
        seq_q=q.shape[1],
        seq_kv=inputs.k.shape[1],
        head_dim=q.shape[3],
        causal=causal,
        dtype_str=dtype_str,
        num_cu=96,
        _override_flags=override,
    )
    out = torch.empty_like(q, device=device)
    launcher(q, inputs.k, inputs.v, out)
    torch.cuda.synchronize()
    return out


def test_use_perm_p_switch_is_bit_exact():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=128,
        seq_kv=128,
        head_dim=128,
        dtype=torch.float16,
        device=device,
    )
    out_perm = _run_flash_attn_with_override(inputs, causal=False, override={"use_perm_p": True})
    out_no_perm = _run_flash_attn_with_override(inputs, causal=False, override={"use_perm_p": False})
    torch.testing.assert_close(out_perm, out_no_perm, rtol=0.0, atol=0.0)


def test_use_vector_store_switch_is_bit_exact():
    device = torch.device("cuda")
    _seed_all(20261029)
    inputs = _make_inputs(
        batch=1,
        num_heads=4,
        seq_q=128,
        seq_kv=128,
        head_dim=64,
        dtype=torch.float16,
        device=device,
    )
    base_override = {"use_perm_p": True}
    out_vector = _run_flash_attn_with_override(
        inputs,
        causal=False,
        override={**base_override, "use_vector_store": True},
    )
    out_scalar = _run_flash_attn_with_override(
        inputs,
        causal=False,
        override={**base_override, "use_vector_store": False},
    )
    torch.testing.assert_close(out_vector, out_scalar, rtol=0.0, atol=0.0)


def test_dense_matrix_accuracy():
    res = _run_isolated("dense_matrix")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["max_ulp"] <= 1.0, payload


def test_unaligned_lengths_accuracy():
    res = _run_isolated("unaligned_lengths")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["max_ulp"] <= 1.0, payload


def test_cross_attention_bottom_right():
    res = _run_isolated("cross_attention")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["max_ulp"] <= 1.0, payload
    assert payload["fully_masked_zero"], payload


def test_kernel_deterministic():
    res = _run_isolated("deterministic")
    assert res.returncode == 0, f"subprocess failed: {res.stderr or res.stdout}"
    payload = json.loads(res.stdout or "{}")
    assert payload["deterministic"], payload
    assert payload["max_ulp"] <= 1.0, payload


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _main()
