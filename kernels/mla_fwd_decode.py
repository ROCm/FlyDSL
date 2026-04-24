# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode launcher.  Uses aiter for device queries."""

import functools
import re
import shutil
import subprocess

import torch


def _gcn_arch_base(arch_name: str) -> str:
    """Strip target features (':sramecc+:xnack-') from a gcnArchName."""
    return arch_name.split(":", 1)[0]


@functools.lru_cache(maxsize=None)
def _get_lds_size_per_cu(arch: str) -> int:
    """Return the LDS (shared memory) size per CU in bytes for ``arch``.

    Cached per arch so a mixed-GPU process (or one that switches devices)
    gets the right LDS budget for the active device — not whichever GPU
    rocminfo happens to list first. Caller must pass the current device's
    base gcnArchName (e.g. ``"gfx942"``).

    Parses the GROUP segment pool size from ``rocminfo`` output, picking
    the first GPU agent whose name matches ``arch``.
    """
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        raise RuntimeError("rocminfo not found on PATH")
    result = subprocess.run(
        [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    agents = re.split(r"Agent\s*\d+", result.stdout)
    for agent in agents:
        if "Device Type" not in agent or agent.find("GPU") == -1:
            continue
        # Match this agent's Name (e.g. "gfx942") against the requested arch.
        name_m = re.search(r"^\s*Name:\s*(\S+)", agent, re.MULTILINE)
        if not name_m or name_m.group(1) != arch:
            continue
        lines = agent.split("\n")
        for i, line in enumerate(lines):
            if re.search(r"Segment\s*:\s*GROUP", line) and i + 1 < len(lines):
                m = re.search(r"Size\s*:\s*(\d+)", lines[i + 1])
                if m:
                    return int(m.group(1)) * 1024  # KB -> bytes
    raise RuntimeError(
        f"No GPU GROUP segment found in rocminfo output for arch {arch!r}"
    )


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def flydsl_mla_fwd_decode(
    query: torch.Tensor,        # [num_seqs, num_heads, head_size]
    kv_buffer: torch.Tensor,    # [num_page, page_size, num_kv_heads, head_size]
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    final_output: torch.Tensor, # [num_seqs, num_heads, v_head_dim]
    split_output: torch.Tensor, # [num_partial_slots, 1, num_heads, v_head_dim]
    split_lse: torch.Tensor,    # [num_partial_slots, 1, num_heads, 1]
    softmax_scale: float,
) -> None:
    """Launch the FlyDSL MLA decode forward kernel."""
    num_heads = query.size(1)
    q_dtype = query.dtype
    kv_dtype = kv_buffer.dtype

    if num_heads == 128 and _is_fp8(q_dtype) and _is_fp8(kv_dtype):
        from .mla_fwd_decode_m16x8_fp8_fp8 import (
            OCCUPANCY,
            QK_HEAD_DIM,
            V_HEAD_DIM,
            launch_mla_fwd_decode_m16x8_fp8_fp8,
        )

        # ── shape validation ──
        assert query.ndim == 3, (
            f"query: expected 3D [num_seqs, num_heads, qk_head_dim], got shape {list(query.shape)}"
        )
        assert query.size(2) == QK_HEAD_DIM, (
            f"query: head_dim={query.size(2)}, expected {QK_HEAD_DIM}"
        )
        assert kv_buffer.ndim == 4, (
            f"kv_buffer: expected 4D [num_page, page_size, num_kv_heads, qk_head_dim], "
            f"got shape {list(kv_buffer.shape)}"
        )
        assert kv_buffer.size(1) * kv_buffer.size(2) == 1, (
            f"kv_buffer: page_size*num_kv_heads must be 1, "
            f"got page_size={kv_buffer.size(1)}, num_kv_heads={kv_buffer.size(2)}"
        )
        assert kv_buffer.size(3) == QK_HEAD_DIM, (
            f"kv_buffer: head_dim={kv_buffer.size(3)}, expected {QK_HEAD_DIM}"
        )
        num_seqs = query.size(0)
        assert final_output.shape == (num_seqs, num_heads, V_HEAD_DIM), (
            f"final_output: expected shape [{num_seqs}, {num_heads}, {V_HEAD_DIM}], "
            f"got {list(final_output.shape)}"
        )
        num_partial = split_output.size(0)
        assert split_output.ndim == 4 and split_output.shape[1:] == (1, num_heads, V_HEAD_DIM), (
            f"split_output: expected [N, 1, {num_heads}, {V_HEAD_DIM}], "
            f"got {list(split_output.shape)}"
        )
        assert split_lse.ndim == 4 and split_lse.shape[1:] == (1, num_heads, 1), (
            f"split_lse: expected [N, 1, {num_heads}, 1], got {list(split_lse.shape)}"
        )
        assert split_lse.size(0) == num_partial, (
            f"split_lse batch dim ({split_lse.size(0)}) != split_output batch dim ({num_partial})"
        )
        dev = query.device
        for name, t in [("kv_buffer", kv_buffer), ("kv_page_indices", kv_page_indices),
                        ("work_indptr", work_indptr), ("work_info_set", work_info_set),
                        ("final_output", final_output), ("split_output", split_output),
                        ("split_lse", split_lse)]:
            assert t.device == dev, f"{name}: expected device {dev}, got {t.device}"

        # Output tensors must be contiguous: reshape() on a non-contiguous
        # output would silently materialize a copy, the kernel would write
        # into the copy, and the caller's original tensor would never be
        # updated. Use view() after asserting contiguity so any layout
        # mismatch fails loudly here instead.
        for name, t in [("final_output", final_output),
                        ("split_output", split_output),
                        ("split_lse", split_lse)]:
            assert t.is_contiguous(), (
                f"{name}: must be contiguous (stride={list(t.stride())}, "
                f"shape={list(t.shape)}); reshape() would silently copy and "
                f"the kernel's writes would not be visible to the caller"
            )

        num_pages = kv_buffer.size(0)

        query_flat = query.reshape(num_seqs * num_heads, QK_HEAD_DIM)
        kv_flat = kv_buffer.reshape(num_pages, QK_HEAD_DIM)
        final_flat = final_output.view(num_seqs * num_heads, V_HEAD_DIM)
        split_o_flat = split_output.view(num_partial * num_heads, V_HEAD_DIM)
        split_lse_flat = split_lse.view(num_partial * num_heads)

        work_indptr_flat = work_indptr.contiguous()
        work_info_flat = work_info_set.contiguous().view(-1)
        kv_idx_flat = kv_page_indices.contiguous()

        from aiter.jit.utils.chip_info import get_cu_num

        num_cus = get_cu_num()
        arch = _gcn_arch_base(torch.cuda.get_device_properties(dev).gcnArchName)
        lds_size = _get_lds_size_per_cu(arch) // OCCUPANCY

        launch_mla_fwd_decode_m16x8_fp8_fp8(
            query_flat,
            kv_flat,
            kv_idx_flat,
            work_indptr_flat,
            work_info_flat,
            final_flat,
            split_o_flat,
            split_lse_flat,
            softmax_scale,
            num_cus,
            lds_size,
            stream=torch.cuda.current_stream(),
        )
    else:
        raise NotImplementedError(
            f"flydsl_mla_fwd_decode: unsupported num_heads={num_heads}, "
            f"q_dtype={q_dtype}, kv_dtype={kv_dtype}"
        )
