# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Softmax autotuning through the normal direct JIT path."""

from flydsl.autotune import Config, autotune
from kernels.norm.softmax_kernel import (
    BLOCK_THREADS,
    TUNING_SCHEMA,
    WARP_SIZE,
    softmax_direct,
)

# HIP's hard workgroup ceiling. There is no device-property helper in the repo and this
# adopter is not the place to add the first one.
MAX_BLOCK_THREADS = 1024

_BLOCK_THREADS_AXIS = (64, 128, 256, 512)
_WAVES_PER_EU_AXIS = (None, 1, 2)

# Per-dtype relative tolerance for the candidate correctness gate. Pinned from the worst
# error measured over every candidate x {f32,f16,bf16} x 7 shapes on gfx950 (f32 1.3e-7,
# f16 2.4e-4, bf16 1.9e-3), which tracks each dtype's output quantization: bf16 keeps 8
# mantissa bits, f16 keeps 11. The same values gate the compatibility default in
# tests/kernels/test_softmax.py, so no candidate is held to a weaker standard.
_RTOL = {"f32": 1e-5, "f16": 5e-3, "bf16": 2e-2}
# Row sums accumulate the per-element error across N terms; measured worst case was
# f32 2.4e-7, f16 1.2e-4, bf16 8.3e-4.
_ROW_SUM_TOL = {"f32": 1e-5, "f16": 2e-3, "bf16": 1e-2}

_SUPPORTED_DTYPES = ("f16", "bf16", "f32")


def elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def tile_cols(dtype_str: str, block_threads: int) -> int:
    """Columns one full vectorized tile covers, from the 128-bit transaction contract."""
    return block_threads * (128 // elem_bits(dtype_str))


def uses_fast_path(N: int, dtype_str: str, block_threads: int) -> bool:
    """Whether this candidate takes the vectorized path rather than the scalar one.

    N > 0 and exact divisibility already imply N >= tile_cols.
    """
    return N % tile_cols(dtype_str, block_threads) == 0


def is_legal(block_threads: int, warp_size: int = WARP_SIZE) -> bool:
    """Objective legality only. ``warp_size`` is a parameter because the module-level
    WARP_SIZE is resolved at import time and cannot be patched per test."""
    if block_threads <= 0 or block_threads > MAX_BLOCK_THREADS:
        return False
    if block_threads % warp_size != 0:
        return False
    # The block reduction's second stage runs on a single wave indexing lane < RED_SLOTS.
    red_slots = -(-block_threads // warp_size)
    return red_slots <= warp_size


def candidate_configs(warp_size: int = WARP_SIZE):
    """The one owner of candidate generation: a bounded Cartesian product with only
    illegal and duplicate entries removed, in a deterministic order."""
    seen = set()
    configs = []
    for block_threads in _BLOCK_THREADS_AXIS:
        if not is_legal(block_threads, warp_size):
            continue
        for waves_per_eu in _WAVES_PER_EU_AXIS:
            identity = (block_threads, waves_per_eu)
            if identity in seen:
                continue
            seen.add(identity)
            configs.append(Config(BLOCK_THREADS=block_threads, waves_per_eu=waves_per_eu))
    if (BLOCK_THREADS, None) not in seen:
        raise RuntimeError(f"the compatibility default BLOCK_THREADS={BLOCK_THREADS} was pruned")
    return configs


def _default_config(*_args, **_kwargs):
    return Config(BLOCK_THREADS=BLOCK_THREADS)


def _search_configs(*_args, **_kwargs):
    return candidate_configs()


def _validate_candidate(sig_args):
    """Untimed correctness gate: reject a candidate that launches but computes wrongly.

    The comparison is scale-aware. Softmax elements are O(1/N), so an absolute bound
    sized for O(1) values would accept an all-zero output.
    """
    import torch

    dtype_str = sig_args["dtype_str"]
    out = sig_args["C"].float()
    reference = torch.softmax(sig_args["A"].float(), dim=-1)

    finite_ref = torch.isfinite(reference)
    if not bool(torch.isfinite(out)[finite_ref].all()):
        raise ValueError(f"candidate produced non-finite output where the reference is finite ({dtype_str})")

    tol = _RTOL[dtype_str]
    row_scale = reference.amax(dim=-1, keepdim=True)
    error = (out - reference).abs()
    bound = tol * (reference.abs() + row_scale)
    if bool((error > bound).any()):
        worst = (error / (reference.abs() + row_scale)).max().item()
        raise ValueError(f"candidate exceeded the {dtype_str} tolerance {tol:g}: relative error {worst:.3e}")

    row_sum_error = (out.sum(dim=-1) - 1.0).abs().max().item()
    if row_sum_error > _ROW_SUM_TOL[dtype_str]:
        raise ValueError(f"candidate broke the row-sum invariant ({dtype_str}): |sum - 1| = {row_sum_error:.3e}")


_softmax_tuner = autotune(
    configs=_search_configs,
    key=["m_in", "N", "dtype_str", "tuning_schema"],
    default=_default_config,
    artifact_name="softmax_fwd",
    validate_hook=_validate_candidate,
)(softmax_direct)


def _overlaps(a, b) -> bool:
    """Whether two contiguous tensors share any byte of storage."""
    a_start = a.data_ptr()
    a_end = a_start + a.numel() * a.element_size()
    b_start = b.data_ptr()
    b_end = b_start + b.numel() * b.element_size()
    return a_start < b_end and b_start < a_end


def softmax_autotuned(input_t, output, stream=None):
    """Row-wise Softmax with opt-in autotuning.

    Normal calls follow the scratch-cache -> offline-artifact -> compatibility-default
    ordering and never benchmark. ``FLYDSL_AUTOTUNE=1`` is the explicit forced search.
    """
    import torch

    from kernels.norm.rmsnorm_common import torch_dtype_to_str

    if input_t.dim() != 2 or output.dim() != 2:
        raise ValueError(f"softmax_autotuned expects row-wise 2-D tensors, got {input_t.dim()}-D and {output.dim()}-D")
    if input_t.shape != output.shape:
        raise ValueError(f"shape mismatch: input {tuple(input_t.shape)} vs output {tuple(output.shape)}")
    if input_t.dtype != output.dtype:
        raise ValueError(f"dtype mismatch: input {input_t.dtype} vs output {output.dtype}")
    if input_t.device != output.device:
        raise ValueError(f"device mismatch: input {input_t.device} vs output {output.device}")
    if not input_t.is_contiguous() or not output.is_contiguous():
        raise ValueError("softmax_autotuned requires row-major contiguous storage")

    dtype_str = torch_dtype_to_str(input_t.dtype)
    if dtype_str not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype_str!r} (expected one of {_SUPPORTED_DTYPES})")

    M, N = int(input_t.shape[0]), int(input_t.shape[1])
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if M == 0:
        return None
    if _overlaps(input_t, output):
        raise ValueError("softmax_autotuned is out-of-place; input and output storage must not overlap")

    with torch.cuda.device(input_t.device):
        launch_stream = torch.cuda.current_stream() if stream is None else stream
        return _softmax_tuner(
            input_t,
            output,
            M,
            N=N,
            dtype_str=dtype_str,
            tuning_schema=TUNING_SCHEMA,
            stream=launch_stream,
        )
