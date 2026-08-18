# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Softmax autotuning through the normal direct JIT path."""

import os

from flydsl.autotune import Config, _tuning_enabled, autotune
from kernels.norm.softmax_kernel import (
    BLOCK_THREADS,
    TUNING_SCHEMA,
    WARP_SIZE,
    softmax_direct,
)

# HIP's hard workgroup ceiling. There is no device-property helper in the repo and this
# adopter is not the place to add the first one.
MAX_BLOCK_THREADS = 1024

_FULL_ROW_THREADS = (64, 128, 256, 512)
_WAVES_PER_EU_AXIS = (None, 1, 2, 4)
_SUBGROUP_THREADS = (8, 16, 32, 64)
_MULTI_ROW_BLOCK_THREADS = (128, 256)
_TIE_RELATIVE = 0.02

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
_FAST_CONTEXT_ENV_VARS = (
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_LLVM_DIR",
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "FLYDSL_EXTRA_SOURCE_DIRS",
    "FLYDSL_GPU_ARCH",
    "FLYDSL_RUNTIME_KIND",
    "FLYDSL_AUTOTUNE_CONFIG_DIR",
    "ARCH",
    "HSA_OVERRIDE_GFX_VERSION",
    "COMPILE_ONLY",
)
_FAST_CONTEXT_ENV_VARS_BYTES = tuple(os.fsencode(name) for name in _FAST_CONTEXT_ENV_VARS)
_softmax_hot_cache = {}


def elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def tile_cols(dtype_str: str, threads_per_row: int) -> int:
    """Columns one full vectorized tile covers, from the 128-bit transaction contract."""
    return threads_per_row * (128 // elem_bits(dtype_str))


def uses_fast_path(N: int, dtype_str: str, threads_per_row: int) -> bool:
    """Whether this candidate takes the vectorized path rather than the scalar one.

    N > 0 and exact divisibility already imply N >= tile_cols.
    """
    return N % tile_cols(dtype_str, threads_per_row) == 0


def is_legal(
    block_threads: int,
    warp_size: int = WARP_SIZE,
    *,
    threads_per_row: int | None = None,
    rows_per_block: int = 1,
) -> bool:
    """Objective legality only. ``warp_size`` is a parameter because the module-level
    WARP_SIZE is resolved at import time and cannot be patched per test."""
    if block_threads <= 0 or block_threads > MAX_BLOCK_THREADS:
        return False
    threads_per_row = block_threads if threads_per_row is None else threads_per_row
    if block_threads % warp_size != 0 or block_threads != threads_per_row * rows_per_block:
        return False
    if threads_per_row <= 0 or threads_per_row & (threads_per_row - 1):
        return False
    if rows_per_block <= 0:
        return False
    if threads_per_row <= warp_size:
        return warp_size % threads_per_row == 0
    if rows_per_block != 1 or threads_per_row % warp_size != 0:
        return False
    # The block reduction's second stage runs on a single wave indexing lane < RED_SLOTS.
    red_slots = -(-threads_per_row // warp_size)
    return red_slots <= warp_size


def _quack_threads_per_row(N: int) -> int:
    """Portable part of Quack's row-width heuristic (cluster splitting excluded)."""
    for limit, threads in ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)):
        if N <= limit:
            return threads
    return 256


def _multi_row_geometries(m_in: int, N: int, dtype_str: str, warp_size: int):
    """Return bounded subgroup geometries for packing independent short rows."""
    heuristic = _quack_threads_per_row(N)
    if heuristic > warp_size or m_in < 2:
        return []

    vector_count = (N + (128 // elem_bits(dtype_str)) - 1) // (128 // elem_bits(dtype_str))
    row_threads = {
        threads
        for threads in (heuristic // 2, heuristic, heuristic * 2)
        if threads in _SUBGROUP_THREADS and threads <= max(8, vector_count)
    }
    geometries = []
    for threads_per_row in sorted(row_threads):
        totals = _MULTI_ROW_BLOCK_THREADS if threads_per_row == heuristic else (128,)
        for block_threads in totals:
            rows_per_block = block_threads // threads_per_row
            if rows_per_block <= 1 or rows_per_block > m_in:
                continue
            if is_legal(
                block_threads,
                warp_size,
                threads_per_row=threads_per_row,
                rows_per_block=rows_per_block,
            ):
                geometries.append((block_threads, threads_per_row, rows_per_block))
    return geometries


def candidate_configs(
    warp_size: int = WARP_SIZE,
    *,
    m_in: int = 64,
    N: int = 4096,
    dtype_str: str = "bf16",
):
    """Generate a bounded, shape-aware set of legal forward configurations.

    The search is layered rather than one large Cartesian product: full-row
    geometry crosses occupancy hints, while Quack-style multi-row geometry
    explores subgroup packing.
    """
    seen = set()
    configs = []

    def append(
        block_threads,
        threads_per_row,
        rows_per_block=1,
        *,
        waves_per_eu=None,
    ):
        if not is_legal(
            block_threads,
            warp_size,
            threads_per_row=threads_per_row,
            rows_per_block=rows_per_block,
        ):
            return
        identity = (
            block_threads,
            threads_per_row,
            rows_per_block,
            waves_per_eu,
        )
        if identity in seen:
            return
        seen.add(identity)
        configs.append(
            Config(
                BLOCK_THREADS=block_threads,
                THREADS_PER_ROW=threads_per_row,
                ROWS_PER_BLOCK=rows_per_block,
                waves_per_eu=waves_per_eu,
            )
        )

    # Existing one-row kernel geometry, now including the gfx950 WPE=4 bound.
    for block_threads in _FULL_ROW_THREADS:
        for waves_per_eu in _WAVES_PER_EU_AXIS:
            append(block_threads, block_threads, waves_per_eu=waves_per_eu)

    # Quack's most transferable idea: decouple the row-reduction subgroup from
    # total CTA size so one wave/block can process several independent rows.
    multi_row = _multi_row_geometries(m_in, N, dtype_str, warp_size)
    for geometry in multi_row:
        append(*geometry)
    heuristic = _quack_threads_per_row(N)
    preferred = next((geometry for geometry in multi_row if geometry[1] == heuristic), None)
    if preferred is not None:
        append(*preferred, waves_per_eu=2)

    compatibility = (BLOCK_THREADS, BLOCK_THREADS, 1, None)
    if compatibility not in seen:
        raise RuntimeError(f"the compatibility default BLOCK_THREADS={BLOCK_THREADS} was pruned")
    return configs


def _default_config(*_args, **_kwargs):
    return Config(BLOCK_THREADS=BLOCK_THREADS)


def _search_configs(A, C, m_in, N, dtype_str, **_kwargs):
    return candidate_configs(m_in=int(m_in), N=int(N), dtype_str=dtype_str)


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


def _softmax_select_config(results):
    """Prefer a stable ABI-compatible config inside a measurement tie.

    Event granularity and clock movement can reorder 6--10 us candidates whose
    true latency differs by less than a percent. Keep the exact winner unless
    another result is within two percent; inside that band prefer the existing
    256-thread default, then no compiler occupancy override, then more packed
    rows per block.
    """
    best_time = min(elapsed for _config, elapsed in results)
    contenders = [pair for pair in results if pair[1] <= best_time * (1.0 + _TIE_RELATIVE)]

    def priority(pair):
        config, elapsed = pair
        block_threads = int(config.kwargs["BLOCK_THREADS"])
        threads_per_row = int(config.kwargs.get("THREADS_PER_ROW", block_threads))
        rows_per_block = int(config.kwargs.get("ROWS_PER_BLOCK", 1))
        compatibility = (
            block_threads == BLOCK_THREADS
            and threads_per_row == BLOCK_THREADS
            and rows_per_block == 1
            and config.waves_per_eu is None
        )
        return (
            not compatibility,
            config.waves_per_eu is not None,
            -rows_per_block,
            elapsed,
        )

    return min(contenders, key=priority)


_softmax_tuner = autotune(
    configs=_search_configs,
    key=["m_in", "N", "dtype_str", "tuning_schema"],
    warmup=10,
    rep=100,
    default=_default_config,
    artifact_name="softmax_fwd",
    validate_hook=_validate_candidate,
    select_config=_softmax_select_config,
)(softmax_direct)


def _fast_context_token():
    """Cheap invalidation axes for the adapter-level compiled-call cache."""
    data = getattr(os.environ, "_data", None)
    if data is None:
        environment = tuple(os.environ.get(name, "") for name in _FAST_CONTEXT_ENV_VARS)
    else:
        environment = tuple(data.get(name, b"") for name in _FAST_CONTEXT_ENV_VARS_BYTES)
    compile_hints = getattr(softmax_direct, "compile_hints", {})
    return environment, repr(sorted(compile_hints.items()))


def _hot_key(input_t, output, m, n, dtype_str, stream):
    device_index = input_t.device.index
    if device_index is None:
        import torch

        device_index = torch.cuda.current_device()
    return (
        device_index,
        type(input_t),
        type(output),
        tuple(input_t.shape),
        tuple(input_t.stride()),
        tuple(output.stride()),
        str(input_t.dtype),
        m,
        n,
        dtype_str,
        type(stream),
        TUNING_SCHEMA,
        _fast_context_token(),
    )


def _config_options(config):
    return (
        int(config.kwargs["BLOCK_THREADS"]),
        int(config.kwargs.get("THREADS_PER_ROW", config.kwargs["BLOCK_THREADS"])),
        int(config.kwargs.get("ROWS_PER_BLOCK", 1)),
    )


def _compile_resolved(config, input_t, output, m, n, dtype_str, stream):
    """Compile and execute one resolved config, returning its hot callable."""
    import flydsl.compiler as flyc
    from flydsl.compiler.kernel_function import CompilationContext

    block_threads, threads_per_row, rows_per_block = _config_options(config)
    positional = (
        input_t,
        output,
        m,
        n,
        dtype_str,
        block_threads,
        TUNING_SCHEMA,
        stream,
        threads_per_row,
        rows_per_block,
    )
    with CompilationContext.compile_hints(config.compiler_opts()):
        compiled = flyc.compile(softmax_direct, *positional)
    return compiled, positional


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
        call_kwargs = {
            "N": N,
            "dtype_str": dtype_str,
            "tuning_schema": TUNING_SCHEMA,
            "stream": launch_stream,
        }
        hot_key = _hot_key(input_t, output, M, N, dtype_str, launch_stream)
        if not _tuning_enabled():
            entry = _softmax_hot_cache.get(hot_key)
            if entry is not None:
                compiled, static = entry
                return compiled(
                    input_t,
                    output,
                    *static[:5],
                    launch_stream,
                    *static[5:],
                )

        config = _softmax_tuner.resolve_config(
            input_t,
            output,
            M,
            **call_kwargs,
        )
        compiled, positional = _compile_resolved(config, input_t, output, M, N, dtype_str, launch_stream)
        if compiled is not None:
            # Only input/output pointers and the stream vary on the hot path.
            static = (*positional[2:7], *positional[8:])
            _softmax_hot_cache[hot_key] = (compiled, static)
        return None
