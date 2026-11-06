# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Public MXFP4 preparation, GEMM launch and compile-only startup warmup."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .heuristic import _normalize_config, choose_config
from .hybrid import build_hybrid
from .register import build_register


@lru_cache(maxsize=None)
def _check_device(index):
    if not torch.version.hip:
        raise ValueError("RDNA4 MXFP4 requires ROCm and gfx1200/gfx1201")
    properties = torch.cuda.get_device_properties(index)
    arch = properties.gcnArchName.split(":")[0]
    if arch not in ("gfx1200", "gfx1201"):
        raise ValueError(f"RDNA4 MXFP4 requires gfx1200/gfx1201, got {arch}")
    return properties


@lru_cache(maxsize=None)
def _device_capacities(index):
    properties = _check_device(index)
    cache_bytes = getattr(properties, "L2_cache_size", 0)
    try:
        import amdsmi

        # Torch maps logical HIP ordinals, including visible-device remapping,
        # to AMD SMI handles. Discovery runs once at weight preparation.
        handle = torch.cuda._get_amdsmi_handler(index)
        entries = [
            entry
            for entry in amdsmi.amdsmi_get_gpu_cache_info(handle)["cache"]
            if "DATA_CACHE" in entry["cache_properties"]
            and "CPU_CACHE" not in entry["cache_properties"]
            and 2 <= entry["cache_level"] < 2**32 - 1
            and 0 < entry["cache_size"] < 2**32 - 1
            and 0 < entry["num_cache_instance"] < 2**32 - 1
        ]
        level = max((entry["cache_level"] for entry in entries), default=0)
        size = sum(
            entry["cache_size"] * entry["num_cache_instance"] * 1024
            for entry in entries
            if entry["cache_level"] == level
        )
        if size > 0:
            cache_bytes = size
    except Exception:
        pass
    return properties.multi_processor_count, cache_bytes, properties.shared_memory_per_block


def get_device_capabilities(device_id=0):
    """Read and cache launch capacities during initialization on the owning GPU.

    Pass these values to ``get_warmup_keys`` in compile-only worker processes.
    Last-level data cache uses AMD SMI where available, then Torch's L2 size.
    This function may initialize HIP; the pure selector and warmup do not.
    """
    processors, cache_bytes, lds_bytes = _device_capacities(device_id)
    return dict(num_processors=processors, cache_size_bytes=cache_bytes, lds_size_bytes=lds_bytes)


@dataclass(frozen=True)
class PreparedMXFP4:
    """Immutable compressed weights and metadata returned by ``prepare_weights``."""

    packed: torch.Tensor
    scales: torch.Tensor
    n: int
    k: int
    mode: str = "grouped"
    base_exponents: torch.Tensor | None = None
    table: torch.Tensor | None = None
    normal_only: bool = False
    nibble_layout: str = "linear"

    @property
    def storage_bytes(self):
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
            + (0 if self.base_exponents is None else self.base_exponents.numel() * 4)
        )


_tables = {}


def _fold_table(device, mode):
    key = (device, mode)
    if key not in _tables:
        low, high, dtype = (-8, 6, torch.float8_e4m3fn) if mode == "e4m3" else (-15, 13, torch.float8_e5m2)
        values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        raw = torch.stack([values * (2.0**shift) for shift in range(low, high + 1)]).to(dtype).view(torch.uint8)
        # Every table entry in the advertised range must be lossless.
        decoded = raw.view(dtype).float()
        original = torch.stack([values * (2.0**shift) for shift in range(low, high + 1)])
        assert torch.equal(decoded, original)
        _tables[key] = raw.contiguous().view(torch.int32).to(device)
    return _tables[key]


def prepare_weights(packed, scales, *, mode="auto", nibble_layout="split"):
    """Prepare OCP MXFP4 weights once, preserving their represented values.

    Args:
        packed: RDNA4 uint8 tensor [N, K/2], even K in the low nibble.
        scales: E8M0 uint8 tensor [N, K/32] on the same device.
        mode: ``auto`` chooses lossless E4M3, E5M2, or grouped K32 scaling.
            Forced E4M3/E5M2 raises if the scale span cannot be represented.
        nibble_layout: ``split`` enables byte-permute decoding; ``linear``
            retains interleaved nibbles inside each packed word.

    Returns:
        Compressed, padded and preshuffled weights with folding metadata.

    Raises:
        ValueError: For unsupported devices, formats, or forced folding.
    """
    if packed.ndim != 2 or packed.dtype != torch.uint8 or not packed.is_cuda:
        raise ValueError("packed weights must be CUDA uint8 [N,K/2]")
    get_device_capabilities(packed.device.index)
    n, kp = packed.shape
    k = kp * 2
    if n <= 0 or k <= 0 or k % 32:
        raise ValueError("N must be positive and K a positive multiple of 32")
    if scales.shape != (n, k // 32) or scales.dtype != torch.uint8 or scales.device != packed.device:
        raise ValueError("scales must be uint8 [N,K/32] on the weight device")
    npad = (n + 15) // 16 * 16
    w = torch.nn.functional.pad(packed, (0, 0, 0, npad - n)).contiguous()
    s = torch.nn.functional.pad(scales, (0, 0, 0, npad - n), value=127).contiguous()
    if nibble_layout not in ("linear", "split"):
        raise ValueError("unknown nibble layout")
    if nibble_layout == "split":
        # Each dword's low nibbles hold k[0:4], high nibbles k[4:8].
        # Two byte-permute lookups then produce the two FP8 dwords directly.
        v = w.view(npad, k // 8, 4)
        w = torch.stack(
            (
                (v[..., 0] & 15) | ((v[..., 2] & 15) << 4),
                (v[..., 0] >> 4) | (v[..., 2] & 240),
                (v[..., 1] & 15) | ((v[..., 3] & 15) << 4),
                (v[..., 1] >> 4) | (v[..., 3] & 240),
            ),
            -1,
        ).view(npad, k // 2)
    # One dword per WMMA lane, low/even nibble first; no expanded weights.
    w = w.view(npad // 16, 16, k // 16, 2, 4).permute(0, 2, 3, 1, 4).contiguous()
    s = s.view(npad // 16, 16, k // 32).transpose(1, 2).contiguous()
    if mode not in ("auto", "grouped", "e4m3", "e5m2"):
        raise ValueError("unknown weight preparation mode")
    selected = mode
    base = None
    table = None
    normal_only = False
    if mode != "grouped":
        maxima = s.to(torch.int32).amax(1).flatten()
        minima = s.to(torch.int32).amin(1).flatten()
        # Flatten order is tile-N then column, matching WMMA lane columns.
        valid = bool((maxima < 255).all())
        for candidate, headroom, lower in [("e4m3", 6, -8), ("e5m2", 13, -15)]:
            if mode not in ("auto", candidate):
                continue
            candidate_base = (maxima - headroom).clamp_min(1)
            lossless = valid and bool(((minima - candidate_base) >= lower).all())
            if lossless:
                selected = candidate
                base = candidate_base.contiguous()
                table = _fold_table(packed.device, candidate)
                threshold = -5 if candidate == "e4m3" else -13
                normal_only = bool(((minima - candidate_base) >= threshold).all())
                break
        else:
            if mode != "auto":
                raise ValueError(f"scale range cannot be represented losslessly in {mode}")
            selected = "grouped"
    return PreparedMXFP4(w, s, n, k, selected, base, table, normal_only, nibble_layout)


_executables = {}


def specialization_key(
    m,
    n,
    k,
    *,
    mode="e4m3",
    normal_only=False,
    nibble_layout="split",
    dtype=torch.bfloat16,
    device_id=0,
    config=None,
    cache_policy="auto",
    num_processors=32,
    cache_size_bytes=64 * 1024**2,
    lds_size_bytes=65536,
):
    """Return the exact executable identity used by runtime and startup warmup."""
    options = _normalize_config(
        config
        if config
        else choose_config(
            m,
            n,
            k,
            mode,
            cache_policy=cache_policy,
            num_processors=num_processors,
            cache_size_bytes=cache_size_bytes,
            lds_size_bytes=lds_size_bytes,
        ),
        m,
        k,
    )
    options.update(
        fold_mode=mode,
        normal_only=normal_only,
        output_dtype="bf16" if dtype == torch.bfloat16 else "fp16",
        device_id=device_id,
        separated_nibbles=nibble_layout == "split",
    )
    engine = options.pop("engine")
    if engine == "hybrid" and mode == "grouped":
        raise ValueError("hybrid requires losslessly folded FP8 weights")
    if engine == "hybrid" and (options.get("tile_k", 64) not in (32, 64, 128, 256) or k % options.get("tile_k", 64)):
        raise ValueError("hybrid requires a complete power-of-two K tile (32, 64, 128 or 256)")
    kernel_m = m if m in (1, 2, 4, 8, 16) else 0
    if kernel_m == 0:
        bm = (
            options.get("tile_m", 256)
            if engine == "hybrid"
            else 16 * options.get("waves_m", 1) * options.get("repeat_m", 1)
        )
        options["aligned_m"] = m % bm == 0
    return engine, kernel_m, n, k, tuple(sorted(options.items()))


def _m_alignment(engine, options):
    if not options.get("aligned_m", False):
        return 1
    bm = (
        options.get("tile_m", 256)
        if engine == "hybrid"
        else 16 * options.get("waves_m", 1) * options.get("repeat_m", 1)
    )
    return bm & -bm


def _arguments(x, packed, scales, x_scale, out, base_exponents, table, kernel_m, alignment=1):
    m = x.shape[0]
    packed_words = packed.view(torch.int32)
    args = [
        flyc.from_torch_tensor(t, use_32bit_stride=True)
        for t in (
            x.view(torch.int32),
            packed_words,
            scales,
            x_scale.view(m),
            out,
            base_exponents if base_exponents is not None else packed_words,
            table if table is not None else packed_words,
        )
    ]
    if kernel_m == 0:
        for index in (0, 3, 4):
            args[index].mark_shape_dynamic(0, divisibility=alignment)
    return args


def gemm(x, weights, x_scale, *, out=None, dtype=None, config=None, cache_policy="auto"):
    """Multiply FP8 activations by prepared MXFP4 weights on the current stream.

    Args:
        x: Contiguous E4M3 tensor [M, K].
        weights: Result of ``prepare_weights`` on the activation device.
        x_scale: Contiguous FP32 tensor [M] or [M, 1].
        out: Optional contiguous output tensor [M, N].
        dtype: BF16 (default) or FP16, matching ``out`` when supplied.
        config: Optional register/hybrid factory parameters. Otherwise use
            analytical host rules with cached device capacities.
        cache_policy: ``auto`` balances hot and rotating weights; ``hot`` is
            for repeated reuse, ``rotating`` for weights cycled between layers.

    Returns:
        Tensor [M, N] containing the represented matrix product with per-row
        activation scales, FP32 accumulation and output-dtype rounding. No
        dense weight buffer is materialized.

    Raises:
        ValueError: For incompatible shapes, devices, formats, or strategies.
    """
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("activations must be contiguous FP8 E4M3 [M,K]")
    m, k = x.shape
    if m <= 0 or k != weights.k or x.device != weights.packed.device:
        raise ValueError("activation shape/device mismatch")
    if (
        x_scale.numel() != m
        or x_scale.dtype != torch.float32
        or not x_scale.is_contiguous()
        or x_scale.device != x.device
    ):
        raise ValueError("activation scales must be contiguous FP32 [M] or [M,1]")
    _check_device(x.device.index)
    n = weights.n
    if max(m * k, m * n * 2, weights.packed.numel(), weights.scales.numel()) >= 2**32:
        raise ValueError("current kernels require each addressed buffer to be smaller than 4 GiB")
    if dtype is None:
        dtype = out.dtype if out is not None else torch.bfloat16
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("output dtype must be BF16 or FP16")
    if out is None:
        out = torch.empty(m, n, device=x.device, dtype=dtype)
    if out.shape != (m, n) or out.dtype != dtype or not out.is_contiguous() or out.device != x.device:
        raise ValueError("output must be contiguous BF16/FP16 [M,N] on the activation device")
    if config:
        options = dict(config)
    else:
        options = choose_config(
            m, n, k, weights.mode, cache_policy=cache_policy, **get_device_capabilities(x.device.index)
        )
    key = specialization_key(
        m,
        n,
        k,
        mode=weights.mode,
        normal_only=weights.normal_only,
        nibble_layout=weights.nibble_layout,
        dtype=dtype,
        device_id=x.device.index,
        config=options,
    )
    engine, kernel_m, _, _, parameters = key
    options = dict(parameters)
    factory = build_hybrid if engine == "hybrid" else build_register
    args = _arguments(
        x,
        weights.packed,
        weights.scales,
        x_scale,
        out,
        weights.base_exponents,
        weights.table,
        kernel_m,
        _m_alignment(engine, options),
    )
    stream = fx.Stream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _executables.get(key)
    if compiled is None:
        launcher = factory(kernel_m, n, k, **options)
        with torch.cuda.device(x.device):
            compiled = flyc.compile(launcher, *args, stream)
        _executables[key] = compiled
    else:
        if torch.cuda.current_device() == x.device.index:
            compiled(*args, stream)
        else:
            with torch.cuda.device(x.device):
                compiled(*args, stream)
    return out


_compile_lock = threading.RLock()


def get_warmup_keys(
    n,
    k,
    max_m,
    *,
    mode="e4m3",
    normal_only=False,
    nibble_layout="split",
    dtype=torch.bfloat16,
    device_id=0,
    cache_policy="auto",
    num_processors=32,
    cache_size_bytes=64 * 1024**2,
    lds_size_bytes=65536,
):
    """Enumerate analytical launch keys without querying or initializing a GPU.

    Capacity defaults describe R9700. Pass ``get_device_capabilities`` values
    from initialization to match runtime selection on the target device.
    """

    if max_m < 1:
        raise ValueError("max_m must be positive")
    keys = dict.fromkeys(
        specialization_key(
            m,
            n,
            k,
            mode=mode,
            normal_only=normal_only,
            nibble_layout=nibble_layout,
            dtype=dtype,
            device_id=device_id,
            cache_policy=cache_policy,
            num_processors=num_processors,
            cache_size_bytes=cache_size_bytes,
            lds_size_bytes=lds_size_bytes,
        )
        for m in range(1, max_m + 1)
    )
    return list(keys)


def compile_gemm(key):
    """Compile one key using meta tensors and FlyDSL's compile-only mode.

    Call during single-threaded startup, before inference or graph capture.
    No real tensors or dummy GPU launches are required. Runtime materializes
    the already compiled artifact on its first use.
    """
    engine, kernel_m, n, k, parameters = key
    options = _normalize_config(dict(parameters, engine=engine), kernel_m, k)
    options.pop("engine")
    alignment = _m_alignment(engine, options)
    if kernel_m:
        m = kernel_m
    elif options.get("aligned_m", False):
        m = (
            options.get("tile_m", 256)
            if engine == "hybrid"
            else 16 * options.get("waves_m", 1) * options.get("repeat_m", 1)
        )
    else:
        m = 1
    npad = (n + 15) // 16 * 16
    mode = options["fold_mode"]
    dtype = torch.bfloat16 if options["output_dtype"] == "bf16" else torch.float16
    with torch.device("meta"):
        x = torch.empty(m, k, dtype=torch.float8_e4m3fn)
        packed = torch.empty(npad // 16, k // 16, 2, 16, 4, dtype=torch.uint8)
        scales = torch.empty(npad // 16, k // 32, 16, dtype=torch.uint8)
        x_scale = torch.empty(m, dtype=torch.float32)
        out = torch.empty(m, n, dtype=dtype)
        refs = None if mode == "grouped" else torch.empty(npad, dtype=torch.int32)
        table = None if mode == "grouped" else torch.empty(29 if mode == "e5m2" else 15, 2, dtype=torch.int32)
    args = _arguments(x, packed, scales, x_scale, out, refs, table, kernel_m, alignment)
    factory = build_hybrid if engine == "hybrid" else build_register
    launcher = factory(kernel_m, n, k, **options)
    with _compile_lock:
        previous = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            flyc.compile(launcher, *args, fx.Stream(None))
        finally:
            if previous is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = previous
