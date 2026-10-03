# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X INT8 ConvRot weight/act quantize (Hadamard rotate + rowwise INT8).

Matches HIP ``quantize_int8_convrot_weight`` / fused
``quantize_and_rotate_rowwise`` semantics:

* Regular Hadamard of size ``group_size`` in {16, 64, 256} via radix-4 FWHT
  (Kronecker of H4); normalize by ``1/sqrt(G)``.
* Per-row scale ``amax/127`` (floor ``1e-30``); ``q = roundeven(x * rcp(scale))``.

Weights are rotated offline (``W @ H``; H is symmetric). Activations use the
same kernel online before ``int8_linear``. Packed ``convrot_w4a4`` /
``asym_w4a8`` live in ``rdna4_convrot_w4a4`` / ``rdna4_asym_w4a8``.

"""

import math
from collections.abc import Callable
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor


def _kernel_signature(**params: object) -> str:
    """Specialization suffix for kernel/launch names (matches kernels.common.gfx120x_buf_helpers)."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


KERNEL_NAME = "int8_convrot_quant_gfx120x"
WARP = 32
_SUPPORTED_GROUPS = (16, 64, 256)
# LDS row staging budget (f32). Matches typical ConvRot card budgets.
_MAX_K_F32 = 8192
_INV_127 = 1.0 / 127.0


def _block_threads(k: int) -> int:
    if k <= 256:
        return 32
    if k <= 512:
        return 64
    if k <= 2048:
        return 128
    return 256


def _dtype_name(dtype: torch.dtype) -> str:
    import torch

    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]


@lru_cache(maxsize=64)
def build_int8_convrot_quant_module(
    in_dtype: str = "bfloat16",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Compile one-row-per-block ConvRot rotate + rowwise INT8 quant kernel."""
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if k > _MAX_K_F32:
        raise ValueError(f"K={k} exceeds LDS staging budget {_MAX_K_F32}")

    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    n_stages = int(math.log(group_size, 4))
    # Butterflies per stage across the whole row: K/4.
    n_butterflies = k // 4
    bf_steps = (n_butterflies + block_threads - 1) // block_threads
    load_steps = (k + block_threads - 1) // block_threads
    reduction_slots = (block_threads + WARP - 1) // WARP
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    row_bytes = k * 4
    red_bytes = max(reduction_slots, 1) * 4
    sig = _kernel_signature(
        block=block_threads,
        dtype=in_dtype,
        group_size=group_size,
        k=k,
        op="int8_convrot",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_quant_kernel(
        In: fx.Pointer,
        OutQ: fx.Pointer,
        OutScale: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
    ) -> None:
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        c0 = fx.Float32(0.0)
        # Grid is exactly n_rows (one block per row).
        active_row = bid < n_rows

        alloc = fx.SharedAllocator()
        row_base = alloc.allocate(row_bytes)._ptr
        red_base = alloc.allocate(red_bytes)._ptr

        def as_f32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, ptr.address_space), ptr)

        row_f = as_f32(row_base)
        red_f = as_f32(red_base)

        def lds_load(idx: object) -> object:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(k))
            off = safe.select(fx.Int64(idx), fx.Int64(0))
            v = fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1)).load()
            return safe.select(fx.Float32(v[0]), c0)

        def lds_store(idx: object, val: object) -> None:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(k))
            off = safe.select(fx.Int64(idx), fx.Int64(0))
            view = fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1))
            # Always store; inactive lanes write index 0 repeatedly (harmless after load).
            view.store(Vec.from_elements([val], fx.Float32))

        def wave_reduce_max(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = fx.max(w, gpu.shuffle_xor(w, off, WARP))
            return w

        def block_reduce_max(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_max(val)
            if lane == 0:
                off = fx.Int64(wave)
                fx.make_view(fx.add_offset(red_f, off), fx.make_layout(1, 1)).store(Vec.from_elements([w], fx.Float32))
            gpu.barrier()
            if wave == 0:
                in_range = lane < fx.Int32(reduction_slots)
                lane_safe = in_range.select(lane, fx.Int32(0))
                vv = fx.make_view(fx.add_offset(red_f, fx.Int64(lane_safe)), fx.make_layout(1, 1)).load()
                ww = in_range.select(fx.Float32(vv[0]), c0)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.make_view(fx.add_offset(red_f, fx.Int64(0)), fx.make_layout(1, 1)).store(
                        Vec.from_elements([ww], fx.Float32)
                    )
            gpu.barrier()
            top = fx.make_view(fx.add_offset(red_f, fx.Int64(0)), fx.make_layout(1, 1)).load()
            return fx.Float32(top[0])

        # --- load row to LDS as f32 ---
        in_buf = ptr_buf_tensor(
            In,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K) * fx.Int64(in_bytes),
        )
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            inb = active_row & (col < fx.Int32(k))
            gidx = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col)
            safe_g = inb.select(gidx, fx.Int64(0))
            raw = buf_copy_load(in_buf, safe_g, elem=InTy, unit_elems=1)
            if const_expr(in_dtype != "float32"):
                val = fx.Float32(InTy(raw).to(fx.Float32))
            else:
                val = fx.Float32(raw)
            val = inb.select(val, c0)
            # Store only when in-bounds for this row; out-of-range cols skip.
            if col < fx.Int32(k):
                off = fx.Int64(col)
                fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1)).store(
                    Vec.from_elements([val], fx.Float32)
                )
        gpu.barrier()

        # --- radix-4 FWHT within each group (H4 Kronecker) ---
        # stages: stride = 4^s for s in 0..n_stages-1
        for stage in range_constexpr(n_stages):
            stride = 4**stage
            block = 4 * stride
            for step in range_constexpr(bf_steps):
                bf = tid + fx.Int32(step * block_threads)
                inb = active_row & (bf < fx.Int32(n_butterflies))
                # Map butterfly index → (group, base, i)
                # Butterflies are packed as: for each group, for each base in 0..G step block,
                # for each i in 0..stride-1 → one butterfly. Count per group = G/4.
                bf_in_group = bf % fx.Int32(group_size // 4)
                group = bf // fx.Int32(group_size // 4)
                # Within group: bf_in_group = (base/block)*stride + i
                base_idx = (bf_in_group // fx.Int32(stride)) * fx.Int32(block)
                i = bf_in_group % fx.Int32(stride)
                gs = group * fx.Int32(group_size)
                idx0 = gs + base_idx + i
                idx1 = idx0 + fx.Int32(stride)
                idx2 = idx0 + fx.Int32(2 * stride)
                idx3 = idx0 + fx.Int32(3 * stride)
                a = lds_load(idx0)
                b = lds_load(idx1)
                c = lds_load(idx2)
                d = lds_load(idx3)
                y0 = a + b + c - d
                y1 = a + b - c + d
                y2 = a - b + c + d
                y3 = -a + b + c + d
                # Mask: inactive threads rewrite zeros at index 0 — skip stores when !inb.
                # Use select to keep SSA; only inb lanes' stores matter for their indices.
                if inb:
                    lds_store(idx0, y0)
                    lds_store(idx1, y1)
                    lds_store(idx2, y2)
                    lds_store(idx3, y3)
            gpu.barrier()

        # Normalize by 1/sqrt(G)
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            if col < fx.Int32(k):
                v = lds_load(col) * fx.Float32(inv_sqrt_g)
                off = fx.Int64(col)
                fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1)).store(Vec.from_elements([v], fx.Float32))
        gpu.barrier()

        # --- amax + scale ---
        thread_amax = c0
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            inb = active_row & (col < fx.Int32(k))
            v = lds_load(col)
            thread_amax = fx.max(thread_amax, inb.select(fmath.absf(v), c0))
        amax = block_reduce_max(thread_amax)
        scale = fx.max(amax * fx.Float32(_INV_127), fx.Float32(1e-30))
        inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))

        if active_row & (tid == fx.Int32(0)):
            sc_buf = ptr_buf_tensor(
                OutScale,
                elem=fx.Float32,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(n_rows) * fx.Int64(4),
            )
            buf_copy_store(sc_buf, fx.Int64(bid), scale, elem=fx.Float32, unit_elems=1)

        gpu.barrier()

        # --- quantize to int8 ---
        q_buf = ptr_buf_tensor(
            OutQ,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K),
        )
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            inb = active_row & (col < fx.Int32(k))
            v = lds_load(col)
            qf = fmath.roundeven(v * inv)
            qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
            qi = qf.to(fx.Int8)
            gidx = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col)
            safe_g = inb.select(gidx, fx.Int64(0))
            # Store: inactive writes to index 0 — only store when inb.
            if inb:
                buf_copy_store(q_buf, safe_g, qi, elem=fx.Int8, unit_elems=1)

    convrot_quant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Pointer,
        OutQ: fx.Pointer,
        OutScale: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        convrot_quant_kernel(In, OutQ, OutScale, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def _ptr(t: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int = 256,
    *,
    stochastic_rounding: int | None = 0,
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object]:
    """Offline ConvRot weight rotation + rowwise INT8 quantize (HIP-compatible API).

    Args:
        weight: floating weight ``[..., K]`` (bf16/fp16/fp32) on CUDA.
        group_size: Hadamard size ∈ {16, 64, 256}; must divide ``K``.
        stochastic_rounding: must be 0 / None (FlyDSL path is deterministic).

    Returns:
        ``(q_int8, scale_f32)`` with ``scale`` shaped ``[*weight.shape[:-1], 1]``.
    """
    require_gfx120x(weight.device, what="quantize_int8_convrot_weight (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if stochastic_rounding:
        raise ValueError("quantize_int8_convrot_weight FlyDSL path does not support stochastic_rounding")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if weight.dim() < 1:
        raise ValueError("weight must have at least 1 dim")
    if weight.device.type != "cuda":
        raise ValueError("weight must be on CUDA/ROCm")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {weight.dtype}")

    orig_shape = tuple(weight.shape)
    k = int(orig_shape[-1])
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if k > _MAX_K_F32:
        raise ValueError(f"K={k} exceeds staging budget {_MAX_K_F32}")

    w2d = weight.reshape(-1, k).contiguous()
    m = w2d.shape[0]
    q = torch.empty((m, k), device=w2d.device, dtype=torch.int8)
    scales = torch.empty((m,), device=w2d.device, dtype=torch.float32)

    launch = build_int8_convrot_quant_module(in_dtype=_dtype_name(w2d.dtype), group_size=group_size, k=k)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        _ptr(w2d),
        _ptr(q),
        _ptr(scales),
        m,
        k,
        stream,
    )
    scale_out = scales.reshape(*orig_shape[:-1], 1)
    return q.reshape(orig_shape), scale_out


def quantize_and_rotate_rowwise(
    x: object,
    group_size: int = 256,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Online activation ConvRot rotate + rowwise INT8 (same kernel as weights)."""
    return quantize_int8_convrot_weight(x, group_size=group_size, stream=stream)


def reference_quantize_int8_convrot_weight(weight: torch.Tensor, group_size: int = 256) -> tuple[object, object]:
    """Torch reference matching eager ConvRot + rowwise INT8."""
    import torch

    def _build_hadamard(size: int, device: object, dtype: object) -> torch.Tensor:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        h = h4
        cur = 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        return h / (size**0.5)

    orig_shape = tuple(weight.shape)
    k = orig_shape[-1]
    w2d = weight.reshape(-1, k).to(torch.float32)
    h = _build_hadamard(group_size, w2d.device, torch.float32)
    n_groups = k // group_size
    wg = w2d.reshape(-1, n_groups, group_size)
    # H symmetric → W @ H.T == W @ H
    rotated = torch.matmul(wg, h.T).reshape(-1, k)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30)
    # Match HIP: round-half-to-even after exact divide (ref uses IEEE /).
    q = torch.round(rotated / scale).clamp(-128, 127).to(torch.int8)
    return q.reshape(orig_shape), scale.reshape(*orig_shape[:-1], 1).to(torch.float32)


def dequantize_int8_convrot_weight(
    q: torch.Tensor, scale: torch.Tensor | float | None, group_size: int = 256, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequant INT8 ConvRot weights and un-rotate (torch reference host path).

    FlyDSL dequant kernel is a follow-up; this host path matches eager
    for tests and requant round-trips.
    """
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16

    def _build_hadamard(size: int, device: object, dtype: object) -> torch.Tensor:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        h = h4
        cur = 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        return h / (size**0.5)

    orig = tuple(q.shape)
    k = orig[-1]
    q2 = q.reshape(-1, k).float()
    sc = scale.to(device=q.device, dtype=torch.float32).reshape(-1, 1)
    deq = q2 * sc
    h = _build_hadamard(group_size, q.device, torch.float32)
    n_groups = k // group_size
    dg = deq.reshape(-1, n_groups, group_size)
    # Inverse rotate: same H (orthogonal, symmetric).
    out = torch.matmul(dg, h.T).reshape(orig)
    return out.to(out_dtype)


def int8_linear_convrot(
    x: object,
    weight_q: object,
    weight_scale: object,
    *,
    group_size: int = 256,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """ConvRot INT8 linear: online act rotate+quant + iu8 WMMA GEMM.

    ``weight_q`` must already be offline-ConvRot-quantized (same ``group_size``).
    Requires the gfx120x iu8 WMMA atom (iu8 / kit 13).

    Host path is intentionally lean for short dual-launch shapes (FlyDSL
    ``do_bench`` / HIP launch-overhead guidance): one act-quant launch then one
    iu8 GEMM, with flat scale buffers and no redundant ``[..., 1]`` round-trips.
    Allocates act-q / scales / out each call (same contract as HIP).
    """
    require_gfx120x(x.device, what="int8_linear_convrot (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled
    from kernels.gemm.rdna4_int8_linear import (
        create_wmma_int8_linear_module,
        pick_tile_config,
    )

    if out_dtype is None:
        out_dtype = torch.bfloat16
    if x.dim() < 1:
        raise ValueError("x must be at least 1D")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    orig_shape = tuple(x.shape)
    k = int(orig_shape[-1])
    if weight_q.dim() != 2:
        raise ValueError("weight_q must be 2D [N, K]")
    n = int(weight_q.shape[0])
    if int(weight_q.shape[1]) != k:
        raise ValueError(f"K mismatch: x {k} vs weight {weight_q.shape[1]}")
    if k % 16 != 0:
        raise ValueError(f"int8_linear_convrot requires K divisible by 16, got {k}")
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if k > _MAX_K_F32:
        raise ValueError(f"K={k} exceeds staging budget {_MAX_K_F32}")
    if x.device.type != "cuda":
        raise ValueError("x must be on CUDA/ROCm")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")

    # Contiguous 2D view without an extra copy when already packed.
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    m = int(x2d.shape[0])

    if stream is None:
        stream = torch.cuda.current_stream()

    # Direct quant launch into flat (m,k)/(m,) — avoids quantize_int8_convrot_weight
    # reshaping scales to [...,1] then immediately flattening again (host enqueue gap
    # on sub-100µs dual-launch paths; see FlyDSL docs/autotune_guide.md).
    aq = torch.empty((m, k), device=x2d.device, dtype=torch.int8)
    a_scale = torch.empty((m,), device=x2d.device, dtype=torch.float32)
    q_launch = build_int8_convrot_quant_module(in_dtype=_dtype_name(x2d.dtype), group_size=group_size, k=k)
    _run_compiled(q_launch, _ptr(x2d), _ptr(aq), _ptr(a_scale), m, k, stream)

    if weight_q.is_contiguous():
        wq = weight_q
    else:
        wq = weight_q.contiguous()
    if (
        weight_scale.dtype == torch.float32
        and weight_scale.device == x.device
        and weight_scale.is_contiguous()
        and weight_scale.numel() in (1, n)
        and weight_scale.dim() == 1
    ):
        w_scale = weight_scale
    else:
        w_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"weight_scale must be scalar or [N]={n}, got {w_scale.numel()}")

    out_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[out_dtype]
    cfg = pick_tile_config(m, n, k)
    skip_bounds = (m % cfg.bm == 0) and (n % cfg.bn == 0)
    g_launch = create_wmma_int8_linear_module(out_name, cfg, skip_bounds=skip_bounds, w_scale_per_n=w_per_n)
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    _run_compiled(
        g_launch,
        _ptr(aq),
        _ptr(wq),
        _ptr(out),
        _ptr(a_scale),
        _ptr(w_scale),
        m,
        n,
        k,
        stream,
    )
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype).reshape(1, -1)
    return out.reshape(*orig_shape[:-1], n)
