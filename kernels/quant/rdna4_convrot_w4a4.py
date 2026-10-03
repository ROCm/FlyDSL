# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X ConvRot W4A4 weight/act quantize (Hadamard + signed INT4 pack).

Hadamard of size ``convrot_groupsize`` in {16, 64, 256} (radix-4 FWHT), then
per-row scale ``amax/7``, round-to-nearest, clamp to signed ``[-7, 7]``, and
row-major nibble pack ``qdata[N, K//2]`` (low nibble = even column).

**Linear default is native int4:** ``convrot_w4a4_linear`` uses
``linear_dtype="int4"`` — device act ConvRot-i4 + ``rdna4_iu4_gemm``. That
matches the HIP ConvRot-W4A4 linear idea and ties or beats HIP on an otherwise idle
shapes we timed. Pass ``linear_dtype="int8"`` only for unpack→iu8. If the
native shape gate fails (``K % 16 != 0``), the call falls back to unpack→iu8.

``quant_group_size`` must be 64. AWQ and SVDQuant stay on their own modules.

Call examples: ``docs/prebuilt_kernels_guide.md`` (gfx120x ConvRot / iu4).
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
from kernels.common.dispatch_mode import get_dispatch_mode
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

# Reference int4 MMA group (layout contract); quantize itself is rowwise.
_INT4_GROUP_SIZE = 64
_INT4_MAX = 7
_INV_7 = 1.0 / float(_INT4_MAX)
_SCALE_FLOOR = 1e-10


def _kernel_signature(**params: object) -> str:
    """Specialization suffix for kernel/launch names (matches kernels.common.gfx120x_buf_helpers)."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


KERNEL_NAME = "convrot_w4a4_quant_gfx120x"
WARP = 32
_SUPPORTED_GROUPS = (16, 64, 256)
# LDS row staging budget (f32). Same card budget as int8 ConvRot.
_MAX_K_F32 = 8192


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
def build_convrot_w4a4_quant_module(
    in_dtype: str = "bfloat16",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Compile one-row-per-block ConvRot rotate + rowwise signed INT4 pack."""
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if k % 2 != 0:
        raise ValueError(f"K={k} must be even for INT4 packing")
    if k > _MAX_K_F32:
        raise ValueError(f"K={k} exceeds LDS staging budget {_MAX_K_F32}")

    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    n_stages = int(math.log(group_size, 4))
    n_butterflies = k // 4
    bf_steps = (n_butterflies + block_threads - 1) // block_threads
    load_steps = (k + block_threads - 1) // block_threads
    packed_k = k // 2
    pack_steps = (packed_k + block_threads - 1) // block_threads
    reduction_slots = (block_threads + WARP - 1) // WARP
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    row_bytes = k * 4
    red_bytes = max(reduction_slots, 1) * 4
    sig = _kernel_signature(
        block=block_threads,
        dtype=in_dtype,
        group_size=group_size,
        k=k,
        op="convrot_w4a4",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_w4a4_quant_kernel(
        In: fx.Pointer,
        OutQ: fx.Pointer,
        OutScale: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
    ) -> None:
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        c0 = fx.Float32(0.0)
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
            if col < fx.Int32(k):
                off = fx.Int64(col)
                fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1)).store(
                    Vec.from_elements([val], fx.Float32)
                )
        gpu.barrier()

        # --- radix-4 FWHT within each group (H4 Kronecker) ---
        for stage in range_constexpr(n_stages):
            stride = 4**stage
            block = 4 * stride
            for step in range_constexpr(bf_steps):
                bf = tid + fx.Int32(step * block_threads)
                inb = active_row & (bf < fx.Int32(n_butterflies))
                bf_in_group = bf % fx.Int32(group_size // 4)
                group = bf // fx.Int32(group_size // 4)
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

        # --- amax + scale (amax/7, floor 1e-10 — layout contract) ---
        thread_amax = c0
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            inb = active_row & (col < fx.Int32(k))
            v = lds_load(col)
            thread_amax = fx.max(thread_amax, inb.select(fmath.absf(v), c0))
        amax = block_reduce_max(thread_amax)
        scale = fx.max(amax * fx.Float32(_INV_7), fx.Float32(_SCALE_FLOOR))
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

        # --- quantize to signed int4 in LDS (reuse f32 slots as i32 codes) ---
        for step in range_constexpr(load_steps):
            col = tid + fx.Int32(step * block_threads)
            inb = active_row & (col < fx.Int32(k))
            v = lds_load(col)
            qf = fmath.roundeven(v * inv)
            qf = fx.max(fx.min(qf, fx.Float32(float(_INT4_MAX))), fx.Float32(float(-_INT4_MAX)))
            # Store as f32-coded int for pack stage (keeps one LDS buffer).
            if col < fx.Int32(k):
                off = fx.Int64(col)
                fx.make_view(fx.add_offset(row_f, off), fx.make_layout(1, 1)).store(
                    Vec.from_elements([inb.select(qf, c0)], fx.Float32)
                )
        gpu.barrier()

        # --- pack two signed nibbles per byte (low=even, high=odd) ---
        q_buf = ptr_buf_tensor(
            OutQ,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        for step in range_constexpr(pack_steps):
            pc = tid + fx.Int32(step * block_threads)
            inb = active_row & (pc < fx.Int32(packed_k))
            c0i = pc * fx.Int32(2)
            c1i = c0i + fx.Int32(1)
            lo_f = lds_load(c0i)
            hi_f = lds_load(c1i)
            # Convert to int32 then mask nibble (two's complement & 0xF).
            lo_i = lo_f.to(fx.Int32) & fx.Int32(0xF)
            hi_i = hi_f.to(fx.Int32) & fx.Int32(0xF)
            packed = (lo_i | (hi_i << fx.Int32(4))).to(fx.Int8)
            gidx = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(pc)
            safe_g = inb.select(gidx, fx.Int64(0))
            if inb:
                buf_copy_store(q_buf, safe_g, packed, elem=fx.Int8, unit_elems=1)

    convrot_w4a4_quant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Pointer,
        OutQ: fx.Pointer,
        OutScale: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        convrot_w4a4_quant_kernel(In, OutQ, OutScale, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def _ptr(t: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import torch

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


def _pack_int4_row_major(values: object) -> torch.Tensor:
    """Wire format: low nibble = even column, high = odd."""
    import torch

    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Signed-nibble unpack (full ``[-8, 7]`` storage range)."""
    import torch

    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    *,
    stochastic_rounding: int | None = 0,
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object]:
    """Offline ConvRot + signed INT4 pack (HIP-compatible API).

    Returns:
        ``(qdata[N, K//2] int8, scales[N] float32)``.
    """
    require_gfx120x(weight.device, what="quantize_convrot_w4a4_weight (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA contract requires quant_group_size {_INT4_GROUP_SIZE}, " f"got {quant_group_size}")
    if stochastic_rounding:
        raise ValueError("quantize_convrot_w4a4_weight FlyDSL path does not support stochastic_rounding")
    if convrot_groupsize not in _SUPPORTED_GROUPS:
        raise ValueError(f"convrot_groupsize must be one of {_SUPPORTED_GROUPS}, got {convrot_groupsize}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N, K], got shape {tuple(weight.shape)}")
    if weight.device.type != "cuda":
        raise ValueError("weight must be on CUDA/ROCm")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {weight.dtype}")

    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % convrot_groupsize != 0:
        raise ValueError(f"K={k} not divisible by convrot_groupsize={convrot_groupsize}")
    if k % quant_group_size != 0:
        raise ValueError(f"K={k} not divisible by quant_group_size={quant_group_size}")
    if k > _MAX_K_F32:
        raise ValueError(f"K={k} exceeds staging budget {_MAX_K_F32}")

    w2d = weight.contiguous()
    q = torch.empty((n, k // 2), device=w2d.device, dtype=torch.int8)
    scales = torch.empty((n,), device=w2d.device, dtype=torch.float32)

    launch = build_convrot_w4a4_quant_module(in_dtype=_dtype_name(w2d.dtype), group_size=convrot_groupsize, k=k)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(launch, _ptr(w2d), _ptr(q), _ptr(scales), n, k, stream)
    return q, scales


def reference_quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
) -> tuple[object, object]:
    """Torch reference matching eager reference ConvRot W4A4."""
    import torch

    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    n, k = weight.shape
    w2d = weight.reshape(-1, k).to(torch.float32)
    h = _build_hadamard(convrot_groupsize, w2d.device, torch.float32)
    n_groups = k // convrot_groupsize
    wg = w2d.reshape(-1, n_groups, convrot_groupsize)
    rotated = torch.matmul(wg, h.T).reshape(-1, k)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=_SCALE_FLOOR)
    scale = abs_max / float(_INT4_MAX)
    q = torch.round(rotated / scale).clamp(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return _pack_int4_row_major(q).reshape(n, k // 2), scale.reshape(n).to(torch.float32)


def dequantize_convrot_w4a4_weight(
    qdata: object,
    scales: object,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: object = None,
) -> torch.Tensor:
    """Dequant packed W4 + un-rotate (torch host path; matches eager reference)."""
    import torch

    if output_dtype is None:
        output_dtype = torch.float32
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    w_int = _unpack_int4_row_major(qdata).to(torch.float32)
    w_rot = w_int * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    h = _build_hadamard(convrot_groupsize, qdata.device, torch.float32)
    n, k = w_rot.shape
    n_groups = k // convrot_groupsize
    wg = w_rot.reshape(n, n_groups, convrot_groupsize)
    return torch.matmul(wg, h.T).reshape(n, k).to(output_dtype)


def _convrot_w4a4_native_iu4(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    *,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Native iu4 path: device act ConvRot-i4 + ``rdna4_iu4_gemm``.

    Raises when shapes fail ``shapes_ok_for_native_iu4``; the public linear
    wrapper catches that and falls back to unpack→iu8.
    """
    import torch

    from kernels.gemm.rdna4_iu4_gemm import iu4_gemm, shapes_ok_for_native_iu4

    if out_dtype is None:
        out_dtype = x.dtype
    orig = tuple(x.shape)
    k = int(orig[-1])
    n = int(qweight.shape[0])
    if int(qweight.shape[1]) * 2 != k:
        raise ValueError(f"Input K={k} does not match qweight K={qweight.shape[1] * 2}")
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    m = int(x2d.shape[0])
    if not shapes_ok_for_native_iu4(m, n, k):
        raise ValueError(f"native iu4 gate miss for MNK=({m},{n},{k})")
    if k % convrot_groupsize != 0:
        raise ValueError(f"K={k} not divisible by convrot_groupsize={convrot_groupsize}")

    # Device act path: same ConvRot rotate+i4 pack kernel as weights (reference
    # HIP uses convrot_quant_int4). Avoid host Hadamard tax that slowed the idle microbench.
    a_packed, a_scale = quantize_convrot_w4a4_weight(x2d, convrot_groupsize=convrot_groupsize, stream=stream)
    w_scale = wscales.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    out2d = iu4_gemm(
        a_packed,
        qweight.contiguous(),
        a_scale.contiguous(),
        w_scale,
        out_dtype=out_dtype,
        prefer_native=True,
        stream=stream,
    )
    if bias is not None:
        out2d = out2d + bias.to(device=out2d.device, dtype=out2d.dtype)
    return out2d.reshape(*orig[:-1], n)


def convrot_w4a4_linear(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
    *,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """ConvRot W4A4 linear: bf16/fp16 ``x`` × packed W4 weights → ``[M, N]``.

    Typical call (default = native int4)::

        qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
        y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)

    Args:
        x: ``[..., K]`` activations (bf16/fp16). On ``linear_dtype='int4'``
            the host ConvRot-quantizes acts to packed int4 on device.
        qweight: ``[N, K//2]`` int8 from ``quantize_convrot_w4a4_weight``.
        wscales: ``[N]`` float32 weight scales from the same quantize call.
        convrot_groupsize: Hadamard group in ``{16, 64, 256}``; must divide K.
        linear_dtype: ``'int4'`` (default) = act ConvRot-i4 + native
            ``iu4_gemm``. ``'int8'`` = unpack weight nibbles → int8-act ConvRot
            linear. If the native shape gate fails under ``'int4'``, the host
            falls back to the ``'int8'`` path automatically.
        out_dtype: output dtype (default = ``x.dtype``).

    See ``docs/prebuilt_kernels_guide.md`` for call examples.
    Requires gfx120x iu4 (native) and iu8 (fallback) WMMA atoms.
    """
    require_gfx120x(x.device, what="convrot_w4a4_linear (gfx120x)")
    import torch

    from kernels.quant.rdna4_int8_convrot import int8_linear_convrot

    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    if out_dtype is None:
        out_dtype = x.dtype
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")

    # FLYDSL_DISPATCH_MODE: force_flydsl → native int4; force_hip → unpack int8 path.
    mode = get_dispatch_mode()
    if mode == "force_flydsl":
        linear_dtype = "int4"
    elif mode == "force_hip":
        linear_dtype = "int8"

    if linear_dtype == "int4":
        try:
            return _convrot_w4a4_native_iu4(
                x,
                qweight,
                wscales,
                bias=bias,
                convrot_groupsize=convrot_groupsize,
                out_dtype=out_dtype,
                stream=stream,
            )
        except Exception:
            # Default-safe: unpack→iu8 when native misses shapes / fails.
            pass

    w_int8 = _unpack_int4_row_major(qweight).to(torch.int8)
    return int8_linear_convrot(
        x,
        w_int8,
        wscales,
        group_size=convrot_groupsize,
        bias=bias,
        out_dtype=out_dtype,
        stream=stream,
    )
