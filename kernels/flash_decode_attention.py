"""Flash Decode Attention kernel builder.

Single-query (decode-phase) attention using online softmax:
  O[h,d] = sum_j( softmax(Q[h,:] . K[h,j,:] / sqrt(d_k))_j * V[h,j,d] )

Architecture:
  Grid:  (total_heads, 1, 1)  -- one wavefront per (batch, head)
  Block: (WARP_SIZE, 1, 1)    -- AMD wave64, barrier-free dot product reduction

Each thread owns ELEMS_PER_THREAD = head_dim / WARP_SIZE output elements.
Dot products Q.K[j] use intra-warp xor-shuffle sum reduction so all lanes
see the same score without shared-memory barriers.
Online softmax avoids materializing the full attention score matrix.

Memory layout (row-major, batch and heads flattened into dim-0):
  Q: [total_heads, head_dim]
  K: [total_heads, seq_len, head_dim]
  V: [total_heads, seq_len, head_dim]
  O: [total_heads, head_dim]

where total_heads = batch_size * num_heads.
"""

from _mlir import ir

from flydsl.dialects.ext import flir, arith, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
import _mlir.extras.types as T


KERNEL_NAME = "flash_decode_attention"

WARP_SIZE = 64


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32()
    if dtype_str == "f16":
        return T.f16()
    if dtype_str == "bf16":
        return T.bf16()
    raise ValueError(f"unsupported dtype: {dtype_str}")


def build_flash_decode_attention_module(
    seq_len: int,
    head_dim: int,
    dtype_str: str = "f16",
):
    """Build MLIR module for decode-phase flash attention.

    Args:
        seq_len:   KV cache sequence length (compile-time constant).
        head_dim:  per-head dimension, must be divisible by WARP_SIZE (64).
        dtype_str: element type for Q/K/V/O ("f32", "f16", or "bf16").

    Returns:
        An MlirModule instance whose ``__call__`` launches the kernel.
    """
    if head_dim % WARP_SIZE != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by WARP_SIZE ({WARP_SIZE})"
        )

    arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    BLOCK_THREADS = WARP_SIZE
    ELEMS_PER_THREAD = head_dim // WARP_SIZE

    _state = {}

    class _FlashDecodeAttn(flir.MlirModule):
        GPU_MODULE_NAME = "flash_decode_attn"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}", abi = "500">']

        def init_gpu_module(self):
            _state["elem_type"] = dtype_to_elem_type(dtype_str)
            _state["compute_type"] = T.f32()

        @flir.kernel
        def flash_decode_attention_kernel(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, head_dim, _state["elem_type"]),
            K: lambda: T.memref(DYN, seq_len, head_dim, _state["elem_type"]),
            V: lambda: T.memref(DYN, seq_len, head_dim, _state["elem_type"]),
            O: lambda: T.memref(DYN, head_dim, _state["elem_type"]),
            total_heads: lambda: T.index(),
        ):
            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]
            fm_fast = flir.arith.FastMathFlags.fast

            h = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            c_neg_inf = arith.constant(float("-inf"), type=compute_type)
            c_zero_f = arith.constant(0.0, type=compute_type)
            c_log2e = arith.constant(1.4426950408889634, type=compute_type)
            rsqrt_d = arith.constant(1.0 / (head_dim ** 0.5), type=compute_type)

            # Thread t owns output elements [d_base .. d_base + ELEMS_PER_THREAD).
            c_ept = flir.const_index(ELEMS_PER_THREAD)
            d_base = flir.arith.MulIOp(
                arith.as_value(tid), arith.as_value(c_ept)
            ).result

            # Pre-compute per-element head-dim indices.
            d_indices = []
            for e in range_constexpr(ELEMS_PER_THREAD):
                d_off = flir.const_index(e)
                d_indices.append(
                    flir.arith.AddIOp(
                        arith.as_value(d_base), arith.as_value(d_off)
                    ).result
                )

            # Load this thread's Q elements into registers.
            q_local = []
            for e in range_constexpr(ELEMS_PER_THREAD):
                q_e = flir.memref.load(Q, [arith.as_value(h), d_indices[e]])
                q_f = (
                    q_e
                    if dtype_str == "f32"
                    else flir.arith.extf(compute_type, arith.as_value(q_e))
                )
                q_local.append(q_f)

            # ---- online softmax state ----
            m = c_neg_inf  # running max
            l = c_zero_f   # running denominator (sum of exp)
            acc = [c_zero_f] * ELEMS_PER_THREAD  # weighted V accumulator

            width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))

            # ---- main loop over KV-cache positions (compile-time unrolled) ----
            for j_py in range_constexpr(seq_len):
                j = flir.const_index(j_py)

                # Partial dot product: Q[d_base:d_base+EPT] . K[h, j, d_base:d_base+EPT]
                partial = c_zero_f
                for e in range_constexpr(ELEMS_PER_THREAD):
                    k_e = flir.memref.load(
                        K, [arith.as_value(h), arith.as_value(j), d_indices[e]]
                    )
                    k_f = (
                        k_e
                        if dtype_str == "f32"
                        else flir.arith.extf(
                            compute_type, arith.as_value(k_e)
                        )
                    )
                    qk = flir.arith.MulFOp(
                        arith.as_value(q_local[e]),
                        arith.as_value(k_f),
                        fastmath=fm_fast,
                    ).result
                    partial = flir.arith.AddFOp(
                        arith.as_value(partial), qk, fastmath=fm_fast
                    ).result

                # Warp-wide sum reduction (xor-shuffle, wave64).
                w = arith.as_value(partial)
                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.as_value(arith.constant(sh, type=T.i32()))
                    peer = arith.as_value(
                        gpu.ShuffleOp(
                            arith.as_value(w), off, width_i32, mode="xor"
                        ).shuffleResult
                    )
                    w = flir.arith.AddFOp(
                        arith.as_value(w), peer, fastmath=fm_fast
                    ).result

                # score = dot(Q, K_j) / sqrt(head_dim)
                score = flir.arith.MulFOp(
                    arith.as_value(w),
                    arith.as_value(rsqrt_d),
                    fastmath=fm_fast,
                ).result

                # Online softmax update:
                #   m_new     = max(m, score)
                #   correction = exp2((m_old - m_new) * log2e)
                #   p          = exp2((score - m_new) * log2e)
                #   l          = l * correction + p
                #   acc[e]     = acc[e] * correction + p * V[h, j, e]
                m_new = flir.arith.MaximumFOp(
                    arith.as_value(m), arith.as_value(score)
                ).result

                diff_m = flir.arith.SubFOp(
                    arith.as_value(m), m_new, fastmath=fm_fast
                ).result
                corr_arg = flir.arith.MulFOp(
                    diff_m, arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                correction = flir.math.exp2(corr_arg, fastmath=fm_fast)

                diff_s = flir.arith.SubFOp(
                    arith.as_value(score), m_new, fastmath=fm_fast
                ).result
                p_arg = flir.arith.MulFOp(
                    diff_s, arith.as_value(c_log2e), fastmath=fm_fast
                ).result
                p = flir.math.exp2(p_arg, fastmath=fm_fast)

                l_corr = flir.arith.MulFOp(
                    arith.as_value(l),
                    arith.as_value(correction),
                    fastmath=fm_fast,
                ).result
                l = flir.arith.AddFOp(
                    l_corr, arith.as_value(p), fastmath=fm_fast
                ).result

                # Update accumulator with weighted V.
                new_acc = []
                for e in range_constexpr(ELEMS_PER_THREAD):
                    v_e = flir.memref.load(
                        V, [arith.as_value(h), arith.as_value(j), d_indices[e]]
                    )
                    v_f = (
                        v_e
                        if dtype_str == "f32"
                        else flir.arith.extf(
                            compute_type, arith.as_value(v_e)
                        )
                    )
                    a_corr = flir.arith.MulFOp(
                        arith.as_value(acc[e]),
                        arith.as_value(correction),
                        fastmath=fm_fast,
                    ).result
                    pv = flir.arith.MulFOp(
                        arith.as_value(p),
                        arith.as_value(v_f),
                        fastmath=fm_fast,
                    ).result
                    new_acc.append(
                        flir.arith.AddFOp(a_corr, pv, fastmath=fm_fast).result
                    )

                acc = new_acc
                m = m_new

            # ---- store output: O[h, d] = acc[d] / l ----
            for e in range_constexpr(ELEMS_PER_THREAD):
                out_f32 = flir.arith.DivFOp(
                    arith.as_value(acc[e]),
                    arith.as_value(l),
                    fastmath=fm_fast,
                ).result
                if dtype_str != "f32":
                    out_e = flir.arith.truncf(elem_type, out_f32)
                else:
                    out_e = out_f32
                flir.memref.store(
                    arith.as_value(out_e),
                    O,
                    [arith.as_value(h), d_indices[e]],
                )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, head_dim, _state["elem_type"]),
            K: lambda: T.memref(DYN, seq_len, head_dim, _state["elem_type"]),
            V: lambda: T.memref(DYN, seq_len, head_dim, _state["elem_type"]),
            O: lambda: T.memref(DYN, head_dim, _state["elem_type"]),
            total_heads: lambda: T.index(),
        ):
            c1 = flir.arith_ext.index(1)
            gx = total_heads
            bx = flir.arith_ext.index(BLOCK_THREADS)
            flir.gpu_ext.LaunchFuncOp(
                ["flash_decode_attn", "flash_decode_attention_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Q, K, V, O, total_heads],
            )

    return _FlashDecodeAttn()
