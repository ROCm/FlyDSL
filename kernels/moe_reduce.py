"""MoE Reduction Kernel — reduce-sum over the topk dimension.

Performs:  Y[t, d] = sum(X[t, :, d])  for all t, d.

Input shape:  X [tokens, topk, model_dim]   (3-D, contiguous, row-major)
Output shape: Y [tokens, model_dim]          (2-D, contiguous, row-major)

Design constraints (inherited from MoE stage-2 use-case):
  - ``topk`` is compile-time constant and typically small.
    The kernel fully unrolls the topk loop, so large topk values will
    increase code size without benefit.
  - ``model_dim`` must be a multiple of ``VEC_WIDTH * BLOCK_SIZE``
    (currently 8 * 256 = 2048) for full utilisation; shorter tails are
    handled by a bounds check but waste threads.
  - ``model_dim`` should be at least 2048 for the vectorized loads to be
    worthwhile.  For very small model_dim, ``torch.sum`` may be faster.

Typical shapes (DeepSeek-V3 style):
  - (tokens, 8, 7168)  — TP8 dense-shared-expert config
  - (tokens, 6, 5120)  — EP / K=6 config

The kernel is compiled and cached via ``compile_moe_reduction()``.
"""

import functools

import flydsl
from flydsl.dialects.ext import flir, arith, scf
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from _mlir import ir
import _mlir.extras.types as T

from kernels.kernels_common import stream_ptr_to_async_token


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.

    Args:
        topk: Number of expert slots to reduce.
        model_dim: Hidden dimension (should be large and aligned; see module docstring).
        dtype_str: Element type — ``"f16"`` | ``"bf16"`` | ``"f32"``.

    Returns:
        A compiled executable callable as ``exe(X, Y, m_tokens, stream_ptr)``.
    """
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    USE_NONTEMPORAL = True
    VEC_ALIGN = 16

    assert model_dim % VEC_WIDTH == 0, f"Unsupported model_dim: {model_dim} (must be divisible by {VEC_WIDTH})"

    _state = {}

    class _MoeReduction(flir.MlirModule):
        GPU_MODULE_NAME = f"moe_reduction_{topk}_{model_dim}_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            if dtype_str == "f32":
                elem_type = T.f32()
            elif dtype_str == "f16":
                elem_type = T.f16()
            elif dtype_str == "bf16":
                elem_type = ir.BF16Type.get()
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")

            compute_type = T.f32()
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

        @flir.kernel
        def moe_reduction_kernel(
            self: flir.T.i64,
            X: lambda: T.memref(DYN, topk, model_dim, _state["elem_type"]),
            Y: lambda: T.memref(DYN, model_dim, _state["elem_type"]),
            m_tokens: lambda: T.index(),
        ):
            from _mlir.dialects import vector as mlir_vector

            token_idx = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            tensor_X = flir.make_tensor(X, shape=(m_tokens, topk, model_dim), strides=(topk * model_dim, model_dim, 1))
            tensor_Y = flir.make_tensor(Y, shape=(m_tokens, model_dim), strides=(model_dim, 1))

            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_SIZE * VEC_WIDTH
            gX = flir.zipped_divide(tensor_X, (1, 1, tile_cols))
            gY = flir.zipped_divide(tensor_Y, (1, tile_cols))

            copy_atom_load = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            copy_atom_store = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_X = flir.make_tiled_copy_tv(
                copy_atom_load,
                flir.make_ordered_layout((1, 1, BLOCK_SIZE), order=(2, 1, 0)),
                flir.make_ordered_layout((1, 1, VEC_WIDTH), order=(2, 1, 0)),
                thr_shape=(1, 1, BLOCK_SIZE),
                val_shape=(1, 1, VEC_WIDTH),
            )
            tiled_copy_Y = flir.make_tiled_copy_tv(
                copy_atom_store,
                flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0)),
                flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0)),
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )

            thr_copy_X = tiled_copy_X.get_slice(tid)
            thr_copy_Y = tiled_copy_Y.get_slice(tid)

            thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value

            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_f32 = ir.VectorType.get([VEC_WIDTH], compute_type)

            c_model_dim = arith.index(model_dim).value

            tile_n_idx = arith.ArithValue(flir.block_idx("y"))
            c_base = tile_n_idx * BLOCK_SIZE * VEC_WIDTH
            curr_idx = (c_base + thread_offset_base).value

            # Bounds check
            c_end_idx = (arith.ArithValue(curr_idx) + arith.index(VEC_WIDTH)).value
            in_bounds = arith.CmpIOp(arith.CmpIPredicate.sle, c_end_idx, c_model_dim)

            _if_in_bounds = scf.IfOp(arith.as_value(in_bounds))
            with _if_in_bounds.then():
                c_zero_f32 = arith.constant(0.0, type=compute_type)
                acc = mlir_vector.splat(vec_type_f32, arith.as_value(c_zero_f32))

                for k in range_constexpr(topk):
                    blkX = gX[(token_idx, k, tile_n_idx)]
                    thrX = thr_copy_X.partition_S(blkX)
                    frgX = flir.make_fragment_like(thrX, elem_type)
                    flir.copy(
                        tiled_copy_X,
                        thrX,
                        frgX,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                    vec_val_e = mlir_vector.load(vec_type_e, frgX.memref, [c0_idx, c0_idx, c0_idx], alignment=VEC_ALIGN)
                    if dtype_str in ("f16", "bf16"):
                        vec_val = flir.arith.extf(vec_type_f32, arith.as_value(vec_val_e))
                    else:
                        vec_val = vec_val_e
                    acc = (arith.ArithValue(acc) + arith.ArithValue(vec_val)).value

                if dtype_str in ("f16", "bf16"):
                    out_vec = flir.arith.truncf(vec_type_e, arith.as_value(acc))
                else:
                    out_vec = acc

                blkY = gY[(token_idx, tile_n_idx)]
                thrY = thr_copy_Y.partition_S(blkY)
                frgY = flir.make_fragment_like(thrY, elem_type)
                mlir_vector.store(arith.as_value(out_vec), frgY.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                flir.copy(
                    tiled_copy_Y,
                    frgY,
                    thrY,
                    nontemporal=USE_NONTEMPORAL,
                    alignment=VEC_ALIGN,
                )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            X: lambda: T.memref(DYN, topk, model_dim, _state["elem_type"]),
            Y: lambda: T.memref(DYN, model_dim, _state["elem_type"]),
            m_tokens: lambda: T.index(),
            stream_ptr: lambda: T.i64(),  # PyTorch stream pointer
        ):
            from flydsl.dialects.ext import arith as arith_ext
            c1 = arith.as_value(arith_ext.index(1))
            gx = arith.as_value(m_tokens)
            tile_size = BLOCK_SIZE * VEC_WIDTH
            gy = arith.as_value(arith_ext.index((model_dim + tile_size - 1) // tile_size))
            bx = arith.as_value(arith_ext.index(BLOCK_SIZE))

            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [f"moe_reduction_{topk}_{model_dim}_{dtype_str}", "moe_reduction_kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[X, Y, m_tokens],
                async_dependencies=[stream_token],
            )

    m = _MoeReduction()
    exe = flydsl.compile(m)
    return exe
