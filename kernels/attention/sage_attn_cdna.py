# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SageAttention kernel for AMD MI308X (CDNA gfx942)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from kernels.attention.sage_attn_helpers import SageKernelFacade, _make_sage_launch_config
from kernels.common.tensor_shim import _run_compiled


def build_sage_attn_cdna_module(
    num_q_heads,
    num_kv_heads,
    head_dim,
    causal=False,
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    use_bias=False,
    v_transposed: bool = False,
    bias_block_m: int | None = None,
    gfx942_w256=True,
):
    """Build FlyDSL sage attention kernel for AMD MI308X (CDNA gfx942)."""
    del gfx942_w256  # MI308X path is always 32x32x16 w256; kept for harness/API compat
    cfg = _make_sage_launch_config(
        num_q_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        block_m=block_m,
        block_n=block_n,
        flat_work_group_size=flat_work_group_size,
        use_bias=use_bias,
        v_transposed=v_transposed,
        bias_block_m=bias_block_m,
        path_tag=path_tag,
    )
    t = cfg["t"]
    allocator = cfg["allocator"]
    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_SIZE = cfg["BLOCK_SIZE"]
    NUM_Q_HEADS = cfg["NUM_Q_HEADS"]
    impl = SageKernelFacade(t, allocator)
    k_loader = impl.k_loader
    v_loader = impl.v_loader
    qk_softmax = impl.qk_softmax
    pv_gemm = impl.pv_gemm
    store = impl.store

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def sage_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        Bias: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
    ):
        const_expr(cfg["cache_tag"])
        impl.setup(Q, K, V, O, Q_descale, K_descale, V_descale, Bias, batch_size, seq_len_q, seq_len_k, num_q_blocks)
        if const_expr(impl.t.NUM_PIPE_STAGES == 2):
            k_loader.coop_load_k(impl.ZERO_INDEX, impl.ZERO_INDEX)
            v_loader.coop_load_v(impl.ZERO_INDEX, impl.ZERO_INDEX)
            k_loader.coop_load_k(fx.Int64(impl.t.BLOCK_N), impl.K_BUF1_OFF)
            v_loader.coop_load_v(fx.Int64(impl.t.BLOCK_N), impl.V_BUF1_OFF)
            gpu.barrier()
            c0_i32_init = fx.Int32(0).ir_value()
            init_args = [c0_i32_init, _raw(impl.c_neg_inf), _raw(impl.c_zero_f)]
            for _ in range_constexpr(impl.t.D_CHUNKS):
                init_args.append(_raw(impl.c_zero_v16f32))
            _OFF_CUR_BUF = 0
            _OFF_M = 1
            _OFF_L = 2
            _OFF_O_ACCS = 3
            loop_results = init_args
            for kv_block_start, inner_iter_args in range(0, impl.kv_upper, impl.t.BLOCK_N, init=init_args):
                cur_buf_i32 = inner_iter_args[_OFF_CUR_BUF]
                m_running = inner_iter_args[_OFF_M]
                l_running = inner_iter_args[_OFF_L]
                o_accs = [inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(impl.t.D_CHUNKS)]
                cur_k_off = impl._buf_off(cur_buf_i32, impl.K_BUF1_OFF)
                cur_v_off = impl._buf_off(cur_buf_i32, impl.V_BUF1_OFF)
                next_buf_i32 = (fx.Int32(cur_buf_i32) ^ fx.Int32(1)).ir_value()
                m_new, l_new, corr, p_words = qk_softmax.emit_qk_softmax_pquant(
                    kv_block_start, cur_k_off, m_running, l_running
                )
                corr_vec16 = Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
                for dc in range_constexpr(impl.t.D_CHUNKS):
                    o_accs[dc] = impl._fmul(o_accs[dc], corr_vec16)
                kv_block_after_next = fx.Int64(kv_block_start) + fx.Int64(2 * impl.t.BLOCK_N)
                o_accs = pv_gemm.run_pv_mfma(p_words, cur_v_off, o_accs)
                gpu.barrier()
                k_loader.coop_load_k(kv_block_after_next, cur_k_off)
                v_loader.coop_load_v(kv_block_after_next, cur_v_off)
                _yield_args = [next_buf_i32, _raw(m_new), _raw(l_new)]
                for dc in range_constexpr(impl.t.D_CHUNKS):
                    _yield_args.append(o_accs[dc])
                loop_results = yield _yield_args
            _FINAL_OFF_L = _OFF_L
            _FINAL_OFF_O_ACCS = _OFF_O_ACCS
        else:
            init_args = [_raw(impl.c_neg_inf), _raw(impl.c_zero_f)]
            for _ in range_constexpr(impl.t.D_CHUNKS):
                init_args.append(_raw(impl.c_zero_v16f32))
            _OFF_M = 0
            _OFF_L = 1
            _OFF_O_ACCS = 2
            loop_results = init_args
            for kv_block_start, inner_iter_args in range(0, impl.kv_upper, impl.t.BLOCK_N, init=init_args):
                m_running = inner_iter_args[_OFF_M]
                l_running = inner_iter_args[_OFF_L]
                o_accs = [inner_iter_args[_OFF_O_ACCS + i] for i in range_constexpr(impl.t.D_CHUNKS)]
                k_loader.coop_load_k(kv_block_start, impl.ZERO_INDEX)
                v_loader.coop_load_v(kv_block_start, impl.ZERO_INDEX)
                gpu.barrier()
                m_new, l_new, corr, p_words = qk_softmax.emit_qk_softmax_pquant(
                    kv_block_start, impl.ZERO_INDEX, m_running, l_running
                )
                corr_vec16 = Vec.from_elements([corr], fx.Float32).broadcast_to(16).ir_value()
                for dc in range_constexpr(impl.t.D_CHUNKS):
                    o_accs[dc] = impl._fmul(o_accs[dc], corr_vec16)
                o_accs = pv_gemm.run_pv_mfma(p_words, impl.ZERO_INDEX, o_accs)
                gpu.barrier()
                _yield_args = [_raw(m_new), _raw(l_new)]
                for dc in range_constexpr(impl.t.D_CHUNKS):
                    _yield_args.append(o_accs[dc])
                loop_results = yield _yield_args
            _FINAL_OFF_L = _OFF_L
            _FINAL_OFF_O_ACCS = _OFF_O_ACCS
        if impl.q_in_bounds:
            store.write_output(loop_results, _FINAL_OFF_L, _FINAL_OFF_O_ACCS)

    @flyc.jit
    def launch_sage_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Q_descale: fx.Tensor,
        K_descale: fx.Tensor,
        V_descale: fx.Tensor,
        Bias: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_q_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        bs_idx = fx.Int64(batch_size)
        slq_idx = fx.Int64(seq_len_q)
        num_q_tiles = (slq_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_Q_HEADS
        launcher = sage_attn_kernel(
            Q, K, V, O, Q_descale, K_descale, V_descale, Bias, batch_size, seq_len_q, seq_len_k, num_q_blocks
        )
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get("denormal-fp-math-f32"), ir.StringAttr.get("preserve-sign,preserve-sign")]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": True, "lsr-drop-solution": True},
    }

    def _launch(
        Q,
        K,
        V,
        O,  # noqa: E741
        Q_descale,
        K_descale,
        V_descale,
        Bias,
        batch_size,
        seq_len_q,
        seq_len_k,
        num_q_blocks,
        stream=None,
    ):
        with CompilationContext.compile_hints(_compile_hints):
            return _run_compiled(
                launch_sage_attn,
                Q,
                K,
                V,
                O,
                Q_descale,
                K_descale,
                V_descale,
                Bias,
                batch_size,
                seq_len_q,
                seq_len_k,
                num_q_blocks,
                stream,
            )

    def _compile(
        Q,
        K,
        V,
        O,  # noqa: E741
        Q_descale,
        K_descale,
        V_descale,
        Bias,
        batch_size,
        seq_len_q,
        seq_len_k,
        num_q_blocks,
        stream=None,  # noqa: E741
    ):
        with CompilationContext.compile_hints(_compile_hints):
            return flyc.compile(
                launch_sage_attn,
                Q,
                K,
                V,
                O,
                Q_descale,
                K_descale,
                V_descale,
                Bias,
                batch_size,
                seq_len_q,
                seq_len_k,
                num_q_blocks,
                fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch
