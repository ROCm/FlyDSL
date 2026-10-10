# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x split-K combine for Flash Attention (block=64).

The attention kernel writes fp32 partials ``(m_nat, l, unnormalized O)`` per
KV split. This kernel reduces them, folds an optional attention sink, and
stores bf16/fp16 O plus optional LSE. It is a device kernel, not a host
online-softmax loop. Launch uses a 64-thread block (two gfx120x waves).
"""

from collections.abc import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Vector as Vec
from kernels.common.kernels_common import LOG2E as _LOG2E
from kernels.common.tensor_shim import _run_compiled


def build_splitk_combine_module(
    num_heads: int,
    head_dim: int,
    dtype_str: str = "bf16",
    has_sink: bool = False,
    return_lse: bool = False,
    daz: bool = True,
    unsafe_fp_math: bool = True,
) -> Callable[..., None]:
    """Build the gfx120x split-K combine (block=64). It writes the merged output and LSE."""
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    HAS_SINK = bool(has_sink)
    RETURN_LSE = bool(return_lse)
    if dtype_str == "bf16":
        elem_dtype = fx.BFloat16
    elif dtype_str == "f16":
        elem_dtype = fx.Float16
    else:
        raise ValueError(f"gfx120x split-K combine: unsupported dtype {dtype_str}")
    BLOCK = 64

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def splitk_combine_kernel(
        WsM: fx.Pointer,
        WsL: fx.Pointer,
        WsO: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        Sink: fx.Pointer,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        num_splits: fx.Int32,
        ws_rows: fx.Int32,
    ) -> None:
        def _fadd(a: object, b: object) -> object:
            return a + b

        def _fsub(a: object, b: object) -> object:
            return a - b

        def _fmul(a: object, b: object) -> object:
            return a * b

        def _fmax(a: object, b: object) -> fx.Float32:
            return fx.max(fx.Float32(a), fx.Float32(b))

        def _as_f32(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, ptr.address_space),
                ptr,
            )

        def _as_out(ptr: fx.Pointer) -> fx.Pointer:
            return fx.recast_iter(
                fx.PointerType.get(elem_dtype.ir_type, ptr.address_space),
                ptr,
            )

        m_ptr = _as_f32(WsM)
        l_ptr = _as_f32(WsL)
        o_ptr = _as_f32(WsO)
        out_ptr = _as_out(O)
        lse_ptr = _as_f32(LSE)
        sink_ptr = _as_f32(Sink)

        c_neg = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        c_log2e = fx.Float32(_LOG2E)

        tid = fx.Uint64(gpu.thread_idx.x)
        bid = fx.Uint64(gpu.block_idx.x)
        gid = bid * fx.Uint64(BLOCK) + tid
        total = fx.Uint64(batch_size) * fx.Uint64(NUM_HEADS) * fx.Uint64(seq_len) * fx.Uint64(HEAD_DIM)
        if gid < total:
            d = gid % fx.Uint64(HEAD_DIM)
            rowpack = gid // fx.Uint64(HEAD_DIM)
            q = rowpack % fx.Uint64(seq_len)
            tmp = rowpack // fx.Uint64(seq_len)
            h = tmp % fx.Uint64(NUM_HEADS)
            b = tmp // fx.Uint64(NUM_HEADS)
            row_in_split = (
                fx.Int64(b) * fx.Int64(NUM_HEADS) * fx.Int64(seq_len) + fx.Int64(h) * fx.Int64(seq_len) + fx.Int64(q)
            )

            def _row(s: fx.Int32 | int) -> fx.Int64:
                return fx.Int64(s) * fx.Int64(ws_rows) + row_in_split

            # Single-element scf.yield unwraps to a scalar (not a 1-list).
            m_carry = c_neg
            for s_i, m_iter in range(fx.Int32(0), num_splits, fx.Int32(1), init=[c_neg]):
                m_running = m_iter[0]
                rr = _row(s_i)
                ls = fx.Float32(fx.ptr_load(l_ptr + rr))
                ms = fx.Float32(fx.ptr_load(m_ptr + rr))
                ms_use = (ls > c_zero).select(ms, c_neg)
                m_running = _fmax(m_running, ms_use)
                m_carry = yield [m_running]
            m_final = m_carry

            if const_expr(HAS_SINK):
                _sidx = fx.Int64(b) * fx.Int64(NUM_HEADS) + fx.Int64(h)
                sink_logit = fx.Float32(fx.ptr_load(sink_ptr + _sidx))
                m_final = _fmax(m_final, sink_logit)

            acc_carry = [c_zero, c_zero]
            for s_i, acc_iter in range(fx.Int32(0), num_splits, fx.Int32(1), init=[c_zero, c_zero]):
                acc_r = acc_iter[0]
                l_r = acc_iter[1]
                rr = _row(s_i)
                ls = fx.Float32(fx.ptr_load(l_ptr + rr))
                ms = fx.Float32(fx.ptr_load(m_ptr + rr))
                alive = ls > c_zero
                diff = alive.select(_fmul(_fsub(ms, m_final), c_log2e), c_zero)
                scale = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff).ir_value()))
                scale = alive.select(scale, c_zero)
                oidx = rr * fx.Int64(HEAD_DIM) + fx.Int64(d)
                ov = fx.Float32(fx.ptr_load(o_ptr + oidx))
                acc_r = _fadd(acc_r, _fmul(ov, scale))
                l_r = _fadd(l_r, _fmul(ls, scale))
                acc_carry = yield [acc_r, l_r]
            acc = acc_carry[0]
            l_acc = acc_carry[1]

            if const_expr(HAS_SINK):
                diff_s = _fmul(_fsub(sink_logit, m_final), c_log2e)
                sink_w = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff_s).ir_value()))
                l_acc = _fadd(l_acc, sink_w)

            has = l_acc > c_zero
            l_safe = has.select(l_acc, c_one)
            inv = c_one / l_safe
            out_v = has.select(_fmul(acc, inv), c_zero)
            out_idx = ((fx.Int64(b) * fx.Int64(seq_len) + fx.Int64(q)) * fx.Int64(NUM_HEADS) + fx.Int64(h)) * fx.Int64(
                HEAD_DIM
            ) + fx.Int64(d)
            stored = Vec.from_elements([out_v], fx.Float32).to(elem_dtype)
            view = fx.make_view(
                fx.add_offset(out_ptr, out_idx),
                fx.make_layout(1, 1),
            )
            view.store(stored)
            if const_expr(RETURN_LSE):
                if d == fx.Uint64(0):
                    # Use l_safe (1 when empty) so log never sees 0 under daz /
                    # no-nans-fp-math; empty rows stay finite -inf via has.select.
                    lse_v = has.select(
                        _fadd(m_final, fx.log(l_safe, fastmath=fx.arith.FastMathFlags.fast)),
                        c_neg,
                    )
                    fx.ptr_store(lse_v, lse_ptr + row_in_split)

    @flyc.jit
    def launch_splitk_combine(
        WsM: fx.Pointer,
        WsL: fx.Pointer,
        WsO: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        Sink: fx.Pointer,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        num_splits: fx.Int32,
        ws_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ) -> None:
        ctx = CompilationContext.get_current()
        total = fx.Uint64(batch_size) * fx.Uint64(NUM_HEADS) * fx.Uint64(seq_len) * fx.Uint64(HEAD_DIM)
        grid_x = (total + fx.Uint64(BLOCK - 1)) // fx.Uint64(BLOCK)
        launcher = splitk_combine_kernel(
            WsM,
            WsL,
            WsO,
            O,
            LSE,
            Sink,
            batch_size,
            seq_len,
            num_splits,
            ws_rows,
        )
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            if const_expr(unsafe_fp_math):
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("no-nans-fp-math"),
                            ir.StringAttr.get("true"),
                        ]
                    )
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("unsafe-fp-math"),
                            ir.StringAttr.get("true"),
                        ]
                    )
                )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    _from_c_void_p = flyc.from_c_void_p
    _U8 = fx.Uint8

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return _from_c_void_p(_U8, t.data_ptr())

    def _launch(
        ws_m: torch.Tensor,
        ws_l: torch.Tensor,
        ws_o: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor,
        batch: int,
        seq_len: int,
        num_splits: int,
        ws_rows: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if stream is None:
            stream = torch.cuda.current_stream()
        # Soft-contig workspace/out/sink so a direct module call cannot OOB on strided views.
        if not ws_m.is_contiguous():
            ws_m = ws_m.contiguous()
        if not ws_l.is_contiguous():
            ws_l = ws_l.contiguous()
        if not ws_o.is_contiguous():
            ws_o = ws_o.contiguous()
        if not out.is_contiguous():
            raise ValueError("splitk combine out must be contiguous")
        if lse is not None and lse.numel() and not lse.is_contiguous():
            lse = lse.contiguous()
        if sink is not None and sink.numel() and not sink.is_contiguous():
            sink = sink.contiguous()
        _run_compiled(
            launch_splitk_combine,
            _ptr(ws_m),
            _ptr(ws_l),
            _ptr(ws_o),
            _ptr(out),
            _ptr(lse),
            _ptr(sink),
            int(batch),
            int(seq_len),
            int(num_splits),
            int(ws_rows),
            stream,
        )

    # AOT compiles the jit launcher. The host still calls this wrapper.
    _launch.jit_function = launch_splitk_combine
    return _launch
