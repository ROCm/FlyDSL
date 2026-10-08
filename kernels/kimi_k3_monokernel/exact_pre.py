"""Opt163 bitwise-validated pre-AttnRes primitives."""
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
H = 7168

def width(rows):
    return 512 // min(8, 1 << (rows.bit_length() - 1))

def rounded(op, a, b):
    return fx.Float32(llvm.inline_asm(
        T.f32, [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
        'v_' + op + '_f32 $0, $1, $2', '=v,v,v', has_side_effects=False))

def add(a, b):
    return rounded('add', a, b)

def mul(a, b):
    return rounded('mul', a, b)

def inverse_rms(total):
    x = add(mul(total, 1.0 / H), 1e-5)
    return (fx.Float64(1.0) / fx.sqrt(x.to(fx.Float64))).to(fx.Float32)

def load4(src, offset):
    a = fx.Int32(bo.buffer_load(src, offset // 2, vec_width=1, dtype=T.i32))
    b = fx.Int32(bo.buffer_load(src, offset // 2 + 1, vec_width=1, dtype=T.i32))
    return ((a << 16).bitcast(fx.Float32), (a & fx.Int32(-65536)).bitcast(fx.Float32),
            (b << 16).bitcast(fx.Float32), (b & fx.Int32(-65536)).bitcast(fx.Float32))


def scalar_bf16(src,j):
    word=fx.Int32(bo.buffer_load(src,j//2,vec_width=1,dtype=T.i32))
    return ((j%2)==0).select((word<<16).bitcast(fx.Float32),(word&fx.Int32(-65536)).bitcast(fx.Float32))
