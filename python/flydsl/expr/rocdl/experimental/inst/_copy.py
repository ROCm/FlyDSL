# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Copy instructions (CDNA): buffer loads/stores/atomics, scalar buffer loads, async LDS DMA, LDS reads/writes/atomics,
LDS transpose reads, and the DS cross-lane instructions.

One wrapper per ISA instruction, named by its mnemonic. Pointer operands accept an ``fx.Pointer`` (a buffer-descriptor
pointer from ``make_buffer_ptr`` / ``make_buffer_tensor`` for ``rsrc``, a shared pointer for ``lds_ptr`` / ``ptr``, a
global pointer for ``global_ptr``) or an ``!llvm.ptr`` value. Data operands accept any DSL value of the instruction's
width (``BUFFER_STORE_DWORDX4`` stores an ``fx.Vector[fx.BFloat16, 8]``) or a register ``fx.Tensor`` of it; loads
return the native type unless ``dtype=`` names another type of the same width, or store into ``out=``, a register
``fx.Tensor`` of that width (``out=frag`` loads ``frag``'s element type). Offsets are byte offsets; ``soffset`` is
wave-uniform.

``aux``: cache policy, ``"sc0"``, ``"sc1"``, ``"nt"`` joined with ``|`` (or the raw bits); 0 is the default policy.
"""

from ....._mlir.dialects import llvm
from ....._mlir.dialects.fly_rocdl import CopyOpCDNA4LdsReadTransposeType
from ._core import Arg, CopyInstruction, Dtype, Imm, Instruction, IrInstruction, mlir_type

_AUX = Imm("aux", "cpol_gfx942", 0)

# (mnemonic suffix, bytes, native type) of the VMEM data widths.
_WIDTHS = {
    "UBYTE": (1, "i8"),
    "USHORT": (2, "i16"),
    "DWORD": (4, "i32"),
    "DWORDX2": (8, "2xi32"),
    "DWORDX3": (12, "3xi32"),
    "DWORDX4": (16, "4xi32"),
}
_STORE_WIDTHS = {
    "BYTE": "i8",
    "SHORT": "i16",
    "DWORD": "i32",
    "DWORDX2": "2xi32",
    "DWORDX3": "3xi32",
    "DWORDX4": "4xi32",
}

# --------------------------------------------------------------------------------------------- buffer loads/stores


def _buffer_load(width):
    native = _WIDTHS[width][1]
    return Instruction(
        f"BUFFER_LOAD_{width}",
        "rocdl.raw.ptr.buffer.load",
        args=(Arg("rsrc", "ptr8"), Arg("offset", "i32"), Arg("soffset", "i32", default=0)),
        imms=(_AUX,),
        result=Dtype(native),
        doc=f"Load {_WIDTHS[width][0]} bytes per lane from ``rsrc + offset + soffset``; out-of-range lanes read 0.",
        sched="vmem_read",
    )


def _buffer_store(width):
    return Instruction(
        f"BUFFER_STORE_{width}",
        "rocdl.raw.ptr.buffer.store",
        args=(
            Arg("vdata", _STORE_WIDTHS[width]),
            Arg("rsrc", "ptr8"),
            Arg("offset", "i32"),
            Arg("soffset", "i32", default=0),
        ),
        imms=(_AUX,),
        doc="Store ``vdata`` to ``rsrc + offset + soffset``; out-of-range lanes are dropped.",
        sched="vmem_write",
    )


BUFFER_LOAD_UBYTE = _buffer_load("UBYTE")
BUFFER_LOAD_USHORT = _buffer_load("USHORT")
BUFFER_LOAD_DWORD = _buffer_load("DWORD")
BUFFER_LOAD_DWORDX2 = _buffer_load("DWORDX2")
BUFFER_LOAD_DWORDX3 = _buffer_load("DWORDX3")
BUFFER_LOAD_DWORDX4 = _buffer_load("DWORDX4")
BUFFER_STORE_BYTE = _buffer_store("BYTE")
BUFFER_STORE_SHORT = _buffer_store("SHORT")
BUFFER_STORE_DWORD = _buffer_store("DWORD")
BUFFER_STORE_DWORDX2 = _buffer_store("DWORDX2")
BUFFER_STORE_DWORDX3 = _buffer_store("DWORDX3")
BUFFER_STORE_DWORDX4 = _buffer_store("DWORDX4")

# --------------------------------------------------------------------------------------------- buffer atomics


def _atomic(name, op, type, doc):
    return Instruction(
        name,
        op,
        args=(Arg("vdata", type), Arg("rsrc", "ptr8"), Arg("offset", "i32"), Arg("soffset", "i32", default=0)),
        imms=(_AUX,),
        result="=vdata",
        doc=doc + " at ``rsrc + offset + soffset``. Returns the previous memory value.",
        sched="vmem_read|vmem_write",
    )


BUFFER_ATOMIC_ADD_F32 = _atomic("BUFFER_ATOMIC_ADD_F32", "rocdl.raw.ptr.buffer.atomic.fadd", "f32", "Atomic f32 add")
BUFFER_ATOMIC_ADD_F64 = _atomic("BUFFER_ATOMIC_ADD_F64", "rocdl.raw.ptr.buffer.atomic.fadd", "f64", "Atomic f64 add")
BUFFER_ATOMIC_PK_ADD_F16 = _atomic(
    "BUFFER_ATOMIC_PK_ADD_F16", "rocdl.raw.ptr.buffer.atomic.fadd", "2xf16", "Atomic packed 2 x f16 add"
)
BUFFER_ATOMIC_PK_ADD_BF16 = _atomic(
    "BUFFER_ATOMIC_PK_ADD_BF16", "rocdl.raw.ptr.buffer.atomic.fadd", "2xbf16", "Atomic packed 2 x bf16 add (gfx950)"
)
BUFFER_ATOMIC_MAX_F64 = _atomic("BUFFER_ATOMIC_MAX_F64", "rocdl.raw.ptr.buffer.atomic.fmax", "f64", "Atomic f64 max")
BUFFER_ATOMIC_SMAX = _atomic("BUFFER_ATOMIC_SMAX", "rocdl.raw.ptr.buffer.atomic.smax", "i32", "Atomic signed i32 max")
BUFFER_ATOMIC_SMAX_X2 = _atomic(
    "BUFFER_ATOMIC_SMAX_X2", "rocdl.raw.ptr.buffer.atomic.smax", "i64", "Atomic signed i64 max"
)
BUFFER_ATOMIC_UMIN = _atomic("BUFFER_ATOMIC_UMIN", "rocdl.raw.ptr.buffer.atomic.umin", "i32", "Atomic unsigned i32 min")
BUFFER_ATOMIC_UMIN_X2 = _atomic(
    "BUFFER_ATOMIC_UMIN_X2", "rocdl.raw.ptr.buffer.atomic.umin", "i64", "Atomic unsigned i64 min"
)


def _cmpswap(name, type):
    return Instruction(
        name,
        "rocdl.raw.ptr.buffer.atomic.cmpswap",
        args=(
            Arg("src", type),
            Arg("cmp", like="src"),
            Arg("rsrc", "ptr8"),
            Arg("offset", "i32"),
            Arg("soffset", "i32", default=0),
        ),
        imms=(_AUX,),
        result="=src",
        doc=f"Atomic {type} compare-and-swap: store ``src`` where memory equals ``cmp``. Returns the previous value.",
        sched="vmem_read|vmem_write",
    )


BUFFER_ATOMIC_CMPSWAP = _cmpswap("BUFFER_ATOMIC_CMPSWAP", "i32")
BUFFER_ATOMIC_CMPSWAP_X2 = _cmpswap("BUFFER_ATOMIC_CMPSWAP_X2", "i64")

# --------------------------------------------------------------------------------------------- scalar buffer loads


def _s_buffer_load(width, native):
    return Instruction(
        f"S_BUFFER_LOAD_{width}",
        "rocdl.ptr.s.buffer.load",
        args=(Arg("rsrc", "ptr8"), Arg("offset", "i32")),
        imms=(_AUX,),
        result=Dtype(native),
        doc="Scalar (SMEM) load from ``rsrc + offset``; ``rsrc`` and ``offset`` must be wave-uniform, otherwise the "
        "backend falls back to a vector buffer load.",
    )


S_BUFFER_LOAD_DWORD = _s_buffer_load("DWORD", "i32")
S_BUFFER_LOAD_DWORDX2 = _s_buffer_load("DWORDX2", "2xi32")
S_BUFFER_LOAD_DWORDX4 = _s_buffer_load("DWORDX4", "4xi32")
S_BUFFER_LOAD_DWORDX8 = _s_buffer_load("DWORDX8", "8xi32")
S_BUFFER_LOAD_DWORDX16 = _s_buffer_load("DWORDX16", "16xi32")

# --------------------------------------------------------------------------------------------- async LDS DMA
# Each lane copies its bytes to LDS at ``lds_ptr`` (wave-uniform) + lane * size, bypassing registers. Only the async
# form is provided: the compiler does not wait for the copy, so group copies with ``asyncmark()`` and wait with
# ``wait_asyncmark(n)`` before reading the LDS data.

_LDS_DOC = (
    "Async copy of {n} bytes per lane from {src} to LDS at ``lds_ptr`` (wave-uniform) + lane * {n}. Track completion "
    "with ``asyncmark()`` / ``wait_asyncmark(n)``."
)


def _buffer_load_lds(width):
    n = _WIDTHS[width][0]
    return Instruction(
        f"BUFFER_LOAD_{width}_LDS",
        "rocdl.raw.ptr.buffer.load.async.lds",
        args=(Arg("rsrc", "ptr8"), Arg("lds_ptr", "ptr3"), Arg("voffset", "i32"), Arg("soffset", "i32", default=0)),
        imms=(Imm("size", operand=True, fixed=n), Imm("offset", "i32", 0, operand=True), _AUX),
        doc=_LDS_DOC.format(n=n, src="``rsrc + voffset + soffset + offset``"),
        sched="ldsdma",
    )


def _global_load_lds(width):
    n = _WIDTHS[width][0]
    return Instruction(
        f"GLOBAL_LOAD_LDS_{width}",
        "rocdl.global.load.async.lds",
        args=(Arg("global_ptr", "ptr1"), Arg("lds_ptr", "ptr3")),
        imms=(Imm("size", fixed=n), Imm("offset", "i32", 0), _AUX),
        doc=_LDS_DOC.format(n=n, src="``global_ptr + offset`` (per lane)"),
        sched="ldsdma",
    )


BUFFER_LOAD_UBYTE_LDS = _buffer_load_lds("UBYTE")
BUFFER_LOAD_USHORT_LDS = _buffer_load_lds("USHORT")
BUFFER_LOAD_DWORD_LDS = _buffer_load_lds("DWORD")
BUFFER_LOAD_DWORDX3_LDS = _buffer_load_lds("DWORDX3")  # gfx950
BUFFER_LOAD_DWORDX4_LDS = _buffer_load_lds("DWORDX4")  # gfx950
GLOBAL_LOAD_LDS_UBYTE = _global_load_lds("UBYTE")
GLOBAL_LOAD_LDS_USHORT = _global_load_lds("USHORT")
GLOBAL_LOAD_LDS_DWORD = _global_load_lds("DWORD")
GLOBAL_LOAD_LDS_DWORDX3 = _global_load_lds("DWORDX3")  # gfx950
GLOBAL_LOAD_LDS_DWORDX4 = _global_load_lds("DWORDX4")  # gfx950

# --------------------------------------------------------------------------------------------- LDS reads/writes
# Plain LDS accesses at ``ptr``, an LDS pointer aligned to the access size (16 bytes for B96 / B128). A constant added to
# the pointer becomes the instruction's offset field.


def _ds_align(size):
    return 16 if size == 12 else size


def _ds_read(width, mem, size, extend=None):
    def build(ods):
        if extend is None:
            return llvm.load(ods["res"], ods["ptr"], alignment=_ds_align(size))
        return extend(ods["res"], llvm.load(mlir_type(mem), ods["ptr"], alignment=size))

    return IrInstruction(
        f"DS_READ_{width}",
        "llvm.load",
        args=(Arg("ptr", "ptr3"),),
        result="i32" if extend else Dtype(mem),
        doc=f"Load {size} bytes per lane from LDS at ``ptr``" + (", extended to i32." if extend else "."),
        sched="ds_read",
        build=build,
    )


def _ds_write(width, data, size):
    def build(ods):
        llvm.store(ods["vdata"], ods["ptr"], alignment=_ds_align(size))

    return IrInstruction(
        f"DS_WRITE_{width}",
        "llvm.store",
        args=(Arg("vdata", data), Arg("ptr", "ptr3")),
        doc=f"Store {size} bytes per lane (``vdata``) to LDS at ``ptr``.",
        sched="ds_write",
        build=build,
    )


DS_READ_U8 = _ds_read("U8", "i8", 1, llvm.zext)
DS_READ_I8 = _ds_read("I8", "i8", 1, llvm.sext)
DS_READ_U16 = _ds_read("U16", "i16", 2, llvm.zext)
DS_READ_I16 = _ds_read("I16", "i16", 2, llvm.sext)
DS_READ_B32 = _ds_read("B32", "i32", 4)
DS_READ_B64 = _ds_read("B64", "2xi32", 8)
DS_READ_B96 = _ds_read("B96", "3xi32", 12)
DS_READ_B128 = _ds_read("B128", "4xi32", 16)
DS_WRITE_B8 = _ds_write("B8", "i8", 1)
DS_WRITE_B16 = _ds_write("B16", "i16", 2)
DS_WRITE_B32 = _ds_write("B32", "i32", 4)
DS_WRITE_B64 = _ds_write("B64", "2xi32", 8)
DS_WRITE_B96 = _ds_write("B96", "3xi32", 12)
DS_WRITE_B128 = _ds_write("B128", "4xi32", 16)

# --------------------------------------------------------------------------------------------- LDS atomics
# Each operation is two instructions: ``DS_<OP>_<TYPE>`` returns nothing, ``DS_<OP>_RTN_<TYPE>`` returns the previous
# value. ``ptr`` is an LDS pointer aligned to the data size.

_SIZES = {"i32": 4, "f32": 4, "2xf16": 4, "2xbf16": 4, "i64": 8, "f64": 8}


def _ds_atomic(name, bin_op, data, returns):
    def build(ods):
        order = llvm.AtomicOrdering.monotonic
        return llvm.atomicrmw(bin_op, ods["ptr"], ods["vdata"], order, syncscope="workgroup", alignment=_SIZES[data])

    return IrInstruction(
        name,
        "llvm.atomicrmw",
        args=(Arg("vdata", data), Arg("ptr", "ptr3")),
        result="=vdata" if returns else None,
        doc=f"Atomic {bin_op.name.strip('_')} of ``vdata`` into LDS at ``ptr``"
        + (". Returns the previous value." if returns else "."),
        sched="ds_read|ds_write",
        build=build,
    )


def _ds_cmpst(bits):
    def build(ods):
        order = llvm.AtomicOrdering.monotonic
        pair = llvm.cmpxchg(
            ods["ptr"], ods["cmp"], ods["src"], order, order, syncscope="workgroup", alignment=bits // 8
        )
        return llvm.extractvalue(ods["res"], pair, [0])

    return IrInstruction(
        f"DS_CMPST_RTN_B{bits}",
        "llvm.cmpxchg",
        args=(Arg("src", f"i{bits}"), Arg("cmp", like="src"), Arg("ptr", "ptr3")),
        result="=src",
        doc="Atomic compare-and-store in LDS at ``ptr``: store ``src`` where memory equals ``cmp``. Returns the previous "
        "value.",
        sched="ds_read|ds_write",
        build=build,
    )


_B = llvm.AtomicBinOp
# (mnemonic with ``{}`` where ``_RTN`` goes and without the bit count, atomicrmw operation)
_DS_RMW = (
    ("ADD{}_U", _B.add),
    ("SUB{}_U", _B.sub),
    ("INC{}_U", _B.uinc_wrap),
    ("DEC{}_U", _B.udec_wrap),
    ("MIN{}_I", _B.min),
    ("MAX{}_I", _B.max),
    ("MIN{}_U", _B.umin),
    ("MAX{}_U", _B.umax),
    ("AND{}_B", _B._and),
    ("OR{}_B", _B._or),
    ("XOR{}_B", _B._xor),
    ("ADD{}_F", _B.fadd),
    ("MIN{}_F", _B.fmin),
    ("MAX{}_F", _B.fmax),
)
_ds_atomics = [
    _ds_atomic(f"DS_{form.format(rtn)}{bits}", op, ("f" if form.endswith("_F") else "i") + str(bits), bool(rtn))
    for bits in (32, 64)
    for form, op in _DS_RMW
    for rtn in ("", "_RTN")
]
_ds_atomics += [
    _ds_atomic(f"DS_PK_ADD{rtn}_{ty.upper()}", _B.fadd, f"2x{ty}", bool(rtn))
    for ty in ("f16", "bf16")
    for rtn in ("", "_RTN")
]
_ds_atomics += [_ds_atomic(f"DS_WRXCHG_RTN_B{bits}", _B.xchg, f"i{bits}", True) for bits in (32, 64)]
_ds_atomics += [_ds_cmpst(32), _ds_cmpst(64)]
#: The LDS atomics, also bound by name in this module.
DS_ATOMICS = tuple(_ds_atomics)
globals().update((i.__name__, i) for i in DS_ATOMICS)

# --------------------------------------------------------------------------------------------- DS cross-lane
# Lane exchange through the LDS crossbar; LDS memory is not accessed. ``addr`` is a byte address, 4 * lane.

DS_SWIZZLE_B32 = Instruction(
    "DS_SWIZZLE_B32",
    "rocdl.ds_swizzle",
    args=(Arg("src", "i32"),),
    imms=(Imm("offset", "i32", values=range(0x10000), operand=True),),
    result="=src",
    doc="Swizzle ``src`` across lanes by the 16-bit ``offset`` pattern: with bit 15 set, a quad permute "
    "(``offset[7:0]``); otherwise and / or / xor lane masks within 32 lanes (``offset[4:0]``, ``[9:5]``, ``[14:10]``).",
    sched="all_ds",
)
DS_BPERMUTE_B32 = Instruction(
    "DS_BPERMUTE_B32",
    "rocdl.ds_bpermute",
    args=(Arg("addr", "i32", ods="index"), Arg("src", "i32")),
    result="=src",
    doc="Backward permute: each lane reads ``src`` of lane ``addr / 4`` (modulo the wave size).",
    sched="all_ds",
)
DS_PERMUTE_B32 = IrInstruction(
    "DS_PERMUTE_B32",
    "llvm.amdgcn.ds.permute",
    args=(Arg("addr", "i32"), Arg("src", "i32")),
    result="=src",
    doc="Forward permute: each lane sends its ``src`` to lane ``addr / 4`` (modulo the wave size); a lane no lane "
    "sends to reads 0.",
    sched="all_ds",
    build=lambda ods: llvm.call_intrinsic(ods["res"], "llvm.amdgcn.ds.permute", [ods["addr"], ods["src"]], [], []),
)

# --------------------------------------------------------------------------------------------- LDS transpose reads
# 16 consecutive lanes read a 16-row block cooperatively and receive it transposed (gfx950). The layouts are those of
# fx.rocdl.cdna4.LDSReadTrans(granularity, bits); ``val_bits`` is the element width they count in.


def _ds_read_tr(bits, granularity, native):
    return CopyInstruction(
        f"DS_READ_B{bits}_TR_B{granularity}",
        f"rocdl.ds.read.tr{granularity}.b{bits}",
        args=(Arg("ptr", "ptr3"),),
        result=Dtype(native),
        doc=f"Transposing LDS read of {bits} bits per lane in {granularity}-bit elements.",
        copy_op=lambda: CopyOpCDNA4LdsReadTransposeType.get(granularity, bits),
        val_bits=granularity,
        sched="ds_read",
    )


DS_READ_B64_TR_B4 = _ds_read_tr(64, 4, "2xi32")
DS_READ_B64_TR_B8 = _ds_read_tr(64, 8, "2xi32")
DS_READ_B96_TR_B6 = _ds_read_tr(96, 6, "3xi32")
DS_READ_B64_TR_B16 = _ds_read_tr(64, 16, "4xi16")
