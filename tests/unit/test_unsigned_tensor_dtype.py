#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""An unsigned tensor keeps its signedness across the kernel boundary (ROCm/FlyDSL#701).

A ``torch.uint8`` tensor used to arrive as ``Int8``, so every byte with the high bit set was
treated as negative: widening sign-extended and int-to-float used ``sitofp``. These tests
run both signedness families through the same kernel on the device and check the values,
not just the label -- reading unsigned memory, writing unsigned memory, and reinterpreting
signed storage as unsigned with ``recast_iter``.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

# Four bytes per thread, so the copy also exercises the vectorized load path.
PER_THREAD = 4
BYTES = [0x00, 0x01, 0x7F, 0x80, 0xC0, 0xFE, 0xFF, 0x87]
NUM_THREADS = len(BYTES)

_ATOM_FOR_WIDTH = {8: fx.UniversalCopy8b, 16: fx.UniversalCopy16b, 32: fx.UniversalCopy32b, 64: fx.UniversalCopy64b}


def _straddling_values(bits):
    """Values on both sides of the sign bit, so signed and unsigned answers differ."""
    high = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    return [0, 1, sign_bit - 1, sign_bit, (3 << (bits - 2)) & high, high - 1, high, sign_bit + 7]


def _widen_and_scale(src, dst, element_type):
    """dst[i] = int(float(src[i])) * 2 -- wrong for src[i] >= 0x80 if src is read as signed."""
    tid = fx.thread_idx.x
    layout = fx.make_layout(PER_THREAD, 1)
    frag_in = fx.make_rmem_tensor(layout, element_type)
    frag_out = fx.make_rmem_tensor(layout, fx.Int32)
    view_in = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), fx.make_int_tuple(tid * PER_THREAD)), layout))
    view_out = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(dst), fx.make_int_tuple(tid * PER_THREAD)), layout))

    fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), element_type), view_in, frag_in)
    values = frag_in.load()
    frag_out.store((values.to(fx.Float32) * fx.Float32(2.0)).to(fx.Int32))
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), frag_out, view_out)


@flyc.kernel
def _widen_kernel(src: fx.Tensor, dst: fx.Tensor):
    _widen_and_scale(src, dst, src.element_type)


@flyc.jit
def _launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _widen_kernel(src, dst).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


def _run(torch_dtype):
    # Both dtypes get the identical bytes, so only their signedness can explain a
    # difference in the result. ``torch.tensor(..., dtype=torch.int8)`` refuses 0x80..0xFF,
    # hence the reinterpreting view.
    values = BYTES * PER_THREAD
    src = torch.tensor(values, dtype=torch.uint8, device="cuda").view(torch_dtype)
    dst = torch.zeros(len(values), dtype=torch.int32, device="cuda")
    _launch(src, dst, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return src.tolist(), dst.tolist()


@pytest.mark.parametrize("torch_dtype", [torch.uint8, torch.int8])
def test_widen_and_convert_follow_tensor_signedness(torch_dtype):
    """Each dtype must reproduce what torch itself computes for the same bytes."""
    values, got = _run(torch_dtype)
    assert got == [v * 2 for v in values]


def _elementwise_launcher(apply, src_bits, dst_bits, recast=None, recast_out=None, make_frag=None):
    """Build a launcher for ``dst[i] = apply(src[i])``, one element per thread.

    The fragments come from :func:`make_fragment_like`, so the destination fragment --
    and therefore the store -- inherits the destination tensor's signedness, unless
    *make_frag* builds the source fragment with an explicit dtype.

    *recast*, *recast_out* and *make_frag* are functions rather than dtypes on purpose: a
    captured class does not participate in the JIT cache key, so two launchers differing
    only in a captured dtype would share one compiled artifact.
    """
    src_atom, dst_atom = _ATOM_FOR_WIDTH[src_bits], _ATOM_FOR_WIDTH[dst_bits]

    @flyc.kernel
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        tid = fx.thread_idx.x
        layout = fx.make_layout(1, 1)
        src_iter = fx.get_iter(src)
        dst_iter = fx.get_iter(dst)
        if fx.const_expr(recast is not None):
            # Reinterpret the raw storage as another element type, the way the packed-code
            # kernels do; the recast element type is what decides signedness downstream.
            src_iter = recast(src_iter)
        if fx.const_expr(recast_out is not None):
            dst_iter = recast_out(dst_iter)
        view_in = fx.Tensor(fx.make_view(fx.add_offset(src_iter, fx.make_int_tuple(tid)), layout))
        view_out = fx.Tensor(fx.make_view(fx.add_offset(dst_iter, fx.make_int_tuple(tid)), layout))
        if fx.const_expr(make_frag is not None):
            # An explicit dtype makes the fragment's own storage decide signedness, which is
            # how a kernel reads bytes it knows are unsigned out of signed storage.
            frag_in = make_frag(view_in)
        else:
            frag_in = fx.make_fragment_like(view_in)
        frag_out = fx.make_fragment_like(view_out)
        fx.copy(fx.make_copy_atom(src_atom(), frag_in.element_type), view_in, frag_in)
        frag_out.store(apply(frag_in.load(), frag_in.element_type, view_out.element_type))
        fx.copy(fx.make_copy_atom(dst_atom(), view_out.element_type), frag_out, view_out)

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


def _run_elementwise(launch, src, dst_torch_dtype):
    dst = torch.zeros(src.numel(), dtype=dst_torch_dtype, device="cuda")
    launch(src, dst, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return dst.tolist()


def _shift_right(values, src_type, dst_type):
    return values >> dst_type(4)


def _widen_to_i64(values, src_type, dst_type):
    return values.to(fx.Int64)


def _widen_to_i32(values, src_type, dst_type):
    return values.to(fx.Int32)


def _recast_to_unsigned_bytes(iterator):
    return fx.recast_iter(fx.Uint8, iterator)


def _recast_to_unsigned_16(iterator):
    return fx.recast_iter(fx.Uint16, iterator)


def _recast_to_unsigned_32(iterator):
    return fx.recast_iter(fx.Uint32, iterator)


def _recast_to_unsigned_64(iterator):
    return fx.recast_iter(fx.Uint64, iterator)


def _recast_to_signed_bytes(iterator):
    return fx.recast_iter(fx.Int8, iterator)


# Only ``torch.uint8`` is accepted at the tensor boundary, so the wider unsigned dtypes are
# reached the way real kernels reach them: signed storage reinterpreted with ``recast_iter``.
_RECAST_UNSIGNED = {8: _recast_to_unsigned_bytes, 16: _recast_to_unsigned_16, 32: _recast_to_unsigned_32}
_RECAST_UNSIGNED[64] = _recast_to_unsigned_64
_INT_WIDTHS = [(bits, getattr(torch, f"int{bits}")) for bits in (8, 16, 32, 64)]


def _twos_complement(value, bits):
    """The signed spelling of *value*'s bits, so torch will accept it at that width."""
    return value - (1 << bits) if value >> (bits - 1) else value


def _straddling_tensor(bits, signed_dtype):
    values = _straddling_values(bits)
    return values, torch.tensor([_twos_complement(v, bits) for v in values], dtype=signed_dtype, device="cuda")


@pytest.mark.parametrize("bits, signed_dtype", _INT_WIDTHS)
def test_unsigned_source_and_destination_at_every_width(bits, signed_dtype):
    """``v >> 4`` written back through storage of the same width, for every integer width.

    Both the load and the store go through unsigned storage, so this covers the
    destination side as well as the source side. The same bits read as signed must still
    shift arithmetically.
    """
    values, src = _straddling_tensor(bits, signed_dtype)
    recast = _RECAST_UNSIGNED[bits]

    unsigned = _elementwise_launcher(_shift_right, bits, bits, recast=recast, recast_out=recast)
    assert _run_elementwise(unsigned, src, signed_dtype) == [v >> 4 for v in values]

    signed = _elementwise_launcher(_shift_right, bits, bits)
    assert _run_elementwise(signed, src, signed_dtype) == (src >> 4).tolist()


@pytest.mark.parametrize("bits, signed_dtype", [w for w in _INT_WIDTHS if w[0] < 64])
def test_widening_covers_the_full_unsigned_range_at_every_width(bits, signed_dtype):
    """Zero- versus sign-extension into a destination wide enough to hold either answer.

    Every unsigned width has to widen correctly, not just ``uint8``: ``0xFFFFFFFF`` must
    arrive as ``4294967295`` and the same bits read as signed as ``-1``. 64 bits is
    excluded because there is no wider signed destination to observe it in.
    """
    values, src = _straddling_tensor(bits, signed_dtype)

    unsigned = _elementwise_launcher(_widen_to_i64, bits, 64, recast=_RECAST_UNSIGNED[bits])
    assert _run_elementwise(unsigned, src, torch.int64) == values

    signed = _elementwise_launcher(_widen_to_i64, bits, 64)
    assert _run_elementwise(signed, src, torch.int64) == src.tolist()


def test_recast_iter_reinterprets_storage_as_unsigned():
    """``recast_iter(fx.Uint8, ...)`` -- how packed codes are read -- yields unsigned bytes."""
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda").view(torch.int8)

    as_unsigned = _elementwise_launcher(_widen_to_i32, 8, 32, recast=_recast_to_unsigned_bytes)
    assert _run_elementwise(as_unsigned, src, torch.int32) == BYTES

    as_signed = _elementwise_launcher(_widen_to_i32, 8, 32, recast=_recast_to_signed_bytes)
    assert _run_elementwise(as_signed, src, torch.int32) == src.tolist()


def _unsigned_byte_fragment(view):
    return fx.make_fragment_like(view, fx.Uint8)


def _signed_byte_fragment(view):
    return fx.make_fragment_like(view, fx.Int8)


def test_make_fragment_like_with_an_explicit_dtype_decides_signedness():
    """``make_fragment_like(view, fx.Uint8)`` makes the register storage unsigned.

    The source tensor is the same bytes both times, so only the fragment's declared dtype
    can explain the difference.
    """
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda")

    as_unsigned = _elementwise_launcher(_widen_to_i32, 8, 32, make_frag=_unsigned_byte_fragment)
    assert _run_elementwise(as_unsigned, src, torch.int32) == BYTES

    as_signed = _elementwise_launcher(_widen_to_i32, 8, 32, make_frag=_signed_byte_fragment)
    assert _run_elementwise(as_signed, src, torch.int32) == src.view(torch.int8).tolist()


def _scalar_load_launcher(load):
    """``dst[i] = int32(src[i])`` where the element is read one scalar at a time.

    *load* is a function rather than a flag so that the two index paths of
    :func:`memref_load` compile to separate kernels instead of sharing a JIT cache entry.
    """

    @flyc.kernel
    def kernel(src: fx.Tensor, dst: fx.Tensor):
        tid = fx.thread_idx.x
        layout = fx.make_layout(1, 1)
        view_in = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), fx.make_int_tuple(tid)), layout))
        view_out = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(dst), fx.make_int_tuple(tid)), layout))
        frag_in = fx.make_fragment_like(view_in)
        frag_out = fx.make_fragment_like(view_out)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy8b(), frag_in.element_type), view_in, frag_in)
        frag_out[0] = load(frag_in).to(fx.Int32)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), view_out.element_type), frag_out, view_out)

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(src, dst).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


def _load_at_static_index(frag):
    # A Python index: memref_load builds the int tuple itself and checks the profile.
    return fx.memref_load(frag, 0)


def _load_at_dynamic_index(frag):
    # An int-tuple value built from a run-time operand -- the other branch of memref_load.
    # thread_idx.y is 0 for a one-dimensional block, but the compiler cannot know that.
    return fx.memref_load(frag, fx.make_int_tuple(fx.thread_idx.y))


@pytest.mark.parametrize("load", [_load_at_static_index, _load_at_dynamic_index])
@pytest.mark.parametrize("torch_dtype", [torch.uint8, torch.int8])
def test_scalar_memref_load_keeps_the_storage_signedness(load, torch_dtype):
    """One element at a time, the scalar counterpart of the vector load already covered.

    ``memref_load`` yields a signless value, so widening it can only pick ``extui`` over
    ``extsi`` if the dtype came from the memref it was loaded from. The same bytes are read
    through unsigned and signed storage, so only that can explain a difference.
    """
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda").view(torch_dtype)
    got = _run_elementwise(_scalar_load_launcher(load), src, torch.int32)
    assert got == src.tolist()


def _ordered_against_64(values, src_type, dst_type):
    bound = src_type(0x40)
    return (values > bound).select(values, bound).to(fx.Int32)


@pytest.mark.parametrize("torch_dtype", [torch.uint8, torch.int8])
def test_ordering_executes_with_the_dtype_signedness(torch_dtype):
    """Comparison picks ``ugt`` or ``sgt`` from the dtype; checked on hardware.

    ``0x40`` sits below the sign bit, so the same bytes order differently under the two
    readings and only the dtype can explain the result.
    """
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda").view(torch_dtype)
    got = _run_elementwise(_elementwise_launcher(_ordered_against_64, 8, 32), src, torch.int32)
    assert got == [max(v, 0x40) for v in src.tolist()]


def _predicate_widened(values, src_type, dst_type):
    return (values > src_type(0x40)).to(fx.Int32)


def test_widening_a_predicate_uses_the_unsigned_domain():
    """``i1`` carries no sign bit, so widening ``true`` must give 1, not -1.

    A width-1 integer's only bit is its value. Reading it as signed would sign-extend
    through ``extsi`` and turn every set predicate into -1, so ``ArithValue`` and
    ``int_to_int`` both normalise width-1 signedness away. The device shows the
    difference directly.
    """
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda")
    got = _run_elementwise(_elementwise_launcher(_predicate_widened, 8, 32), src, torch.int32)
    assert got == [int(v > 0x40) for v in src.tolist()]


@flyc.kernel
def _pointer_shift_kernel(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32):
    tid = fx.thread_idx.x
    if tid < n:
        dst[tid] = (src[tid] >> 4).to(fx.Int32)


@flyc.jit
def _launch_pointer_shift(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _pointer_shift_kernel(src, dst, n).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


@pytest.mark.parametrize("dsl_dtype, torch_dtype", [(fx.Uint8, torch.uint8), (fx.Int8, torch.int8)])
def test_raw_pointer_argument_keeps_its_signedness(dsl_dtype, torch_dtype):
    """A raw pointer argument carries a dtype too, and loading through it must honour it."""
    src = torch.tensor(BYTES, dtype=torch.uint8, device="cuda")
    dst = torch.zeros(NUM_THREADS, dtype=torch.int32, device="cuda")

    _launch_pointer_shift(
        flyc.from_c_void_p(dsl_dtype, src.data_ptr()),
        flyc.from_c_void_p(fx.Int32, dst.data_ptr()),
        NUM_THREADS,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert dst.tolist() == [v >> 4 for v in src.view(torch_dtype).tolist()]


@flyc.kernel
def _ptr_load_scalar_kernel(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32):
    tid = fx.thread_idx.x
    if tid < n:
        # The pointer's element type is signed; the ``ui8`` result type is what asks for an
        # unsigned read. It names the dtype only -- the SSA value it produces stays signless.
        value = fx.ptr_load(src + tid, result_type=ir.IntegerType.get_unsigned(8))
        dst[tid] = (value >> 4).to(fx.Int32)


@flyc.kernel
def _ptr_load_vector_kernel(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32):
    tid = fx.thread_idx.x
    if tid < n:
        values = fx.ptr_load(src + tid * PER_THREAD, result_type=ir.Type.parse("vector<4xui8>"))
        for i in fx.range_constexpr(PER_THREAD):
            dst[tid * PER_THREAD + i] = (values[i] >> 4).to(fx.Int32)


@flyc.jit
def _launch_ptr_load_scalar(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _ptr_load_scalar_kernel(src, dst, n).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


@flyc.jit
def _launch_ptr_load_vector(src: fx.Pointer, dst: fx.Pointer, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _ptr_load_vector_kernel(src, dst, n).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


@pytest.mark.parametrize("launch, per_thread", [(_launch_ptr_load_scalar, 1), (_launch_ptr_load_vector, PER_THREAD)])
def test_ptr_load_with_an_unsigned_result_type(launch, per_thread):
    """``ptr_load`` may be handed a raw ``uiN`` MLIR type; it must mean unsigned, and must
    not leak into the SSA value, which ``arith`` would reject."""
    values = BYTES * per_thread
    src = torch.tensor(values, dtype=torch.uint8, device="cuda").view(torch.int8)
    dst = torch.zeros(len(values), dtype=torch.int32, device="cuda")

    launch(
        flyc.from_c_void_p(fx.Int8, src.data_ptr()),
        flyc.from_c_void_p(fx.Int32, dst.data_ptr()),
        NUM_THREADS,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert dst.tolist() == [v >> 4 for v in values]
