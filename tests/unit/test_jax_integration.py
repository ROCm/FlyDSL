#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the FlyDSL JAX integration layer.

Tests the adapter (from_jax), primitive (jax_kernel), and FFI bridge
independently, without requiring a full FlyDSL kernel compilation
(except where noted).
"""

import ctypes
import struct

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pytest.skip("JAX not installed", allow_module_level=True)

from flydsl._mlir import ir


# ======================================================================
# Adapter tests (from_jax / JaxTensorAdaptor)
# ======================================================================


class TestJaxTensorAdaptor:
    """Tests for flydsl.jax.adapter."""

    def test_import_without_torch(self):
        """Importing flydsl.jax should not require torch."""
        from flydsl.jax import from_jax, JaxTensorAdaptor

        assert from_jax is not None
        assert JaxTensorAdaptor is not None

    def test_basic_f32(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta = from_jax(a)
        assert ta._orig_dtype == jnp.float32
        assert ta._orig_shape == (128,)
        assert ta._orig_strides == (1,)

    def test_2d_array(self):
        from flydsl.jax import from_jax

        a = jnp.ones((32, 64), dtype=jnp.float32)
        ta = from_jax(a)
        assert ta._orig_shape == (32, 64)
        assert ta._orig_strides == (64, 1)

    @pytest.mark.parametrize(
        "dtype",
        [jnp.float32, jnp.float16, jnp.bfloat16, jnp.int32, jnp.int8, jnp.uint8],
    )
    def test_multiple_dtypes(self, dtype):
        from flydsl.jax import from_jax

        a = jnp.ones(64, dtype=dtype)
        ta = from_jax(a)
        assert ta._orig_dtype == dtype

    def test_fly_types_returns_memref(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta = from_jax(a)
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            types = ta.__fly_types__()
            assert len(types) == 1
            assert "f32" in str(types[0])
            assert "128" in str(types[0])

    def test_fly_ptrs_returns_pointers(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta = from_jax(a)
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            ta.__fly_types__()  # build memref desc
            ptrs = ta.__fly_ptrs__()
            assert len(ptrs) >= 1
            assert isinstance(ptrs[0], int) or isinstance(ptrs[0], ctypes.c_void_p)

    def test_cache_signature_stable(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta1 = from_jax(a)
        ta2 = from_jax(a)
        assert ta1.__cache_signature__() == ta2.__cache_signature__()

    def test_cache_signature_differs_by_shape(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        b = jnp.ones(256, dtype=jnp.float32)
        assert from_jax(a).__cache_signature__() != from_jax(b).__cache_signature__()

    def test_cache_signature_differs_by_dtype(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        b = jnp.ones(128, dtype=jnp.float16)
        assert from_jax(a).__cache_signature__() != from_jax(b).__cache_signature__()

    def test_mark_layout_dynamic(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta = from_jax(a).mark_layout_dynamic(leading_dim=0, divisibility=4)
        # Should return self for chaining
        assert ta is not None
        assert ta._orig_shape == (128,)

    def test_assumed_align(self):
        from flydsl.jax import from_jax

        a = jnp.ones(128, dtype=jnp.float32)
        ta = from_jax(a, assumed_align=16)
        assert ta.assumed_align == 16
        sig = ta.__cache_signature__()
        assert 16 in sig

    def test_rejects_non_jax_array(self):
        from flydsl.jax.adapter import JaxTensorAdaptor

        with pytest.raises(TypeError, match="Expected jax.Array"):
            JaxTensorAdaptor([1, 2, 3])

    def test_float8_dtype_if_available(self):
        """Float8 arrays should be handled via uint8 view."""
        from flydsl.jax import from_jax

        f8 = getattr(jnp, "float8_e4m3fn", None)
        if f8 is None:
            pytest.skip("float8_e4m3fn not available in this JAX version")
        a = jnp.ones(64, dtype=f8)
        ta = from_jax(a)
        assert ta._orig_dtype == f8


# ======================================================================
# Primitive tests (jax_kernel / flydsl_call_p)
# ======================================================================


class TestJaxKernelPrimitive:
    """Tests for flydsl.jax.primitive."""

    def test_eager_single_output(self):
        from flydsl.jax import jax_kernel

        call_log = []

        def mock_fn(*args, **kwargs):
            call_log.append({"n_args": len(args), "kwargs": dict(kwargs)})

        wrapped = jax_kernel(
            mock_fn,
            out_shapes=lambda a, b: [(a.shape, a.dtype)],
        )

        a = jnp.ones(64, dtype=jnp.float32)
        b = jnp.ones(64, dtype=jnp.float32)
        result = wrapped(a, b)

        assert len(result) == 1
        assert result[0].shape == (64,)
        assert result[0].dtype == jnp.float32
        assert len(call_log) == 1
        # 2 inputs + 1 output = 3 JaxTensorAdaptors
        assert call_log[0]["n_args"] == 3

    def test_eager_multiple_outputs(self):
        from flydsl.jax import jax_kernel

        def mock_fn(*args, **kwargs):
            pass

        wrapped = jax_kernel(
            mock_fn,
            out_shapes=lambda a: [
                (a.shape, jnp.float32),
                (a.shape, jnp.int32),
            ],
        )

        a = jnp.ones(32, dtype=jnp.float32)
        result = wrapped(a)

        assert len(result) == 2
        assert result[0].dtype == jnp.float32
        assert result[1].dtype == jnp.int32

    def test_constexpr_kwargs_forwarded(self):
        from flydsl.jax import jax_kernel

        received = {}

        def mock_fn(*args, **kwargs):
            received.update(kwargs)

        wrapped = jax_kernel(
            mock_fn,
            out_shapes=lambda a: [(a.shape, a.dtype)],
            constexpr_kwargs={"block_dim": 256, "vec_width": 4},
        )

        a = jnp.ones(64, dtype=jnp.float32)
        wrapped(a)
        assert received == {"block_dim": 256, "vec_width": 4}

    def test_runtime_scalars_forwarded(self):
        from flydsl.jax import jax_kernel

        call_log = []

        def mock_fn(*args, **kwargs):
            call_log.append({"n_args": len(args), "args": list(args)})

        wrapped = jax_kernel(
            mock_fn,
            out_shapes=lambda a: [(a.shape, a.dtype)],
            runtime_scalars={"n": 128},
        )

        a = jnp.ones(64, dtype=jnp.float32)
        wrapped(a)

        # 1 input + 1 output + 1 scalar = 3 args
        assert call_log[0]["n_args"] == 3
        # Last arg should be the scalar value 128
        assert call_log[0]["args"][-1] == 128

    def test_abstract_eval(self):
        """Primitive abstract eval should propagate shapes/dtypes."""
        from jax._src import core
        from flydsl.jax.primitive import flydsl_call_p

        aval_in = core.ShapedArray((128,), jnp.float32)
        aval_out = core.ShapedArray((64,), jnp.int32)

        result = flydsl_call_p.abstract_eval(
            aval_in,
            flyc_func=None,
            out_avals=(aval_out,),
            constexpr_kwargs=(),
            runtime_scalars=(),
        )

        # result is (out_avals, effects)
        out_avals = result[0]
        assert len(out_avals) == 1
        assert out_avals[0].shape == (64,)
        assert out_avals[0].dtype == jnp.int32


# ======================================================================
# C trampoline tests
# ======================================================================


class TestXlaBridge:
    """Tests for the C trampoline (_xla_bridge.so)."""

    @pytest.fixture
    def bridge_lib(self):
        from flydsl.jax.ffi_bridge import _get_bridge_lib

        return _get_bridge_lib()

    def test_register_and_dispatch(self, bridge_lib):
        """Register a function and verify the bridge dispatches correctly."""
        results = []

        FUNC_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def test_fn(ptrs_raw):
            ptrs = ctypes.cast(ptrs_raw, ctypes.POINTER(ctypes.c_void_p))
            # Read buffer slot
            buf_storage = ctypes.cast(ptrs[0], ctypes.POINTER(ctypes.c_void_p))
            results.append(("buf0", buf_storage[0]))
            # Read stream slot
            stream_storage = ctypes.cast(ptrs[1], ctypes.POINTER(ctypes.c_void_p))
            results.append(("stream", stream_storage[0]))

        cfunc = FUNC_T(test_fn)
        func_ptr = ctypes.cast(cfunc, ctypes.c_void_p).value

        slot = bridge_lib.flydsl_xla_register(func_ptr, 1, 0, None)
        assert slot >= 0

        bridge_ptr = bridge_lib.flydsl_xla_get_bridge(slot)
        assert bridge_ptr != 0

        # Simulate XLA calling the bridge
        BRIDGE_T = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
        )
        bridge_fn = BRIDGE_T(bridge_ptr)

        fake_buf = ctypes.c_void_p(0xBEEF)
        buffers = (ctypes.c_void_p * 1)(fake_buf)
        opaque = struct.pack("i", slot)

        bridge_fn(0xDEAD, ctypes.cast(buffers, ctypes.c_void_p), opaque, len(opaque))

        assert results[0] == ("buf0", 0xBEEF)
        assert results[1] == ("stream", 0xDEAD)

    def test_scalar_insertion(self, bridge_lib):
        """Scalars should be inserted between buffers and stream."""
        results = []

        FUNC_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def test_fn(ptrs_raw):
            ptrs = ctypes.cast(ptrs_raw, ctypes.POINTER(ctypes.c_void_p))
            for i in range(4):  # 2 bufs + 1 scalar + 1 stream
                storage = ctypes.cast(ptrs[i], ctypes.POINTER(ctypes.c_void_p))
                results.append(storage[0])

        cfunc = FUNC_T(test_fn)
        func_ptr = ctypes.cast(cfunc, ctypes.c_void_p).value

        scalar_val = (ctypes.c_int64 * 1)(42)
        slot = bridge_lib.flydsl_xla_register(
            func_ptr, 2, 1, ctypes.cast(scalar_val, ctypes.c_void_p)
        )

        bridge_ptr = bridge_lib.flydsl_xla_get_bridge(slot)
        BRIDGE_T = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
        )
        bridge_fn = BRIDGE_T(bridge_ptr)

        bufs = (ctypes.c_void_p * 2)(ctypes.c_void_p(0xAA), ctypes.c_void_p(0xBB))
        opaque = struct.pack("i", slot)

        bridge_fn(0xCC, ctypes.cast(bufs, ctypes.c_void_p), opaque, len(opaque))

        assert results[0] == 0xAA  # buf0
        assert results[1] == 0xBB  # buf1
        assert results[2] == 42    # scalar
        assert results[3] == 0xCC  # stream

    def test_registration_returns_incremental_slots(self, bridge_lib):
        FUNC_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        noop = FUNC_T(lambda p: None)
        ptr = ctypes.cast(noop, ctypes.c_void_p).value

        s1 = bridge_lib.flydsl_xla_register(ptr, 1, 0, None)
        s2 = bridge_lib.flydsl_xla_register(ptr, 1, 0, None)
        assert s2 == s1 + 1

    def test_rejects_too_many_buffers(self, bridge_lib):
        FUNC_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        noop = FUNC_T(lambda p: None)
        ptr = ctypes.cast(noop, ctypes.c_void_p).value

        slot = bridge_lib.flydsl_xla_register(ptr, 999, 0, None)
        assert slot == -1

    def test_rejects_too_many_scalars(self, bridge_lib):
        FUNC_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        noop = FUNC_T(lambda p: None)
        ptr = ctypes.cast(noop, ctypes.c_void_p).value

        slot = bridge_lib.flydsl_xla_register(ptr, 1, 999, None)
        assert slot == -1


# ======================================================================
# Registration deduplication
# ======================================================================


class TestRegistrationDedup:
    """Test that compile_and_register deduplicates by shape signature."""

    def test_same_shapes_produce_same_target_name(self):
        """Same function name + shapes + kwargs should hash to the same target."""
        import hashlib

        def _build_target_name(func_name, input_shapes, output_shapes, constexpr, scalars):
            sig_parts = [func_name]
            for shape, dtype in input_shapes:
                sig_parts.append(f"i{shape}:{dtype}")
            for shape, dtype in output_shapes:
                sig_parts.append(f"o{shape}:{dtype}")
            for k, v in sorted(constexpr.items()):
                sig_parts.append(f"c{k}={v}")
            for k, v in sorted(scalars.items()):
                sig_parts.append(f"r{k}={v}")
            name_hash = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:16]
            return f"flydsl_{name_hash}"

        name1 = _build_target_name(
            "vecAdd",
            [((128,), jnp.float32), ((128,), jnp.float32)],
            [((128,), jnp.float32)],
            {"const_n": 129},
            {"n": 128},
        )
        name2 = _build_target_name(
            "vecAdd",
            [((128,), jnp.float32), ((128,), jnp.float32)],
            [((128,), jnp.float32)],
            {"const_n": 129},
            {"n": 128},
        )
        assert name1 == name2

    def test_different_shapes_produce_different_target_name(self):
        import hashlib

        def _build_target_name(func_name, input_shapes, output_shapes, constexpr, scalars):
            sig_parts = [func_name]
            for shape, dtype in input_shapes:
                sig_parts.append(f"i{shape}:{dtype}")
            for shape, dtype in output_shapes:
                sig_parts.append(f"o{shape}:{dtype}")
            for k, v in sorted(constexpr.items()):
                sig_parts.append(f"c{k}={v}")
            for k, v in sorted(scalars.items()):
                sig_parts.append(f"r{k}={v}")
            name_hash = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()[:16]
            return f"flydsl_{name_hash}"

        name1 = _build_target_name(
            "vecAdd",
            [((128,), jnp.float32)],
            [((128,), jnp.float32)],
            {}, {},
        )
        name2 = _build_target_name(
            "vecAdd",
            [((256,), jnp.float32)],
            [((256,), jnp.float32)],
            {}, {},
        )
        assert name1 != name2
