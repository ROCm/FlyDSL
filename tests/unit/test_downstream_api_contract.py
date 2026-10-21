# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Freeze the FlyDSL API surface that downstream repositories import directly.

vLLM and ATOM author their own FlyDSL kernels, and those kernels reach past the
documented DSL into compiler, runtime and raw MLIR-binding internals. Because the
call sites live in another repository, nothing in this repo turns red when one of
the symbols below is renamed, moved, or has its signature changed: the breakage
only surfaces downstream, after the fact.

Downstream call sites mirrored here (vllm-project/vLLM main, scanned 2026-10-10):

``flydsl.utils.smem_allocator.SmemAllocator`` / ``SmemPtr``
    ``vllm/v1/attention/ops/flydsl_kernels/tq_decode.py``
    ``vllm/v1/attention/ops/flydsl_kernels/tq_decode_gqa6.py``
    ``vllm/v1/attention/ops/flydsl_kernels/ultraquant_decode_hd256.py``
``flydsl.expr.typing.as_ir_value``, ``flydsl.compiler.ast_rewriter.ASTRewriter``
    ``vllm/models/deepseek_v41/amd/mono/common/ops.py``
    ``vllm/models/deepseek_v41/amd/mono/attention/device.py``
``flydsl.compiler.kernel_function.CompilationContext``
    ``vllm/models/deepseek_v41/amd/mono/`` (mono layer construction)
``flydsl.runtime.device.get_rocm_arch``
    mono, tq_decode, ultraquant_decode_hd256
``flydsl._mlir.ir``, ``flydsl._mlir.dialects.{scf, llvm}``
    mono ``attention/device.py`` / ``common/ops.py``, tq_decode, ultraquant

Changing any of these is allowed. Updating the downstream repository in the same
change, and updating this file, is the point. A failure here means "this change
is about to break vLLM", not "this change is wrong".
"""

import inspect
import warnings

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]


def _params(obj):
    """Ordered parameter names of ``obj``, minus ``self``."""
    return [p for p in inspect.signature(obj).parameters if p != "self"]


# ---------------------------------------------------------------------------
# Legacy LDS allocator
#
# CLAUDE.md marks SmemAllocator/SmemPtr legacy and SmemAllocator warns on every
# first use, but *nothing* under kernels/ or tests/ imports it any more -- only
# vLLM does, from three paged-decode kernels. Deleting the module per the
# deprecation plan would leave this repo's CI green and break those kernels at
# import time. Migrate vLLM to fx.SharedAllocator before removing it.
# ---------------------------------------------------------------------------


def _new_legacy_allocator(**kwargs):
    """Construct SmemAllocator the way vLLM does, muting the legacy warning.

    ``_warn_legacy_smem`` fires at most once per process, so asserting on the
    warning would make this test depend on collection order. Suppress instead.
    """
    from flydsl.utils.smem_allocator import SmemAllocator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SmemAllocator(None, **kwargs)


def test_smem_allocator_constructor_signature():
    from flydsl.utils.smem_allocator import SmemAllocator

    # vLLM calls SmemAllocator(None, arch=arch, global_sym_name="tq_smem"):
    # ctx positional, the other two by keyword.
    assert _params(SmemAllocator.__init__) == ["ctx", "arch", "global_sym_name"]


def test_smem_allocator_manual_byte_cursor():
    """vLLM lays out LDS by hand through the mutable ``.ptr`` cursor.

    tq_decode / tq_decode_gqa6 / ultraquant_decode_hd256 read and assign
    ``allocator.ptr`` 17 times between them; it is load-bearing public state, not
    an implementation detail.
    """
    allocator = _new_legacy_allocator(arch="gfx950", global_sym_name="tq_smem")

    assert allocator.ptr == 0
    allocator.ptr = 1024
    q_off = allocator.ptr
    allocator.ptr += 2048
    kv_off = allocator.ptr
    allocator.ptr += 512

    assert (q_off, kv_off) == (1024, 3072)
    assert allocator.ptr == 3584
    assert allocator.global_sym_name == "tq_smem"
    assert allocator.arch == "gfx950"


def test_smem_allocator_emitters_exist():
    """``get_base`` / ``finalize`` need an MLIR insertion point, so only their
    presence and arity are frozen here."""
    from flydsl.utils.smem_allocator import SmemAllocator

    assert callable(SmemAllocator.get_base) and _params(SmemAllocator.get_base) == []
    assert callable(SmemAllocator.finalize) and _params(SmemAllocator.finalize) == []


def test_smem_ptr_signature():
    """vLLM: ``SmemPtr(base, byte_offset, T.f32, shape=(N_CENTROIDS,)).get()``."""
    from flydsl.utils.smem_allocator import SmemPtr

    assert _params(SmemPtr.__init__) == ["base_memref", "byte_offset", "element_type", "shape"]
    for name in ("get", "load", "store"):
        assert callable(getattr(SmemPtr, name)), f"SmemPtr.{name} is gone"
    assert _params(SmemPtr.load) == ["idxs"]
    assert _params(SmemPtr.store) == ["val", "idxs"]


# ---------------------------------------------------------------------------
# Compiler internals reached by the DeepSeek-V4.1 mono kernels
# ---------------------------------------------------------------------------


def test_as_ir_value_signature():
    """mono passes every ROCDL intrinsic operand through ``as_ir_value``.

    No test in this repo called it before this one; only kernels/ and vLLM did.
    """
    from flydsl.expr.typing import as_ir_value

    sig = inspect.signature(as_ir_value)
    assert list(sig.parameters) == ["value", "keep_static"]
    assert sig.parameters["keep_static"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["keep_static"].default is False


def test_ast_rewriter_transform_is_a_single_argument_entry_point():
    """mono's ``traced(fn)`` helper is exactly ``ASTRewriter.transform(fn)``."""
    from flydsl.compiler.ast_rewriter import ASTRewriter

    assert callable(ASTRewriter.transform)
    assert _params(ASTRewriter.transform) == ["f"]

    def _plain(x):
        return x + 1

    # With no transformers registered transform() is identity; either way it must
    # return something callable rather than raising.
    assert callable(ASTRewriter.transform(_plain))


def test_compilation_context_is_importable_and_default_constructible():
    from flydsl.compiler.kernel_function import CompilationContext

    assert _params(CompilationContext.__init__) == []
    assert CompilationContext() is not None


def test_get_rocm_arch_returns_a_gfx_string():
    from flydsl.runtime.device import get_rocm_arch

    assert _params(get_rocm_arch) == []
    arch = get_rocm_arch()
    assert isinstance(arch, str) and arch.startswith("gfx"), arch


# ---------------------------------------------------------------------------
# Raw MLIR bindings
#
# mono and the TurboQuant/UltraQuant kernels build ops against these directly,
# so the binding package layout is part of the downstream contract.
# ---------------------------------------------------------------------------


def test_mlir_binding_entry_points_are_importable():
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm, scf

    assert hasattr(ir, "Type") and hasattr(ir.Type, "parse")
    # mono: ir.Type.parse("!llvm.ptr<1>"), llvm.StructType.get_literal([...]),
    # llvm.extractvalue(...); tq_decode: scf for its hand-built loops.
    assert hasattr(llvm, "StructType") and hasattr(llvm.StructType, "get_literal")
    assert hasattr(llvm, "extractvalue")
    assert hasattr(scf, "ForOp")
