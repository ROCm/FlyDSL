#!/usr/bin/env python3
"""
Regression tests for the co_freevars mismatch fix in ast_rewriter.py.

When a kernel is a closure and AST rewriting eliminates a captured free
variable (e.g. const_expr(True) unpacked to True), the new code object
has fewer co_freevars than the original __closure__.  The fix in
ASTRewriter.transform detects this mismatch and rebuilds the function
via types.FunctionType with a matching closure.

These tests verify that the fix works for:
  - single const_expr elimination
  - multiple const_expr eliminations
  - non-closure kernels (no rebuild needed)
  - functools.update_wrapper preserves __annotations__ across rebuild (E2E)
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l1a_compile]


def _has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def test_closure_const_expr_freevars_mismatch():
    """Closure kernel with const_expr(True) unpacked: co_freevars shrinks from 2 to 1."""

    def make_kernel(batch_size):
        from flydsl.expr import const_expr

        @flyc.jit
        def kernel(x: fx.Int32) -> fx.Int32:
            if const_expr(True):
                return batch_size + x
            return x

        return kernel

    make_kernel(32)


def test_closure_multiple_const_expr_eliminated():
    """Multiple const_expr calls eliminated, co_freevars shrinks further."""

    def make_kernel(batch_size):
        from flydsl.expr import const_expr

        @flyc.jit
        def kernel(x: fx.Int32) -> fx.Int32:
            if const_expr(True):
                y = batch_size + x
            if const_expr(False):
                y = x
            return y

        return kernel

    make_kernel(64)


def test_non_closure_kernel_no_issue():
    """Top-level kernel (no closure) should not hit the enclosing_mod path."""

    @flyc.jit
    def kernel(x: fx.Int32) -> fx.Int32:
        return x + 1

    assert kernel.func.__closure__ is None


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(not _has_gpu(), reason="CUDA/ROCm not available")
def test_constexpr_annotation_survives_freevar_rebuild():
    """E2E: functools.update_wrapper preserves __annotations__ across rebuild.

    const_expr is imported locally → becomes a closure variable for kernel.
    AST rewriter inlines const_expr(True) → drops const_expr from co_freevars
    → triggers types.FunctionType rebuild.

    Without functools.update_wrapper, __annotations__ is lost:
      - Constexpr[int] annotation vanishes
      - BLOCK_SIZE is treated as a regular kernel parameter
      - TypeError: Cannot derive IR types from 64
    """
    from flydsl.expr import buffer_ops
    from flydsl.expr.typing import Constexpr

    def make_kernel():
        from flydsl.expr import const_expr as const_expr  # noqa: PLR0124

        @flyc.kernel
        def kernel(Out: fx.Tensor, n: fx.Int32, BLOCK_SIZE: Constexpr[int]):
            val = n
            if const_expr(True):
                val = val + fx.Int32(1)
            rsrc = buffer_ops.create_buffer_resource(Out)
            buffer_ops.buffer_store(val, rsrc, fx.Int32(0))

        @flyc.jit
        def launch(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            kernel(Out, n, BLOCK_SIZE=64).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

        return launch

    import torch

    launch = make_kernel()
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    launch(t_out, fx.Int32(10))
    torch.cuda.synchronize()
    assert out.item() == 11  # 10 + 1
