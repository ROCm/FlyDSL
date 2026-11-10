# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``ASTRewriter.transform`` rewrites a code object once and reuses it for every function built from it."""

import functools
import gc
import importlib.util
import os

import pytest

from flydsl.compiler.ast_rewriter import ASTRewriter

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _no_ast_diff(monkeypatch):
    # FLYDSL_DEBUG_AST_DIFF bypasses the cache.
    monkeypatch.delenv("FLYDSL_DEBUG_AST_DIFF", raising=False)


def _factory(scale):
    def scaled(x):
        return x * scale

    return scaled


def test_functions_from_one_factory_share_the_rewritten_code():
    double, triple = ASTRewriter.transform(_factory(2)), ASTRewriter.transform(_factory(3))

    assert double.__code__ is triple.__code__
    assert double.__code__ is not _factory(2).__code__
    assert (double(5), triple(5)) == (10, 15)


def _traced(fn):
    @functools.wraps(fn)
    def kernel(*args, **kwargs):
        return fn(*args, **kwargs)

    return kernel


def _make_add():
    @_traced
    def kernel(x):
        return x + 1

    return kernel


def _make_mul():
    @_traced
    def kernel(x):
        return x * 100

    return kernel


def test_wrappers_from_one_decorator_keep_their_own_body():
    # functools.wraps wrappers from one decorator share a code object. The inner def has the same name as the
    # functions it wraps, so the name check cannot tell them apart and the unwrap check must.
    add, mul = ASTRewriter.transform(_make_add()), ASTRewriter.transform(_make_mul())

    assert (add(3), mul(3)) == (4, 300)


def _nested_factory(offset):
    def outer(x):
        def inner(y):
            return y - 1000

        return inner(x) + offset + 2000

    return outer


def test_renamed_function_does_not_change_later_rewrites():
    # Named after its nested def, the function's rewrite picks the nested function out of the module.
    renamed = _nested_factory(1)
    renamed.__name__ = "inner"
    ASTRewriter.transform(renamed)

    assert ASTRewriter.transform(_nested_factory(5))(10) == 1015


def test_ast_diff_bypasses_the_cache(monkeypatch):
    # The diff is logged during the rewrite, so every function must be rewritten again to show it.
    monkeypatch.setenv("FLYDSL_DEBUG_AST_DIFF", "1")
    double, triple = ASTRewriter.transform(_factory(2)), ASTRewriter.transform(_factory(3))

    assert double.__code__ is not triple.__code__


def test_registering_a_transformer_clears_the_cache(monkeypatch):
    monkeypatch.setattr(ASTRewriter, "transformers", list(ASTRewriter.transformers))
    ASTRewriter.transform(_factory(2))
    assert ASTRewriter._rewrite_cache

    ASTRewriter.register(type("NoOp", (), {}))

    assert not ASTRewriter._rewrite_cache


def _load_factory_module(tmp_path, name):
    path = tmp_path / f"{name}.py"
    path.write_text("def factory():\n    def body(x):\n        return x + 1\n\n    return body\n")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_an_entry_is_dropped_when_its_code_is_freed(tmp_path):
    _, module = _load_factory_module(tmp_path, "freed_kernels")
    before = len(ASTRewriter._rewrite_cache)
    ASTRewriter.transform(module.factory())
    assert len(ASTRewriter._rewrite_cache) == before + 1

    del module
    gc.collect()

    assert len(ASTRewriter._rewrite_cache) == before


def test_edited_source_is_rewritten_again(tmp_path):
    # The rewrite reads the file on disk, as the JIT cache key does, so an edit must not reuse the old rewrite.
    path, module = _load_factory_module(tmp_path, "edited_kernels")
    assert ASTRewriter.transform(module.factory())(1) == 2

    path.write_text(path.read_text().replace("x + 1", "x - 1"))
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert ASTRewriter.transform(module.factory())(1) == 0
