"""
Tests for flyc.jit shmem support — verifying the new compilation path
that auto-detects mori_shmem_* extern calls and links bitcode.

Tests are organised in three tiers:

  1. Unit tests (no GPU, no mori): exercise CompilationContext.extern_symbols
     tracking, _pipeline_fragments link_libs generation, and CompiledArtifact
     serialization of needs_shmem.

  2. Compilation test (single GPU, needs mori): verify the full pipeline
     compiles a @flyc.jit kernel that uses ExternFunction with shmem calls.
     Uses COMPILE_ONLY=1 so no actual kernel launch happens.

Run:
  pytest tests/test_shmem_jit_support.py -v
"""
import pytest


# ============================================================
# Tier 1 — Unit tests (no GPU, no mori required)
# ============================================================

class TestPipelineFragmentsLinkLibs:
    """Verify _pipeline_fragments generates correct rocdl-attach-target options."""

    def test_no_link_libs(self):
        from flydsl.compiler.jit_function import MlirCompiler
        frags = MlirCompiler._pipeline_fragments(chip="gfx942")
        rocdl_frag = [f for f in frags if "rocdl-attach-target" in f][0]
        # Should NOT contain any l= option
        assert " l=" not in rocdl_frag
        # Should end with wave64=true}}
        assert rocdl_frag.rstrip().endswith("wave64=true}")

    def test_single_link_lib(self):
        from flydsl.compiler.jit_function import MlirCompiler
        frags = MlirCompiler._pipeline_fragments(
            chip="gfx942", link_libs=["/path/to/lib.bc"]
        )
        rocdl_frag = [f for f in frags if "rocdl-attach-target" in f][0]
        assert "l=/path/to/lib.bc" in rocdl_frag

    def test_multiple_link_libs(self):
        from flydsl.compiler.jit_function import MlirCompiler
        frags = MlirCompiler._pipeline_fragments(
            chip="gfx942", link_libs=["/a.bc", "/b.bc"]
        )
        rocdl_frag = [f for f in frags if "rocdl-attach-target" in f][0]
        assert "l=/a.bc" in rocdl_frag
        assert "l=/b.bc" in rocdl_frag

    def test_empty_link_libs_same_as_none(self):
        from flydsl.compiler.jit_function import MlirCompiler
        frags_none = MlirCompiler._pipeline_fragments(chip="gfx942", link_libs=None)
        frags_empty = MlirCompiler._pipeline_fragments(chip="gfx942", link_libs=[])
        rocdl_none = [f for f in frags_none if "rocdl-attach-target" in f][0]
        rocdl_empty = [f for f in frags_empty if "rocdl-attach-target" in f][0]
        assert rocdl_none == rocdl_empty


class TestCompilationContextExternSymbols:
    """Verify CompilationContext tracks extern symbols."""

    def test_extern_symbols_initially_empty(self):
        from flydsl.compiler.kernel_function import CompilationContext
        with CompilationContext.create() as ctx:
            assert ctx.extern_symbols == set()

    def test_extern_symbols_add(self):
        from flydsl.compiler.kernel_function import CompilationContext
        with CompilationContext.create() as ctx:
            ctx.extern_symbols.add("mori_shmem_my_pe")
            ctx.extern_symbols.add("mori_shmem_ptr_p2p")
            assert "mori_shmem_my_pe" in ctx.extern_symbols
            assert len(ctx.extern_symbols) == 2

    def test_extern_symbols_shmem_detection(self):
        from flydsl.compiler.kernel_function import CompilationContext
        with CompilationContext.create() as ctx:
            ctx.extern_symbols.add("mori_shmem_my_pe")
            ctx.extern_symbols.add("some_other_func")
            has_shmem = any(
                s.startswith("mori_shmem_") for s in ctx.extern_symbols
            )
            assert has_shmem

    def test_no_shmem_symbols(self):
        from flydsl.compiler.kernel_function import CompilationContext
        with CompilationContext.create() as ctx:
            ctx.extern_symbols.add("some_other_func")
            has_shmem = any(
                s.startswith("mori_shmem_") for s in ctx.extern_symbols
            )
            assert not has_shmem

    def test_get_current_returns_active_context(self):
        from flydsl.compiler.kernel_function import CompilationContext
        assert CompilationContext.get_current() is None
        with CompilationContext.create() as ctx:
            assert CompilationContext.get_current() is ctx
        assert CompilationContext.get_current() is None


class TestExternFunctionRegistersSymbol:
    """Verify ExternFunction._ensure_declared registers into CompilationContext."""

    def test_symbol_registered_on_declare(self):
        from flydsl._mlir import ir
        from flydsl._mlir.dialects import gpu
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl.expr.extern import ExternFunction

        ext_fn = ExternFunction(
            symbol="mori_shmem_my_pe",
            arg_types=[],
            ret_type="int32",
        )

        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    gpu_mod = gpu.GPUModuleOp("test_gpu", targets=None)
                    gpu_mod.regions[0].blocks.append()
                    gpu_body = gpu_mod.regions[0].blocks[0]

                    with CompilationContext.create() as comp_ctx:
                        comp_ctx.gpu_module_body = gpu_body
                        ext_fn._ensure_declared(gpu_body)
                        assert "mori_shmem_my_pe" in comp_ctx.extern_symbols


class TestCompiledArtifactShmemFlag:
    """Verify CompiledArtifact serialization of needs_shmem."""

    def test_default_needs_shmem_false(self):
        from flydsl.compiler.jit_executor import CompiledArtifact
        # We can't easily create a real CompiledArtifact without a compiled module,
        # but we can test serialization logic directly.
        state = {
            "ir_text": "module {}",
            "entry": "test",
            "source_ir": None,
            "needs_shmem": False,
        }
        art = CompiledArtifact.__new__(CompiledArtifact)
        art.__setstate__(state)
        assert art._needs_shmem is False

    def test_needs_shmem_true_roundtrip(self):
        from flydsl.compiler.jit_executor import CompiledArtifact
        state = {
            "ir_text": "module {}",
            "entry": "test",
            "source_ir": None,
            "needs_shmem": True,
        }
        art = CompiledArtifact.__new__(CompiledArtifact)
        art.__setstate__(state)
        assert art._needs_shmem is True
        out_state = art.__getstate__()
        assert out_state["needs_shmem"] is True

    def test_backward_compat_missing_needs_shmem(self):
        """Old pickled artifacts won't have needs_shmem; should default to False."""
        from flydsl.compiler.jit_executor import CompiledArtifact
        state = {
            "ir_text": "module {}",
            "entry": "test",
            "source_ir": None,
            # no needs_shmem key
        }
        art = CompiledArtifact.__new__(CompiledArtifact)
        art.__setstate__(state)
        assert art._needs_shmem is False


class TestModuleLoadHookSymbol:
    """Verify that libfly_jit_runtime.so exports mgpuSetModuleLoadHook."""

    def test_hook_symbol_exported(self):
        import ctypes
        from flydsl.compiler.jit_executor import _resolve_runtime_libs
        lib_path = _resolve_runtime_libs()[0]
        lib = ctypes.CDLL(lib_path)
        # Should not raise — symbol must exist
        fn = lib.mgpuSetModuleLoadHook
        assert fn is not None


# ============================================================
# Tier 2 — Compilation test (needs GPU + mori)
# ============================================================

def _has_mori():
    try:
        from mori.ir.bitcode import find_bitcode
        from flydsl.compiler.jit_function import _FLYDSL_COV
        find_bitcode(cov=_FLYDSL_COV)
        return True
    except Exception:
        return False


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _has_gpu() or not _has_mori(),
                    reason="Needs GPU + mori with shmem bitcode")
class TestShmemJitCompilation:
    """Full pipeline compilation test with ExternFunction + shmem bitcode.

    Uses COMPILE_ONLY=1 so no kernel launch / shmem init required.
    """

    def test_auto_detection_passes_link_libs(self, monkeypatch):
        """Verify that JitFunction auto-detects shmem externs and resolves bitcode.

        This test traces a @flyc.jit kernel with mori_shmem_* ExternFunction,
        verifies that:
        1. CompilationContext.extern_symbols contains the shmem symbol
        2. MlirCompiler.compile is called with link_libs pointing to the bitcode
        """
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.compiler.jit_function import MlirCompiler
        from flydsl.expr.extern import ExternFunction

        shmem_my_pe = ExternFunction(
            symbol="mori_shmem_my_pe",
            arg_types=[],
            ret_type="int32",
        )

        @flyc.kernel
        def trivial_shmem_kernel(out: fx.Tensor):
            tid = fx.thread_idx.x
            pe = shmem_my_pe()

        @flyc.jit
        def launch(out: fx.Tensor):
            trivial_shmem_kernel(out).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        class _CompileCaptureDone(Exception):
            pass

        captured = {}

        @classmethod
        def mock_compile(cls, module, *, chip=None, func_name="", link_libs=None):
            captured["link_libs"] = link_libs
            captured["chip"] = chip
            raise _CompileCaptureDone()

        monkeypatch.setattr(MlirCompiler, "compile", mock_compile)

        import torch
        out = torch.zeros(64, dtype=torch.int32, device="cuda")

        with pytest.raises(_CompileCaptureDone):
            launch(out)

        # Verify link_libs was set to the mori bitcode path
        assert captured["link_libs"] is not None
        assert len(captured["link_libs"]) == 1
        assert captured["link_libs"][0].endswith(".bc")

    def test_no_shmem_kernel_no_link_libs(self, monkeypatch):
        """Verify that non-shmem kernels do NOT get link_libs."""
        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.compiler.jit_function import MlirCompiler

        @flyc.kernel
        def plain_kernel(out: fx.Tensor):
            tid = fx.thread_idx.x

        @flyc.jit
        def launch_plain(out: fx.Tensor):
            plain_kernel(out).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        class _CompileCaptureDone(Exception):
            pass

        captured = {}

        @classmethod
        def mock_compile(cls, module, *, chip=None, func_name="", link_libs=None):
            captured["link_libs"] = link_libs
            raise _CompileCaptureDone()

        monkeypatch.setattr(MlirCompiler, "compile", mock_compile)

        import torch
        out = torch.zeros(64, dtype=torch.int32, device="cuda")

        with pytest.raises(_CompileCaptureDone):
            launch_plain(out)

        # Non-shmem kernel should have link_libs=None
        assert captured["link_libs"] is None

    def test_full_compile_with_shmem_extern(self, monkeypatch):
        """End-to-end: compile a @flyc.jit kernel with mori_shmem calls.

        Runs the real MLIR pipeline (bitcode linking + gpu-module-to-binary)
        with COMPILE_ONLY=1, verifying the ABI version match.
        """
        monkeypatch.setenv("COMPILE_ONLY", "1")

        from flydsl.compiler.jit_function import _FLYDSL_COV

        # Clear any cached bitcode paths from earlier tests
        import mori.ir.bitcode as _bc_mod
        _bc_mod._cached_paths.clear()

        from mori.ir.bitcode import find_bitcode
        bc_path = find_bitcode(cov=_FLYDSL_COV)
        assert f"_cov{_FLYDSL_COV}" in bc_path, f"Expected cov{_FLYDSL_COV} cache dir, got: {bc_path}"
        print(f"\n  bitcode path: {bc_path}")
        _bc_mod._cached_paths.clear()  # clear again so JitFunction resolves fresh

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr.extern import ExternFunction

        shmem_my_pe = ExternFunction(
            symbol="mori_shmem_my_pe",
            arg_types=[],
            ret_type="int32",
        )

        @flyc.kernel
        def shmem_kernel(out: fx.Tensor):
            tid = fx.thread_idx.x
            pe = shmem_my_pe()

        @flyc.jit
        def launch_shmem(out: fx.Tensor):
            shmem_kernel(out).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        import torch
        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        # COMPILE_ONLY=1: returns None on success, raises on failure
        result = launch_shmem(out)
        assert result is None

    def test_compile_allreduce_multi_extern(self, monkeypatch):
        """Compile a kernel using multiple mori shmem externs (allreduce pattern).

        Verifies that the JIT pipeline handles kernels calling several different
        mori_shmem_* functions (my_pe, ptr_p2p, quiet_thread) through a single
        bitcode link.  Uses COMPILE_ONLY=1.
        """
        monkeypatch.setenv("COMPILE_ONLY", "1")

        import mori.ir.bitcode as _bc_mod
        _bc_mod._cached_paths.clear()

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        from flydsl.expr.extern import ExternFunction

        shmem_my_pe = ExternFunction(
            symbol="mori_shmem_my_pe", arg_types=[], ret_type="int32",
        )
        shmem_ptr_p2p = ExternFunction(
            symbol="mori_shmem_ptr_p2p",
            arg_types=["uint64", "int32", "int32"],
            ret_type="uint64",
        )
        shmem_quiet = ExternFunction(
            symbol="mori_shmem_quiet_thread",
            arg_types=[], ret_type="int32",
        )

        @flyc.kernel
        def allreduce_kernel(data: fx.Int64, result: fx.Int64, n: fx.Int32):
            tid = fx.thread_idx.x
            pe = shmem_my_pe()
            # Simulate P2P address resolution
            remote = shmem_ptr_p2p(data, pe, tid)
            shmem_quiet()

        @flyc.jit
        def launch_allreduce(data: fx.Int64, result: fx.Int64, n: fx.Int32):
            allreduce_kernel(data, result, n).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        import torch
        buf = torch.zeros(64, dtype=torch.int32, device="cuda")
        result = launch_allreduce(fx.Int64(buf.data_ptr()), fx.Int64(buf.data_ptr()), 64)
        assert result is None


# ============================================================
# Tier 3 — Regression tests (GPU, no mori required)
# ============================================================

@pytest.mark.skipif(not _has_gpu(), reason="Needs GPU")
class TestNonShmemKernelRegression:
    """Verify that non-shmem kernels still compile and run correctly.

    These tests ensure the shmem JIT support changes did not break
    normal kernel compilation and execution.
    """

    def test_simple_elementwise_compile_and_run(self):
        """A simple kernel that writes thread_idx to output — compile + execute."""
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.kernel
        def write_tid_kernel(out: fx.Tensor):
            tid = fx.thread_idx.x
            # out[tid] = tid  (via memref_store with scalar index)
            fx.memref_store(tid, out, tid)

        @flyc.jit
        def launch_write_tid(out: fx.Tensor):
            write_tid_kernel(out).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        import torch
        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        launch_write_tid(out)

        expected = torch.arange(64, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(out, expected)

    def test_simple_elementwise_compile_only(self, monkeypatch):
        """Compile-only: verify pipeline completes without link_libs."""
        monkeypatch.setenv("COMPILE_ONLY", "1")

        import flydsl.compiler as flyc
        import flydsl.expr as fx

        @flyc.kernel
        def noop_kernel(out: fx.Tensor):
            tid = fx.thread_idx.x

        @flyc.jit
        def launch_noop(out: fx.Tensor):
            noop_kernel(out).launch(
                grid=(1, 1, 1), block=(64, 1, 1),
            )

        import torch
        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        result = launch_noop(out)
        assert result is None

    def test_multi_block_kernel(self):
        """Multi-block kernel with block_idx — no shmem, compile + execute."""
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        BLOCK = 64

        @flyc.kernel
        def multi_block_kernel(out: fx.Tensor):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
            # out[bid * BLOCK + tid] = bid * BLOCK + tid
            global_tid = bid * BLOCK + tid
            fx.memref_store(global_tid, out, global_tid)

        @flyc.jit
        def launch_multi_block(out: fx.Tensor):
            multi_block_kernel(out).launch(
                grid=(4, 1, 1), block=(BLOCK, 1, 1),
            )

        import torch
        N = BLOCK * 4
        out = torch.zeros(N, dtype=torch.int32, device="cuda")
        launch_multi_block(out)

        expected = torch.arange(N, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(out, expected)
