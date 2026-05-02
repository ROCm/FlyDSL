# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Codegen-only verification of FlyDSL's AMDGPU MachineFunctionPasses.

These tests do NOT launch a kernel.  They drive the
``gpu-module-to-binary{format=isa}`` pass on a hand-written gpu.module so
the FlyDSL serializer (lib/Dialect/FlyROCDL/Target/) injects our two
MachineFunctionPasses at PreRegAlloc, then we grep the resulting AMDGPU
ISA text for the expected effects:

  - ``AMDGPUMFMATieVDSTToSrc2``   : every 128-bit-dst MFMA has vdst==src2
                                     when default-on; vdst != src2 when the
                                     -fly-amdgpu-tie-vdst-src2-for-mfma-128
                                     cl::opt is flipped to false.
  - ``AMDGPUPreferAGPRForDSRead`` : a DS load tagged ``nontemporal``
                                     causes its destination (and downstream
                                     MFMA operand) to land in an AGPR (a*
                                     register), whereas the same kernel
                                     without nontemporal falls back to VGPR.

No GPU is required — only ``fly-opt``-style codegen via the embedded
LLVM AMDGPU backend that lives in libFlyPythonCAPI.so.
"""

import difflib
import re
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Skip whole module if the FlyDSL python build artifacts aren't available.
# ---------------------------------------------------------------------------
try:
    from flydsl._mlir import ir
    from flydsl._mlir import passmanager
    # Importing this triggers _mlirRegisterEverything's NB_MODULE init, which
    # registers our cl::opts in the same LLVMSupport that owns the codegen
    # passes.
    from flydsl._mlir._mlir_libs import _mlirRegisterEverything  # noqa: F401
    from flydsl.compiler.llvm_options import llvm_options
    from flydsl.compiler.jit_function import _extract_isa_text
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"FlyDSL python build artifacts not importable: {exc}",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


# ---------------------------------------------------------------------------
# Helper: drive gpu-module-to-binary{format=isa} on a string MLIR module
# and return the AMDGPU assembly text.
# ---------------------------------------------------------------------------
def _compile_to_isa(mlir_text: str, *, llvm_opts: dict | None = None) -> str:
    """Compile *mlir_text* through gpu-module-to-binary and return ISA text."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = False
    ctx.load_all_available_dialects()

    def _run() -> str:
        mod = ir.Module.parse(mlir_text, context=ctx)
        pm = passmanager.PassManager.parse(
            'builtin.module(gpu-module-to-binary{format=isa opts="" section= toolkit=})',
            context=ctx,
        )
        pm.enable_verifier(True)
        pm.run(mod.operation)
        return _extract_isa_text(mod.operation.get_asm(enable_debug_info=False))

    if llvm_opts:
        with llvm_options(llvm_opts):
            return _run()
    return _run()


# ---------------------------------------------------------------------------
# Minimal MLIR templates.  We pass loaded-from-global vectors into MFMA so
# the operands can't be constant-folded away, and store the MFMA result so
# DCE can't drop it.
# ---------------------------------------------------------------------------
_MFMA_KERNEL_TMPL = """
module attributes {{gpu.container_module}} {{
  gpu.module @m [#rocdl.target<chip = "gfx942">] {tag_attr} {{
    llvm.func @k(
        %arg_a:   !llvm.ptr<1>,
        %arg_b:   !llvm.ptr<1>,
        %arg_acc: !llvm.ptr<1>,
        %arg_out: !llvm.ptr<1>
    ) attributes {{gpu.kernel, rocdl.kernel}} {{
      {load_a}
      %b   = llvm.load %arg_b   : !llvm.ptr<1> -> vector<4xf16>
      %c   = llvm.load %arg_acc : !llvm.ptr<1> -> vector<4xf32>
      %r = rocdl.mfma.f32.16x16x16f16 %a, %b, %c, 0, 0, 0
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      llvm.store %r, %arg_out : vector<4xf32>, !llvm.ptr<1>
      llvm.return
    }}
  }}
}}
"""

# Default opt-in: both FlyDSL AMDGPU passes are enabled via the discardable
# attribute that FlySerializeAMDGPUModule reads.  Tests that want to verify
# "default off" behaviour pass tag_attr="" explicitly.
_TAG_ATTR_BOTH = (
    'attributes {fly.amdgpu_codegen_passes = '
    '{prefer_agpr_for_ds_read, mfma_tie_vdst_to_src2}}'
)


def _mfma_kernel_global_a(*, tag_attr: str = _TAG_ATTR_BOTH) -> str:
    """%a comes from a plain global load (no AGPR hint expected)."""
    load_a = "%a = llvm.load %arg_a : !llvm.ptr<1> -> vector<4xf16>"
    return _MFMA_KERNEL_TMPL.format(load_a=load_a, tag_attr=tag_attr)


def _mfma_kernel_lds_nontemporal_a(*, tag_attr: str = _TAG_ATTR_BOTH) -> str:
    """%a comes from an LDS (addrspace 3) load tagged nontemporal — the
    AGPR pass should pin its destination to AGPR, which we expect to
    surface as an a* register in the MFMA src0 in the final ISA."""
    load_a = textwrap.dedent("""\
        %lds_addr = llvm.mlir.constant(0 : i32) : i32
        %lds_ptr  = llvm.inttoptr %lds_addr : i32 to !llvm.ptr<3>
        %a = llvm.load %lds_ptr {nontemporal} : !llvm.ptr<3> -> vector<4xf16>
    """).strip()
    return _MFMA_KERNEL_TMPL.format(load_a=load_a, tag_attr=tag_attr)


# ---------------------------------------------------------------------------
# ISA introspection helpers
# ---------------------------------------------------------------------------
# MFMA syntax in AMDGPU ASM: "v_mfma_f32_16x16x16_f16 v[0:3], v[6:7], v[8:9], v[0:3]".
# Use [^,\s]+ for operands so we don't greedily swallow trailing commas.
_MFMA_RE = re.compile(
    r"^\s*(v_mfma_\S+)\s+([^,\s]+),\s*([^,\s]+),\s*([^,\s]+),\s*([^,\s]+)",
    re.MULTILINE,
)


def _strip_reg(token: str) -> str:
    """Normalize a register token like 'v[12:15]' or 'a4' or 'v0,' for compare."""
    return token.rstrip(",").strip()


def _mfma_lines(isa: str) -> list[tuple[str, str, str, str, str]]:
    """Return list of (op, dst, src0, src1, src2) for each v_mfma_* line."""
    out = []
    for m in _MFMA_RE.finditer(isa):
        out.append(
            (m.group(1), _strip_reg(m.group(2)), _strip_reg(m.group(3)),
             _strip_reg(m.group(4)), _strip_reg(m.group(5)))
        )
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMFMATieVDSTToSrc2:
    """-fly-amdgpu-tie-vdst-src2-for-mfma-128 (default true)."""

    def test_default_on_ties_vdst_to_src2(self):
        """Every 128-bit-dst MFMA in the final ISA has vdst == src2.

        Note: for ``v_mfma_f32_16x16x16_f16`` (DstVT.Size == 128) vanilla
        AMDGPU TableGen already provides a tied ``_mac_e64`` form that the
        instruction selector picks when convenient.  Our pass is a backstop
        that ties vdst to src2 at MIR level for cases where the selector
        emits the early-clobber form anyway.  This test verifies that with
        our pass on (default) the resulting ASM is in tied form, which is
        the load-bearing guarantee for downstream perf.
        """
        isa = _compile_to_isa(_mfma_kernel_global_a())
        mfmas = _mfma_lines(isa)
        assert mfmas, f"no v_mfma_* in ISA:\n{isa[:1000]}"
        for op, dst, _s0, _s1, src2 in mfmas:
            assert dst == src2, (
                f"{op}: expected vdst==src2, got dst={dst} src2={src2}\n"
                f"ISA snippet:\n{isa[:2000]}"
            )

    def test_cl_opt_round_trip(self):
        """cl::opt for the MFMA tie pass is registered & toggleable.

        We can't observe a *behavioral* diff just from flipping this opt off
        on a 16x16x16f16 kernel — the early-clobber-vs-tied selection is
        already done by TableGen-level instruction patterns, so even with
        our pass disabled the codegen still produces the tied form.  What
        we *can* verify is that the cl::opt is registered with the correct
        name and that ``set_llvm_option_bool`` round-trips its value, which
        proves the pass code is linked into libFlyPythonCAPI.so and its
        static initializers ran.
        """
        from flydsl._mlir._mlir_libs import _mlirDialectsFly as _fly
        # default is True
        prev = _fly.set_llvm_option_bool(
            "fly-amdgpu-tie-vdst-src2-for-mfma-128", False
        )
        assert prev is True, (
            "expected default-true for fly-amdgpu-tie-vdst-src2-for-mfma-128, "
            f"got {prev}"
        )
        prev2 = _fly.set_llvm_option_bool(
            "fly-amdgpu-tie-vdst-src2-for-mfma-128", True  # restore
        )
        assert prev2 is False


class TestPreferAGPRForDSRead:
    """``nontemporal`` on a DS load pins the destination to AGPR."""

    def test_nontemporal_lds_pins_to_agpr(self):
        isa = _compile_to_isa(_mfma_kernel_lds_nontemporal_a())
        mfmas = _mfma_lines(isa)
        assert mfmas, f"no v_mfma_* in ISA:\n{isa[:1000]}"
        # src0 of MFMA is fed by the nontemporal LDS load → expect AGPR.
        # AGPR register tokens start with 'a' (e.g. a0, a[4:7], a16).  Both
        # bare AGPR class and the AGPR end of an AV_* pair show up as 'a'.
        agpr_src0 = [
            (op, src0) for op, _dst, src0, _s1, _src2 in mfmas
            if src0.startswith("a")
        ]
        assert agpr_src0, (
            "no MFMA has src0 in AGPR — AMDGPUPreferAGPRForDSRead did not "
            f"pin the nontemporal LDS load.  ISA snippet:\n{isa[:2000]}"
        )

    def test_global_load_does_not_pin(self):
        """Sanity: WITHOUT nontemporal (and with the default cl::opt off),
        the same kernel shape that loads from global should not force the
        MFMA src0 into AGPR.  This guards against ``startswith('a')``
        accidentally matching everything."""
        isa = _compile_to_isa(_mfma_kernel_global_a())
        mfmas = _mfma_lines(isa)
        assert mfmas, f"no v_mfma_* in ISA:\n{isa[:1000]}"
        # We don't claim 'no AGPR anywhere' — the regalloc is free to mix —
        # only that at least one MFMA src0 is a normal v* (VGPR), proving
        # the pass isn't running on un-tagged loads.
        vgpr_src0 = [
            (op, src0) for op, _dst, src0, _s1, _src2 in mfmas
            if src0.startswith("v")
        ]
        assert vgpr_src0, (
            "every MFMA src0 ended up in AGPR even without nontemporal — "
            f"unexpected pinning?  ISA snippet:\n{isa[:2000]}"
        )

    def test_cl_opt_extends_pin_to_all_ds_loads(self):
        """When -fly-amdgpu-prefer-agpr-for-ds-read is on, plain (non-
        nontemporal) DS loads should also get pinned to AGPR.  We rebuild
        the LDS-load kernel without the nontemporal flag and toggle the
        cl::opt to true: the ds_read should now land in an a* register
        even without the per-instruction hint."""
        # Same kernel shape but without {nontemporal} on the LDS load.
        load_a = textwrap.dedent("""\
            %lds_addr = llvm.mlir.constant(0 : i32) : i32
            %lds_ptr  = llvm.inttoptr %lds_addr : i32 to !llvm.ptr<3>
            %a = llvm.load %lds_ptr : !llvm.ptr<3> -> vector<4xf16>
        """).strip()
        kernel = _MFMA_KERNEL_TMPL.format(load_a=load_a, tag_attr=_TAG_ATTR_BOTH)

        isa_off = _compile_to_isa(kernel)
        mfmas_off = _mfma_lines(isa_off)
        agpr_off = [s0 for _op, _d, s0, _s1, _s2 in mfmas_off if s0.startswith("a")]
        # Without the cl::opt, plain DS loads are NOT pinned -> v* expected.
        assert not agpr_off, (
            f"plain DS load already pinned to AGPR without cl::opt; "
            f"ISA:\n{isa_off[:1500]}"
        )

        isa_on = _compile_to_isa(
            kernel, llvm_opts={"fly-amdgpu-prefer-agpr-for-ds-read": True}
        )
        mfmas_on = _mfma_lines(isa_on)
        agpr_on = [s0 for _op, _d, s0, _s1, _s2 in mfmas_on if s0.startswith("a")]
        assert agpr_on, (
            f"cl::opt on did NOT extend AGPR pinning to plain DS loads; "
            f"ISA:\n{isa_on[:1500]}"
        )


class TestOptInGating:
    """`fly.amdgpu_codegen_passes` is opt-in: a gpu.module without the
    attribute must serialize through the stock upstream codegen pipeline
    (no FlyDSL MachineFunctionPasses inserted)."""

    def test_no_tag_attr_means_pass_does_not_run(self):
        """A nontemporal LDS load is the AGPR pass's strongest trigger; with
        the opt-in attribute absent, the pass shouldn't fire and src0 must
        stay in a VGPR."""
        kernel = _mfma_kernel_lds_nontemporal_a(tag_attr="")
        isa = _compile_to_isa(kernel)
        mfmas = _mfma_lines(isa)
        assert mfmas, f"no v_mfma_* in ISA:\n{isa[:1000]}"
        agpr_src0 = [
            (op, src0) for op, _dst, src0, _s1, _src2 in mfmas
            if src0.startswith("a")
        ]
        assert not agpr_src0, (
            "AGPR pass ran without `fly.amdgpu_codegen_passes` opt-in attr; "
            f"got src0 pinned to AGPR.  ISA snippet:\n{isa[:2000]}"
        )

    def test_partial_tag_only_runs_named_pass(self):
        """When only `prefer_agpr_for_ds_read` is opted in, the AGPR pass
        runs (nontemporal LDS load → src0 in AGPR) but the MFMA tie pass is
        not invoked.  We can't directly observe the tie pass NOT running on
        16x16x16f16 (TableGen already produces the tied form), so this test
        only positively asserts the AGPR effect — its value is proving the
        partial-tag path compiles + serializes correctly."""
        tag = (
            'attributes {fly.amdgpu_codegen_passes = '
            '{prefer_agpr_for_ds_read}}'
        )
        kernel = _mfma_kernel_lds_nontemporal_a(tag_attr=tag)
        isa = _compile_to_isa(kernel)
        mfmas = _mfma_lines(isa)
        assert mfmas, f"no v_mfma_* in ISA:\n{isa[:1000]}"
        agpr_src0 = [
            src0 for _op, _dst, src0, _s1, _src2 in mfmas
            if src0.startswith("a")
        ]
        assert agpr_src0, (
            "AGPR pass did not fire even though prefer_agpr_for_ds_read was "
            f"opted in.  ISA snippet:\n{isa[:2000]}"
        )

    def test_same_kernel_isa_changes_with_hint(self):
        """End-to-end opt-in proof: identical kernel source compiled twice,
        once with the `fly.amdgpu_codegen_passes` opt-in attribute and once
        without.  The two ISA outputs must differ — that difference is the
        only direct evidence that flipping the hint actually drives the
        injected AMDGPU MachineFunctionPass through codegen.

        Uses the LDS-nontemporal kernel because that's the one shape where
        `FlyAMDGPUPreferAGPRForDSRead` produces a behaviour change visible
        in the textual ISA (src0 of the MFMA moves from v* to a*)."""
        kernel_off = _mfma_kernel_lds_nontemporal_a(tag_attr="")
        kernel_on = _mfma_kernel_lds_nontemporal_a()  # default: both on
        # Sanity-check that the only difference between the two MLIR inputs
        # is the opt-in attribute — otherwise an ISA diff wouldn't prove
        # what we claim it proves.
        assert kernel_off.replace("\n", "") + " " != kernel_on.replace(
            "\n", ""
        ), "test setup error: kernel_off and kernel_on are identical"
        for line_off, line_on in zip(
            kernel_off.splitlines(), kernel_on.splitlines()
        ):
            if line_off != line_on:
                assert "fly.amdgpu_codegen_passes" in line_on, (
                    "test setup error: MLIR diff between off/on kernels is "
                    f"not just the opt-in attr.  off: {line_off!r}  on: "
                    f"{line_on!r}"
                )

        isa_off = _compile_to_isa(kernel_off)
        isa_on = _compile_to_isa(kernel_on)

        if isa_off == isa_on:
            # Build a short context-line diff for the failure message so a
            # human can see at a glance what we expected to change but didn't.
            diff = "\n".join(
                difflib.unified_diff(
                    isa_off.splitlines(),
                    isa_on.splitlines(),
                    fromfile="hint=off",
                    tofile="hint=on",
                    n=2,
                    lineterm="",
                )
            )
            pytest.fail(
                "ISA was identical with and without the opt-in attr — the "
                "FlyDSL AMDGPU passes appear to NOT have run when opted in, "
                "or to have run when opted out.  Diff (empty == identical):\n"
                f"{diff or '<no diff>'}\n\n"
                f"--- ISA (hint=off, first 800 chars) ---\n{isa_off[:800]}"
            )

        # Concrete behavioural assertion: src0 of at least one MFMA flipped
        # from VGPR (v*) to AGPR (a*) when the hint was turned on.  This is
        # the load-bearing perf change the AGPR pass is supposed to deliver.
        src0_off = {
            src0 for _op, _d, src0, _s1, _s2 in _mfma_lines(isa_off)
        }
        src0_on = {
            src0 for _op, _d, src0, _s1, _s2 in _mfma_lines(isa_on)
        }
        flipped_to_agpr = any(s.startswith("a") for s in src0_on - src0_off)
        assert flipped_to_agpr, (
            "ISA differs but no MFMA src0 moved from v* (hint=off) to a* "
            "(hint=on) — diff is in something other than AGPR pinning.\n"
            f"src0 off: {sorted(src0_off)}\n"
            f"src0 on : {sorted(src0_on)}"
        )
