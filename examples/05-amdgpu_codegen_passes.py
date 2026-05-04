# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Standard usage of FlyDSL's AMDGPU codegen pass plugin.

Two MachineFunctionPasses live in `lib/Codegen/AMDGPU/`:
  * FlyAMDGPUMFMATieVDSTToSrc2     — ties vdst==src2 on 128-bit-dst MFMA
                                     at MIR level (backstop for cases the
                                     selector emits the early-clobber form).
  * FlyAMDGPUPreferAGPRForDSRead   — pins nontemporal ds_read destinations
                                     (and downstream MFMA src) to AGPR,
                                     freeing VGPRs for the rest of the kernel.

Both are opt-in per gpu.module via `compile_hints` on the `@flyc.jit`
launcher.  Two equivalent ways to apply them:

    # (A) Wrap once at the call site:
    flyc.compile[{
        "prefer_agpr_for_ds_read": True,
        "mfma_tie_vdst_to_src2":   True,
    }](my_launch, *args)

    # (B) Wrap once, reuse the hinted callable:
    hinted = flyc.compile[{
        "prefer_agpr_for_ds_read": True,
        "mfma_tie_vdst_to_src2":   True,
    }](my_launch)
    hinted(*args)

The kernel below is the smallest meaningful trigger for the AGPR pin pass:
two `vector.load_op(..., nontemporal=True)` from LDS feed an MFMA.  The
nontemporal flag lowers to `llvm.load {nontemporal}` → `ds_read` carrying
the `MachineMemOperand::isNonTemporal()` bit, which is the hint
`FlyAMDGPUPreferAGPRForDSRead` looks for.

Runs in COMPILE-ONLY mode (`COMPILE_ONLY=1` + `FLYDSL_DUMP_IR=1`): no GPU
launch — just trace, lower, dump ISA for both flavors, and print a
unified diff of the resulting `*final_isa.s`.
"""

# Set BEFORE any flydsl imports so env reader sees them.
import os
os.environ.setdefault("COMPILE_ONLY", "1")
os.environ.setdefault("FLYDSL_DUMP_IR", "1")

import difflib
import re
import sys
import tempfile
from pathlib import Path

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith, buffer_ops, rocdl, vector
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch


# ─── Kernel: nontemporal LDS load → MFMA → buffer_store ─────────────────
_arch = get_rocm_arch()
allocator = SmemAllocator(None, arch=_arch)


@flyc.kernel
def mfma_lds_nontemporal_kernel(C: fx.Tensor):
    base = allocator.get_base()
    # Pack A and B into a single LDS slot loaded as one nontemporal
    # vector<8xf16>, then split into two vector<4xf16> via vector.extract.
    # This avoids LLVM's load-merger combining two adjacent ds_read_b64 into
    # a single ds_read2_b64 (which the AGPR pin pass does not handle).
    lds_ab = SmemPtr(base, 0, T.f16, shape=(8,)).get()

    vec_f16x4 = T.vec(4, T.f16)
    vec_f16x8 = T.vec(8, T.f16)
    vec_f32x4 = T.vec(4, T.f32)
    zero_idx = arith.constant(0, index=True)

    # Single nontemporal LDS load → ds_read_b128 with NonTemporal MMO →
    # FlyAMDGPUPreferAGPRForDSRead pins the destination into AGPR.
    from flydsl._mlir.dialects import vector as _v
    ab_vec = vector.load_op(vec_f16x8, lds_ab, [zero_idx], nontemporal=True)
    a_vec = _v.ExtractStridedSliceOp(vec_f16x4, ab_vec, [0], [4], [1]).result
    b_vec = _v.ExtractStridedSliceOp(vec_f16x4, ab_vec, [4], [4], [1]).result

    # Constant-zero accumulator.  SIFoldOperands folds this to an inline
    # immediate src2 slot on the MFMA between PreRegAlloc and TwoAddress —
    # the MFMA-tie pass guards against that case (see
    # lib/Codegen/AMDGPU/MFMATieVDSTToSrc2.cpp), so this no longer crashes.
    acc = arith.constant_vector(0.0, vec_f32x4)

    out_vec = rocdl.mfma_f32_16x16x16f16(vec_f32x4, [a_vec, b_vec, acc, 0, 0, 0])

    # Anchor the MFMA result so DCE keeps the loads + MFMA alive.
    out_rsrc = buffer_ops.create_buffer_resource(C, max_size=False)
    buffer_ops.buffer_store(out_vec, out_rsrc, fx.Index(0))


@flyc.jit
def launch_bare(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator.finalize()
    mfma_lds_nontemporal_kernel(C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def launch_hinted(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    allocator.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        allocator.finalize()
    mfma_lds_nontemporal_kernel(C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


# Mutate launch_hinted's compile_hints in place.  We use a fresh JitFunction
# (not the bare one) so the hint does not leak into the bare compile.
flyc.compile[{
    "prefer_agpr_for_ds_read": True,
    "mfma_tie_vdst_to_src2":   True,
}](launch_hinted)


# ─── Compile-only driver ────────────────────────────────────────────────
def _compile_and_get_isa(jit_fn, *, label: str) -> tuple[str, Path]:
    dump_dir = Path(tempfile.mkdtemp(prefix=f"flydsl_demo_{label}_"))
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)

    C = torch.zeros(64, dtype=torch.float32)  # CPU is fine — compile-only
    jit_fn(C)  # COMPILE_ONLY=1 → returns before GPU launch

    isa_files = sorted(dump_dir.rglob("*_final_isa.s"))
    if not isa_files:
        sys.exit(f"[{label}] no *_final_isa.s under {dump_dir}")
    return isa_files[-1].read_text(), isa_files[-1]


bare_isa, bare_path     = _compile_and_get_isa(launch_bare,   label="bare")
hinted_isa, hinted_path = _compile_and_get_isa(launch_hinted, label="hinted")


# ─── Behavioural counts ─────────────────────────────────────────────────
def _count(isa: str) -> dict:
    return {
        "ds_read":           len(re.findall(r"\bds_read\w*", isa)),
        "ds_read_to_agpr":   len(re.findall(r"\bds_read\w*\s+a\[", isa))
                           + len(re.findall(r"\bds_read\w*\s+a\d", isa)),
        "v_mfma":            len(re.findall(r"\bv_mfma\w*", isa)),
        "mfma_src_in_agpr":  len(re.findall(r"\bv_mfma\S*\s+\S+,\s*a", isa)),
    }

bare_c, hinted_c = _count(bare_isa), _count(hinted_isa)
print(f"=== ISA counts (bare → hinted) ===")
print(f"  ds_read total       : {bare_c['ds_read']:>3} → {hinted_c['ds_read']:>3}")
print(f"  ds_read → AGPR (a*) : {bare_c['ds_read_to_agpr']:>3} → {hinted_c['ds_read_to_agpr']:>3}")
print(f"  v_mfma total        : {bare_c['v_mfma']:>3} → {hinted_c['v_mfma']:>3}")
print(f"  v_mfma src in AGPR  : {bare_c['mfma_src_in_agpr']:>3} → {hinted_c['mfma_src_in_agpr']:>3}")
print()


# ─── Diff (with cosmetic noise stripped) ────────────────────────────────
_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"\.amdhsa_next_free_(?:vgpr|sgpr|agpr)|"
    r"\.amdhsa_accum_offset|\.amdhsa_kernarg_size|"
    r"\.amdhsa_reserve_\w+|\.amdhsa_user_sgpr_\w+|\.amdhsa_system_\w+|"
    r"\.amdhsa_(?:enable|exception|float|fp16|tg_split|dx10|ieee|uses)_\w+|"
    r"\.size\s+|\.set\s+\.L|\.section\b|\.text\b|\.p2align\b)",
    re.IGNORECASE,
)
def _denoise(s): return [ln for ln in s.splitlines() if not _NOISE_RE.match(ln)]

print(f"=== ISA diff ({bare_path.name}) ===")
diff = list(difflib.unified_diff(
    _denoise(bare_isa), _denoise(hinted_isa),
    fromfile=f"bare/{bare_path.name}", tofile=f"hinted/{hinted_path.name}",
    lineterm="",
))
if not diff:
    print("(no diff)")
else:
    for line in diff:
        print(line)
