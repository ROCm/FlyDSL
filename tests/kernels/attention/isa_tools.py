# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-only access to what the compiler produced: the MLIR stages, the LLVM IR, the final ISA and the code object's
metadata (kernarg list, denorm mode, resources), plus the scanners the hazard tests run over them.

A dump is made by compiling a **fresh** builder (never a memoised one: the in-process JIT cache would skip the dump) with
`FLYDSL_DUMP_IR` pointing at a temp directory. Nothing is launched.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Dump:
    path: Path

    def _read(self, suffix):
        files = sorted(self.path.glob(f"*{suffix}"))
        assert files, f"no *{suffix} in {self.path}"
        return files[-1].read_text()

    @property
    def isa(self):
        return self._read("final_isa.s")

    @property
    def llvm_ir(self):
        return self._read("llvm_ir.ll")

    def mlir(self, stage_substr):
        files = sorted(self.path.glob(f"*{stage_substr}*.mlir"))
        assert files, f"no MLIR stage matching {stage_substr!r} in {self.path}"
        return files[-1].read_text()

    @property
    def metadata(self):
        """The `.amdgpu_metadata` block of the final ISA, parsed: `amdhsa.kernels[0]` as a dict."""
        text = self.isa
        block = text.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0]
        block = block.split("\n...", 1)[0]  # the YAML document end; a tab-indented tail follows
        return yaml.safe_load(block)["amdhsa.kernels"][0]

    @property
    def kernel_args(self):
        """`[(kind, size)]` of the explicit (non-hidden) kernel arguments, in order."""
        return [
            (a[".value_kind"], a[".size"]) for a in self.metadata[".args"] if not a[".value_kind"].startswith("hidden_")
        ]

    @property
    def hidden_args(self):
        return [a[".value_kind"] for a in self.metadata[".args"] if a[".value_kind"].startswith("hidden_")]

    @property
    def denorm_mode_32(self):
        return int(re.search(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", self.isa).group(1))

    def resources(self):
        m = self.metadata
        return {
            k.lstrip("."): m[k]
            for k in (
                ".vgpr_count",
                ".sgpr_count",
                ".vgpr_spill_count",
                ".sgpr_spill_count",
                ".private_segment_fixed_size",
                ".group_segment_fixed_size",
                ".kernarg_segment_size",
            )
        }


def compile_dump(compile_fn, tmp_path, monkeypatch):
    """Run `compile_fn()` (which must compile a fresh builder) with the IR dump on; return the `Dump` of the kernel it made."""
    out = Path(tmp_path)
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(out))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    compile_fn()
    dirs = sorted(p for p in out.iterdir() if p.is_dir())
    assert dirs, f"the compile wrote no dump under {out}"
    return Dump(dirs[-1])


# ---------------------------------------------------------------------------
# ISA scanners
# ---------------------------------------------------------------------------

_VREG = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def _regs(token):
    token = token.strip()
    m = re.match(r"v\[(\d+):(\d+)\]", token)
    if m:
        return set(range(int(m[1]), int(m[2]) + 1))
    m = re.match(r"v(\d+)$", token)
    return {int(m[1])} if m else set()


def instructions(isa):
    """`[(mnemonic, [operand tokens])]` of the real instructions of an assembly listing, in order. The mnemonic drops its
    `_e32`/`_e64` encoding suffix: the listing spells `v_exp_f32_e32`, and a scanner matching `v_exp_f32` saw nothing.
    """
    out = []
    for line in isa.split("\n"):
        s = line.split(";")[0].strip()
        if not s or s.startswith(".") or s.endswith(":") or s.startswith("//"):
            continue
        parts = s.split(None, 1)
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        out.append((re.sub(r"_e(32|64)$", "", parts[0]), ops))
    return out


def is_valu(op):
    return op.startswith("v_") and not op.startswith(("v_mfma", "v_smfmac", "v_accvgpr", "v_readfirstlane"))


def scan_exp2_wait_state(isa):
    """K44: every `v_exp_f32 vN` must be followed by at least one instruction that is not a VALU reading `vN`
    (`v_exp_f32` is quarter-rate and needs one wait state before a VALU consumer; `GCNHazardRecognizer` does not model
    it on gfx950). Returns the zero-gap sites."""
    ins = instructions(isa)
    bad = []
    for i, (op, ops) in enumerate(ins[:-1]):
        if op != "v_exp_f32":
            continue
        dst = _regs(ops[0])
        nxt_op, nxt_ops = ins[i + 1]
        if is_valu(nxt_op) and any(dst & _regs(o) for o in nxt_ops[1:]):
            bad.append((i, f"{op} {', '.join(ops)} ; {nxt_op} {', '.join(nxt_ops)}"))
    return bad


_FMA_OPS = ("v_fma_", "v_pk_fma_", "v_fmac_", "v_mad_", "v_mac_")
_PASS_THROUGH_OPS = ("v_cndmask_b32", "v_mov_b32", "v_mov_b64")


def _vgprs(token):
    return _regs(token.strip("-|"))


def _writer(ins, i, regs):
    """Index of the nearest VALU above `i` that writes any of `regs`, wrapping past the top of the listing to its end: a
    value carried around a loop is written below its reader."""
    for k in range(1, len(ins)):
        j = (i - k) % len(ins)
        op, ops = ins[j]
        if op.startswith("v_") and ops and _regs(ops[0]) & regs:
            return j
    return None


def scan_exp2_fed_by_fma(isa):
    """Aotriton issue 54: the `v_exp_f32` sites whose operand comes from an FMA, i.e. `exp2(fma(s, c, -m))`, an unrounded
    product against a rounded max. The operand is traced to its writer through masks and moves (`v_cndmask`, `v_mov`).
    Returns the sites."""
    ins = instructions(isa)
    bad = []
    for i, (op, ops) in enumerate(ins):
        if op != "v_exp_f32":
            continue
        todo, seen = [(i, _vgprs(ops[1]))], set()
        while todo:
            at, regs = todo.pop()
            j = _writer(ins, at, regs)
            if j is None or j in seen:
                continue
            seen.add(j)
            w_op, w_ops = ins[j]
            if w_op.startswith(_FMA_OPS):
                bad.append((i, f"{w_op} {', '.join(w_ops)} ; {op} {', '.join(ops)}"))
                break
            if w_op.startswith(_PASS_THROUGH_OPS):
                todo += [(j, r) for r in map(_vgprs, w_ops[1:]) if r]
    return bad


def scan_ds_read_tr_waits(isa):
    """K42: the result of every `ds_read_b64_tr_b16` must be waited on (`s_waitcnt lgkmcnt(k)` retiring it) before its first
    reader, `v_accvgpr_write` copies included. Returns `(n_reads, unwaited_uses)`."""
    outstanding, bad, n = [], 0, 0
    for op, ops in instructions(isa):
        if op == "s_waitcnt":
            m = re.search(r"lgkmcnt\((\d+)\)", " ".join(ops))
            if m:
                k = int(m.group(1))
                outstanding = (
                    outstanding[len(outstanding) - k :]
                    if (k and len(outstanding) > k)
                    else ([] if k == 0 else outstanding)
                )
            continue
        if op.startswith(("ds_read", "ds_load")):
            outstanding.append(_regs(ops[0]))
            n += op == "ds_read_b64_tr_b16"
            continue
        srcs = set()
        for o in ops[1:]:
            srcs |= _regs(o.split()[0])
        bad += sum(1 for o in outstanding if o & srcs)
    return n, bad


def scan_cvt_pk_to_mfma(isa, min_wait_states=2):
    """K45: a `v_cvt_pk_bf16_f32 vM` whose result feeds a `v_mfma` SrcA/SrcB needs `min_wait_states` wait states between
    them (an `s_nop N` counts N + 1; each independent VALU counts 1; scalar instructions count 0). Restricted to this
    producer: the general VALU-write-then-MFMA rule has about 7500 benign sites. Returns the sites with fewer."""
    ins = instructions(isa)
    bad = []
    for i, (op, ops) in enumerate(ins):
        if op != "v_cvt_pk_bf16_f32":
            continue
        dst = _regs(ops[0])
        gap = 0
        for nxt_op, nxt_ops in ins[i + 1 :]:
            if nxt_op.startswith("v_mfma") and len(nxt_ops) >= 3 and (dst & (_regs(nxt_ops[1]) | _regs(nxt_ops[2]))):
                if gap < min_wait_states:
                    bad.append((i, gap))
                break
            if nxt_op == "s_nop":
                gap += int(nxt_ops[0]) + 1
            elif is_valu(nxt_op) or nxt_op.startswith("v_mfma"):
                if dst & set().union(*[_regs(o) for o in nxt_ops[1:]]) if nxt_ops[1:] else False:
                    break
                gap += 1
            if gap >= min_wait_states + 8:
                break
    return bad


def inline_asm_strings(mlir_text):
    """`[(asm_string, constraints)]` of every `llvm.inline_asm` in an LLVM-dialect MLIR dump."""
    out = []
    for m in re.finditer(r'llvm\.inline_asm[^"]*"((?:[^"\\]|\\.)*)",\s*"((?:[^"\\]|\\.)*)"', mlir_text):
        out.append((m.group(1), m.group(2)))
    return out


def isa_stats(isa):
    """Instruction-mix counts the structural tests compare."""
    counts = {}
    for op, _ in instructions(isa):
        counts[op] = counts.get(op, 0) + 1
    return counts


def env_flag(name, default=""):
    return os.environ.get(name, default)


def fresh_fwd_dump(backend, arch, meta, tmp_path, monkeypatch, window=None, **pins):
    """Compile a **fresh** forward for `(meta, pins)` with the dump on and return its `Dump` (no launch)."""
    from tests.kernels.attention.attn_testlib import compile_inputs, lse_alloc

    knobs = backend.fwd_knobs(arch, **pins).resolve(meta)
    fn = backend.build_fwd(meta, knobs)
    args, kw = compile_inputs(meta)
    if window is not None:
        kw["window"] = window
    if knobs.RETURN_LSE == "always":
        kw["lse"] = lse_alloc(1, 2, 64)  # this mode requires an LSE tensor
    return compile_dump(lambda: fn.compile(*args, **kw), tmp_path, monkeypatch)
