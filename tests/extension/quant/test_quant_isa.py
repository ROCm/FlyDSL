#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cross-architecture compilation and ISA verification for ``fx.quant``.

Compiles a representative ``fx.quant`` kernel with ``COMPILE_ONLY=1`` across
CDNA3 (``gfx942``), CDNA4 (``gfx950``), RDNA3 (``gfx1100``), RDNA4
(``gfx1201``), and CDNA5 (``gfx1250``) and inspects the emitted AMDGPU assembly:

- ``gfx942``: FNUZ hardware FP8 must NOT be emitted for OCP ``E4M3FN``/``E5M2``;
  ``v_perm_b32`` is emitted for ``E2M1`` table lookups.
- ``gfx950``: emits native OCP ``FP8`` (``v_cvt_pk_fp8_f32``,
  ``v_cvt_pk_f32_fp8``, ``v_cvt_sr_fp8_f32``) AND native ``FP4``
  (``v_cvt_scalef32_pk_fp4_f32``, ``v_cvt_scalef32_pk_f32_fp4``,
  ``v_cvt_scalef32_pk_bf16_fp4``).
- ``gfx1100``: compiles cleanly without hardware ``FP8`` or ``FP4`` instructions
  and uses ``v_perm_b32`` for ``E2M1``.
- ``gfx1201`` and ``gfx1250``: emit native OCP ``FP8`` pack/stochastic-round and
  True16-safe scalar byte-indexed ``v_cvt_f32_fp8`` (avoiding missing ``_t16``
  ISel patterns on ``v_cvt_pk_f32_fp8`` until ``llvm/llvm-project#231721`` /
  ``ROCm/FlyDSL#1376``), plus ``v_perm_b32`` for ``E2M1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_DRIVER = """
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel(known_block_size=[32, 1, 1])
def quant_isa_probe(In: fx.Tensor, Out: fx.Tensor):
    tid = fx.thread_idx.x
    v0 = In[tid * 8 + 0]
    v1 = In[tid * 8 + 1]
    v2 = In[tid * 8 + 2]
    v3 = In[tid * 8 + 3]
    v4 = In[tid * 8 + 4]
    v5 = In[tid * 8 + 5]
    v6 = In[tid * 8 + 6]
    v7 = In[tid * 8 + 7]

    fp8_pk = fx.quant.pack_e4m3x4(v0, v1, v2, v3)
    u0, u1, u2, u3 = fx.quant.unpack_e4m3x4(fp8_pk)
    sr_pk = fx.quant.stoch_round_e4m3x4(u0, u1, u2, u3, fx.Uint32(tid))

    fp4_pk = fx.quant.pack_e2m1x8((v0, v1, v2, v3, v4, v5, v6, v7))
    d0, d1, d2, d3 = fx.quant.unpack_e2m1x4(fp4_pk, half=0)
    bf01, bf23 = fx.quant.unpack_e2m1x4_bf16x2(fp4_pk, half=1)
    fp8_from_fp4 = fx.quant.unpack_e2m1x4_e4m3x4(fp4_pk, half=0)
    p_i4, q_i4 = fx.quant.unpack_e2m1x8_int4_pair(fp4_pk)

    acc = (
        sr_pk
        ^ bf01
        ^ bf23
        ^ fp8_from_fp4
        ^ p_i4
        ^ q_i4
        ^ (d0 + d1 + d2 + d3).bitcast(fx.Uint32)
    )
    Out[tid] = acc.bitcast(fx.Int32)


@flyc.jit
def compile_probe(In: fx.Tensor, Out: fx.Tensor):
    quant_isa_probe(In, Out).launch(grid=(1, 1, 1), block=(32, 1, 1))


inp = torch.zeros(256, dtype=torch.float32, device="meta")
out = torch.zeros(32, dtype=torch.int32, device="meta")
compile_probe(inp, out)
"""


def _compile_arch_asm(arch: str, tmp_path: Path) -> str:
    dump_dir = tmp_path / f"dump_{arch}"
    cache_dir = tmp_path / f"cache_{arch}"
    script_path = tmp_path / f"probe_{arch}.py"
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_DRIVER, encoding="utf-8")

    env = os.environ.copy()
    env["ARCH"] = arch
    env["COMPILE_ONLY"] = "1"
    env["FLYDSL_DUMP_IR"] = "1"
    env["FLYDSL_DUMP_DIR"] = str(dump_dir)
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache_dir)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"cross-compile failed for {arch}:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    asm_files = list(dump_dir.glob("*/*_final_isa.s")) or list(dump_dir.rglob("*.s"))
    assert asm_files, f"no .s assembly dumped for {arch} in {dump_dir}"
    return "\n".join(p.read_text(encoding="utf-8") for p in asm_files)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250"])
def test_quant_cross_arch_isa(arch: str, tmp_path: Path):
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("requires torch for meta tensor tracing")

    asm = _compile_arch_asm(arch, tmp_path)
    assert "_fake16" not in asm
    assert "v_perm_b32" in asm

    if arch in ("gfx942", "gfx1100"):
        assert "v_cvt_pk_fp8_f32" not in asm
        assert "v_cvt_f32_fp8" not in asm
        assert "v_cvt_pk_f32_fp8" not in asm
        assert "v_cvt_sr_fp8_f32" not in asm
        assert "v_cvt_scalef32_" not in asm
    elif arch == "gfx950":
        assert "v_cvt_pk_fp8_f32" in asm
        assert "v_cvt_pk_f32_fp8" in asm
        assert "v_cvt_sr_fp8_f32" in asm
        assert "v_cvt_scalef32_pk_fp4_f32" in asm
        assert "v_cvt_scalef32_pk_f32_fp4" in asm
        assert "v_cvt_scalef32_pk_bf16_fp4" in asm
    elif arch in ("gfx1201", "gfx1250"):
        assert "v_cvt_pk_fp8_f32" in asm
        assert "v_cvt_f32_fp8" in asm
        assert "v_cvt_pk_f32_fp8" not in asm
        assert "v_cvt_sr_fp8_f32" in asm
        assert "v_cvt_scalef32_" not in asm
