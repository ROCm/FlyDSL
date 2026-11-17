# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GPU-free tests: code-object resource reading and the roofline arithmetic."""

import struct

import pytest

from flydsl.tuning.resources import artifact_resources, code_object, kernel_resources, occupancy_waves_per_simd
from flydsl.tuning.roofline import Peaks, Roofline, gemm_roofline, nominal_peaks

pytestmark = pytest.mark.l0_backend_agnostic


# ---- a minimal msgpack encoder, enough to build an amdhsa.kernels note
def _p(v):
    if isinstance(v, bool):
        return b"\xc3" if v else b"\xc2"
    if isinstance(v, int):
        if 0 <= v < 128:
            return bytes([v])
        if v < 256:
            return b"\xcc" + bytes([v])
        if v < 65536:
            return b"\xcd" + struct.pack(">H", v)
        return b"\xce" + struct.pack(">I", v)
    if isinstance(v, str):
        b = v.encode()
        return (bytes([0xA0 | len(b)]) if len(b) < 32 else b"\xd9" + bytes([len(b)])) + b
    if isinstance(v, list):
        return bytes([0x90 | len(v)]) + b"".join(_p(x) for x in v)
    if isinstance(v, dict):
        return bytes([0x80 | len(v)]) + b"".join(_p(k) + _p(x) for k, x in v.items())
    raise TypeError(v)


def _kernel(name, **kw):
    d = {
        ".name": name,
        ".vgpr_count": 96,
        ".agpr_count": 32,
        ".sgpr_count": 40,
        ".private_segment_fixed_size": 0,
        ".group_segment_fixed_size": 16384,
        ".vgpr_spill_count": 0,
        ".sgpr_spill_count": 0,
        ".wavefront_size": 64,
        ".max_flat_workgroup_size": 256,
    }
    d.update(kw)
    return d


def _elf(*kernels):
    note = _p({"amdhsa.version": [1, 2], "amdhsa.kernels": list(kernels)})
    return b"\x7fELF" + b"\0" * 12 + note + b"\0" * 8


def _module_text(elf: bytes) -> str:
    """The ``bin = "..."`` form MLIR prints: non-printables and quotes as \\XX."""
    out = []
    for b in elf:
        if b == 0x22 or b == 0x5C or b < 0x20 or b >= 0x7F:
            out.append(f"\\{b:02X}")
        else:
            out.append(chr(b))
    return 'gpu.binary @k [#gpu.object<#rocdl.target<chip = "gfx950">, bin = "' + "".join(out) + '">]'


def test_code_object_round_trips_the_mlir_escaped_binary():
    elf = _elf(_kernel("k"))
    assert code_object(_module_text(elf)) == elf


def test_code_object_without_a_binary_raises():
    with pytest.raises(RuntimeError, match="no gpu.binary"):
        code_object("module {}")


def test_kernel_resources_reads_every_kernel_and_defaults_missing_spill_fields():
    k2 = _kernel("splitk_reduce", **{".vgpr_count": 24, ".private_segment_fixed_size": 64})
    del k2[".vgpr_spill_count"]
    ks = kernel_resources(_elf(_kernel("main"), k2))
    assert [k["name"] for k in ks] == ["main", "splitk_reduce"]
    assert ks[0]["vgpr"] == 96 and ks[0]["lds_bytes"] == 16384
    assert ks[1]["scratch_bytes"] == 64 and ks[1]["spill_vgpr"] == 0


def test_artifact_resources_takes_the_worst_case_across_kernels():
    class Art:
        _ir_text = _module_text(_elf(_kernel("a", **{".vgpr_spill_count": 3}), _kernel("b", **{".vgpr_count": 200})))

    r = artifact_resources(Art())
    assert r["vgpr"] == 200 and r["spill_vgpr"] == 3 and len(r["kernels"]) == 2


@pytest.mark.parametrize("vgpr, waves", [(64, 8), (128, 4), (129, 3), (256, 2), (512, 1), (0, 8)])
def test_occupancy_follows_the_vgpr_allocation_granule(vgpr, waves):
    assert occupancy_waves_per_simd(vgpr) == waves


# ---- roofline
PEAKS = Peaks("toy", hbm_tbs=5.0, tflops={"bf16": 1000.0})


def test_memory_bound_problem_floors_at_bandwidth():
    # 1 flop/byte is far below the 200 flop/byte crossover
    rl = Roofline(flops=1e9, bytes_moved=1e9, peaks=PEAKS, dtype="bf16")
    assert rl.bound == "memory" and rl.crossover == pytest.approx(200.0)
    assert rl.floor_us == pytest.approx(1e9 / 5e12 * 1e6)


def test_compute_bound_problem_floors_at_matrix_peak():
    rl = Roofline(flops=2e12, bytes_moved=1e9, peaks=PEAKS, dtype="bf16")
    assert rl.bound == "compute"
    assert rl.floor_us == pytest.approx(2e12 / 1e15 * 1e6)


def test_fraction_is_floor_over_time_and_report_agrees():
    rl = Roofline(flops=1e9, bytes_moved=5e9, peaks=PEAKS, dtype="bf16")  # floor 1000 us
    assert rl.fraction(2000.0) == pytest.approx(0.5)
    rep = rl.report(2000.0)
    assert rep["tbs"] == pytest.approx(2.5) and rep["fraction"] == pytest.approx(0.5)


def test_gemm_roofline_counts_inputs_and_output_once():
    rl = gemm_roofline(128, 256, 512, PEAKS, "bf16", a_bytes=2, b_bytes=2, c_bytes=2)
    assert rl.flops == 2 * 128 * 256 * 512
    assert rl.bytes_moved == (128 * 512 + 512 * 256 + 128 * 256) * 2


def test_missing_dtype_peak_is_an_error_not_a_guess():
    with pytest.raises(KeyError, match="no 'fp8' peak"):
        Roofline(1.0, 1.0, PEAKS, "fp8").floor_us


def test_nominal_peaks_match_by_product_name_and_unknown_is_none():
    assert nominal_peaks("AMD Instinct MI355X").hbm_tbs == 8.0
    assert nominal_peaks("Radeon AI PRO R9700") is None
