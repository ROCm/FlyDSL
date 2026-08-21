#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for scripts/isa_resource_table.py.

The ISA text is generated here rather than checked in, so the file states exactly
which parts of LLVM's output the parser depends on: the metadata indentation, the
`.set <kernel>.*` symbols, the `.size` terminator, and the mnemonic spelling.

The one axis that matters is the architecture family, so every assertion runs over
a CDNA shape and an RDNA shape. An earlier version of this tool was tested only
against a CDNA sample and so stayed green while rejecting every kernel on RDNA:
`.agpr_count` is emitted only on MFMA-capable targets, and gfx11 renamed
`ds_read`/`ds_write` to `ds_load`/`ds_store`.

The generated shape was checked against real dumps: its metadata keys are a subset
of the keys LLVM emits, in the same sorted order, and every structural marker the
parser keys off (`.amdgcn_target`, the kernel label, `.Lfunc_endN`, `.size`, the
`.set` block, the `  - `/4-space/6-space indent levels) appears as it does there.
The keys left out are ones the parser never reads.
"""

import pytest

from scripts import isa_resource_table as irt
from scripts.isa_resource_table import VALUE, parse_isa

pytestmark = [pytest.mark.l0_backend_agnostic]

LDS_READS = 3


def make_isa(arch, kernel, *, agpr_field, rdna, num_vgpr, num_agpr, vgpr_count):
    """One kernel's final ISA, in the shape LLVM emits for the given family.

    `agpr_field` and `rdna` are the two real differences between the families:
    only MFMA-capable targets emit `.agpr_count`, and only gfx11+ spell LDS
    access `ds_load`/`ds_store`. Everything else is common to both.
    """
    read = "ds_load_b128" if rdna else "ds_read_b64"
    write = "ds_store_b32" if rdna else "ds_write_b32"
    matrix = "v_wmma_f32_16x16x16_f16" if rdna else "v_mfma_f32_16x16x16_f16"
    lines = [
        f'\t.amdgcn_target "amdgcn-amd-amdhsa--{arch}"',
        "\t.text",
        f"{kernel}:                               ; @{kernel}",
        *[f"\t{read} v[0:1], v2"] * LDS_READS,
        f"\t{write} v3, v4",
        f"\t{matrix} a[0:3], v0, v1, a[0:3]",
        "\ts_endpgm",
        # The parser must skip the descriptor block and bound the body at the
        # `.size` terminator, not at s_endpgm.
        '\t.section\t.rodata,"a",@progbits',
        f"\t.amdhsa_kernel {kernel}",
        f"\t\t.amdhsa_next_free_vgpr {num_vgpr}",
        "\t.end_amdhsa_kernel",
        "\t.text",
        ".Lfunc_end0:",
        f"\t.size\t{kernel}, .Lfunc_end0-{kernel}",
        # Register counts come from these symbols, which every target emits.
        f"\t.set {kernel}.num_vgpr, {num_vgpr}",
        f"\t.set {kernel}.num_agpr, {num_agpr}",
        f"\t.set {kernel}.numbered_sgpr, 53",
        f"\t.set {kernel}.private_seg_size, 0",
        "\t.set amdgpu.max_num_vgpr, 0",  # module-level decoy, must not bind to a kernel
        "\t.amdgpu_metadata",
        "amdhsa.kernels:",
    ]
    # Metadata indentation is a contract: "  - " opens an entry, kernel keys sit
    # at column 4, and argument keys are deeper so they must not be mistaken for
    # kernel keys. Keys are emitted in sorted order, as LLVM does.
    lines += [f"  - .agpr_count:     {num_agpr}", "    .args:"] if agpr_field else ["  - .args:"]
    lines += [
        "      - .offset:         0",
        "        .size:           8",
        "        .value_kind:     global_buffer",
        "    .group_segment_fixed_size: 39936",
        f"    .name:           {kernel}",
        "    .private_segment_fixed_size: 0",
        "    .sgpr_count:     59",
        "    .sgpr_spill_count: 0",
        f"    .symbol:         {kernel}.kd",
        f"    .vgpr_count:     {vgpr_count}",
        "    .vgpr_spill_count: 0",
        f"amdhsa.target:   amdgcn-amd-amdhsa--{arch}",
        ".end_amdgpu_metadata",
    ]
    return "\n".join(lines) + "\n"


FAMILIES = {
    # A CDNA kernel with accumulators in use: `.vgpr_count` is the arch+acc total,
    # which is why it, and not its split, is the VGPR regression trigger.
    "cdna": dict(
        arch="gfx942",
        kernel="gemm_0",
        agpr_field=True,
        rdna=False,
        num_vgpr=256,
        num_agpr=29,
        vgpr_count=285,
    ),
    # An RDNA kernel: no `.agpr_count` anywhere, and LDS spelled ds_load/ds_store.
    "rdna": dict(
        arch="gfx1250",
        kernel="fmha_0",
        agpr_field=False,
        rdna=True,
        num_vgpr=942,
        num_agpr=0,
        vgpr_count=942,
    ),
}


def dump_tree(root, spec, **overrides):
    (root / "k").mkdir(parents=True, exist_ok=True)
    (root / "k" / "21_final_isa.s").write_text(make_isa(**{**spec, **overrides}))
    return root


@pytest.mark.parametrize("family", sorted(FAMILIES), ids=sorted(FAMILIES))
def test_resources_are_read_and_diffed_on_both_architecture_families(tmp_path, family):
    spec = FAMILIES[family]
    before = dump_tree(tmp_path / "before", spec)

    (record,) = parse_isa(before / "k" / "21_final_isa.s").values()
    assert record.name == spec["kernel"], "identity comes from the kernel, not its first argument"

    # Nothing may be unreadable on a healthy dump from either family. The AGPR count
    # in particular must come from the `.set` symbol, which RDNA also emits, rather
    # than from the CDNA-only `.agpr_count` metadata field.
    assert record.unparsed_keys == ()
    assert record.metrics["agpr"] == irt.Cell.of(spec["num_agpr"])
    assert record.metrics["vgpr"] == irt.Cell.of(spec["vgpr_count"])
    assert record.metrics["arch_vgpr"] == irt.Cell.of(spec["num_vgpr"])

    # LDS traffic is counted under whichever spelling this family uses; a parser
    # that knew only one of them would silently report 0 on the other.
    assert record.metrics["lds_read"] == irt.Cell.of(LDS_READS)
    assert record.metrics["lds_write"].state == VALUE

    # An unchanged pair is clean; a higher VGPR total is a regression; a missing
    # input is untrustworthy and must never be reported as either of the first two.
    assert irt.main(["diff", str(before), str(dump_tree(tmp_path / "same", spec))]) == 0
    worse = dump_tree(tmp_path / "worse", spec, vgpr_count=spec["vgpr_count"] + 8)
    assert irt.main(["diff", str(before), str(worse)]) == 1
    assert irt.main(["diff", str(before), str(tmp_path / "nonexistent")]) == 2
