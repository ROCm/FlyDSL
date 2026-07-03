# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for flydsl.utils.kernel_info's AMDGPU ISA "Kernel info" parser."""

import pytest

from flydsl.utils.kernel_info import get_occupancy, parse_kernel_info

pytestmark = [pytest.mark.l0_backend_agnostic]

SAMPLE_ISA = """\
	.text
	.globl	add_kernel
; Kernel info:
; codeLenInByte = 72
; TotalNumSgprs: 14
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 6
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
	.section	.rodata
"""


def test_parse_kernel_info_extracts_all_fields():
    info = parse_kernel_info(SAMPLE_ISA)
    assert info["NumVgprs"] == "6"
    assert info["NumAgprs"] == "0"
    assert info["Occupancy"] == "8"
    assert info["LDSByteSize"] == "0 bytes/workgroup (compile time only)"
    assert info["TotalNumSgprs"] == "14"


def test_parse_kernel_info_missing_block_returns_empty():
    assert parse_kernel_info("no kernel info here") == {}


def test_get_occupancy():
    assert get_occupancy(SAMPLE_ISA) == 8
    assert get_occupancy("nothing") is None
