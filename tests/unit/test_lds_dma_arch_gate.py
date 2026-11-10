# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx11 has no global -> LDS direct-load (``buffer_load_* ... lds``) hardware.

The intrinsic passes MLIR verification there and then aborts the process in LLVM
instruction selection, so the entry points that emit it must reject gfx11 up
front. The gate keys off ``ARCH``, so these run without a device.

Scope is gfx11 only; gfx12 is deliberately left alone.
"""

import pytest

import flydsl.expr as fx
from flydsl.compiler.jit_function import _create_mlir_context
from flydsl.expr.rocdl.utils import require_lds_dma_support

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

NO_LDS_DMA = ["gfx1100", "gfx1101", "gfx1151"]
HAS_LDS_DMA = ["gfx942", "gfx950", "gfx1030"]


@pytest.mark.parametrize("arch", HAS_LDS_DMA)
def test_lds_dma_accepted(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    require_lds_dma_support("raw_ptr_buffer_load_lds")


@pytest.mark.parametrize("arch", NO_LDS_DMA)
def test_buffer_copy_lds_atom_rejected(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    for make_atom in (fx.rocdl.BufferCopyLDS32b, fx.rocdl.BufferCopyLDS128b):
        with pytest.raises(ValueError, match=f"not supported on target arch '{arch}'"):
            make_atom()


@pytest.mark.parametrize("arch", NO_LDS_DMA)
def test_raw_ptr_buffer_load_lds_rejected(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    with _create_mlir_context():
        with pytest.raises(ValueError, match="no global -> LDS direct-load hardware"):
            # Rejected before any operand is touched, so the arguments do not matter.
            fx.rocdl.raw_ptr_buffer_load_lds(None, None, 4, 0, 0, 0)
