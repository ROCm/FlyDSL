# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest
import torch

from tests.unit.test_register_allocator_codegen import SCALAR_TYPES, run_allocator

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize(
    "mode,bank",
    [(mode, "VGPR") for mode in ["scalar", "vector", "packed", "sgpr", "union", "loop", "branch", "fma", *SCALAR_TYPES]]
    + [(mode, "AGPR") for mode in ["scalar", "vector"]],
)
def test_register_allocator_values(mode, bank):
    run_allocator(mode, run=True, bank=bank)
