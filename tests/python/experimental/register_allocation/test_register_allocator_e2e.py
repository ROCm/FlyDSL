# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


import pytest
import torch

from tests.python.experimental.register_allocation.test_register_allocator_codegen import SCALAR_TYPES, run_allocator
from tests.python.experimental.register_allocation.test_register_subword_union import SUPPORTED, UNSUPPORTED, run_case

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize(
    "mode,bank",
    [(mode, "VGPR") for mode in ["scalar", "vector", "packed", "sgpr", "union", "loop", "branch", "fma", *SCALAR_TYPES]]
    + [(mode, "AGPR") for mode in ["scalar", "vector"]],
)
def test_register_allocator_values(mode, bank):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if bank == "AGPR" and not arch.startswith("gfx9"):
        pytest.skip("AGPRs exist only on CDNA (gfx9) targets")
    if mode == "sgpr" and arch.startswith("gfx120"):
        pytest.skip("GFX12 keeps workgroup IDs in trap temporaries; placing them in SGPRs is unsupported")
    run_allocator(mode, run=True, bank=bank)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("mode", ["fixed", "plain"])
@pytest.mark.parametrize("case", sorted(SUPPORTED))
def test_subword_and_union_values(case, mode):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if "sgpr" in case and mode != "plain" and arch.startswith("gfx120"):
        pytest.skip("GFX12 keeps workgroup IDs in trap temporaries; placing them in SGPRs is unsupported")
    run_case(case, mode, run=True)


# Placement rejects these at compile time (see test_register_subword_union.py);
# the same programs without placement must still compute correct values.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("case", UNSUPPORTED)
def test_unsupported_subword_patterns_without_placement(case):
    run_case(case, "plain", run=True)
