# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Architecture-specific shared-memory capacity checks."""

import pytest

from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, check_smem_capacity

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_gfx1100_shared_memory_capacity():
    assert SMEM_CAPACITY_MAP["gfx1100"] == 65536
    check_smem_capacity(65536, "gfx1100")

    with pytest.raises(RuntimeError, match=r"device gfx1100 limit is 65536 bytes"):
        check_smem_capacity(65537, "gfx1100")
