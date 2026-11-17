# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Everything under tests/unit/tuning runs on the host with no GPU: it is L0."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/unit/tuning/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.l0_backend_agnostic)
