# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001

__version__ = "0.2.4"

from .autotune import Config as Config, autotune as autotune  # noqa: E402

# CI trigger test: code change should run the full GPU test matrix (not skipped).
