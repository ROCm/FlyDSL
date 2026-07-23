# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001

# Keep the package version available without importing optional runtime components.
__version__ = "0.2.4"

from .autotune import Config as Config, autotune as autotune  # noqa: E402
