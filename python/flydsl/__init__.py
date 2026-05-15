# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001

__version__ = "0.1.7"

# FFM simulator compatibility shim (no-op outside simulator sessions).
from ._compat import _maybe_preload_system_comgr  # noqa: E402

_maybe_preload_system_comgr()

from .autotune import Config as Config, autotune as autotune  # noqa: E402
