# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

__version__ = "0.1.4"

# FFM simulator compatibility shim (no-op outside simulator sessions).
from ._compat import _maybe_preload_system_comgr  # noqa: E402

_maybe_preload_system_comgr()

from .autotune import autotune, Config
