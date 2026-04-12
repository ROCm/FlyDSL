# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

_BASE_VERSION = "0.1.3.1"

# FFM simulator compatibility shim (no-op outside simulator sessions).
from ._compat import _maybe_preload_system_comgr  # noqa: E402

_maybe_preload_system_comgr()

try:
    from ._version import __version__
except ImportError:
    __version__ = _BASE_VERSION

from .autotune import autotune, Config
