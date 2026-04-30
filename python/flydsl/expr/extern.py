# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""External function expression API."""

from ..compiler.extern import ExternFunction as ffi
from ..compiler.extern_link import ExternFunction, link_extern, register_extern_resolver

__all__ = ["ffi", "ExternFunction", "link_extern", "register_extern_resolver"]
