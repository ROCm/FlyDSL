# SPDX-FileCopyrightText: Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime utilities for flir GPU execution"""

from .device import get_rocm_arch

__all__ = [
    "get_rocm_arch",
]
