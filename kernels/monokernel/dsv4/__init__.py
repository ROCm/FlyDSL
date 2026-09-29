# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 resident decode kernels and ATOM configuration contracts."""

from .config import Dsv4Config
from .op import Dsv4MonoKernel

__all__ = ["Dsv4Config", "Dsv4MonoKernel"]
