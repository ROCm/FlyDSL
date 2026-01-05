"""Runtime utilities for flir GPU execution"""

from .device import get_rocm_arch
from .hip_util import get_hip_arch, hip_check

__all__ = [
    "get_rocm_arch",
    "get_hip_arch",
    "hip_check",
]
