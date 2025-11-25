"""Runtime utilities for rocDSL GPU execution"""

from .hip_util import (
    hip_check,
    launch_kernel,
    get_hip_arch,
)

__all__ = [
    "hip_check",
    "launch_kernel", 
    "get_hip_arch",
]
