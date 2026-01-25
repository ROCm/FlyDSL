"""Runtime utilities for flir GPU execution"""

from .device import get_rocm_arch

__all__ = [
    "get_rocm_arch",
]
