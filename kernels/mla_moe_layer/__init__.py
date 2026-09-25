from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.mla_moe_layer.config import MoeMode

if TYPE_CHECKING:
    from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer

__all__ = ["MoeMode", "SharedReuseMlaMoeLayer"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "SharedReuseMlaMoeLayer":
        from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer

        return SharedReuseMlaMoeLayer
    raise AttributeError(name)
