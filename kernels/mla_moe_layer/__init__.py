from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.mla_moe_layer.config import KIMI_K3_CONFIG, MoeMode

if TYPE_CHECKING:
    from kernels.mla_moe_layer.layer import KimiK3MlaLayer, SharedReuseMlaMoeLayer

__all__ = ["KIMI_K3_CONFIG", "KimiK3MlaLayer", "MoeMode", "SharedReuseMlaMoeLayer"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name in {"KimiK3MlaLayer", "SharedReuseMlaMoeLayer"}:
        from kernels.mla_moe_layer.layer import KimiK3MlaLayer, SharedReuseMlaMoeLayer

        return {"KimiK3MlaLayer": KimiK3MlaLayer, "SharedReuseMlaMoeLayer": SharedReuseMlaMoeLayer}[name]
    raise AttributeError(name)
