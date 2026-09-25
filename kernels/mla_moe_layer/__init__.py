from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.mla_moe_layer.config import KIMI_K3_CONFIG, MoeMode

if TYPE_CHECKING:
    from kernels.mla_moe_layer.layer import Glm5IndexedMlaMoeBlock
    from kernels.mla_moe_layer.kimi_k3 import KimiK3MlaMoeLayer
    from kernels.mla_moe_layer.layer import KimiK3MlaLayer

__all__ = [
    "Glm5IndexedMlaMoeBlock",
    "KIMI_K3_CONFIG",
    "KimiK3MlaLayer",
    "KimiK3MlaMoeLayer",
    "MoeMode",
]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "Glm5IndexedMlaMoeBlock":
        from kernels.mla_moe_layer.layer import Glm5IndexedMlaMoeBlock

        return Glm5IndexedMlaMoeBlock
    if name == "KimiK3MlaLayer":
        from kernels.mla_moe_layer.layer import KimiK3MlaLayer

        return KimiK3MlaLayer
    if name == "KimiK3MlaMoeLayer":
        from kernels.mla_moe_layer.kimi_k3 import KimiK3MlaMoeLayer

        return KimiK3MlaMoeLayer
    raise AttributeError(name)
