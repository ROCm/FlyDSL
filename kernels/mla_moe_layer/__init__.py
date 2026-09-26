# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.mla_moe_layer.config import GLM5_CONFIG, KIMI_K3_CONFIG, MoeMode

if TYPE_CHECKING:
    from kernels.mla_moe_layer.indexed_layer import (
        Glm5IndexedMlaMoeBlock,
        IndexedMlaMoeBlock,
        KimiK3MlaLayer,
    )
    from kernels.mla_moe_layer.kimi_k3 import KimiK3MlaMoeLayer
    from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer

__all__ = [
    "GLM5_CONFIG",
    "Glm5IndexedMlaMoeBlock",
    "IndexedMlaMoeBlock",
    "KIMI_K3_CONFIG",
    "KimiK3MlaLayer",
    "KimiK3MlaMoeLayer",
    "MoeMode",
    "SharedReuseMlaMoeLayer",
]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "SharedReuseMlaMoeLayer":
        from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer

        return SharedReuseMlaMoeLayer
    if name == "Glm5IndexedMlaMoeBlock":
        from kernels.mla_moe_layer.indexed_layer import Glm5IndexedMlaMoeBlock

        return Glm5IndexedMlaMoeBlock
    if name == "IndexedMlaMoeBlock":
        from kernels.mla_moe_layer.indexed_layer import IndexedMlaMoeBlock

        return IndexedMlaMoeBlock
    if name == "KimiK3MlaLayer":
        from kernels.mla_moe_layer.indexed_layer import KimiK3MlaLayer

        return KimiK3MlaLayer
    if name == "KimiK3MlaMoeLayer":
        from kernels.mla_moe_layer.kimi_k3 import KimiK3MlaMoeLayer

        return KimiK3MlaMoeLayer
    raise AttributeError(name)
