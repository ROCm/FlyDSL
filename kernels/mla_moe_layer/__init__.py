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
    from kernels.mla_moe_layer.kda import KimiK3KdaAttention
    from kernels.mla_moe_layer.kda_conv import KimiK3KdaCausalConv
    from kernels.mla_moe_layer.kda_recurrence import KimiK3KdaConvRecurrence, KimiK3KdaRecurrence
    from kernels.mla_moe_layer.kimi_k3 import KimiK3KdaMoeLayer, KimiK3MlaMoeLayer
    from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer
    from kernels.mla_moe_layer.mxfp8_linear import Mxfp8Linear

__all__ = [
    "GLM5_CONFIG",
    "Glm5IndexedMlaMoeBlock",
    "IndexedMlaMoeBlock",
    "KIMI_K3_CONFIG",
    "KimiK3KdaAttention",
    "KimiK3KdaCausalConv",
    "KimiK3KdaConvRecurrence",
    "KimiK3KdaMoeLayer",
    "KimiK3KdaRecurrence",
    "KimiK3MlaLayer",
    "KimiK3MlaMoeLayer",
    "MoeMode",
    "Mxfp8Linear",
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
    if name == "KimiK3KdaAttention":
        from kernels.mla_moe_layer.kda import KimiK3KdaAttention

        return KimiK3KdaAttention
    if name == "KimiK3KdaCausalConv":
        from kernels.mla_moe_layer.kda_conv import KimiK3KdaCausalConv

        return KimiK3KdaCausalConv
    if name == "KimiK3KdaRecurrence":
        from kernels.mla_moe_layer.kda_recurrence import KimiK3KdaRecurrence

        return KimiK3KdaRecurrence
    if name == "KimiK3KdaConvRecurrence":
        from kernels.mla_moe_layer.kda_recurrence import KimiK3KdaConvRecurrence

        return KimiK3KdaConvRecurrence
    if name == "KimiK3KdaMoeLayer":
        from kernels.mla_moe_layer.kimi_k3 import KimiK3KdaMoeLayer

        return KimiK3KdaMoeLayer
    if name == "Mxfp8Linear":
        from kernels.mla_moe_layer.mxfp8_linear import Mxfp8Linear

        return Mxfp8Linear
    raise AttributeError(name)
