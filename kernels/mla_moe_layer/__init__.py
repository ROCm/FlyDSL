from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.mla_moe_layer.config import MoeMode

if TYPE_CHECKING:
    from kernels.mla_moe_layer.layer import Glm5IndexedMlaMoeBlock

__all__ = ["Glm5IndexedMlaMoeBlock", "MoeMode"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "Glm5IndexedMlaMoeBlock":
        from kernels.mla_moe_layer.layer import Glm5IndexedMlaMoeBlock

        return Glm5IndexedMlaMoeBlock
    raise AttributeError(name)
