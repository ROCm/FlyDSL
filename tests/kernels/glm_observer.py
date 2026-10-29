# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only observations of GLM scratch mailboxes; never used in timed graphs."""

import torch

from kernels.monokernel.config import HIDDEN, KV_LORA, MOE_SLOTS, N_EXPERTS, NOPE_DIM, PE_DIM, Q_LORA, V_DIM
from kernels.monokernel.glm.layout import INDEX_DIM


def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False, _tile=None) -> torch.Tensor:
    """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
    ``bf2``: each pair's value word packs two bf16 elements)."""
    if "_tile_rows" in self.scr_layout and _tile is None:
        tile = self.scr_layout["_tile_rows"]
        if shape[0] != self.S:
            raise ValueError("tiled debug expects a leading sample dimension")
        return torch.cat(
            [
                self.debug(name, (tile, *shape[1:]), dtype, pairs, bf2, _tile=part)
                for part in range(self.scr_layout["_tile_count"])
            ]
        )
    part = 0 if _tile is None else _tile
    off = self.scr_layout[name] + part * self.scr_layout.get("_tile_stride", 0)
    storage = self.scratch
    if name == "index_q" and "index_cp_q" in self.sym_layout:
        # Debug reads the most recently published slot; production forward
        # never reads device epochs on the host.
        off = self.sym_layout["index_cp_q"] + part * self.sym_layout.get("_tile_stride", 0)
        stride = self.sym_layout["index_cp_q_stride"]
        storage = self.sym_storage
        first_tag = storage[off + 4 : off + 8].view(torch.int32).item()
        second_tag = storage[off + stride + 4 : off + stride + 8].view(torch.int32).item()
        off += stride if second_tag > first_tag else 0
    n = 1
    for d in shape:
        n *= d
    if not pairs:
        return storage[off : off + n * 4].view(dtype).view(shape)
    if bf2:
        words = storage[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
        return words.view(torch.bfloat16).float().view(shape)
    words = storage[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
    return words.view(dtype).view(shape)


def intermediates(self):
    S, H = self.S, self.W.heads
    result = dict(
        q_a=self.debug("q_a", (S, Q_LORA)),
        kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
        q_nope=self.debug("q_nope", (S, H, NOPE_DIM), bf2=True),
        q_pe=self.debug("q_pe", (S, H, PE_DIM), bf2=True),
        q_lat=self.debug("q_lat", (S, H, KV_LORA), bf2=True),
        o=self.debug("o", (S, self.output_heads * V_DIM), bf2=True),
        a=self.debug("a", (S, HIDDEN), bf2=True).to(torch.bfloat16),
        scores=self.debug("scores", (S, N_EXPERTS)),
        sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
        prob=self.debug("prob", (S, MOE_SLOTS)),
        mid=self.debug("mid", (S, MOE_SLOTS, self.W.config.inter)),
        xq=self.debug("xqd", (S, HIDDEN), pairs=False),
    )
    if self.with_indexer:
        result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
        result["index_w"] = self.debug("index_w", (S, 32))
        result["indices"] = self.debug("indices", (S, self.topk), torch.int32, pairs=False)
    return result


def attach_observer(operator):
    """Add read-only numerical observations to a test's operator class."""
    operator.debug = debug
    operator.intermediates = intermediates
    return operator
