# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compatibility entry point for PR #1256's TP4 paged interface."""

from kernels.monokernel.config import AttentionWeight, KvCacheLayout, glm5_tp_config
from kernels.monokernel.glm.op import Glm5MonoKernel, prepare_glm5_weights

__all__ = ["Glm5TP4MonoKernel", "prepare_glm5_weights"]


class Glm5TP4MonoKernel(Glm5MonoKernel):
    """Keep the TP4 positional API and its BF16-RoPE/MTP4 defaults."""

    def __init__(
        self,
        W,
        samples,
        rank=0,
        npes=1,
        group=None,
        topk=2048,
        launches_per_step=1,
        with_indexer=False,
        index_max_seq=4096,
        index_request_width=None,
        index_page_size=16,
        index_block_table_stride=None,
        attention_weight=AttentionWeight.FP8_BLOCK128,
        kv_cache_layout=KvCacheLayout.SPLIT,
        kv_cache_dtype="bf16",
        prepared_weights=None,
        runtime=None,
        dcp_size=1,
        native_fp4_mfma=False,
        timeline=False,
    ):
        if npes != 4:
            raise ValueError(f"Glm5TP4MonoKernel requires TP4, got TP{npes}")
        if W.config != glm5_tp_config(4):
            raise ValueError("Glm5TP4MonoKernel requires a TP4 weight shard")
        if index_request_width is None:
            index_request_width = 5 if samples in (5, 10) else samples
        super().__init__(
            W,
            samples,
            rank,
            npes,
            group,
            topk,
            launches_per_step,
            with_indexer,
            index_max_seq,
            timeline,
            index_request_width=index_request_width,
            index_page_size=index_page_size,
            index_block_table_stride=index_block_table_stride,
            attention_weight=attention_weight,
            kv_cache_layout=kv_cache_layout,
            kv_cache_dtype=kv_cache_dtype,
            prepared_weights=prepared_weights,
            runtime=runtime,
            dcp_size=dcp_size,
            native_fp4_mfma=native_fp4_mfma,
            rope_dtype="bf16",
        )
