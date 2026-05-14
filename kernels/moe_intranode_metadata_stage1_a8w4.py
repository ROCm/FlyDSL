# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Intranode MoE metadata path (recv_meta + peer gather) for A8W4 / MX-FP8 activations.

This module wires together:

* :mod:`kernels.moe_metadata_dispatch_recv_meta` — metadata-only dispatch (no ``out_tok`` payload)
* :mod:`kernels.moe_peer_gather_mxfp8_a8w4` — device gather into dense ``uint8`` staging
* :mod:`kernels.recv_meta_a8w4` — row packing helpers for host-side checks

Downstream **stage1 MFMA** uses :func:`kernels.mixed_moe_gemm_2stage.compile_mixed_moe_gemm1` on the
**baseline** path, or :func:`kernels.moe_fused_dispatch_gather_gemm1_a8w4.compile_fused_dispatch_gather_gemm1_a8w4`
for fused gather+GEMM (``intranode_peer_gather`` default True; use ``False`` only for single-PE).
Callers should build ``sorted_token_ids`` / ``arg_x`` using the same conventions as the existing MoE
tests once staging matches the legacy token×topk layout (``dst_slot`` rows).
"""

from __future__ import annotations

from kernels.recv_meta_a8w4 import (
    RECV_META_ROW_BYTES,
    pack_recv_meta_row_i32,
    sort_recv_meta_rows_by_src_pe,
    unpack_recv_meta_row_i32,
)
from kernels.moe_peer_gather_mxfp8_a8w4 import compile_peer_gather_mxfp8_rows
from kernels.moe_metadata_dispatch_recv_meta import make_metadata_dispatch_recv_meta_jit

__all__ = [
    "RECV_META_ROW_BYTES",
    "pack_recv_meta_row_i32",
    "unpack_recv_meta_row_i32",
    "sort_recv_meta_rows_by_src_pe",
    "compile_peer_gather_mxfp8_rows",
    "make_metadata_dispatch_recv_meta_jit",
]
