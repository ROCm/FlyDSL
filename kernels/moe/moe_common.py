# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types + low-level helpers shared across MoE FlyDSL kernel modules."""

import os
from enum import Enum

from flydsl.expr import rocdl


# Build the async gmem->LDS DMA destination pointer via a provenance-preserving
# GEP from the LDS memref base instead of an inttoptr round-trip. Keeping SSA
# provenance lets LLVM AA prove the buffer_load_lds write to one ping/pong buffer
# does not alias ds_reads of the other, dropping the expcnt(0) it would otherwise
# insert before every ds_read. Set FLIR_LDS_PROVENANCE_GEP=0 to fall back.
LDS_PROVENANCE_GEP = os.environ.get("FLIR_LDS_PROVENANCE_GEP", "1") in (
    "1", "true", "True", "yes", "YES",
)


def normalize_mfma_k64(op):
    """Wrap the raw ``rocdl.mfma_i32_16x16x64_i8`` ODS class so call sites can use the
    uniform ``fn(res_ty, [a, b, acc, cbsz, abid, blgp])`` signature.

    Some installed flydsl builds expose ``rocdl.mfma_i32_16x16x64_i8`` as the raw
    nanobind ODS class (positional ``a, b, c, cbsz, abid, blgp``) rather than the
    operands-list helper used by the K32 path. Wrap the raw class so kernel call
    sites can use ``mfma_fn_k64(res_ty, [a, b, acc, 0, 0, 0])`` uniformly.
    """
    if op is None:
        return None
    import inspect
    try:
        params = list(inspect.signature(op).parameters.values())
        if len(params) >= 2 and params[1].name == "operands":
            return op
    except (TypeError, ValueError):
        pass
    _split = getattr(rocdl, "_split_mfma_operands", None)

    def _wrapped(result_type, operands, *, loc=None, ip=None):
        if _split is not None:
            a, b, c, cbsz, abid, blgp = _split(operands, loc=loc)
        else:
            a, b, c = operands[0], operands[1], operands[2]
            cbsz = int(operands[3]) if len(operands) > 3 else 0
            abid = int(operands[4]) if len(operands) > 4 else 0
            blgp = int(operands[5]) if len(operands) > 5 else 0
        res = op(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)
        return getattr(res, "result", res)

    return _wrapped


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"
