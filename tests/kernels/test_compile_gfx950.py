#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-only tests for gfx950 target.

Validates that kernels compile successfully for gfx950 without requiring a GPU.
Run with: COMPILE_ONLY=1 ARCH=gfx950 PYTHONPATH=./ pytest tests/kernels/test_compile_gfx950.py -v
"""

import os
import pytest

# Force compile-only mode and gfx950 target
os.environ.setdefault("COMPILE_ONLY", "1")
os.environ.setdefault("ARCH", "gfx950")

pytestmark = [pytest.mark.compile_only]

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T


def _dummy_tensor(*shape, dtype=torch.float32):
    """Create a small dummy tensor for tracing.

    Uses CUDA tensors because DLPack with stream=-1 requires a CUDA device.
    compile-only mode skips actual kernel launch, so the data content doesn't matter.
    """
    return torch.zeros(*shape, dtype=dtype, device="cuda")


class TestVecAddCompile:
    def test_vecadd_compiles_gfx950(self):
        from tests.kernels.test_vec_add import vecAdd

        SIZE = 1024
        BLOCK_DIM = 256
        VEC_WIDTH = 4
        a = _dummy_tensor(SIZE)
        b = _dummy_tensor(SIZE)
        c = _dummy_tensor(SIZE)
        tA = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
        vecAdd(tA, b, c, SIZE, SIZE, BLOCK_DIM, VEC_WIDTH)


class TestQuantCompile:
    def test_quant_compiles_gfx950(self):
        # Enable quant test
        os.environ["FLYDSL_RUN_QUANT"] = "1"
        from tests.kernels.test_quant import build_quant_module

        launch_fn, config = build_quant_module(N=4096)
        M = 32
        inp = _dummy_tensor(M, 4096, dtype=torch.float16)
        out = _dummy_tensor(M, 4096, dtype=torch.int8)
        scales = _dummy_tensor(M, dtype=torch.float32)
        launch_fn(inp, out, scales, M)


class TestMoeReduceCompile:
    def test_moe_reduce_compiles_gfx950(self):
        from kernels.moe_gemm_2stage import compile_moe_reduction

        reduce_fn = compile_moe_reduction(topk=8, model_dim=7168, dtype_str="f16")
        tokens = 4
        x = _dummy_tensor(tokens, 8, 7168, dtype=torch.float16)
        y = _dummy_tensor(tokens, 7168, dtype=torch.float16)
        mask = _dummy_tensor(0, 8, dtype=torch.uint8)
        stream = torch.cuda.current_stream()
        reduce_fn(x, y, mask, tokens, stream)


class TestRopeCompile:
    def test_rope_compiles_gfx950(self):
        from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module

        launch_fn = build_fused_rope_cache_module(
            head_dim=64, num_q_heads=8, num_kv_heads=1,
            block_size=16, flash_layout=True, dtype_str="bf16",
        )
        T_tok = 4
        Q = _dummy_tensor(T_tok, 8, 64, dtype=torch.bfloat16)
        K = _dummy_tensor(T_tok, 1, 64, dtype=torch.bfloat16)
        V = _dummy_tensor(T_tok, 1, 64, dtype=torch.bfloat16)
        positions = _dummy_tensor(T_tok, dtype=torch.int32)
        cos_cache = _dummy_tensor(128, 32, dtype=torch.bfloat16)
        sin_cache = _dummy_tensor(128, 32, dtype=torch.bfloat16)
        slot_mapping = _dummy_tensor(T_tok, dtype=torch.int32)
        key_cache = _dummy_tensor(8, 16, 1, 64, dtype=torch.bfloat16)
        value_cache = _dummy_tensor(8, 16, 1, 64, dtype=torch.bfloat16)
        Q_out = _dummy_tensor(T_tok, 8, 64, dtype=torch.bfloat16)
        K_out = _dummy_tensor(T_tok, 1, 64, dtype=torch.bfloat16)
        launch_fn(
            Q, K, V, positions, cos_cache, sin_cache, slot_mapping,
            key_cache, value_cache, Q_out, K_out, T_tok,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
