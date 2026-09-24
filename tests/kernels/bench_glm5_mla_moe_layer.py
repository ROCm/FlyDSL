# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Lean driver for profiling the GLM-5 MLA+MoE layer monokernel (no torch golden).

python3 tests/kernels/bench_glm5_mla_moe_layer.py [-S 1] [--pos 3000] [--iters 3]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.mla_moe_layer.op import Glm5MlaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.reference import KV_LORA, PE_DIM, make_weights, rope_table  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=1)
    ap.add_argument("--pos", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    W = make_weights(0, heads=8, device=dev)
    cos, sin = rope_table(4096, device=dev)
    kv = torch.randn(4096, KV_LORA, device=dev).to(torch.bfloat16)
    pe = torch.randn(4096, PE_DIM, device=dev).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(max(a.pos + s + 1, 2048), device=dev)[:2048] for s in range(a.S)]).int()
    op = Glm5MlaMoeLayer(W, a.S)
    h = torch.randn(a.S, 6144, device=dev).to(torch.bfloat16)
    pos = torch.tensor([a.pos], dtype=torch.int32, device=dev)
    for _ in range(a.iters):
        op.forward(h, pos, kv, pe, idx, cos, sin)
    torch.cuda.synchronize()
    print("done")
