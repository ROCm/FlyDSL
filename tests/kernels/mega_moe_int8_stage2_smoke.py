#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Standalone gfx950 INT8 Stage2 compile/correctness smoke."""

import argparse

import torch

import flydsl.expr as fx
from kernels.mega_moe.mega_moe_stage2 import run_mega_moe_stage2
from kernels.mega_moe.quant import (
    convert_aiter_lqq_to_megamoe,
    repack_megamoe_lqq_for_int8_loader,
)
from tests.utils import shuffle_weight


def _case(mode: str, iterations: int, model_dim: int, inter_dim: int):
    torch.manual_seed(2026)
    device = torch.device("cuda", 0)
    experts = 2
    rows, topk, max_tok, block_m = 3, 1, 32, 32

    a2 = torch.zeros((max_tok, inter_dim), dtype=torch.int8, device=device)
    a2[:rows] = torch.randint(-8, 9, (rows, inter_dim), dtype=torch.int8, device=device)
    sx = torch.ones(max_tok, dtype=torch.float32, device=device)
    sx[:rows] = torch.tensor([0.01, 0.015, 0.02], device=device)
    sw = torch.linspace(0.005, 0.02, experts * model_dim, dtype=torch.float32, device=device).view(experts, model_dim)

    qscale = qzero = None
    if mode == "w8a8smooth":
        logical_w = torch.randint(
            -8,
            9,
            (experts, model_dim, inter_dim),
            dtype=torch.int8,
            device=device,
        )
        kernel_w = shuffle_weight(logical_w).contiguous().view(-1)
        reference_w = logical_w.float()
    else:
        u4 = torch.randint(
            0,
            16,
            (experts, model_dim, inter_dim),
            dtype=torch.uint8,
            device=device,
        )
        raw_scale = torch.randint(
            1,
            3,
            (experts, model_dim, inter_dim // 64),
            dtype=torch.uint8,
            device=device,
        )
        raw_zero = torch.randint(0, 16, raw_scale.shape, dtype=torch.uint8, device=device)
        kernel_w, qscale, qzero = convert_aiter_lqq_to_megamoe(u4, raw_scale, raw_zero)
        kernel_w = repack_megamoe_lqq_for_int8_loader(kernel_w, model_dim, inter_dim)
        scale_k = raw_scale.repeat_interleave(64, dim=2).to(torch.int32)
        zero_k = raw_zero.repeat_interleave(64, dim=2).to(torch.int32)
        reference_w = (
            ((u4.to(torch.int32) * scale_k + zero_k).clamp(0, 255).to(torch.uint8) ^ 0x80).view(torch.int8).float()
        )

    expert_ids = torch.tensor([1], dtype=torch.int32, device=device)
    num_valid = torch.tensor([rows], dtype=torch.int32, device=device)
    max_expert_tiles = torch.ones(1, dtype=torch.int32, device=device)
    sorted_ids = torch.full((block_m,), 0x00FFFFFF, dtype=torch.int32, device=device)
    sorted_ids[:rows] = torch.arange(rows, dtype=torch.int32, device=device)
    routing = torch.zeros(block_m, dtype=torch.float32, device=device)
    routing[:rows] = torch.tensor([0.5, 0.75, 1.25], device=device)
    tile_row_base = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.zeros((max_tok * topk, model_dim), dtype=torch.bfloat16, device=device)
    p2p_table = torch.tensor([output.data_ptr()], dtype=torch.int64, device=device)
    stream = fx.Stream(torch.cuda.current_stream())

    def launch():
        run_mega_moe_stage2(
            fx.Int64(a2.data_ptr()),
            fx.Int64(sx.data_ptr()),
            fx.Int64(kernel_w.data_ptr()),
            fx.Int64(sw.data_ptr()),
            fx.Int64(expert_ids.data_ptr()),
            fx.Int64(num_valid.data_ptr()),
            fx.Int64(max_expert_tiles.data_ptr()),
            fx.Int64(sorted_ids.data_ptr()),
            fx.Int64(routing.data_ptr()),
            fx.Int64(tile_row_base.data_ptr()),
            fx.Int64(p2p_table.data_ptr()),
            block_m,
            inter_dim,
            model_dim,
            stream,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            rank=0,
            npes=1,
            max_tok=max_tok,
            recv_cap=max_tok,
            comb_inp_nbytes=output.numel() * output.element_size(),
            BM=block_m,
            SBM=block_m,
            BN=128,
            BK=256,
            HIDDEN_MAX=model_dim,
            INTER_MAX=inter_dim,
            cu_num=torch.cuda.get_device_properties(0).multi_processor_count,
            persist=False,
            quant_mode=mode,
            qscale_w=qscale,
            qzero_w=qzero,
        )

    # First call is the required compile smoke; second validates execution.
    launch()
    launch()
    torch.cuda.synchronize()
    reference = (a2[:rows].float() @ reference_w[1].T) * sx[:rows, None] * sw[1][None, :] * routing[:rows, None]
    actual = output[:rows].float()
    error = actual - reference
    max_abs = error.abs().max().item()
    rel_l2 = (error.square().sum() / reference.square().sum().clamp_min(1e-12)).sqrt().item()
    if not torch.allclose(actual, reference, rtol=0.01, atol=0.05):
        raise AssertionError(f"{mode} mismatch: max_abs={max_abs:.6g}, rel_l2={rel_l2:.6g}")

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        launch()
    end.record()
    torch.cuda.synchronize()
    latency_us = begin.elapsed_time(end) * 1000.0 / iterations
    print(
        f"{mode}: compile=PASS correctness=PASS max_abs={max_abs:.6g} rel_l2={rel_l2:.6g} stage2_us={latency_us:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("a8w4smooth", "w8a8smooth", "all"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--inter-dim", type=int, default=256)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    modes = ("a8w4smooth", "w8a8smooth") if args.mode == "all" else (args.mode,)
    for mode in modes:
        _case(mode, args.iterations, args.model_dim, args.inter_dim)


if __name__ == "__main__":
    main()
