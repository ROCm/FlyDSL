#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Small distributed compile smoke for MegaMoEV2 INT8 Stage1."""

import os

import mori.shmem as ms
import torch
import torch.distributed as dist

from kernels.mega_moe.mega_moe import MegaMoEV2
from kernels.mega_moe.quant import convert_aiter_lqq_to_megamoe


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    quant = os.environ.get("MEGAMOE_INT8_QUANT", "w8a8smooth")
    run_e2e = os.environ.get("MEGAMOE_INT8_E2E", "0") == "1"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world,
        device_id=device,
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

    model_dim = inter_dim = 256
    topk, mtpr, experts_per_rank = 2, 16, 8
    experts = experts_per_rank * world
    weight_rows = 2 * inter_dim
    if quant == "a8w4smooth":
        w1_u4 = torch.zeros(
            (experts_per_rank, weight_rows, model_dim),
            dtype=torch.uint8,
            device=device,
        )
        w2_u4 = torch.zeros(
            (experts_per_rank, model_dim, inter_dim),
            dtype=torch.uint8,
            device=device,
        )
        w1_raw_q = torch.ones(
            (experts_per_rank, weight_rows, model_dim // 64),
            dtype=torch.uint8,
            device=device,
        )
        w2_raw_q = torch.ones(
            (experts_per_rank, model_dim, inter_dim // 64),
            dtype=torch.uint8,
            device=device,
        )
        w1_raw_z = torch.full_like(w1_raw_q, 128)
        w2_raw_z = torch.full_like(w2_raw_q, 128)
        w1, w1_qscale, w1_qzero = convert_aiter_lqq_to_megamoe(
            w1_u4, w1_raw_q, w1_raw_z
        )
        w2, w2_qscale, w2_qzero = convert_aiter_lqq_to_megamoe(
            w2_u4, w2_raw_q, w2_raw_z
        )
        lqq = dict(
            w1_lqq_scale=w1_qscale,
            w1_lqq_zero=w1_qzero,
            w2_lqq_scale=w2_qscale,
            w2_lqq_zero=w2_qzero,
        )
    else:
        fill = 0 if run_e2e else None
        w1 = (
            torch.zeros(
                (experts_per_rank * weight_rows * model_dim,),
                dtype=torch.int8,
                device=device,
            )
            if fill == 0
            else torch.randint(
                -8,
                8,
                (experts_per_rank * weight_rows * model_dim,),
                dtype=torch.int8,
                device=device,
            )
        )
        w2 = torch.zeros(
            (experts_per_rank * model_dim * inter_dim,),
            dtype=torch.int8,
            device=device,
        )
        lqq = {}

    if quant == "a8w4smooth" and not run_e2e:
        # Non-zero payload keeps the Stage1-only smoke sensitive to data flow.
        w1.random_(-128, 127)
    elif quant == "w8a8smooth" and not run_e2e:
        w2 = torch.randint(
            -8,
            8,
            (experts_per_rank * model_dim * inter_dim,),
            dtype=torch.int8,
            device=device,
        )

    op = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        w1=w1,
        w1_scale=torch.full(
            (experts_per_rank, weight_rows),
            1e-3,
            dtype=torch.float32,
            device=device,
        ),
        w2=w2,
        w2_scale=torch.full(
            (experts_per_rank, model_dim),
            1e-3,
            dtype=torch.float32,
            device=device,
        ),
        max_tok_per_rank=mtpr,
        fc1_smooth_scale=torch.ones(
            (experts, model_dim), dtype=torch.float32, device=device
        ),
        fc2_smooth_scale=torch.ones(
            (experts, inter_dim), dtype=torch.float32, device=device
        ),
        **lqq,
    )
    front_q = torch.randint(
        -8, 8, (1, topk, model_dim), dtype=torch.int8, device=device
    )
    front_scale = torch.ones(
        (1, topk, 1), dtype=torch.float32, device=device
    )
    topk_ids = torch.tensor(
        [[rank * experts_per_rank, ((rank + 1) % world) * experts_per_rank]],
        dtype=torch.int32,
        device=device,
    )
    weights = torch.ones((1, topk), dtype=torch.float32, device=device)
    if run_e2e:
        x = torch.randn((1, model_dim), dtype=torch.bfloat16, device=device)
        result = op.forward(x, weights, topk_ids)
        torch.cuda.synchronize()
        output = op._int8_stage1_output
        padded_rows = int(output.num_valid_ids[0].item())
        if padded_rows < topk:
            raise AssertionError(
                f"{quant} E2E produced {padded_rows} sorted rows, expected >= {topk}"
            )
        if not torch.isfinite(result).all() or result.abs().max().item() != 0.0:
            raise AssertionError(
                f"{quant} zero-weight E2E mismatch: max={result.abs().max().item()}"
            )
    else:
        output = None
        for _ in range(2):
            output = op._run_int8_stage1(
                front_q, front_scale, weights, topk_ids
            )
        torch.cuda.synchronize()
    if rank == 0:
        print(
            f"{quant}: num_valid={output.num_valid_ids.tolist()} "
            f"total_recv={output.total_recv.tolist()} "
            f"e2e={'PASS' if run_e2e else 'SKIP'}",
            flush=True,
        )
    ms.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
