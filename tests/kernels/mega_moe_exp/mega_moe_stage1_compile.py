# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Representative COMPILE_ONLY gfx950 checks for MegaMoE v2 stage1."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "64M")

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mori.shmem as ms  # noqa: E402
import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from kernels.mega_moe.mega_moe_exp.mega_moe_stage1 import compile_mega_moe_stage1  # noqa: E402
from kernels.mega_moe.mega_moe_exp.planner import (  # noqa: E402
    DISPATCH_TABLE_SIZE,
    make_stage1_dispatch_plan,
)

MODEL_DIM = 7168
INTER_DIM = 3072
TOPK = 6
SCALE_DIM = MODEL_DIM // 32


def _alloc():
    dev = torch.device("cuda", 0)
    return {
        "out": torch.zeros(1, dtype=torch.float8_e4m3fn, device=dev),
        "x": torch.zeros(1, dtype=torch.float8_e4m3fn, device=dev),
        "w": torch.zeros(1, dtype=torch.uint8, device=dev),
        "scale_x": torch.zeros(1, dtype=torch.uint8, device=dev),
        "scale_w": torch.zeros(1, dtype=torch.uint8, device=dev),
        "trb": torch.zeros(1, dtype=torch.int32, device=dev),
        "se": torch.zeros(1, dtype=torch.int32, device=dev),
        "nv": torch.zeros(4, dtype=torch.int32, device=dev),
        "out_scale": torch.zeros(1, dtype=torch.uint8, device=dev),
    }


def _compile_case(tensors, common, case):
    plan = make_stage1_dispatch_plan(
        batch_size=case["max_batch"],
        npes=case["npes"],
        experts_per_rank=case["epr"],
        topk=TOPK,
        tile_m=case["sort_block_m"],
        row_bytes=MODEL_DIM,
        use_per_tile_payload_resource=case["use_tile_resource"],
    )
    launch = compile_mega_moe_stage1(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        rank=0,
        experts_per_rank=case["epr"],
        fuse_npes=case["npes"],
        fuse_topk=TOPK,
        fuse_cap=case["npes"] * case["max_batch"],
        fuse_mtpr=case["max_batch"],
        fuse_scale_dim=SCALE_DIM,
        sort_block_m=case["sort_block_m"],
        tile_n=case["tile_n"],
        tile_k=256,
        num_waves=case["num_waves"],
        grid_mult=case["grid_mult"],
        wgm=2,
        sched_nmajor=False,
        pipe_weights=True,
        mfma_amajor=True,
        swizzle_a=True,
        use_xcd=True,
        use_tile_resource=case["use_tile_resource"],
        waves_per_eu_hint=2,
        num_cu=256,
        num_dispatch_cu=case["num_dispatch_cu"],
        small_fixed=case["small_fixed"],
        small_fixed_route_tokens=case["run_tokens"],
    )
    args = (
        tensors["out"],
        tensors["x"],
        tensors["w"],
        tensors["scale_x"],
        tensors["scale_w"],
        tensors["trb"],
        tensors["se"],
        tensors["nv"],
        tensors["out_scale"],
        fx.Int32(plan.max_rows),
        fx.Int64(common["disp"].data_ptr()),
        fx.Int32(case["run_tokens"]),
        fx.Int64(common["in_tok"].data_ptr()),
        fx.Int64(common["in_idx"].data_ptr()),
        fx.Int64(common["in_wts"].data_ptr()),
        fx.Int64(common["in_sc"].data_ptr()),
        fx.Int64(common["parity"].data_ptr()),
        fx.Int64(common["expected"].data_ptr()),
        common["stream"],
    )
    flyc.compile(launch, *args)
    print(f"[OK] {case['name']} compiled")


def _compile_representative_paths():
    tensors = _alloc()
    dev = torch.device("cuda", 0)
    common = {
        "disp": torch.zeros(DISPATCH_TABLE_SIZE, dtype=torch.int64, device=dev),
        "in_tok": torch.zeros(1, dtype=torch.uint8, device=dev),
        "in_idx": torch.zeros(1, dtype=torch.int32, device=dev),
        "in_wts": torch.zeros(1, dtype=torch.float32, device=dev),
        "in_sc": torch.zeros(1, dtype=torch.uint8, device=dev),
        "parity": torch.zeros(1, dtype=torch.int32, device=dev),
        "expected": torch.zeros(2, dtype=torch.int32, device=dev),
        "stream": fx.Stream(torch.cuda.current_stream().cuda_stream),
    }
    cases = (
        {
            "id": "small-fixed",
            "name": "EP4 small-fixed BS64/max8192",
            "npes": 4,
            "epr": 48,
            "run_tokens": 64,
            "max_batch": 8192,
            "sort_block_m": 32,
            "tile_n": 128,
            "num_waves": 4,
            "grid_mult": 4,
            "num_dispatch_cu": 128,
            "use_tile_resource": True,
            "small_fixed": True,
        },
        {
            "id": "compact-m32",
            "name": "EP4 compact M32 BS512/max8192",
            "npes": 4,
            "epr": 48,
            "run_tokens": 512,
            "max_batch": 8192,
            "sort_block_m": 32,
            "tile_n": 256,
            "num_waves": 4,
            "grid_mult": 4,
            "num_dispatch_cu": 64,
            "use_tile_resource": False,
            "small_fixed": False,
        },
        {
            "id": "calibrated",
            "name": "EP4 calibrated compact BS8192",
            "npes": 4,
            "epr": 48,
            "run_tokens": 8192,
            "max_batch": 8192,
            "sort_block_m": 128,
            "tile_n": 512,
            "num_waves": 8,
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
            "small_fixed": False,
        },
        {
            "id": "large-payload",
            "name": "EP8 large-payload compact BS16384",
            "npes": 8,
            "epr": 48,
            "run_tokens": 16384,
            "max_batch": 16384,
            "sort_block_m": 128,
            "tile_n": 512,
            "num_waves": 8,
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
            "small_fixed": False,
        },
    )
    selected = {
        value.strip()
        for value in os.environ.get(
            "MEGA_V2_COMPILE_CASES",
            ",".join(case["id"] for case in cases),
        ).split(",")
        if value.strip()
    }
    for case in cases:
        if case["id"] in selected:
            _compile_case(tensors, common, case)


def main():
    if os.environ.get("COMPILE_ONLY", "0") != "1":
        raise RuntimeError("This harness requires COMPILE_ONLY=1")
    unique_id = ms.shmem_get_unique_id()
    status = ms.shmem_init_attr(ms.MORI_SHMEM_INIT_WITH_UNIQUEID, 0, 1, unique_id)
    if status != 0:
        raise RuntimeError(f"Mori SHMEM compile bootstrap failed with status {status}")
    try:
        _compile_representative_paths()
    finally:
        ms.shmem_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
