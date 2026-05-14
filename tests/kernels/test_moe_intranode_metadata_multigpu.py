# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Multi-GPU correctness for MoE intranode metadata dispatch (cross-rank recv_meta).

Uses ``torchrun`` + NCCL + mori symmetric heap, launches
``make_metadata_dispatch_recv_meta_jit`` on each rank, then checks **recv_meta** rows,
**TIS**, **total_recv**, **dest_tok_map**, and handshake cleanup.

Pytest spawns ``torchrun`` in a subprocess (same pattern as ``test_flydsl_shmem.py``)
so ``HIP_VISIBLE_DEVICES`` from the outer harness does not hide GPUs.

Manual run on MI355X (8 GPUs)::

    MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 \\
        python tests/kernels/test_moe_intranode_metadata_multigpu.py

Pytest::

    pytest tests/kernels/test_moe_intranode_metadata_multigpu.py -m multi_gpu -v
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from collections import Counter
from typing import Any

import pytest
import torch

pytest.importorskip("flydsl._mlir", reason="FlyDSL MLIR stack required")

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLYDSL_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

_SUCCESS_BANNER = "MoE intranode multi-GPU CROSS_RANK_METADATA_OK"


def _count_physical_gpus() -> int:
    """Physical GPU count (subprocess to bypass HIP_VISIBLE_DEVICES)."""
    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:
        return 0


def setup_distributed(rank: int, world_size: int, master_port: int) -> tuple[int, int]:
    import mori.shmem as ms

    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(master_port),
            }
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=dev,
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", torch.distributed.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return local_rank, world_size


def cleanup() -> None:
    import mori.shmem as ms

    try:
        ms.shmem_finalize()
    except Exception:
        pass
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        pass


def _expected_recv_meta_keys(
    *,
    recv_rank: int,
    npes: int,
    epr: int,
    ept: int,
    mtpr: int,
    cur_tok: int,
    idx_fn: Any,
    wts_fn: Any,
) -> Counter[tuple[int, int, int, int, int, int]]:
    """Multiset of recv_meta rows (no ``dst_slot``).

    Matches ``ep_metadata_dispatch_recv_meta``: routing uses ``idx`` at flat index
    ``i``; when that destination is ``recv_rank``, **each** ``lane < ept`` may emit one
    row (same ``dest_tok_all`` per ``i``, distinct ``dst_slot``).
    """
    keys: list[tuple[int, int, int, int, int, int]] = []
    limit = cur_tok * ept
    for s in range(npes):
        for i in range(limit):
            src_tok = i // ept
            route_col = i % ept
            dest_exp_route = int(idx_fn(s, src_tok, route_col))
            dep = dest_exp_route // epr
            if dep != recv_rank:
                continue
            src_enc = s * mtpr + src_tok
            for lane in range(ept):
                ix = int(idx_fn(s, src_tok, lane))
                w = float(wts_fn(s, src_tok, lane))
                w_bits = struct.unpack("<I", struct.pack("<f", w))[0]
                keys.append(
                    (
                        s & 0xFF,
                        lane & 0xFF,
                        src_tok & 0xFFFFFFFF,
                        ix & 0xFFFFFFFF,
                        w_bits & 0xFFFFFFFF,
                        src_enc & 0xFFFFFFFF,
                    )
                )
    return Counter(keys)


def _expected_total_recv_count(
    *,
    recv_rank: int,
    npes: int,
    epr: int,
    ept: int,
    cur_tok: int,
    idx_fn: Any,
) -> int:
    """Kernel ``total_recv`` sums ``recv_cnt = nsig - 1`` once per remote handshake.

    ``dest_ctr`` increments once per routing slot ``i`` (lane-0 path), not once per
    emitted recv_meta row, so when ``ept > 1`` there can be multiple rows per slot.
    """
    n = 0
    limit = cur_tok * ept
    for s in range(npes):
        for i in range(limit):
            src_tok = i // ept
            route_col = i % ept
            dest_exp_route = int(idx_fn(s, src_tok, route_col))
            if dest_exp_route // epr == recv_rank:
                n += 1
    return n


def _row_from_gpu_bytes(b: bytes) -> dict[str, Any]:
    from kernels.recv_meta_a8w4 import unpack_recv_meta_row_i32

    i32s = struct.unpack("<8I", b)
    return unpack_recv_meta_row_i32(i32s)


def _recv_meta_row_key(d: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    w_bits = struct.unpack("<I", struct.pack("<f", float(d["router_w"])))[0]
    return (
        int(d["src_pe"]) & 0xFF,
        int(d["kp"]) & 0xFF,
        int(d["src_token"]) & 0xFFFFFFFF,
        int(d["expert_id"]) & 0xFFFFFFFF,
        w_bits & 0xFFFFFFFF,
        int(d["dest_encode"]) & 0xFFFFFFFF,
    )


def _reset_state(
    *,
    dev: torch.device,
    dest_pe_ctr: torch.Tensor,
    disp_bar: torch.Tensor,
    total_recv: torch.Tensor,
    dest_tok_map: torch.Tensor,
    shmem_recv_num: torch.Tensor,
    shmem_tok_off: torch.Tensor,
    shmem_tis: torch.Tensor,
    shmem_recv_meta: torch.Tensor,
    sentinel: int,
) -> None:
    dest_pe_ctr.fill_(0)
    disp_bar.fill_(0)
    total_recv.fill_(0)
    dest_tok_map.fill_(sentinel)
    shmem_recv_num.fill_(0)
    shmem_tok_off.fill_(0)
    shmem_tis.fill_(0)
    shmem_recv_meta.fill_(0)


def _run_dispatch_case(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    dev: torch.device,
    epr: int,
    ept: int,
    mtpr: int,
    cur_tok: int,
    block_num: int,
    warp_num_per_block: int,
    idx_fn: Any,
    wts_fn: Any,
    case_tag: str,
) -> None:
    import flydsl.expr as fx
    import mori.shmem as ms
    import torch.distributed as dist

    from kernels.moe_metadata_dispatch_recv_meta import make_metadata_dispatch_recv_meta_jit
    from kernels.recv_meta_a8w4 import RECV_META_ROW_BYTES

    npes = world_size
    max_recv = npes * mtpr
    sentinel = npes * max_recv

    shmem_tok_off = ms.mori_shmem_create_tensor((1,), torch.int32)
    shmem_recv_num = ms.mori_shmem_create_tensor((npes,), torch.int32)
    shmem_tis = ms.mori_shmem_create_tensor((max_recv,), torch.int32)
    shmem_recv_meta = ms.mori_shmem_create_tensor((max_recv * RECV_META_ROW_BYTES,), torch.int8)
    ms.shmem_barrier_all()

    dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=dev)
    disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
    dest_tok_map = torch.full((mtpr * ept,), sentinel, dtype=torch.int32, device=dev)

    p2p_tok_off = torch.zeros(npes, dtype=torch.int64, device=dev)
    p2p_tis = torch.zeros(npes, dtype=torch.int64, device=dev)
    p2p_recv_num = torch.zeros(npes, dtype=torch.int64, device=dev)
    p2p_recv_meta = torch.zeros(npes, dtype=torch.int64, device=dev)
    for pe in range(npes):
        p2p_tok_off[pe] = ms.shmem_ptr_p2p(shmem_tok_off.data_ptr(), rank, pe)
        p2p_tis[pe] = ms.shmem_ptr_p2p(shmem_tis.data_ptr(), rank, pe)
        p2p_recv_num[pe] = ms.shmem_ptr_p2p(shmem_recv_num.data_ptr(), rank, pe)
        p2p_recv_meta[pe] = ms.shmem_ptr_p2p(shmem_recv_meta.data_ptr(), rank, pe)

    jit = make_metadata_dispatch_recv_meta_jit(
        rank=rank,
        npes=npes,
        experts_per_rank=epr,
        experts_per_token=ept,
        max_tok_per_rank=mtpr,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )

    idx = torch.zeros((mtpr, ept), dtype=torch.int32, device=dev)
    wts = torch.zeros((mtpr, ept), dtype=torch.float32, device=dev)
    for t in range(cur_tok):
        for k in range(ept):
            idx[t, k] = int(idx_fn(rank, t, k))
            wts[t, k] = float(wts_fn(rank, t, k))

    _reset_state(
        dev=dev,
        dest_pe_ctr=dest_pe_ctr,
        disp_bar=disp_bar,
        total_recv=total_recv,
        dest_tok_map=dest_tok_map,
        shmem_recv_num=shmem_recv_num,
        shmem_tok_off=shmem_tok_off,
        shmem_tis=shmem_tis,
        shmem_recv_meta=shmem_recv_meta,
        sentinel=sentinel,
    )
    ms.shmem_barrier_all()

    jit(
        fx.Int64(idx.data_ptr()),
        fx.Int64(wts.data_ptr()),
        fx.Int64(shmem_tok_off.data_ptr()),
        fx.Int64(shmem_recv_num.data_ptr()),
        fx.Int64(dest_pe_ctr.data_ptr()),
        fx.Int64(disp_bar.data_ptr()),
        fx.Int64(dest_tok_map.data_ptr()),
        fx.Int64(total_recv.data_ptr()),
        fx.Int64(p2p_tok_off.data_ptr()),
        fx.Int64(p2p_tis.data_ptr()),
        fx.Int64(p2p_recv_num.data_ptr()),
        fx.Int64(p2p_recv_meta.data_ptr()),
        fx.Int32(cur_tok),
        stream=fx.Stream(None),
    )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    exp_ctr = _expected_recv_meta_keys(
        recv_rank=rank,
        npes=npes,
        epr=epr,
        ept=ept,
        mtpr=mtpr,
        cur_tok=cur_tok,
        idx_fn=idx_fn,
        wts_fn=wts_fn,
    )
    n_rows = int(sum(exp_ctr.values()))
    n_sig = _expected_total_recv_count(
        recv_rank=rank,
        npes=npes,
        epr=epr,
        ept=ept,
        cur_tok=cur_tok,
        idx_fn=idx_fn,
    )
    assert int(total_recv.item()) == n_sig, (
        f"[{case_tag}] rank={rank} total_recv={int(total_recv.item())} expected {n_sig} (handshakes)"
    )

    recv_flat = shmem_recv_meta.cpu().numpy().tobytes()
    tis_cpu = shmem_tis.cpu()
    got_rows: list[dict[str, Any]] = []
    for slot in range(max_recv):
        off = slot * RECV_META_ROW_BYTES
        chunk = recv_flat[off : off + RECV_META_ROW_BYTES]
        if chunk == bytes(RECV_META_ROW_BYTES):
            continue
        got = _row_from_gpu_bytes(chunk)
        got_rows.append(got)
    assert len(got_rows) == n_rows, (
        f"[{case_tag}] rank={rank} non-empty recv_meta rows={len(got_rows)} expected {n_rows}"
    )
    got_ctr = Counter(_recv_meta_row_key(r) for r in got_rows)
    assert got_ctr == exp_ctr, f"[{case_tag}] rank={rank} recv_meta multiset mismatch"

    for got in got_rows:
        dst_slot = int(got["dst_slot"]) & 0xFFFFFFFF
        dest_tok_all = dst_slot // ept
        ti = int(tis_cpu[dest_tok_all].item()) & 0xFFFFFFFF
        assert ti == (int(got["dest_encode"]) & 0xFFFFFFFF), (
            f"[{case_tag}] rank={rank} tis[{dest_tok_all}]={ti} row={got}"
        )

    for i in range(cur_tok * ept):
        got_map = int(dest_tok_map[i].item()) & 0xFFFFFFFF
        assert got_map != (sentinel & 0xFFFFFFFF), (
            f"[{case_tag}] rank={rank} dest_tok_map[{i}] should not stay sentinel"
        )
        src_tok = i // ept
        k = i % ept
        dest_exp = int(idx_fn(rank, src_tok, k))
        dep = dest_exp // epr
        slot = got_map - dep * max_recv
        assert got_map // max_recv == dep, (
            f"[{case_tag}] rank={rank} dest_tok_map[{i}]={got_map} dep={dep} max_recv={max_recv}"
        )
        assert 0 <= slot < mtpr, (
            f"[{case_tag}] rank={rank} dest slot {slot} out of range mtpr={mtpr}"
        )
    for i in range(cur_tok * ept, mtpr * ept):
        got_map = int(dest_tok_map[i].item()) & 0xFFFFFFFF
        assert got_map == (sentinel & 0xFFFFFFFF), (
            f"[{case_tag}] rank={rank} dest_tok_map[{i}] should stay sentinel"
        )

    assert bool((shmem_recv_num == 0).all().item()), f"[{case_tag}] rank={rank} recv_num not cleared"

    dist.barrier()


def main() -> None:
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_port = int(os.environ.get("MASTER_PORT", "29850"))
    local_rank, ws = setup_distributed(rank, world_size, master_port)
    assert ws == world_size
    dev = torch.device("cuda", local_rank)

    try:
        t = torch.full((4,), float(rank + 1), device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expect = sum(i + 1 for i in range(world_size)) * torch.ones_like(t)
        assert torch.allclose(t, expect), (t, expect)

        epr = 1
        mtpr = 8
        block_num = 1
        wpb = 1

        def idx_ring(r: int, tok: int, k: int) -> int:
            dest_pe = (r + tok + k + 1) % world_size
            return dest_pe * epr

        def wts_a(r: int, tok: int, k: int) -> float:
            return 0.25 + 0.01 * r + 0.001 * tok + 0.0001 * k

        _run_dispatch_case(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            dev=dev,
            epr=epr,
            ept=1,
            mtpr=mtpr,
            cur_tok=1,
            block_num=block_num,
            warp_num_per_block=wpb,
            idx_fn=idx_ring,
            wts_fn=wts_a,
            case_tag="ring_ept1_tok1",
        )

        _run_dispatch_case(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            dev=dev,
            epr=epr,
            ept=1,
            mtpr=mtpr,
            cur_tok=min(3, mtpr),
            block_num=block_num,
            warp_num_per_block=wpb,
            idx_fn=idx_ring,
            wts_fn=wts_a,
            case_tag="ring_ept1_tokN",
        )

        if world_size >= 3:
            ept2 = 2

            def idx_two_dest(r: int, tok: int, k: int) -> int:
                d0 = (r + tok + 1) % world_size
                d1 = (r + tok + 2) % world_size
                return (d0 if k == 0 else d1) * epr

            _run_dispatch_case(
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                dev=dev,
                epr=epr,
                ept=ept2,
                mtpr=mtpr,
                cur_tok=2,
                block_num=block_num,
                warp_num_per_block=wpb,
                idx_fn=idx_two_dest,
                wts_fn=wts_a,
                case_tag="two_dest_ept2",
            )

        if rank == 0:
            print(
                f"{_SUCCESS_BANNER}: world_size={world_size}, "
                "recv_meta + TIS + total_recv + tok_map + recv_num",
                flush=True,
            )
    finally:
        cleanup()


def _run_torchrun_smoke(*, world_size: int, timeout: int = 600) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    port = 29850 + (os.getpid() % 3000)
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        f"--master_port={port}",
        __file__,
    ]
    return subprocess.run(
        cmd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


@pytest.mark.multi_gpu
def test_moe_intranode_metadata_8gpu_cross_rank_accuracy():
    phys = _count_physical_gpus()
    if phys < 8:
        pytest.skip(f"Requires >= 8 physical GPUs, found {phys}.")
    result = _run_torchrun_smoke(world_size=8)
    assert result.returncode == 0, (
        f"8-GPU metadata cross-rank FAILED (exit {result.returncode})\n"
        f"stdout (last 6000 chars):\n{result.stdout[-6000:]}\n"
        f"stderr (last 6000 chars):\n{result.stderr[-6000:]}"
    )
    assert _SUCCESS_BANNER in result.stdout, result.stdout[-6000:]


@pytest.mark.multi_gpu
def test_moe_intranode_metadata_4gpu_cross_rank_accuracy():
    phys = _count_physical_gpus()
    if phys < 4:
        pytest.skip(f"Requires >= 4 physical GPUs, found {phys}.")
    result = _run_torchrun_smoke(world_size=4)
    assert result.returncode == 0, (
        f"4-GPU metadata cross-rank FAILED (exit {result.returncode})\n"
        f"stdout (last 6000 chars):\n{result.stdout[-6000:]}\n"
        f"stderr (last 6000 chars):\n{result.stderr[-6000:]}"
    )
    assert _SUCCESS_BANNER in result.stdout, result.stdout[-6000:]


if __name__ == "__main__":
    if "RANK" not in os.environ:
        sys.stderr.write(
            "Run under torchrun, e.g.:\n"
            "  MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 "
            f"{__file__}\n"
        )
        sys.exit(2)
    main()
