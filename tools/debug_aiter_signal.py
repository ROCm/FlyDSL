#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Align with tests' embedded flydsl path logic.
_repo = Path(__file__).resolve().parents[1]
_embedded = _repo / "build" / "python_packages" / "flydsl"
_embedded2 = _repo / ".flir" / "build" / "python_packages" / "flydsl"
_pick = _embedded if _embedded.exists() else _embedded2
if _pick.exists():
    os.environ.setdefault("FLYDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_pick))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kernels.custom_all_reduce import init_custom_ar
from flydsl.runtime import ipc as fly_ipc


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = int(s.getsockname()[1])
    s.close()
    return p


def _dump_meta_debug(meta: torch.Tensor, *, meta_size_bytes: int):
    # start debug at meta_size+64; end debug at meta_size+128
    # Each slot is 16B (4 u32). We write [tag, target, last_loaded].
    i32 = meta.view(torch.int32)
    base0 = (meta_size_bytes + 64) // 4
    base1 = (meta_size_bytes + 128) // 4
    # Print a few slots: bid=0 lane=0..3
    def slot(base, lane):
        off = base + lane * 4
        return i32[off : off + 4].detach().cpu().tolist()

    print("  [dbg start] lane0..3:", [slot(base0, i) for i in range(4)], flush=True)
    print("  [dbg end  ] lane0..3:", [slot(base1, i) for i in range(4)], flush=True)


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Inputs
    x = torch.randn((128, 8192), device=device, dtype=torch.float16).contiguous().reshape(-1)

    # Exchange input IPC handles (same as test harness)
    my_ipc = fly_ipc.get_ipc_handle(x)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (my_ipc.handle, int(my_ipc.offset_bytes)), group=dist.group.WORLD)
    handles = [torch.tensor(list(hb), device="cpu", dtype=torch.uint8) for hb, _ in gathered]
    offsets = [int(off) for _, off in gathered]

    # Force our path
    os.environ["FLYDSL_CUSTOM_ALL_REDUCE_BACKEND"] = "aiter"
    os.environ["FLYDSL_AITER_IMPL"] = "flydsl"

    meta = torch.empty((0,), device=device, dtype=torch.int8)
    fa = init_custom_ar(meta, x, handles, offsets, rank=rank, full_nvlink=True)

    out = torch.empty_like(x)
    fa.all_reduce_reg(x, out, open_fp8_quant=False)
    torch.cuda.synchronize()

    # Print error vs reference on each rank
    gathered_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered_x, x)
    ref = torch.zeros_like(x, dtype=torch.float32)
    for t in gathered_x:
        ref += t.to(torch.float32)
    max_err = (out.to(torch.float32) - ref).abs().max().item()
    max_err_local_only = (out.to(torch.float32) - x.to(torch.float32)).abs().max().item()
    if world_size == 2:
        peer = gathered_x[1 - rank]
        max_err_peer_only = (out.to(torch.float32) - (x.to(torch.float32) + peer.to(torch.float32))).abs().max().item()
    else:
        max_err_peer_only = float("nan")

    # Dump debug
    if hasattr(fa, "meta"):
        meta_sz = int(getattr(fa, "_meta_size", 5504))
        print(
            f"[rank={rank}] max_err={max_err:.6f} "
            f"max_err_local_only={max_err_local_only:.6f} "
            f"max_err_peer_only={max_err_peer_only:.6f} "
            f"meta_size={meta_sz}",
            flush=True,
        )
        _dump_meta_debug(fa.meta, meta_size_bytes=meta_sz)
    else:
        print(
            f"[rank={rank}] max_err={max_err:.6f} max_err_local_only={max_err_local_only:.6f} (no fa.meta)",
            flush=True,
        )

    dist.destroy_process_group()


def main():
    ws = int(os.environ.get("WS", "2"))
    # Reduce spin to make timeout+debug trigger quickly if protocol is broken.
    os.environ.setdefault("FLYDSL_AITER_SIGNAL_MAX_SPIN", "2000000")
    port = _free_port()
    mp.spawn(worker, args=(ws, port), nprocs=ws, join=True)


if __name__ == "__main__":
    main()

