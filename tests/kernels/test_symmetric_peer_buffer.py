# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""HIP IPC storage isolation, peer reads, and collective close/retry."""

import ctypes
import datetime

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kernels.monokernel import runtime
from kernels.monokernel.ipc import get_allocation_base, load_runtime

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower, pytest.mark.multi_gpu]


def _check_peers(buffer, iteration, slot):
    buffer.storage.fill_(iteration * 16 + buffer.rank * 2 + slot)
    torch.cuda.synchronize()
    dist.barrier()
    hip = load_runtime()
    hip.hipMemcpy.restype = ctypes.c_int
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    for peer, address in enumerate(buffer.addresses.cpu().tolist()):
        for offset in (0, buffer.storage.numel() // 2, buffer.storage.numel() - 1):
            value = ctypes.c_ubyte()
            status = hip.hipMemcpy(ctypes.byref(value), ctypes.c_void_p(address + offset), 1, 2)
            assert status == 0, (peer, offset, status)
            assert value.value == iteration * 16 + peer * 2 + slot, (peer, offset, value.value)
    dist.barrier()


def _run_peer_buffers(rank, npes, init_method):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=npes,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        for iteration in range(8):
            # Prime a large ordinary segment, then retain a prefix so a normal
            # tensor allocation would become a nonzero-offset suballocation.
            ordinary = torch.empty(62 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            del ordinary
            prefix = torch.empty(40498688, dtype=torch.uint8, device="cuda")
            sizes = (512, 3 * 1024 * 1024) if iteration % 2 == 0 else (1572864, 13369344)
            buffers = [runtime.SymmetricPeerBuffer(size, rank, npes) for size in sizes]
            bases = [get_allocation_base(buffer.local_address) for buffer in buffers]
            assert len(set(bases + [get_allocation_base(prefix.data_ptr())])) == 3
            for slot, buffer in enumerate(buffers):
                _check_peers(buffer, iteration, slot)

            if iteration == 0:
                # One rank fails once. All ranks must retain their storage and
                # be able to retry, even when their own mappings already closed.
                original_close = runtime.close_ipc_handle
                failed_once = False

                def fail_once(base):
                    nonlocal failed_once
                    if rank == 0 and not failed_once:
                        failed_once = True
                        raise RuntimeError("injected IPC close failure")
                    original_close(base)

                runtime.close_ipc_handle = fail_once
                try:
                    with pytest.raises(RuntimeError, match="injected IPC close failure"):
                        buffers[-1].close()
                finally:
                    runtime.close_ipc_handle = original_close

            # Release each pool immediately after close, without a caller-side
            # barrier. Closing one buffer must leave its sibling readable.
            while buffers:
                buffer = buffers.pop()
                buffer.close()
                buffer.close()
                del buffer
                if buffers:
                    _check_peers(buffers[0], iteration, 0)
            del prefix
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("npes", [2, 8])
def test_symmetric_peer_buffer(npes, tmp_path):
    if not torch.version.hip or torch.cuda.device_count() < npes:
        pytest.skip(f"requires {npes} ROCm GPUs")
    mp.spawn(_run_peer_buffers, args=(npes, f"file://{tmp_path / 'store'}"), nprocs=npes)
