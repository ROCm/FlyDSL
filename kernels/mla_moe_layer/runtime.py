# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Owned HIP IPC peer memory used by persistent multi-GPU kernels."""

from __future__ import annotations

import torch

from kernels.comm.custom_all_reduce import FlyDSLAllreduce as _Hip


class SymmetricPeerBuffer:
    """Allocate one symmetric buffer and exchange its address with every rank.

    Remote HIP IPC mappings are closed by :meth:`close`. The local allocation is
    owned by ``storage`` and stays alive for the wrapper's lifetime.
    """

    def __init__(self, nbytes: int, rank: int = 0, npes: int = 1, group=None):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        device = torch.device("cuda", torch.cuda.current_device())
        self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.local_address = self.storage.data_ptr()
        self._remote_bases: list[int] = []

        if npes == 1:
            addresses = [self.local_address]
        else:
            import torch.distributed as dist

            base = _Hip._get_alloc_base_ptr(self.local_address)
            mine = (_Hip._get_mem_handle_bytes(base), self.local_address - base)
            peers = [None] * npes
            dist.all_gather_object(peers, mine, group=group)
            addresses = []
            for peer_rank, (handle, offset) in enumerate(peers):
                if peer_rank == rank:
                    addresses.append(self.local_address)
                else:
                    remote_base = _Hip._open_mem_handle(handle)
                    self._remote_bases.append(remote_base)
                    addresses.append(remote_base + offset)
            dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Close remote mappings; repeated calls are safe."""

        bases, self._remote_bases = self._remote_bases, []
        for base in bases:
            _Hip._close_mem_handle(base)

    def __enter__(self) -> SymmetricPeerBuffer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
