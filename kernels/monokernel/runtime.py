# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Owned HIP IPC peer memory used by fused multi-GPU kernels."""

from __future__ import annotations

import torch

from kernels.monokernel.ipc import close_ipc_handle, get_allocation_base, get_ipc_handle, open_ipc_handle


class SymmetricPeerBuffer:
    """Allocate one symmetric buffer and exchange its address with every rank.

    Remote HIP IPC mappings are closed by :meth:`close`. A private allocator pool
    keeps exported storage separate from ordinary tensor allocations. Both the
    pool and ``storage`` stay alive for the wrapper's lifetime.
    """

    def __init__(self, nbytes: int, rank: int = 0, npes: int = 1, group=None):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        device = torch.device("cuda", torch.cuda.current_device())
        self.rank = rank
        self.npes = npes
        self.group = group
        self._pool = None
        if npes > 1:
            # Exporting suballocations of large, reused tensor segments can give
            # stale HIP IPC mappings on ROCm. Keep peer memory in its own pool;
            # weights, scratch, and graph allocations must not share its segment.
            self._pool = torch.cuda.MemPool()
            with torch.cuda.use_mem_pool(self._pool):
                self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        else:
            self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.local_address = self.storage.data_ptr()
        self._remote_bases: list[int] = []
        self._safety_barrier_complete = False
        self._closed = False

        if npes == 1:
            addresses = [self.local_address]
        else:
            import torch.distributed as dist

            base = get_allocation_base(self.local_address)
            mine = (get_ipc_handle(base), self.local_address - base)
            peers = [None] * npes
            dist.all_gather_object(peers, mine, group=group)
            addresses = []
            try:
                for peer_rank, (handle, offset) in enumerate(peers):
                    if peer_rank == rank:
                        addresses.append(self.local_address)
                    else:
                        remote_base = open_ipc_handle(handle)
                        self._remote_bases.append(remote_base)
                        addresses.append(remote_base + offset)
            except Exception:
                for remote_base in self._remote_bases:
                    try:
                        close_ipc_handle(remote_base)
                    except Exception:
                        pass
                self._remote_bases.clear()
                raise
            dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Synchronize every rank, then collectively close remote mappings.

        Every rank in the peer group must call this method in the same order.
        Repeated calls are safe. Failed closes remain tracked; every rank must
        retry if any rank fails. Storage can be released after all ranks succeed.
        """

        if self._closed:
            return
        if not self._safety_barrier_complete:
            torch.cuda.synchronize(self.storage.device)
            if self.npes > 1:
                import torch.distributed as dist

                if not dist.is_initialized():
                    raise RuntimeError("the distributed process group must remain initialized until peer buffers close")
                dist.barrier(group=self.group)
            self._safety_barrier_complete = True
        failed = []
        first_error = None
        for base in self._remote_bases:
            try:
                close_ipc_handle(base)
            except Exception as exc:
                failed.append(base)
                if first_error is None:
                    first_error = exc
        self._remote_bases = failed
        if self.npes > 1:
            import torch.distributed as dist

            # A fast rank must not release its pool while peers still have its
            # allocation mapped. Propagate failures so all ranks retain storage
            # and participate in a retry, including ranks whose closes succeeded.
            errors = [None] * self.npes
            dist.all_gather_object(errors, str(first_error) if first_error is not None else None, group=self.group)
            if first_error is None and any(error is not None for error in errors):
                first_error = RuntimeError(f"peer HIP IPC close failed: {errors}")
        if first_error is not None:
            raise first_error
        self._closed = True

    def __enter__(self) -> SymmetricPeerBuffer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
