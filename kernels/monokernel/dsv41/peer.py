# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The TP group's symmetric peer buffer on gfx1250: HIP virtual memory, not HIP
IPC. ``hipIpcGetMemHandle`` / ``hipIpcOpenMemHandle`` are not available on
gfx1250 (aiter ``vmm_allocator``; ``runtime.SymmetricPeerBuffer`` hangs in its
exchange there), so each rank ``hipMemCreate``s its buffer, exports it as a POSIX
fd, passes the fd to every peer over a Unix socket (SCM_RIGHTS) and maps the
peers' fds into its own address space.

    with VmmPeerBuffer(nbytes, rank, npes) as buf:
        buf.addresses   # int64 device tensor: every rank's base, as this rank sees it
        buf.local_address
"""

import array
import ctypes
import os
import socket

import torch

_PINNED, _POSIX_FD, _DEVICE, _READ_WRITE, _RECOMMENDED = 1, 1, 1, 3, 1
_D2H = 2
_VP, _SZ = ctypes.c_void_p, ctypes.c_size_t


class _Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("win32_handle", ctypes.c_void_p),
        ("win32_name", ctypes.c_void_p),
        ("compression", ctypes.c_ubyte),
        ("rdma", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("_pad", ctypes.c_byte * 4),
    ]


class _Access(ctypes.Structure):
    _fields_ = [("location_type", ctypes.c_int), ("location_id", ctypes.c_int), ("flags", ctypes.c_int)]


def _hip():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise OSError("no HIP runtime library")
    sig = {
        "hipMemGetAllocationGranularity": [ctypes.POINTER(_SZ), ctypes.POINTER(_Prop), ctypes.c_int],
        "hipMemCreate": [ctypes.POINTER(_VP), _SZ, ctypes.POINTER(_Prop), ctypes.c_ulonglong],
        "hipMemAddressReserve": [ctypes.POINTER(_VP), _SZ, _SZ, _VP, ctypes.c_ulonglong],
        "hipMemMap": [_VP, _SZ, _SZ, _VP, ctypes.c_ulonglong],
        "hipMemSetAccess": [_VP, _SZ, ctypes.POINTER(_Access), _SZ],
        "hipMemExportToShareableHandle": [_VP, _VP, ctypes.c_int, ctypes.c_ulonglong],
        "hipMemImportFromShareableHandle": [ctypes.POINTER(_VP), _VP, ctypes.c_int],
        "hipMemUnmap": [_VP, _SZ],
        "hipMemAddressFree": [_VP, _SZ],
        "hipMemRelease": [_VP],
        "hipMemset": [_VP, ctypes.c_int, _SZ],
        "hipMemcpy": [_VP, _VP, _SZ, ctypes.c_int],
        "hipMemsetAsync": [_VP, ctypes.c_int, _SZ, _VP],
        "hipDeviceSynchronize": [],
    }
    for fn, args in sig.items():
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = ctypes.c_int
    return lib


_HIP = None


def _call(fn, *args):
    global _HIP
    if _HIP is None:
        _HIP = _hip()
    err = getattr(_HIP, fn)(*args)
    if err != 0:
        raise RuntimeError(f"{fn} failed: hipError {err}")


def memset_async(ptr: int, nbytes: int, stream) -> None:
    """Zero ``nbytes`` at ``ptr`` on ``stream`` (a torch stream): captured into a
    graph as a memset node."""
    _call("hipMemsetAsync", _VP(ptr), 0, nbytes, _VP(stream.cuda_stream))


def _prop(dev):
    p = _Prop()
    p.type, p.requestedHandleType, p.location_type, p.location_id = _PINNED, _POSIX_FD, _DEVICE, dev
    return p


class _Mapping:
    """A physical allocation (own, or imported from a peer's fd) mapped here."""

    def __init__(self, size, dev, fd=None):
        p = _prop(dev)
        gran = _SZ()
        _call("hipMemGetAllocationGranularity", ctypes.byref(gran), ctypes.byref(p), _RECOMMENDED)
        self.size = -(-size // gran.value) * gran.value
        self.handle = _VP()
        if fd is None:
            _call("hipMemCreate", ctypes.byref(self.handle), self.size, ctypes.byref(p), 0)
        else:
            _call("hipMemImportFromShareableHandle", ctypes.byref(self.handle), _VP(fd), _POSIX_FD)
        self.va = _VP()
        _call("hipMemAddressReserve", ctypes.byref(self.va), self.size, gran.value, None, 0)
        _call("hipMemMap", self.va, self.size, 0, self.handle, 0)
        acc = _Access(_DEVICE, dev, _READ_WRITE)
        _call("hipMemSetAccess", self.va, self.size, ctypes.byref(acc), 1)

    def export_fd(self) -> int:
        fd = ctypes.c_int()
        _call("hipMemExportToShareableHandle", ctypes.byref(fd), self.handle, _POSIX_FD, 0)
        return fd.value

    def close(self):
        if self.va.value:
            _call("hipMemUnmap", self.va, self.size)
            _call("hipMemAddressFree", self.va, self.size)
            self.va = _VP()
        if self.handle.value:
            _call("hipMemRelease", self.handle)
            self.handle = _VP()


def _send_fd(path, fd):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(60)
        s.connect(path)
        s.sendmsg([b"\x00"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])


def _recv_fd(server) -> int:
    conn, _ = server.accept()
    with conn:
        _, anc, _, _ = conn.recvmsg(1, socket.CMSG_LEN(ctypes.sizeof(ctypes.c_int)))
    for level, kind, data in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(data[: ctypes.sizeof(ctypes.c_int)])
            return fds[0]
    raise RuntimeError("no fd received")


class VmmPeerBuffer:
    """``nbytes`` zeroed on this rank's device, mapped by every rank of the group;
    ``addresses[p]``: rank p's buffer as this rank addresses it."""

    def __init__(self, nbytes: int, rank: int, npes: int, group=None):
        import torch.distributed as dist

        self.rank, self.npes, self.group = rank, npes, group
        dev = torch.cuda.current_device()
        self.own = _Mapping(nbytes, dev)
        self.nbytes = nbytes
        _call("hipMemset", self.own.va, 0, self.own.size)
        _call("hipDeviceSynchronize")
        self.local_address = int(self.own.va.value)
        self.peers = []
        addresses = [self.local_address] * npes
        if npes > 1:
            tag = [f"/tmp/dsv41_vmm_{os.getpid()}_{id(self)}"] if rank == 0 else [None]
            dist.broadcast_object_list(tag, src=0, group=group)
            servers = {}
            for src in range(npes):
                if src != rank:
                    path = f"{tag[0]}_r{rank}_from_r{src}"
                    if os.path.exists(path):
                        os.unlink(path)
                    sv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sv.bind(path)
                    sv.listen(1)
                    sv.settimeout(60)
                    servers[src] = (sv, path)
            sizes = [None] * npes
            dist.all_gather_object(sizes, self.own.size, group=group)
            dist.barrier(group=group)  # every rank listening
            fd = self.own.export_fd()
            try:
                for dst in range(npes):
                    if dst != rank:
                        _send_fd(f"{tag[0]}_r{dst}_from_r{rank}", fd)
            finally:
                os.close(fd)
            for src, (sv, path) in servers.items():
                with sv:
                    got = _recv_fd(sv)
                os.unlink(path)
                try:
                    m = _Mapping(sizes[src], dev, fd=got)
                finally:
                    os.close(got)
                self.peers.append(m)
                addresses[src] = int(m.va.value)
            dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device="cuda")

    def to_host(self) -> torch.Tensor:
        """This rank's buffer, copied to a CPU uint8 tensor."""
        out = torch.empty(self.nbytes, dtype=torch.uint8)
        _call("hipMemcpy", _VP(out.data_ptr()), self.own.va, self.nbytes, _D2H)
        return out

    def close(self):
        torch.cuda.synchronize()
        if self.npes > 1:
            import torch.distributed as dist

            dist.barrier(group=self.group)
        for m in self.peers:
            m.close()
        self.peers = []
        self.own.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CcoPeerBuffer:
    """The same buffer through mori's cco symmetric memory, as aiter's gfx1250
    MegaMoE builds its arena (``Communicator.init`` from rank 0's unique id,
    ``alloc_mem`` + ``register_window``): every rank's allocation sits in one
    flat VA, rank r's slot at ``flat_base + r per_rank_size``, so peer p's copy
    of this rank's buffer is at the same offset in slot p (what the device-side
    ``lsa_ptr(p, off)`` computes)."""

    PER_RANK_VMM = 1 << 30

    def __init__(self, nbytes: int, rank: int, npes: int, group=None):
        import torch.distributed as dist
        from mori.cco import Communicator

        self.rank, self.npes, self.group, self.nbytes = rank, npes, group, nbytes
        uid = [Communicator.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=0, group=group)
        self.comm = Communicator.init(npes, rank, uid[0], per_rank_vmm=self.PER_RANK_VMM)
        self.mem = self.comm.alloc_mem(nbytes)
        self.window = self.comm.register_window(self.mem.ptr, nbytes)
        self.dev = self.comm.create_dev_comm()
        base, per, me = self.dev.flat_base, self.dev.per_rank_size, self.dev.lsa_rank
        self.local_address = int(self.window.local_ptr)
        off = self.local_address - (base + me * per)
        if not 0 <= off < per:
            raise RuntimeError(f"cco: local buffer {self.local_address:#x} outside its flat slot {base + me * per:#x}+{per:#x}")
        _call("hipMemset", _VP(self.local_address), 0, nbytes)
        _call("hipDeviceSynchronize")
        self.comm.barrier()
        self.addresses = torch.tensor([base + p * per + off for p in range(npes)], dtype=torch.int64, device="cuda")

    def to_host(self) -> torch.Tensor:
        out = torch.empty(self.nbytes, dtype=torch.uint8)
        _call("hipMemcpy", _VP(out.data_ptr()), _VP(self.local_address), self.nbytes, _D2H)
        return out

    def close(self):
        torch.cuda.synchronize()
        self.comm.barrier()
        self.comm.close_all_resources()
        self.comm.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
