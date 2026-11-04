# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side coverage for the monokernel peer-memory plumbing.

``kernels/monokernel/ipc.py`` (a ctypes wrapper over HIP allocation and IPC) and
``kernels/monokernel/layout.py``'s ``symmetric_allreduce_nbytes`` had no test
referencing them anywhere in this repo: they are exercised only through the GLM-5
and Kimi-K3 monokernel tests, which are all ``multi_gpu`` 8-GPU cases and so never
run in the default marker set.

The same HIP IPC contract -- hipExtMallocWithFlags(UNCACHED), hipPointerGetAttribute
(RANGE_START_ADDR), hipIpcGetMemHandle / OpenMemHandle / CloseMemHandle -- is what
aiter's ``quick_allreduce_int4_ipc.UncachedIpcHeap`` implements for vLLM's mono
layer. That is a parallel implementation in another repository, not this module, so
these tests cover this repo's copy of the contract rather than vLLM's import.

Everything here runs on a single GPU; the peer mapping itself needs two processes
and stays with the ``multi_gpu`` tests.
"""

import ctypes

import pytest

_HIP_MEMCPY_DEVICE_TO_HOST = 2


# ---------------------------------------------------------------------------
# Pure host helpers (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_symmetric_allreduce_nbytes_sizes_two_epoch_slots_per_region():
    """Two tagged-mailbox epoch slots per region, each npes * max_pairs * 8 bytes."""
    from kernels.monokernel.layout import symmetric_allreduce_nbytes

    # (1024, 2048), npes=4 -> max_pairs 1024, slot 4*1024*8, 2 regions * 2 slots.
    assert symmetric_allreduce_nbytes((1024, 2048), 4) == 2 * 2 * (4 * 1024 * 8)
    assert symmetric_allreduce_nbytes((1024,), 8) == 1 * 2 * (8 * 512 * 8)

    # The widest region sets max_pairs for every region, so order must not matter.
    assert symmetric_allreduce_nbytes((2048, 1024), 4) == symmetric_allreduce_nbytes((1024, 2048), 4)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("sizes", [(), (0,), (-2,), (1024, 1025)], ids=["empty", "zero", "negative", "odd"])
def test_symmetric_allreduce_nbytes_rejects_invalid_sizes(sizes):
    from kernels.monokernel.layout import symmetric_allreduce_nbytes

    with pytest.raises(ValueError, match="positive and even"):
        symmetric_allreduce_nbytes(sizes, 4)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("npes", [1, 3, 6, 16])
def test_symmetric_allreduce_nbytes_rejects_unsupported_npes(npes):
    from kernels.monokernel.layout import symmetric_allreduce_nbytes

    with pytest.raises(ValueError, match="npes must be one of"):
        symmetric_allreduce_nbytes((1024,), npes)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("nbytes", [0, 10, 63, 65])
def test_open_ipc_handle_rejects_a_wrong_sized_handle(nbytes):
    """Length is validated before HIP is touched, so a truncated handle from a
    peer exchange fails with a readable error instead of a HIP fault."""
    from kernels.monokernel import ipc

    with pytest.raises(ValueError, match=f"expected a {ipc.HIP_IPC_HANDLE_BYTES}-byte HIP IPC handle"):
        ipc.open_ipc_handle(b"\x00" * nbytes)


@pytest.mark.l0_backend_agnostic
def test_check_hip_error_is_a_noop_on_success_and_names_the_operation_on_failure():
    from kernels.monokernel import ipc

    assert ipc.check_hip_error(0, operation="probe") is None

    with pytest.raises(RuntimeError, match=r"^probe_op failed: "):
        ipc.check_hip_error(1, operation="probe_op")


# ---------------------------------------------------------------------------
# HIP host operations (single GPU)
# ---------------------------------------------------------------------------


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_hip_runtime_library_loads_once():
    from kernels.monokernel import ipc

    assert ipc.load_runtime() is ipc.load_runtime()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_allocation_base_is_stable_across_offsets_into_one_allocation():
    """Peers exchange the allocation base, not the tensor pointer, so every offset
    inside one allocation must resolve to the same base."""
    import torch

    from kernels.monokernel import ipc

    tensor = torch.zeros(4096, dtype=torch.int32, device="cuda")
    base = ipc.get_allocation_base(tensor.data_ptr())

    assert base != 0
    assert base <= tensor.data_ptr()
    for offset in (0, 4, 512, 4095):
        assert ipc.get_allocation_base(tensor.data_ptr() + offset) == base


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_ipc_handle_is_exactly_the_declared_handle_size():
    """The handle is broadcast as fixed-width bytes; a size drift would corrupt
    every peer's view of the exchange."""
    import torch

    from kernels.monokernel import ipc

    tensor = torch.zeros(1024, dtype=torch.int32, device="cuda")
    handle = ipc.get_ipc_handle(ipc.get_allocation_base(tensor.data_ptr()))

    assert isinstance(handle, bytes)
    assert len(handle) == ipc.HIP_IPC_HANDLE_BYTES == 64


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_allocate_uncached_zeroes_memory_it_did_not_get_clean():
    """``allocate_uncached`` promises zeroed storage, and the mailbox epoch tags are
    read before they are first written -- a non-zero allocation would make a reduce
    observe a stale completion tag.

    Simply allocating and checking for zeros proves nothing: fresh device pages come
    back zeroed anyway, so that assertion passes even with the hipMemset removed
    (verified). Dirty an allocation, free it, and allocate the same size again. HIP
    hands the freed address straight back, so zeros at a reused address can only come
    from this function's own hipMemset.
    """
    from kernels.monokernel import ipc

    size = 256
    library = ipc.load_runtime()
    # hipMemset/hipMemcpy are not part of the module's configured surface; set them
    # up here purely to dirty and read back the allocation.
    library.hipMemcpy.restype = ctypes.c_int
    library.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    library.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

    def _read_back(pointer):
        host = (ctypes.c_ubyte * size)()
        error = library.hipMemcpy(
            ctypes.byref(host), ctypes.c_void_p(pointer), ctypes.c_size_t(size), _HIP_MEMCPY_DEVICE_TO_HOST
        )
        assert error == 0
        return bytes(host)

    first = ipc.allocate_uncached(size)
    assert first != 0
    assert library.hipMemset(ctypes.c_void_p(first), 0xFF, ctypes.c_size_t(size)) == 0
    assert _read_back(first) == b"\xff" * size, "could not dirty the allocation"
    ipc.free_device_memory(first)

    second = ipc.allocate_uncached(size)
    try:
        if second != first:
            pytest.skip("HIP did not reuse the freed address; zeroing cannot be distinguished here")
        assert _read_back(second) == bytes(size), "allocate_uncached did not zero reused memory"
    finally:
        ipc.free_device_memory(second)


# NOTE: there is deliberately no test that opens a handle this process exported
# itself. hipIpcOpenMemHandle rejects that with hipErrorInvalidContext and leaves
# the HIP context unusable for the rest of the process: every later
# torch.*(device="cuda") in the same pytest session then raises
# "HIP error: invalid device context". An earlier revision of this file did exactly
# that and took tests/unit/test_multi_stream_launch.py down with it in the full
# suite, while passing when run alone. The error-wrapping path it was meant to
# cover is already exercised by
# test_check_hip_error_is_a_noop_on_success_and_names_the_operation_on_failure,
# which drives check() with a non-zero status and changes no HIP state.
