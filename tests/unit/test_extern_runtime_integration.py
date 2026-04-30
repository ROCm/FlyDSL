# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import threading

from flydsl.compiler.jit_executor import _pack_ciface_args
from flydsl.compiler.kernel_function import CompilationContext


def test_explicit_module_loader_args_follow_ciface_packing():
    module = ctypes.c_void_p()
    err = ctypes.c_int32()

    packed = _pack_ciface_args(module, err)
    packed_addr_ptr = ctypes.cast(packed, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    slots = ctypes.cast(packed_addr_ptr.contents, ctypes.POINTER(ctypes.c_void_p))

    assert slots[0] == ctypes.addressof(module)
    assert slots[1] == ctypes.addressof(err)
    assert getattr(packed, "_keepalive")


def test_compilation_context_current_is_thread_local():
    barrier = threading.Barrier(2)
    results = []

    def worker():
        with CompilationContext.create() as ctx:
            barrier.wait(timeout=5)
            results.append(CompilationContext.get_current() is ctx)
            barrier.wait(timeout=5)
        results.append(CompilationContext.get_current() is None)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(results) == 4
    assert all(results)


def test_link_libs_preserve_first_use_order_and_dedupe():
    ctx = CompilationContext()

    ctx.add_link_lib("/tmp/b.bc")
    ctx.add_link_lib("/tmp/a.bc")
    ctx.add_link_lib("/tmp/b.bc")

    assert ctx.link_libs == ["/tmp/b.bc", "/tmp/a.bc"]
