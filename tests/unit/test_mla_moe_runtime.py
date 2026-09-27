# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from types import SimpleNamespace

import pytest
import torch

import kernels.mla_moe_layer.runtime as runtime_module
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

pytestmark = pytest.mark.l0_backend_agnostic


def test_close_synchronizes_peers_and_retries_only_failed_mappings(monkeypatch):
    events = []
    fail_once = {0x1000}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("synchronize", device)))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: events.append(("barrier", group)))

    def close_mapping(base):
        events.append(("close", base))
        if base in fail_once:
            fail_once.remove(base)
            raise RuntimeError("transient close failure")

    monkeypatch.setattr(runtime_module, "close_ipc_handle", close_mapping)
    buffer = object.__new__(SymmetricPeerBuffer)
    buffer.storage = SimpleNamespace(device="cuda:3")
    buffer.npes = 3
    buffer.group = "tp-group"
    buffer._remote_bases = [0x1000, 0x2000]
    buffer._safety_barrier_complete = False

    with pytest.raises(RuntimeError, match="transient close failure"):
        buffer.close()

    assert buffer._remote_bases == [0x1000]
    assert events == [
        ("synchronize", "cuda:3"),
        ("barrier", "tp-group"),
        ("close", 0x1000),
        ("close", 0x2000),
    ]

    buffer.close()
    assert buffer._remote_bases == []
    assert events[-1:] == [("close", 0x1000)]
