#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Pytest configuration for the cooperative-algorithm tests."""

import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.fixture(autouse=True)
def host_default_device():
    """Keep the default device on the host, whatever the rest of the session set."""
    if torch is None:
        yield
        return
    with torch.device("cpu"):
        yield
