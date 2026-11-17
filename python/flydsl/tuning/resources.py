# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""What a compiled FlyDSL kernel asks of the GPU, read from its code object.

No GPU and no profiler: the numbers come from the ``amdhsa.kernels`` note of
the ELF that FlyDSL embeds in the compiled module (``gpu.binary``, ``bin =
"..."``). The search uses them to reject configs that spill or exceed LDS
before spending any GPU time on them.
"""

from __future__ import annotations

import re
import struct
from typing import Any

__all__ = [
    "code_object",
    "kernel_resources",
    "artifact_resources",
    "occupancy_waves_per_simd",
]

_BIN = re.compile(r'bin\s*=\s*"')
_KERNELS_KEY = b"amdhsa.kernels"


def code_object(module_text: str) -> bytes:
    """The ``gpu.binary`` ELF in a compiled module's text: the ``bin = "..."``
    string, MLIR-escaped (``\\XX`` a byte, ``\\\\`` and ``\\"`` themselves)."""
    m = _BIN.search(module_text)
    if m is None:
        raise RuntimeError("no gpu.binary object in the compiled module")
    out = bytearray()
    i = m.end()
    n = len(module_text)
    while i < n and module_text[i] != '"':
        ch = module_text[i]
        if ch != "\\":
            out.append(ord(ch))
            i += 1
        elif module_text[i + 1] in '\\"':
            out.append(ord(module_text[i + 1]))
            i += 2
        else:
            out.append(int(module_text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


# ---------------------------------------------------------------- msgpack
def _unpack(buf: bytes, i: int) -> tuple[Any, int]:
    """One msgpack value at ``buf[i:]``; returns ``(value, next index)``."""
    t = buf[i]
    i += 1
    if t < 0x80:
        return t, i
    if t >= 0xE0:
        return t - 0x100, i
    if 0x80 <= t <= 0x8F:
        return _map(buf, i, t & 0x0F)
    if 0x90 <= t <= 0x9F:
        return _array(buf, i, t & 0x0F)
    if 0xA0 <= t <= 0xBF:
        n = t & 0x1F
        return buf[i : i + n].decode("utf-8", "replace"), i + n
    if t == 0xC0:
        return None, i
    if t in (0xC2, 0xC3):
        return t == 0xC3, i
    fixed = {
        0xCC: (">B", 1),
        0xCD: (">H", 2),
        0xCE: (">I", 4),
        0xCF: (">Q", 8),
        0xD0: (">b", 1),
        0xD1: (">h", 2),
        0xD2: (">i", 4),
        0xD3: (">q", 8),
        0xCA: (">f", 4),
        0xCB: (">d", 8),
    }
    if t in fixed:
        fmt, w = fixed[t]
        return struct.unpack_from(fmt, buf, i)[0], i + w
    lens = {0xD9: 1, 0xDA: 2, 0xDB: 4, 0xC4: 1, 0xC5: 2, 0xC6: 4}
    if t in lens:
        n = int.from_bytes(buf[i : i + lens[t]], "big")
        i += lens[t]
        raw = buf[i : i + n]
        return (raw.decode("utf-8", "replace") if t >= 0xD9 else bytes(raw)), i + n
    if t in (0xDC, 0xDD):
        w = 2 if t == 0xDC else 4
        return _array(buf, i + w, int.from_bytes(buf[i : i + w], "big"))
    if t in (0xDE, 0xDF):
        w = 2 if t == 0xDE else 4
        return _map(buf, i + w, int.from_bytes(buf[i : i + w], "big"))
    raise ValueError(f"unsupported msgpack tag {t:#x}")


def _array(buf: bytes, i: int, n: int) -> tuple[list, int]:
    out = []
    for _ in range(n):
        v, i = _unpack(buf, i)
        out.append(v)
    return out, i


def _map(buf: bytes, i: int, n: int) -> tuple[dict, int]:
    out = {}
    for _ in range(n):
        k, i = _unpack(buf, i)
        v, i = _unpack(buf, i)
        out[k] = v
    return out, i


def _kernel_entries(elf: bytes) -> list[dict]:
    at = elf.find(_KERNELS_KEY)
    if at < 0:
        raise RuntimeError("no amdhsa.kernels metadata in the code object")
    # The key is a msgpack str; its value (an array of maps) follows it.
    value, _ = _unpack(elf, at + len(_KERNELS_KEY))
    if not isinstance(value, list):
        raise RuntimeError("malformed amdhsa.kernels metadata")
    return value


def kernel_resources(elf: bytes) -> list[dict]:
    """One dict per kernel in the code object: ``name``, ``vgpr``, ``agpr``,
    ``sgpr``, ``scratch_bytes`` (private segment bytes a lane), ``lds_bytes``,
    ``spill_vgpr``, ``spill_sgpr`` (0 when the code object predates the
    fields), ``wavefront_size`` and ``max_flat_workgroup_size``."""
    out = []
    for e in _kernel_entries(elf):
        out.append(
            {
                "name": str(e.get(".name", "")),
                "vgpr": int(e.get(".vgpr_count", 0)),
                "agpr": int(e.get(".agpr_count", 0)),
                "sgpr": int(e.get(".sgpr_count", 0)),
                "scratch_bytes": int(e.get(".private_segment_fixed_size", 0)),
                "lds_bytes": int(e.get(".group_segment_fixed_size", 0)),
                "spill_vgpr": int(e.get(".vgpr_spill_count", 0)),
                "spill_sgpr": int(e.get(".sgpr_spill_count", 0)),
                "wavefront_size": int(e.get(".wavefront_size", 64)),
                "max_flat_workgroup_size": int(e.get(".max_flat_workgroup_size", 0)),
            }
        )
    return out


def artifact_resources(artifact: Any) -> dict:
    """Resources of a compiled FlyDSL artifact (``CompiledFunction._keepalive``
    or a jit ``_mem_cache`` value), worst case across its kernels, plus
    ``kernels`` (the per-kernel list). Spill and scratch counts take the
    maximum; ``vgpr`` is the code object's ``.vgpr_count``, the unified
    allocation on gfx90a and later (arch VGPR plus AGPR)."""
    ks = kernel_resources(code_object(artifact._ir_text))
    if not ks:
        raise RuntimeError("no kernels in the code object")
    worst = {
        key: max(k[key] for k in ks)
        for key in (
            "vgpr",
            "agpr",
            "sgpr",
            "scratch_bytes",
            "lds_bytes",
            "spill_vgpr",
            "spill_sgpr",
        )
    }
    worst["kernels"] = ks
    return worst


def occupancy_waves_per_simd(vgpr: int, *, vgprs_per_simd: int = 512, granule: int = 8) -> int:
    """Waves per SIMD the VGPR file admits (CDNA3/4: 512 unified VGPRs per
    lane, allocated in 8-register granules; capped at 8 waves). Ignores LDS
    and SGPR limits, so it is an upper bound; ``CompiledFunction
    .max_blocks_per_cu()`` is the runtime's own answer when a device is
    present."""
    if vgpr <= 0:
        return 8
    alloc = -(-vgpr // granule) * granule
    return max(1, min(8, vgprs_per_simd // alloc))
