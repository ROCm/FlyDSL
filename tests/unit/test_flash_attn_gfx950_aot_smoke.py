# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AOT smoke for the gfx950 attention builders (forward, dQ and dK/dV): the flow AOTriton's generator and `flyc_compile` follow, with no
tensors and no launch.

1. The config is imported with flydsl blocked and `resolve`d to flat psels (what the generator records beside each hsaco).
2. In a second step (flydsl available) the metadata is **recomputed** from the description's choices alone, the psels are
   replayed as pins, and the builder is made.
3. The launcher is compiled the way the driver does: null pointers typed from the annotation, `0` for every `Constexpr`, no
   torch tensors. Nothing is launched.

Checks the kernarg list against the ELF's `.args`, that no hidden kernargs appear, that `known_block_size` is set and the
code object exports the one uniquely named kernel.
"""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

REPO = Path(__file__).resolve().parents[2]

_DESCRIBE = """
import json, sys
import kernels.attention.flash_attn_gfx950_config as c
assert 'flydsl' not in sys.modules and 'torch' not in sys.modules
choices = json.loads(sys.argv[1])
meta = c.FmhaInputMetadata(dtype_str=choices['dtype'], head_dim=choices['BLOCK_DMODEL'],
                           window=choices['causal_type'] == 3, bias=bool(choices['bias']), dropout=bool(choices['dropout']))
make = getattr(c, choices['kind'] + '_knobs')
knobs = make('gfx950', BLOCK_DMODEL=choices['BLOCK_DMODEL'], PADDED_HEAD=choices['PADDED_HEAD']).resolve(meta)
print(json.dumps(knobs.as_psels()))
"""

CHOICES = [
    dict(kind=kind, **base)
    for kind in ("fwd", "dq", "dkdv")
    for base in (
        dict(dtype="bf16", BLOCK_DMODEL=64, PADDED_HEAD=False, causal_type=0, bias=0, dropout=0),
        dict(dtype="f16", BLOCK_DMODEL=128, PADDED_HEAD=True, causal_type=3, bias=0, dropout=1),
        dict(dtype="bf16", BLOCK_DMODEL=256, PADDED_HEAD=False, causal_type=0, bias=1, dropout=0),
    )
]
KERNEL_NAMES = {"fwd": "FWD_KERNEL_NAME", "dq": "DQ_KERNEL_NAME", "dkdv": "DKDV_KERNEL_NAME"}


def _synthesised_args(launcher, dtype_str):
    """What `flyc_compile` hands the driver: a typed null pointer per pointer parameter (the tensor operands typed from
    the build's dtype, LSE as f32, the rest as bytes: their alignment contract is the element's), `0` per `Constexpr`,
    scalars as numbers."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    elem = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    typed = {n: elem for n in ("Q", "K", "V", "O", "B", "DO", "DQ", "DK", "DV", "DB")}
    typed.update(LSE=fx.Float32, Delta=fx.Float32)
    args = []
    for p in inspect.signature(launcher.func).parameters.values():
        ann = p.annotation
        if ann is fx.Stream:
            args.append(fx.Stream(None))
        elif ann is fx.Pointer:
            args.append(flyc.from_c_void_p(typed.get(p.name, fx.Uint8), 0))
        elif ann in (fx.Int32, fx.Int64):
            args.append(1)
        elif ann is fx.Float32:
            args.append(1.0)
        else:  # fx.Constexpr and the per-build annotations: the driver passes 0
            args.append(0)
    return args


@pytest.mark.parametrize(
    "choices",
    CHOICES,
    ids=lambda c: f"{c['kind']}-{c['dtype']}-{c['BLOCK_DMODEL']}-c{c['causal_type']}-b{c['bias']}-d{c['dropout']}",
)
def test_aot_flow_compiles_without_a_launch(choices, tmp_path, monkeypatch):
    from kernels.attention import dispatch

    arch = dispatch.current_arch()
    backend = dispatch.backend_for(arch)
    if backend is None:
        pytest.skip(f"no attention backend for {arch}")
    out = subprocess.run(
        [sys.executable, "-c", _DESCRIBE, json.dumps(choices)], cwd=REPO, check=True, capture_output=True, text=True
    )
    psels = json.loads(out.stdout.strip().splitlines()[-1])
    kind = choices["kind"]
    assert psels["BLOCK_DMODEL"] == choices["BLOCK_DMODEL"] and psels["GRID_AXIS_ORDER"] == 0

    from kernels.attention import flash_attn_gfx950_config as cfg
    from tests.kernels.attention import isa_tools

    # Step 2: metadata recomputed from the choices; the psels replayed as pins give back the same knobs.
    meta = cfg.FmhaInputMetadata(
        dtype_str=choices["dtype"],
        head_dim=choices["BLOCK_DMODEL"],
        window=choices["causal_type"] == 3,
        bias=bool(choices["bias"]),
        dropout=bool(choices["dropout"]),
    )
    pins = {k: v for k, v in psels.items() if k != "GRID_AXIS_ORDER"}
    knobs = getattr(backend, f"{kind}_knobs")(arch, **pins).resolve(meta)
    assert knobs.as_psels() == psels
    fn = getattr(backend, f"build_{kind}")(meta, knobs)

    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("ARCH", "gfx950")
    import flydsl.compiler as flyc

    dump = isa_tools.compile_dump(
        lambda: flyc.compile(fn.launcher, *_synthesised_args(fn.launcher, choices["dtype"])), tmp_path, monkeypatch
    )
    assert dump.hidden_args == []
    assert dump.metadata[".name"].startswith(getattr(cfg, KERNEL_NAMES[kind]))
    assert dump.metadata[".max_flat_workgroup_size"] == fn.traits.BLOCK_SIZE
    assert dump.metadata[".kernarg_segment_size"] == sum(s for _, s in dump.kernel_args) + 4
