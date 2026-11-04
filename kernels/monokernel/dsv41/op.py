# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The gfx1250 DeepSeek-V4.1 mono kernels' public entry: their argument lists
(``K1_ABI``, ``K2_ABI``: named once, packed by name, as ATOM's ``KernelAbi``),
their builds, and the optional features a caller turns on (``Features``).

A layer is two launches: K1 (``attn_pre``) then K2 (``layer_post``). Features:

``prefetch_next_layer`` (0 off; ``Features.PREFETCH_*``): as each K2 CTA ends,
the GL2 lines the same CTA reads of the *next* layer's weights -- 1 the next
K1's GEMV weights (wqkv, wq_b, the indexer's wq_b), 2 + K2a's wo_a / wo_b, 3 +
K2b's dense weights (router gate, shared expert). K2's ``nxt`` argument is then
the next layer's pointer table (``next_layer_table``; 0: none, the feature off
for that launch); with the feature off the build ignores ``nxt``. Numerics are
untouched (a prefetch moves no data a kernel reads). Measured at S = 48 TP4 on
MI455X: within noise at every level (2026-10-31: K1 / K2 are latency-bound
there), so it is off by default -- a knob for weight-bound shapes.

    feats = op.Features(prefetch_next_layer=op.Features.PREFETCH_K1)
    k2 = op.build_k2(tokens=48, tp=4, features=feats, tok_par=True)
    nxt = op.next_layer_table(layer_weights[i + 1], device)   # keep it alive
    k2(*op.K2_ABI.pack({..., "nxt": nxt.data_ptr()}), stream=s)
"""

import inspect
from dataclasses import dataclass, field

from kernels.monokernel.dsv41 import attn_pre as k1
from kernels.monokernel.dsv41 import layer_post as lp


def _params(fn) -> tuple:
    """Parameter names of a ``@flyc.kernel`` / ``@flyc.jit`` object's Python function."""
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


@dataclass(frozen=True)
class KernelAbi:
    """A kernel's positional arguments, named once (ATOM's ``KernelAbi``)."""

    names: tuple

    def check_launcher(self, launcher) -> None:
        """The launcher takes exactly ``names``, then ``stream``."""
        got = _params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")

    def pack(self, values: dict) -> list:
        """``values`` (every name, nothing else) in ABI order."""
        extra = sorted(set(values) - set(self.names))
        missing = sorted(set(self.names) - set(values))
        if extra or missing:
            raise TypeError(f"ABI mismatch: missing {missing}, unexpected {extra}")
        return [values[n] for n in self.names]

    def zeros(self) -> list:
        """Every argument 0: a compile-only call's."""
        return [0] * len(self.names)


K1_ABI = KernelAbi(tuple(k1.ARGS))
K2_ABI = KernelAbi(tuple(lp.ARGS))
NEXT_W = lp.NEXT_W  # ``nxt``'s table: the next layer's weight pointers in this order


@dataclass(frozen=True)
class Features:
    """Optional behaviour a caller turns on; the defaults are the measured best."""

    PREFETCH_OFF, PREFETCH_K1, PREFETCH_K2A, PREFETCH_K2B = 0, lp.PF_NEXT_K1, lp.PF_NEXT_K2A, lp.PF_NEXT_K2B
    prefetch_next_layer: int = field(default=0)

    def __post_init__(self):
        if self.prefetch_next_layer not in (0, lp.PF_NEXT_K1, lp.PF_NEXT_K2A, lp.PF_NEXT_K2B):
            raise ValueError(f"prefetch_next_layer {self.prefetch_next_layer}: 0 off, 1 K1, 2 + K2a, 3 + K2b")


def tok_par_ok(tokens: int, tp: int) -> bool:
    """Where MI355 takes token parallelism (runner.TOK_PAR_FROM): TP4, S >= 12, S a multiple of TP."""
    return tp == 4 and tokens >= 12 and tokens % tp == 0


def build_k1(tokens: int, tp: int, *, feeds_index: bool = False, tok_par: bool = False, timeline: bool = False, **build):
    """K1 (``attn_pre``) for a layer; ``feeds_index``: a selecting layer's (its index query too);
    ``build``: further ``AttnPreBuild`` fields (``slice_x8``, ...)."""
    launch = k1.build_attn_pre(
        k1.AttnPreBuild(tokens=tokens, fold=True, tp=tp, feeds_index=feeds_index, tok_par=tok_par, timeline=timeline, **build)
    )
    K1_ABI.check_launcher(launch)
    return launch


def build_k2(tokens: int, tp: int, features: Features = Features(), **build):
    """K2 (``layer_post``) with ``features``; ``build``: further ``LayerPostBuild`` fields
    (``tok_par``, ``index_bound_max``, ...). A K2 built with ``prefetch_next_layer`` > 0 is
    given its next layer's ``nxt`` table, and K1's token parallelism must match its
    ``tok_par`` (the prefetch follows that K1's task placement)."""
    if "pf_next" in build:
        raise TypeError("pf_next: set Features.prefetch_next_layer")
    launch = lp.build_layer_post(
        lp.LayerPostBuild(tokens=tokens, tp=tp, pf_next=features.prefetch_next_layer, **build)
    )
    K2_ABI.check_launcher(launch)
    return launch


def next_layer_table(next_weights: dict, device):
    """K2's ``nxt`` for a layer: the next layer's weights (``NEXT_W`` names -> a tensor or
    a raw pointer; ``iq_w`` 0 or absent past a layer without the indexer) as an int64
    device table. The caller keeps it alive while a launch (or a graph) uses it; the
    weights must be the tensors the next layer's launches read (a prefetch is of those
    exact lines)."""
    import torch

    missing = [n for n in NEXT_W if n != "iq_w" and n not in next_weights]
    if missing:
        raise KeyError(f"next_layer_table: missing {missing}")

    def ptr(v):
        return int(v) if isinstance(v, int) else v.data_ptr()

    return torch.tensor([ptr(next_weights.get(n, 0)) for n in NEXT_W], dtype=torch.int64, device=device)
