# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for `kernels/attention/flash_attn_gfx950_config.py` (metadata, knobs, traits). No GPU.

CFG-01, 03..15 and 17 of the attention test plan. The naming-policy lint (CFG-16) is
`test_flash_attn_gfx950_naming.py`; CFG-02 and 18..20 are `test_attention_abi_host.py`.

CFG-05 compares `resolve` and the traits with a golden generated from AOTriton's vendored tuning modules by
`data/gen_flash_attn_gfx950_golden.py`. The golden uses AOTriton's names; the maps below are the one place the two
vocabularies meet, and `EXPECTED_*` list every deliberate difference with its reason.
"""

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

from kernels.attention import flash_attn_gfx950_config as cfg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).parent / "data" / "flash_attn_gfx950_aotriton_golden.json").read_text())

ARCH = "gfx950"
RUNGS = cfg.LADDER
FEATURES = ("dense", "window", "bias", "dropout")


def meta_for(rung, dtype="bf16", feat="dense", **kw):
    return cfg.FmhaInputMetadata(
        dtype_str=dtype,
        head_dim=rung,
        window=feat == "window",
        bias=feat == "bias",
        dropout=feat == "dropout",
        **kw,
    )


KERNELS = {
    "fwd": (cfg.fwd_knobs, cfg.fwd_traits),
    "dq": (cfg.dq_knobs, cfg.dq_traits),
    "dkdv": (cfg.dkdv_knobs, cfg.dkdv_traits),
}

MATRIX = [(r, d, f) for r in RUNGS for d in ("bf16", "f16") for f in FEATURES]


# ---------------------------------------------------------------------------
# CFG-01
# ---------------------------------------------------------------------------


def test_config_imports_without_flydsl_or_torch():
    """The generator calls the config with no compiler: a meta-path finder rejects flydsl* and torch*."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('flydsl', 'torch', 'mlir_flydsl'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import kernels.attention.flash_attn_gfx950_config as c\n"
        "k = c.fwd_knobs('gfx950').resolve(c.FmhaInputMetadata(dtype_str='bf16', head_dim=100))\n"
        "assert k.as_psels()['BLOCK_DMODEL'] == 128\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True)
    tree = ast.parse((REPO / "kernels/attention/flash_attn_gfx950_config.py").read_text())
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in stdlib for a in node.names), ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] in stdlib, ast.dump(node)


# ---------------------------------------------------------------------------
# CFG-03, CFG-04
# ---------------------------------------------------------------------------


def _render_pon(psels):
    return ";".join(f"{k}={v!r}" for k, v in psels.items())


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_as_psels_is_flat_pon(kernel):
    """Every value is a str/int/bool/float, never None, never nested. Strings satisfy
    `repr(v) == "'" + v + "'"` with no spaces, and a `k=v;...` round trip is the identity."""
    make, _ = KERNELS[kernel]
    for rung, dtype, feat in MATRIX:
        psels = make(ARCH).resolve(meta_for(rung, dtype, feat)).as_psels()
        for k, v in psels.items():
            assert type(v) in (str, int, bool, float), (k, v)
            if isinstance(v, str):
                assert repr(v) == "'" + v + "'" and " " not in v, (k, v)
        parsed = {kv.split("=", 1)[0]: ast.literal_eval(kv.split("=", 1)[1]) for kv in _render_pon(psels).split(";")}
        assert parsed == psels


def test_as_psels_refuses_an_unresolved_knob_set():
    with pytest.raises(ValueError, match="unresolved"):
        cfg.fwd_knobs(ARCH).as_psels()


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_grid_axis_order_is_an_output(kernel):
    """Pinning a GRID_AXIS_ORDER that `resolve` would not pick raises; the resolved value is the launcher's
    contract with the C++ grid calculator. All three gfx950 launchers walk (head, tile, batch): HEAD_FASTEST."""
    make, _ = KERNELS[kernel]
    meta = meta_for(64)
    k = make(ARCH).resolve(meta)
    assert k.GRID_AXIS_ORDER == cfg.GRID_AXIS_HEAD_FASTEST == 0
    with pytest.raises(ValueError, match="GRID_AXIS_ORDER"):
        make(ARCH, GRID_AXIS_ORDER=cfg.GRID_AXIS_TILE_FASTEST).resolve(meta)
    assert make(ARCH, GRID_AXIS_ORDER=0).resolve(meta) == k
    # The psels carry the tile key the C++ side reads: BLOCK_M (fwd, dQ) and BLOCK_N (dK/dV).
    assert {"BLOCK_M", "BLOCK_N", "GRID_AXIS_ORDER"} <= set(k.as_psels())


# ---------------------------------------------------------------------------
# CFG-05
# ---------------------------------------------------------------------------

# AOTriton knob name -> this repository's knob name. Knobs absent from the map were removed or are not comparable.
KNOB_MAP = {
    "fwd": dict(
        block_dmodel="BLOCK_DMODEL",
        block_dmodel_v="BLOCK_DMODEL_V",
        padded_head="PADDED_HEAD",
        block_m="BLOCK_M",
        block_n="BLOCK_N",
        num_waves="num_warps",
        head_dim_granule="HEAD_DIM_GRANULE",
        d_stages="D_STAGES",
        vo_shards="VO_SHARDS",
        waves_per_eu="waves_per_eu",
        daz="daz",
        setprio="SETPRIO",
        stagger="STAGGER",
        GRID_AXIS_ORDER="GRID_AXIS_ORDER",
    ),
    "dq": dict(
        block_dmodel="BLOCK_DMODEL",
        block_dmodel_v="BLOCK_DMODEL_V",
        padded_head="PADDED_HEAD",
        block_m="BLOCK_M",
        block_n="BLOCK_N",
        num_waves="num_warps",
        head_dim_granule="HEAD_DIM_GRANULE",
        mfma_rows="MFMA_ROWS",
        waves_per_eu="waves_per_eu",
        daz="daz",
        setprio="SETPRIO",
        GRID_AXIS_ORDER="GRID_AXIS_ORDER",
    ),
    "dkdv": dict(
        block_dmodel="BLOCK_DMODEL",
        block_dmodel_v="BLOCK_DMODEL_V",
        padded_head="PADDED_HEAD",
        block_kv="BLOCK_N",  # KV rows: the dimension C++ sizes the grid from
        block_q="BLOCK_M",  # Q rows per streamed tile
        num_waves="num_warps",
        head_dim_granule="HEAD_DIM_GRANULE",
        dkv_shards="DKV_SHARDS",
        mfma_rows="MFMA_ROWS",
        num_stream_buffers="NUM_STREAM_BUFFERS",
        tight_registers="TIGHT_REGISTERS",
        waves_per_eu="waves_per_eu",
        daz="daz",
        GRID_AXIS_ORDER="GRID_AXIS_ORDER",
    ),
}

# Deliberate differences from AOTriton's traits. Keys are trait names; values are (golden value -> our value)
# predicates or the reason text for fields that are new here.
EXPECTED_TRAIT_DIFFS = {
    "fwd": {
        # Q16/policy: an LSE pointer that may be null is the default (RETURN_LSE="runtime"); AOTriton's trait
        # said False because the null guard was a runtime branch that did not look at the trait.
        "RETURN_LSE": "the default stores the LSE unless its pointer is null",
        # Policy: longest-first dispatch is on for every windowed build.
        "LPT_TILE_ORDER": "on for window builds",
    },
    "dq": {
        "DUALWAVE_SWP_LAZY_RESCALE": "K58: lazy rescale is a trait fixed False (AOTriton's dQ default was True)",
        "DUALWAVE_SWP_ENABLE_STAGGER": "neither backward kernel uses stagger",
    },
    "dkdv": {
        "DUALWAVE_SWP_ENABLE_STAGGER": "neither backward kernel uses stagger",
        "DUALWAVE_SWP_LAZY_RESCALE": "K58: lazy rescale is a trait fixed False",
    },
}


def _tuplify(value):
    return tuple(_tuplify(v) for v in value) if isinstance(value, list) else value


def _golden_traits(kernel, case):
    spec = GOLDEN["traits"][kernel]
    traits = dict(spec["constant"])
    traits.update(zip(spec["varying"], case[kernel]["traits"]))
    return {k: _tuplify(v) for k, v in traits.items()}


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_resolve_reproduces_aotriton(kernel):
    """`resolve` and the derived traits match AOTriton's for the whole P0 matrix (fields mapped per
    `KNOB_MAP`), plus `HEAD_DIM_V`; every divergence is listed in `EXPECTED_TRAIT_DIFFS`."""
    make, traits_of = KERNELS[kernel]
    assert len(GOLDEN["cases"]) == len(MATRIX)
    differed = set()
    for case in GOLDEN["cases"]:
        rung, dtype, feat = case["rung"], case["dtype"], case["feature"]
        meta = meta_for(rung, dtype, feat)
        knobs = make(ARCH).resolve(meta)
        got = knobs.as_psels()
        for theirs, ours in KNOB_MAP[kernel].items():
            assert got[ours] == case[kernel]["knobs"][theirs], (kernel, rung, dtype, feat, theirs)
        golden = _golden_traits(kernel, case)
        traits = traits_of(meta, knobs)
        mine = dataclasses.asdict(traits)
        for name, value in golden.items():
            if name in EXPECTED_TRAIT_DIFFS[kernel]:
                if mine[name] != value:
                    differed.add(name)
                continue
            if name == "KV_VECTORIZED" and value is None:  # dK/dV: AOTriton leaves it unset for a dense build
                value = False
            assert mine[name] == value, (kernel, rung, dtype, feat, name, mine[name], value)
        # The additions over AOTriton's field set.
        assert traits.HEAD_DIM_V == golden["HEAD_DIM"] == rung
        assert mine["HDIM_QK_FLOOR"] == cfg.rung_below(rung)
    # An allowlist entry that never differs is dead weight.
    assert differed == set(EXPECTED_TRAIT_DIFFS[kernel])


def test_dkdv_roles_are_swapped_in_the_traits():
    """dK/dV streams Q and dO through LDS: `BLOCK_M` (traits) is KV rows, `BLOCK_N` (traits) is Q rows, and the
    knobs carry them as BLOCK_N (KV, what C++ reads for the grid) and BLOCK_M (Q rows)."""
    meta = meta_for(128)
    knobs = cfg.dkdv_knobs(ARCH).resolve(meta)
    traits = cfg.dkdv_traits(meta, knobs)
    assert (traits.BLOCK_KV, traits.BLOCK_Q) == (knobs.BLOCK_N, knobs.BLOCK_M)
    assert (traits.BLOCK_M, traits.BLOCK_N) == (knobs.BLOCK_N, knobs.BLOCK_M)


# ---------------------------------------------------------------------------
# CFG-06, CFG-07
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel,rung,wpe",
    [("fwd", 160, 1), ("fwd", 96, 2), ("fwd", 64, 2), ("fwd", 128, 2), ("fwd", 256, 1), ("dkdv", 160, 1)],
)
def test_waves_per_eu_follows_lds(kernel, rung, wpe):
    """`waves_per_eu=2` only where two workgroups fit in LDS (`2 * LDS_KV_TOTAL_SIZE * 2 B <= 163840`)."""
    make, traits_of = KERNELS[kernel]
    meta = meta_for(rung)
    knobs = make(ARCH).resolve(meta)
    assert knobs.waves_per_eu == wpe
    if kernel == "fwd":
        fits = 2 * traits_of(meta, knobs).LDS_KV_TOTAL_SIZE * cfg.BF16_BYTES <= cfg.LDS_CAP_BYTES
        assert (knobs.waves_per_eu == 2) == fits


@pytest.mark.parametrize("feat", ["dense", "dropout"])
def test_dq_256_takes_the_16_row_family(feat):
    """K47: the 32-row dQ build at 256 with dropout races (507 VGPR + 251 AGPR); 256 and up are 16-row."""
    knobs = cfg.dq_knobs(ARCH).resolve(meta_for(256, feat=feat))
    assert knobs.MFMA_ROWS == 16 and (knobs.BLOCK_M, knobs.BLOCK_N) == (64, 32)
    assert cfg.dq_knobs(ARCH).resolve(meta_for(224, feat=feat)).MFMA_ROWS == 32


# ---------------------------------------------------------------------------
# CFG-08
# ---------------------------------------------------------------------------


def test_lazy_rescale_is_a_trait_fixed_false():
    """K03/K58: lazy rescale is not a knob (f16 output goes non-finite), and every trait carries False."""
    for kernel, (make, traits_of) in KERNELS.items():
        assert "lazy_rescale" not in {f.name for f in dataclasses.fields(make(ARCH))}
        with pytest.raises(TypeError):
            make(ARCH, lazy_rescale=True)
        for rung, dtype, feat in MATRIX[:: len(FEATURES) + 1]:
            meta = meta_for(rung, dtype, feat)
            assert traits_of(meta, make(ARCH).resolve(meta)).DUALWAVE_SWP_LAZY_RESCALE is False
    # The module constant is the one hook, read by the traits constructor.
    assert cfg._LAZY_RESCALE is False


def test_lazy_rescale_hook_reaches_the_traits(monkeypatch):
    meta = meta_for(64)
    knobs = cfg.fwd_knobs(ARCH).resolve(meta)
    before = cfg.fwd_traits(meta, knobs)
    monkeypatch.setattr(cfg, "_LAZY_RESCALE", True)
    after = cfg.fwd_traits(meta, knobs)
    assert after.DUALWAVE_SWP_LAZY_RESCALE is True and before.DUALWAVE_SWP_LAZY_RESCALE is False
    # ... and it changes the cache key, so a cached default binary is never reused.
    assert cfg.build_cache_key(before, knobs) != cfg.build_cache_key(after, knobs)


# ---------------------------------------------------------------------------
# CFG-09
# ---------------------------------------------------------------------------


def _different_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return value + (0,)
    raise AssertionError(type(value))


@pytest.mark.parametrize("kernel,rung", [("fwd", 128), ("dq", 128), ("dkdv", 128)])
def test_cache_key_covers_every_field(kernel, rung):
    """K53: every trait field, and every knob that is not a trait, changes the build's cache key. The key
    carries the field names (a reordering cannot alias two builds)."""
    make, traits_of = KERNELS[kernel]
    meta = meta_for(rung)
    knobs = make(ARCH).resolve(meta)
    traits = traits_of(meta, knobs)
    base = cfg.build_cache_key(traits, knobs)
    names = {f.name for f in dataclasses.fields(traits)}
    flat = {x for pair in base[1] for x in pair if isinstance(x, str)}
    assert names <= flat, "the key must carry field names"
    for f in dataclasses.fields(traits):
        other = dataclasses.replace(traits, **{f.name: _different_value(getattr(traits, f.name))})
        assert cfg.build_cache_key(other, knobs) != base, f"trait {f.name} is not in the cache key"
        assert other.cache_tag != traits.cache_tag
    for name, value in knobs.as_psels().items():
        other = dataclasses.replace(knobs, **{name: _different_value(value)})
        assert cfg.build_cache_key(traits, other) != base, f"knob {name} is not in the cache key"


# ---------------------------------------------------------------------------
# CFG-10
# ---------------------------------------------------------------------------


def test_pins_are_validated():
    """A pinned knob can be slow, never silently wrong: every illegal pin raises at `resolve`."""
    fwd = cfg.fwd_knobs
    with pytest.raises(ValueError, match="does not fit"):
        fwd(ARCH, BLOCK_DMODEL=64).resolve(meta_for(100))  # BLOCK_DMODEL < head_dim
    with pytest.raises(ValueError, match="built rungs"):
        fwd(ARCH, BLOCK_DMODEL=100).resolve(meta_for(100))  # not a rung
    with pytest.raises(ValueError, match="PADDED_HEAD=False"):
        fwd(ARCH, PADDED_HEAD=False).resolve(meta_for(100))  # padded but pinned unpadded
    with pytest.raises(ValueError, match="PADDED_HEAD=False"):
        fwd(ARCH, BLOCK_DMODEL=256, PADDED_HEAD=False).resolve(meta_for(64))  # a wider unpadded tile
    with pytest.raises(ValueError, match="together or not at all"):
        fwd(ARCH, num_warps=4).resolve(meta_for(64))  # partial geometry tuple
    with pytest.raises(NotImplementedError, match="not yet addressable"):
        fwd(ARCH, num_warps=2, BLOCK_M=64, BLOCK_N=64, HEAD_DIM_GRANULE=64).resolve(meta_for(64))
    with pytest.raises(ValueError, match="M extent caps"):
        # 8 waves x 2 shards at 512: BLOCK_M must be (8/2)*32 = 128
        fwd(ARCH, num_warps=8, BLOCK_M=256, BLOCK_N=64, HEAD_DIM_GRANULE=64, VO_SHARDS=2).resolve(meta_for(512))
    with pytest.raises(ValueError, match="RETURN_LSE"):
        fwd(ARCH, RETURN_LSE="sometimes").resolve(meta_for(64))
    with pytest.raises(ValueError, match="STATIC_WINDOW"):
        fwd(ARCH, STATIC_WINDOW=True).resolve(meta_for(64))  # needs meta.window
    with pytest.raises(NotImplementedError, match="BLOCK_DMODEL_V"):
        fwd(ARCH, BLOCK_DMODEL_V=64).resolve(meta_for(128))
    # dQ has no D_STAGES / VO_SHARDS / QK_SHARDS knobs: they are traits fixed at 1.
    for name in ("D_STAGES", "VO_SHARDS", "QK_SHARDS", "lazy_rescale"):
        with pytest.raises(TypeError):
            cfg.dq_knobs(ARCH, **{name: 2})
    # The backward refuses what it does not compute.
    for make in (cfg.dq_knobs, cfg.dkdv_knobs):
        for kw in ({"alibi": True}, {"sink": True}, {"paged": True}):
            with pytest.raises(NotImplementedError):
                make(ARCH).resolve(cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=64, **kw))
    # Bias with a window/causal mask is rejected everywhere (plan 4.10): resolve and the traits functions, all kernels.
    both = cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=64, window=True, bias=True)
    for make in (cfg.fwd_knobs, cfg.dq_knobs, cfg.dkdv_knobs):
        with pytest.raises(ValueError, match="Fold the causal pattern into the bias"):
            make(ARCH).resolve(both)
    ok = meta_for(64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        cfg.fwd_traits(both, cfg.fwd_knobs(ARCH).resolve(ok))
    # More than two D stages races the wide body's DMA waits (finite wrong values): refused.
    for hdim, stages in ((384, 3), (384, 6), (512, 4)):
        with pytest.raises(NotImplementedError, match="at most 2 stages"):
            cfg.fwd_knobs(ARCH, D_STAGES=stages).resolve(meta_for(hdim))
    with pytest.raises(ValueError, match="BLOCK_N"):
        cfg.dkdv_knobs(
            ARCH, num_warps=4, BLOCK_N=100, BLOCK_M=64, HEAD_DIM_GRANULE=64, DKV_SHARDS=1, MFMA_ROWS=32
        ).resolve(meta_for(64))
    with pytest.raises(NotImplementedError):
        cfg.fwd_knobs(ARCH).resolve(cfg.FmhaInputMetadata(dtype_str="fp8", head_dim=64))


# ---------------------------------------------------------------------------
# CFG-11
# ---------------------------------------------------------------------------

PAIRS = [
    (1, 32), (8, 32), (32, 32), (33, 64), (64, 64), (65, 96), (96, 96), (97, 128), (128, 128),
    (129, 160), (160, 160), (161, 192), (192, 192), (193, 224), (224, 224), (225, 256), (256, 256),
    (257, 384), (384, 384), (385, 512), (512, 512),
]  # fmt: skip


def test_ladder_and_floor():
    assert len(PAIRS) == 21
    for hdim, rung in PAIRS:
        assert cfg.tile_width_for(hdim) == rung
        meta = cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=hdim)
        knobs = cfg.fwd_knobs(ARCH).resolve(meta)
        assert knobs.BLOCK_DMODEL == rung and knobs.PADDED_HEAD == (hdim != rung)
        floor = cfg.fwd_traits(meta, knobs).HDIM_QK_FLOOR
        assert floor == cfg.rung_below(rung) and floor < hdim <= rung
    with pytest.raises(ValueError, match="exceeds"):
        cfg.tile_width_for(513)
    with pytest.raises(ValueError):
        cfg.fwd_knobs(ARCH).resolve(cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=600))
    # A wider pinned rung carries no floor: it masks every column.
    meta = meta_for(64)
    knobs = cfg.fwd_knobs(ARCH, BLOCK_DMODEL=128).resolve(meta)
    assert cfg.fwd_traits(meta, knobs).HDIM_QK_FLOOR == 0
    # LADDER is exported as a read-only tuple.
    assert isinstance(cfg.LADDER, tuple) and cfg.LADDER == tuple(sorted(cfg.LADDER))
    assert cfg.rung_below(32) == 0


# ---------------------------------------------------------------------------
# CFG-12
# ---------------------------------------------------------------------------


def test_no_compile_time_head_count_or_scale():
    """K04/K25/K26: head counts and the scale are runtime kernargs, so the metadata carries neither, the
    legacy head-count traits are shape-independent constants, and no kernel body reads one."""
    names = {f.name for f in dataclasses.fields(cfg.FmhaInputMetadata)}
    assert not names & {"sm_scale", "num_heads", "num_kv_heads", "num_heads_q", "num_heads_k", "causal"}
    seen = set()
    for rung in (64, 192):
        meta = meta_for(rung)
        t = cfg.fwd_traits(meta, cfg.fwd_knobs(ARCH).resolve(meta))
        seen.add((t.NUM_HEADS_Q, t.NUM_HEADS_KV, t.GQA_GROUP_SIZE))
    assert seen == {(1, 1, 1)}
    forbidden = {"NUM_HEADS_Q", "NUM_HEADS_KV", "GQA_GROUP_SIZE", "DEFAULT_STRIDE_Q_N", "DEFAULT_STRIDE_KV_N"}
    for path in (REPO / "kernels/attention").glob("flash_attn_gfx950*.py"):
        if path.name == "flash_attn_gfx950_config.py":
            continue
        tree = ast.parse(path.read_text())
        # `flash_attn_gfx950.py` still hosts the legacy `build_flash_attn_dualwave_swp_module` until the interface
        # moves over; only the new builders are held to the rule.
        if path.name == "flash_attn_gfx950.py":
            tree.body = [
                n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("build_flash_attn_gfx950_")
            ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                base = node.value
                is_traits = (isinstance(base, ast.Name) and base.id == "traits") or (
                    isinstance(base, ast.Attribute) and base.attr == "traits"
                )
                assert not is_traits, f"{path.name}:{node.lineno} reads {ast.unparse(node)} (a compile-time head count)"


# ---------------------------------------------------------------------------
# CFG-13
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_resolve_is_pure_and_idempotent(kernel):
    make, _ = KERNELS[kernel]
    for rung, dtype, feat in MATRIX[::3]:
        meta = meta_for(rung, dtype, feat)
        once = make(ARCH).resolve(meta)
        assert once.resolve(meta) == once
        assert make(ARCH).resolve(meta) == once
    # Two processes give equal psels (the metadata is a pure function of the ATI choices).
    code = (
        "import json, kernels.attention.flash_attn_gfx950_config as c\n"
        "m = c.FmhaInputMetadata(dtype_str='bf16', head_dim=96, window=True)\n"
        f"print(json.dumps(c.{kernel}_knobs('gfx950').resolve(m).as_psels(), sort_keys=True))\n"
    )
    out = [subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True).stdout]
    out.append(
        subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True).stdout
    )
    assert out[0] == out[1]
    here = KERNELS[kernel][0](ARCH).resolve(meta_for(96, feat="window")).as_psels()
    assert json.loads(out[0]) == here


# ---------------------------------------------------------------------------
# CFG-14
# ---------------------------------------------------------------------------


def test_defaults_leave_the_aot_abi():
    """A knob's default leaves the ABI exactly AOTriton's: every optional input and feature knob is off."""
    for rung in (64, 192, 384):
        for feat in ("dense", "window"):
            meta = meta_for(rung, feat=feat)
            knobs = cfg.fwd_knobs(ARCH).resolve(meta)
            traits = cfg.fwd_traits(meta, knobs)
            assert (traits.SINK, traits.ALIBI, traits.PAGED, traits.XCD_SWIZZLE) == (False,) * 4
            assert (knobs.NUM_KV_SPLITS, knobs.XCD_SWIZZLE, knobs.RETURN_LSE) == (1, False, "runtime")
            assert knobs.STATIC_WINDOW is False and knobs.daz is True
            assert knobs.LPT_TILE_ORDER == (feat == "window")
            assert traits.SPLITK is False and traits.NUM_KV_SPLITS == 1
            assert traits.LSE_NULL_GUARD and traits.RETURN_LSE


def test_return_lse_modes_map_to_traits():
    meta = meta_for(64)
    flags = {}
    for mode in ("runtime", "always", "never"):
        t = cfg.fwd_traits(meta, cfg.fwd_knobs(ARCH, RETURN_LSE=mode).resolve(meta))
        flags[mode] = (t.RETURN_LSE, t.LSE_NULL_GUARD)
    assert flags == {"runtime": (True, True), "always": (True, False), "never": (False, False)}


# ---------------------------------------------------------------------------
# CFG-15
# ---------------------------------------------------------------------------


def test_family_a_traits_match_main():
    """The standalone traits reproduce main's `_make_dualwave_swp_traits` at family A, on the intersection of
    their fields (`HEAD_DIM_V` included)."""
    utils = pytest.importorskip("kernels.attention.flash_attn_utils")
    for hdim in (64, 128):
        for causal in (False, True):
            meta = cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=hdim, window=causal)
            knobs = cfg.fwd_knobs(ARCH, RETURN_LSE="never").resolve(meta)
            assert (knobs.num_warps, knobs.BLOCK_M, knobs.BLOCK_N) == (8, 256, 64)
            mine = dataclasses.asdict(cfg.fwd_traits(meta, knobs))
            theirs = utils._make_dualwave_swp_traits(
                1,
                1,
                hdim,
                causal=causal,
                dtype_str="bf16",
                waves_per_eu=knobs.waves_per_eu,
                daz=True,
                dualwave_swp_lazy_rescale=False,
                dualwave_swp_setprio=True,
                dualwave_swp_enable_stagger=True,
                num_kv_splits=1,
                varlen=True,
                cross_seqlen=causal,
                paged=False,
                kv_cache_layout="linear",
                kv_vectorized=None,
                return_lse=False,
            )
            common = {f.name for f in dataclasses.fields(theirs)} & set(mine)
            assert {f.name for f in dataclasses.fields(theirs)} <= set(mine), "a main field is missing here"
            diffs = {n: (mine[n], getattr(theirs, n)) for n in common if mine[n] != getattr(theirs, n)}
            # KV_VECTORIZED: main passes None for a dense build, the standalone traits a plain bool.
            diffs.pop("KV_VECTORIZED", None)
            assert not diffs, diffs


# ---------------------------------------------------------------------------
# CFG-17
# ---------------------------------------------------------------------------


def test_stagger_groups_four_waves():
    """K50: the stagger groups are `wave_id // 4`, so 4-wave rungs never shift (a stagger needs two groups
    sharing a SIMD, i.e. 8 waves). The kernel reads this constant."""
    assert cfg.STAGGER_GROUP_WAVES == 4


# ---------------------------------------------------------------------------
# dK/dV: pinned geometry, asymmetric floor
# ---------------------------------------------------------------------------


def test_a_pinned_dkdv_geometry_takes_the_tables_occupancy_hint():
    """The occupancy hint is a per-width table value, not part of the geometry tuple: a pinned geometry resolves with the
    table's `waves_per_eu` (as dQ's takes 1) instead of leaving the build unresolved."""
    pins = dict(MFMA_ROWS=32, DKV_SHARDS=1, num_warps=4, BLOCK_N=128, BLOCK_M=64, HEAD_DIM_GRANULE=64)
    knobs = cfg.dkdv_knobs(ARCH, **pins).resolve(meta_for(128, feat="window"))
    assert knobs.waves_per_eu is not None
    assert cfg.dkdv_knobs(ARCH, waves_per_eu=2, **pins).resolve(meta_for(128, feat="window")).waves_per_eu == 2


def test_dkdv_mask_floor_follows_the_narrower_extent():
    """dK/dV's two extents share one mask floor, so both must sit above it: an asymmetric call whose V width is at or below the
    floor builds with floor 0 (every column masked), where a symmetric one keeps the ladder's floor."""
    sym = meta_for(120)
    asym = cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=120, head_dim_v=8)
    assert cfg.dkdv_traits(sym, cfg.dkdv_knobs(ARCH).resolve(sym)).HDIM_QK_FLOOR == 96
    assert cfg.dkdv_traits(asym, cfg.dkdv_knobs(ARCH).resolve(asym)).HDIM_QK_FLOOR == 0


# ---------------------------------------------------------------------------
# The backward's JIT-only knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["dq", "dkdv"])
def test_bwd_static_knobs(kernel):
    """`STATIC_WINDOW` / `STATIC_SEQLEN` (the Leading_upper_snake_case parameters baked) are off by default, reach the traits when
    set, and `STATIC_WINDOW` needs a window."""
    make, traits_of = KERNELS[kernel]
    windowed = meta_for(64, feat="window")
    base = make(ARCH).resolve(windowed)
    assert base.STATIC_WINDOW is False and base.STATIC_SEQLEN is False
    on = make(ARCH, STATIC_WINDOW=True, STATIC_SEQLEN=True).resolve(windowed)
    traits = traits_of(windowed, on)
    assert traits.STATIC_WINDOW is True and traits.STATIC_SEQLEN is True
    with pytest.raises(ValueError, match="STATIC_WINDOW"):
        make(ARCH, STATIC_WINDOW=True).resolve(meta_for(64))
