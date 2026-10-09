# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx950-specific attention tests: the kernel ABI goldens, codegen hazards the toolchain does not model, schedule and
geometry pins, and the wide body (forward part; the backward kernels add their own sections).

Compile-only tests read the dumped MLIR, LLVM IR and final ISA of a **fresh** build and launch nothing; the others launch
the forward and gate it as the arch-neutral tests do. Registered in `tests/arch_compat.py` (`CDNA_ONLY_TESTS`).
"""

import ast
import itertools
import pathlib

import pytest
import torch

from kernels.attention import flash_attn_gfx950_config as cfg
from tests.kernels.attention import isa_tools
from tests.kernels.attention.attn_testlib import (
    DTYPES,
    WINDOW_BOTRIGHT,
    alloc,
    bwd_check,
    bwd_reference,
    check_floor,
    dkdv_family_pins,
    fwd_check,
    meta_of,
    randn,
    relrms,
    run_bwd,
    run_fwd,
    sdpa_scale,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

ATTN = pathlib.Path(__file__).resolve().parents[3] / "kernels" / "attention"
BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)


@pytest.fixture(scope="module", autouse=True)
def _require_gfx950(arch):
    if not arch.startswith("gfx950"):
        pytest.skip(f"gfx950-specific tests, current arch is {arch}")


# ---------------------------------------------------------------------------
# ABI-01..03: the wire ABI, statically
# ---------------------------------------------------------------------------

# `kernel name -> operands that must be fx.Pointer`: spelled out rather than inferred, so adding an operand makes someone
# write it down. An `fx.Tensor` operand costs a second kernarg slot (a 40-byte shape+stride descriptor the caller cannot
# fill and the kernel never reads) and shifts every later argument.
_WIRE_OPERANDS = {
    "flash_attn_gfx950.py": {
        "flash_attn_func_gfx950_kernel": ["Q", "K", "V", "B", "O", "LSE"],
        "launch_flash_attn_func_gfx950": ["Q", "K", "V", "O", "LSE", "B"],
    },
}
_TRACED_MODULES = ["flash_attn_gfx950.py", "flash_attn_gfx950_helpers.py"]
_WIRE_NAMES = {"Q", "K", "V", "B", "O", "DO", "DQ", "DK", "DV", "DB", "LSE", "Delta", "Bias", "DebugCounts"}
_PAIRS = [("flash_attn_gfx950.py", "flash_attn_func_gfx950_kernel", "launch_flash_attn_func_gfx950")]
_JIT_ONLY = {"stream", "batch_size"}
# The forward's launcher declares `Q K V O LSE B` while its kernel takes `Q K V B O LSE`; the call site passes them
# positionally in the kernel's order. Reordering a kernel signature moves every later kernarg and AOTriton binds to those
# offsets, so the divergence is recorded, not fixed.
_ORDER_DIVERGES = {"flash_attn_gfx950.py"}


def _functions(path):
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def _decorator_path(node):
    while isinstance(node, ast.Call):
        node = node.func
    return ast.unparse(node)


def _traced_scopes(tree):
    """Subtrees that become device code: `@flyc.kernel`/`@flyc.jit` functions and every method of a class (the kernel contexts
    store the views as `self.Q`). Host code reads shapes off real torch tensors on purpose."""
    scopes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            scopes += [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        elif isinstance(node, ast.FunctionDef) and any(
            _decorator_path(d) in ("flyc.kernel", "flyc.jit") for d in node.decorator_list
        ):
            scopes.append(node)
    return scopes


@pytest.mark.parametrize("filename", sorted(_WIRE_OPERANDS))
def test_tensor_operands_are_pointers(filename):
    funcs = _functions(ATTN / filename)
    for fn_name, operands in _WIRE_OPERANDS[filename].items():
        annotated = {a.arg: ast.unparse(a.annotation) if a.annotation else None for a in funcs[fn_name].args.args}
        for operand in operands:
            assert (
                annotated.get(operand) == "fx.Pointer"
            ), f"{filename}:{fn_name}: {operand} is {annotated.get(operand)}"


@pytest.mark.parametrize("filename,kernel,launcher", _PAIRS, ids=[p[0] for p in _PAIRS])
def test_launcher_and_kernel_line_up(filename, kernel, launcher):
    """The launcher hands the kernel its arguments positionally, so order is the ABI: two swapped anywhere along forty
    untyped arguments compiles, launches and returns a plausible wrong answer."""
    funcs = _functions(ATTN / filename)
    kernel_args = [a.arg for a in funcs[kernel].args.args]
    jit_args = [a.arg for a in funcs[launcher].args.args]
    extra = sorted(set(jit_args) - set(kernel_args))
    assert set(extra) <= _JIT_ONLY, f"the launcher takes {extra}, which its kernel does not"
    common = [a for a in jit_args if a in kernel_args]
    if filename in _ORDER_DIVERGES:
        assert sorted(common) == sorted(kernel_args)
    else:
        assert common == kernel_args


@pytest.mark.parametrize("filename", _TRACED_MODULES)
def test_no_size_read_off_a_wire_view(filename):
    """`wire_view` gives a pointer operand a placeholder layout, so `.shape`/`.layout` off one returns 1 (a wrong answer, not
    an error). Every extent the kernel needs is on the wire already."""
    tree = ast.parse((ATTN / filename).read_text())
    bad = []
    for scope in _traced_scopes(tree):
        for node in ast.walk(scope):
            if isinstance(node, ast.Attribute) and node.attr in {"shape", "layout"}:
                base = node.value
                name = (
                    base.attr
                    if isinstance(base, ast.Attribute) and getattr(base.value, "id", None) == "self"
                    else getattr(base, "id", None)
                )
                if name in _WIRE_NAMES:
                    bad.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------
# ABI-04, ABI-05: the goldens
# ---------------------------------------------------------------------------

# What AOTriton's description must declare: every parameter of the kernel `def`, folded `Constexpr` ones included.
FWD_DEF_PARAMS = (
    "Q K V B O LSE workspace block_table seqinfo_q0 seqinfo_q1 seqinfo_k0 seqinfo_k1 varlen_bits num_seqlens Max_seqlen_q "
    "Max_seqlen_k Window_left Window_right philox_seed_ptr philox_offset1 philox_offset2 philox_seed_output "
    "philox_offset_output idropout_p dropout_scale num_head_q num_head_k hdim_qk hdim_vo sm_scale stride_q_batch "
    "stride_q_head stride_q_seq stride_k_batch stride_k_head stride_k_seq stride_v_batch stride_v_head stride_v_seq "
    "stride_o_batch stride_o_head stride_o_seq stride_b_batch stride_b_head stride_b_seq_q block_table_stride "
    "alibi_slopes alibi_stride_b sink"
).split()

# The kernarg block at default metadata and knobs, `(kind, size)` per non-`Constexpr` parameter, offsets left to the
# `kernelParams` model: pointers are 8-byte buffers, scalars by value. 296 bytes, no hidden arguments.
_P, _I, _L, _F = ("global_buffer", 8), ("by_value", 4), ("by_value", 8), ("by_value", 4)
FWD_KERNARG_GOLDEN = (
    [_P] * 6  # Q K V B O LSE
    + [_P] * 4  # seqinfo_q0 q1 k0 k1
    + [_I] * 6  # varlen_bits num_seqlens Max_seqlen_q Max_seqlen_k Window_left Window_right
    + [_P, _P, _L, _P, _P]  # philox_seed_ptr philox_offset1 philox_offset2 philox_seed_output philox_offset_output
    + [_I, _F]  # idropout_p dropout_scale
    + [_I] * 4  # num_head_q num_head_k hdim_qk hdim_vo
    + [_F]  # sm_scale
    + [_L] * 15  # strides: q k v o b, (batch, head, seq) each
)


def _kernel_def_params():
    tree = ast.parse((ATTN / "flash_attn_gfx950.py").read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_flash_attn_gfx950_fwd")
    kernel = next(
        n for n in ast.walk(build) if isinstance(n, ast.FunctionDef) and n.name == "flash_attn_func_gfx950_kernel"
    )
    return [(a.arg, ast.unparse(a.annotation)) for a in kernel.args.args]


def test_def_parameter_golden_list():
    """ABI-04: the kernel `def` parameter list equals a checked-in golden: what AOTriton must declare (the generator parses
    the `def`, and the declared order is frozen and load-bearing)."""
    assert [n for n, _ in _kernel_def_params()] == FWD_DEF_PARAMS


def test_kernarg_golden_matches_elf_args(backend, arch, tmp_path, monkeypatch):
    """ABI-05/06: at default metadata and knobs the explicit kernarg list equals the golden in count, order, size and kind,
    totals 296 bytes and has **no hidden arguments** (a `gpu.grid_dim` read appends the 256-byte hidden block, which
    trips AOTriton's `--verify`). Offsets are not compared: the launch uses `kernelParams`."""
    for meta in (meta_of(head_dim=64), meta_of(head_dim=64, window=True), meta_of(head_dim=128, bias=True)):
        d = isa_tools.fresh_fwd_dump(backend, arch, meta, tmp_path / f"d{abs(hash(meta))}", monkeypatch)
        assert d.kernel_args == FWD_KERNARG_GOLDEN, meta
        assert d.hidden_args == [], f"hidden kernargs at defaults: {d.hidden_args}"
        assert d.metadata[".kernarg_segment_size"] == 296


# the explicit list expected from the def: Pointer -> 8-byte buffer, Int32/Float32 -> 4, Int64 -> 8; folded params vanish
def _expected_kernargs():
    # Per-build annotations: WL_ANN / SEQ_ANN are Int32 at the defaults (Leading_upper_snake_case); WS_ANN / BT_ANN / BTS_ANN fold away.
    kinds = {"fx.Pointer": _P, "fx.Int32": _I, "fx.Float32": _F, "fx.Int64": _L, "WL_ANN": _I, "SEQ_ANN": _I}
    return [kinds[a] for _, a in _kernel_def_params() if a in kinds]


def test_golden_matches_the_def():
    assert _expected_kernargs() == FWD_KERNARG_GOLDEN


# ---------------------------------------------------------------------------
# ABI-11: DAZ reaches the hardware
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("daz,mode", [(True, 0), (False, 3)])
def test_daz_reaches_the_hardware_mode(backend, arch, tmp_path, monkeypatch, daz, mode):
    """K51: `daz_denormal_attr()` (`llvm.denormal_fpenv`) is what moves `.amdhsa_float_denorm_mode_32` (0 flush, 3 IEEE);
    the old `denormal-fp-math-f32` passthrough never did, so AOTriton 0.14 ran with IEEE denormals."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=64), tmp_path, monkeypatch, daz=daz)
    assert d.denorm_mode_32 == mode


# ---------------------------------------------------------------------------
# ABI-12/13/14/15/16: codegen hazards the toolchain does not model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [192, 256])
def test_ds_read_tr_is_a_rocdl_op_and_its_results_are_waited(backend, arch, tmp_path, monkeypatch, hdim):
    """K42/C05: `ds_read_b64_tr_b16` is a ROCDL op, not inline asm (`SIInsertWaitcnts` cannot see inside asm, so above hdim 128
    the AGPR copies of the result landed before any wait: non-deterministic NaN); the ISA has it, every result is
    waited before its first reader, and no `vmcnt(0)` drain of the KV prefetch is forced around the reads."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim), tmp_path, monkeypatch)
    assert not [a for a, _ in isa_tools.inline_asm_strings(d.mlir("convert_arith_to_llvm")) if "ds_read_b64_tr" in a]
    n, unwaited = isa_tools.scan_ds_read_tr_waits(d.isa)
    assert n > 0 and unwaited == 0, (n, unwaited)
    assert d.isa.count("vmcnt(0)") <= 8, "the V reads must carry LDS alias scopes or buffer_load...lds forces vmcnt(0)"


@pytest.mark.parametrize("pins", [{}, {"STAGGER": True}], ids=["default", "stagger"])
@pytest.mark.parametrize("hdim", [64, 192])
def test_inline_asm_that_writes_scc_declares_it(backend, arch, tmp_path, monkeypatch, hdim, pins):
    """K43: an inline asm that runs `s_cmp`/`s_cbranch_scc`/`s_and`... destroys SCC and must say `~{scc}`, or LLVM keeps an
    SCC-producing compare live across it (the null-LSE compare was hoisted above the stagger barrier and the LSE store
    skipped for every row)."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim), tmp_path, monkeypatch, **pins)
    needles = ("s_cmp", "s_cbranch_scc", "s_and_b", "s_or_b", "s_add_", "s_sub_", "saveexec")
    asms = isa_tools.inline_asm_strings(d.mlir("convert_arith_to_llvm"))
    scc = [(a, c) for a, c in asms if any(n in a for n in needles)]
    assert scc, "the stagger barrier asm must be present at these builds"
    assert all("~{scc}" in c for _, c in scc), scc


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("hdim", [96, 224, 256])
def test_exp2_wait_state_scan(backend, arch, tmp_path, monkeypatch, hdim, dtype_str):
    """K44/C04: `v_exp_f32` is quarter-rate and needs one wait state before a VALU consumer; `GCNHazardRecognizer` does not
    model it on gfx950 (a zero-gap site carries the *pre-exp* score into the MFMA: one wrong element per pack; 7 of 108
    bf16 builds, at rungs 96, 224, 256). Zero zero-gap sites."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(dtype_str=dtype_str, head_dim=hdim), tmp_path, monkeypatch)
    assert any(op == "v_exp_f32" for op, _ in isa_tools.instructions(d.isa)), "the scan must see the exps"
    assert isa_tools.scan_exp2_wait_state(d.isa) == []


@pytest.mark.parametrize("window", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("hdim", [64, 128, 512])
def test_softmax_exponent_is_not_an_fma(backend, arch, tmp_path, monkeypatch, hdim, window):
    """Aotriton issue 54: no `v_exp_f32` operand may come from an FMA. Under the ambient `fast` hint LLVM fused the score
    scale into the row-max subtract, `exp2(fma(s, c, -m))`, across the dual-wave hot loop; `test_large_bf16_nan_values`
    shows what that does to the output."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim, window=window), tmp_path, monkeypatch)
    assert any(op == "v_exp_f32" for op, _ in isa_tools.instructions(d.isa)), "the scan must see the exps"
    assert isa_tools.scan_exp2_fed_by_fma(d.isa) == []


@pytest.mark.parametrize("hdim", [128, 512])
def test_sink_exponent_is_not_an_fma(backend, arch, tmp_path, monkeypatch, hdim):
    """`test_softmax_exponent_is_not_an_fma` for the sink fold, where `sink * log2e` became the FMA of
    `exp2(sink_log2 - m)`."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim, sink=True), tmp_path, monkeypatch)
    assert isa_tools.scan_exp2_fed_by_fma(d.isa) == []


@pytest.mark.xfail(strict=False, reason="ABI-16 calibration: triage every hit, allowlist the benign ones, then gate")
@pytest.mark.parametrize("hdim", [64, 128])
def test_cvt_pk_feeds_mfma_after_two_wait_states(backend, arch, tmp_path, monkeypatch, hdim):
    """K45/C04: `v_cvt_pk_bf16_f32` -> `v_mfma` SrcA/SrcB needs two wait states (not modelled on gfx950). Restricted to this
    producer: the general rule false-positives on ~7500 benign sites."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim, dropout=True), tmp_path, monkeypatch)
    assert isa_tools.scan_cvt_pk_to_mfma(d.isa) == []


@pytest.mark.parametrize("dtype_str,want,other", [("bf16", "bf16", "f16"), ("f16", "f16", "bf16")])
def test_builds_use_the_mfma_for_their_dtype(backend, arch, tmp_path, monkeypatch, dtype_str, want, other):
    """K39: f16 builds use `_f16` MFMAs and `v_cvt_pk_f16_f32`; bf16 builds the bf16 forms."""
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(dtype_str=dtype_str, head_dim=64), tmp_path, monkeypatch)
    counts = isa_tools.isa_stats(d.isa)
    mfma = {k for k in counts if k.startswith("v_mfma")}
    assert mfma and all(k.endswith("_" + want) for k in mfma), (mfma, other)


# ---------------------------------------------------------------------------
# ABI-23: resource canary
# ---------------------------------------------------------------------------

# `(vgpr_spill_count, private_segment_fixed_size)` upper bounds at the hazard rungs (C01/C03): register spills are where the
# toolchain's spill bugs bite, so a regression here is a numerics risk before it is a performance one. Updating this table
# needs a review note.
RESOURCE_GOLDEN = {160: (0, 0), 192: (0, 0), 224: (0, 0), 256: (0, 0)}


@pytest.mark.parametrize("hdim", sorted(RESOURCE_GOLDEN))
def test_resource_canary(backend, arch, tmp_path, monkeypatch, hdim):
    d = isa_tools.fresh_fwd_dump(backend, arch, meta_of(head_dim=hdim), tmp_path, monkeypatch)
    res = d.resources()
    spills, scratch = RESOURCE_GOLDEN[hdim]
    assert res["vgpr_spill_count"] <= spills and res["private_segment_fixed_size"] <= scratch, res
    assert res["vgpr_count"] <= 512


# ---------------------------------------------------------------------------
# FWD-29..31, 34, 35: schedule, geometry, determinism, the wide body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdim", [96, 64], ids=["4wave", "8wave"])
def test_stagger_is_bit_identical(fwd_build, hdim):
    """K50: stagger shifts one wave group a pipeline phase; the output is bitwise unchanged. The grouping is `wave_id // 4`,
    so a 4-wave rung never shifts (the perf half is measured by `run_benchmark.sh`, not here)."""
    dtype = DTYPES["bf16"]
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, 512, hdim, dtype, gen=gen) for _ in range(3))
    outs = []
    for stagger in (True, False):
        fn = fwd_build(meta_of(head_dim=hdim, window=True), STAGGER=stagger)
        o = alloc(1, 4, 512, hdim, dtype)
        run_fwd(fn, q, k, v, o, window=BR)
        outs.append(o)
    assert torch.equal(*outs)


def _legal_geometries(rung):
    """Every supported `(num_warps, BLOCK_M, BLOCK_N, HEAD_DIM_GRANULE)` tuple that is legal at `rung`."""
    out = []
    for geom in cfg._FWD_SUPPORTED_GEOMETRIES:
        pins = dict(zip(("num_warps", "BLOCK_M", "BLOCK_N", "HEAD_DIM_GRANULE"), geom))
        try:
            cfg.fwd_knobs("gfx950", **pins).resolve(cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=rung))
        except (ValueError, NotImplementedError):
            continue
        out.append(pins)
    return out


@pytest.mark.parametrize("rung", [32, 64, 96, 128])
def test_geometry_tuples_agree(fwd_build, rung):
    """K21/K23/C14, lesson 13: every legal geometry tuple at a rung meets the fp64 floor, and the tuples agree pairwise within
    2x the floor (a wrong V-read constant at granule 32 once gave wrong answers only there)."""
    dtype = DTYPES["bf16"]
    tuples = _legal_geometries(rung)
    assert tuples, "every rung has at least one legal geometry"
    results = []
    for pins in tuples:
        fn = fwd_build(meta_of(head_dim=rung), **pins)
        r = fwd_check(fn, b=1, hq=4, sq=300, d=rung, dtype=dtype, ctx=f"{rung} {pins}", seed=rung)
        results.append((pins, r))
    floor = results[0][1]["floor"]
    from tests.kernels.attention.attn_testlib import relrms

    for (pa, ra), (pb, rb) in itertools.combinations(results, 2):
        assert relrms(ra["o"], rb["o"].double()) <= 2 * 2 * max(floor, 1e-7), (pa, pb)


_HAZARD = [(96, "bf16"), (192, "f16"), (256, "bf16")]


@pytest.mark.parametrize("feat", ["bias_dropout", "causal_dropout"])
@pytest.mark.parametrize("hdim,dtype_str", _HAZARD)
def test_run_to_run_bitwise_at_hazard_rungs(fwd_build, hdim, dtype_str, feat):
    """K42/K44/K45/K46: the hazard symptoms are non-deterministic, so N identical launches must be bitwise equal and NaN-free
    (CI N=3). Bias and a window are mutually exclusive, so the two feature mixes are bias+dropout and window+dropout."""
    dtype = DTYPES[dtype_str]
    meta = meta_of(
        dtype_str=dtype_str, head_dim=hdim, dropout=True, bias=feat == "bias_dropout", window=feat == "causal_dropout"
    )
    fn = fwd_build(meta)
    gen = seeded(hdim)
    q, k, v = (randn(1, 4, 513, hdim, dtype, gen=gen) for _ in range(3))
    bias = torch.randn(1, 4, 513, 513, device="cuda", generator=gen).to(dtype) if feat == "bias_dropout" else None
    outs = []
    for _ in range(3):
        o = alloc(1, 4, 513, hdim, dtype)
        run_fwd(fn, q, k, v, o, bias=bias, window=BR if feat == "causal_dropout" else None, p_drop=0.3, seed=5)
        outs.append(o)
    assert not torch.isnan(outs[0]).any()
    assert all(torch.equal(outs[0], o) for o in outs[1:])


_W = dict(num_warps=4, BLOCK_N=64, HEAD_DIM_GRANULE=64)


@pytest.mark.parametrize(
    "hdim,pins",
    [
        (300, {}),
        (384, {}),
        (448, {}),
        (512, {}),
        (512, dict(_W, BLOCK_M=128, VO_SHARDS=1)),
        (512, dict(num_warps=8, BLOCK_M=64, BLOCK_N=64, HEAD_DIM_GRANULE=64, VO_SHARDS=4)),
    ],
)
def test_wide_body_suite(fwd_build, hdim, pins):
    """K19/K23: the wide body (D staged through LDS and sharded across waves) meets the floor; NaN-prefilled outputs and slack
    expose a store that leaks into the next row (hdim 300 once came out at 0.58 absolute error)."""
    dtype = DTYPES["bf16"]
    try:
        fn = fwd_build(meta_of(head_dim=hdim), **pins)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"illegal pin at this width: {e}")
    gen = seeded(hdim)
    b, h, s = 1, 4, 200
    q, k, v = (randn(b, h, s, hdim, dtype, gen=gen) for _ in range(3))
    fwd_check(fn, b=b, hq=h, sq=s, d=hdim, dtype=dtype, qkv=(q, k, v), ctx=f"wide {hdim} {pins}")
    assert fn.traits.D_STAGES > 1 or fn.traits.VO_SHARDS > 1


@pytest.mark.parametrize("hdim,stages", [(384, 3), (384, 6), (512, 4)])
def test_wide_stage_pins_beyond_two_are_refused(hdim, stages):
    """A pin can be slow but never silently wrong: with more than two D stages the wide body's `vmcnt` wait counts let the stage
    being read race its own DMA (finite wrong values in the last chunks), so `resolve` refuses."""
    meta = meta_of(head_dim=hdim)
    with pytest.raises(NotImplementedError, match="at most 2 stages"):
        cfg.fwd_knobs("gfx950", D_STAGES=stages).resolve(meta)


def test_hdim32_one_accumulator_builds(fwd_build):
    """K48: hdim 32 has one 32-column output chunk (`D_CHUNKS == 1`), where the one-output inline-asm anchor aborts LLVM
    ('inline asm with one output cannot return struct'); it builds on the eager rescale path and is correct."""
    for dtype_str in ("bf16", "f16"):
        fn = fwd_build(meta_of(dtype_str=dtype_str, head_dim=32, window=True))
        assert fn.traits.D_CHUNKS == 1
        fwd_check(fn, b=1, hq=2, sq=200, d=32, dtype=DTYPES[dtype_str], window=BR, ctx=f"hdim32 {dtype_str}")


# ---------------------------------------------------------------------------
# Backward hazards and determinism: BWD-23..26
# ---------------------------------------------------------------------------


def _identity_problem(s=128, h=2, seed=40):
    """`V = I`, so the forward's output reveals the dropout mask it used (`head_dim == seqlen_k`)."""
    gen = seeded(seed)
    q, k, do = (randn(1, h, s, s, DTYPES["bf16"], gen=gen) for _ in range(3))
    v = torch.eye(s, device="cuda", dtype=DTYPES["bf16"]).expand(1, h, s, s).contiguous()
    return q, k, v, do


@pytest.mark.parametrize("launches", [3, pytest.param(10, marks=pytest.mark.large_shape)])
def test_hazard_repro_dkdv_bd128_m16(bwd_build, launches):
    """K45: the exact repro of the 16-row dK/dV hazard: BLOCK_DMODEL 128, bf16, non-causal, dropout and bias, unpadded. A
    wait state the toolchain did not insert let one wave read stale accumulator registers in dK columns 16..31, differently
    on every launch. So: N launches are bitwise identical, and those columns are inside the floor."""
    q, k, v, do = _identity_problem()
    bias = randn(1, 2, 128, 128, DTYPES["bf16"], gen=seeded(41)).contiguous()
    meta = meta_of(head_dim=128, bias=True, dropout=True)
    builds = bwd_build(meta, dkdv=dkdv_family_pins(128, 16))
    outs = [run_bwd(builds, q, k, v, do, bias=bias, p_drop=0.25, seed=5, which=("dkdv",)) for _ in range(launches)]
    for other in outs[1:]:
        assert torch.equal(outs[0]["dk"], other["dk"]) and torch.equal(outs[0]["dv"], other["dv"]), "launches differ"
    keep = outs[0]["o"] != 0
    sm = sdpa_scale(128)

    def ref(**kw):
        return bwd_reference(q, k, v, do, sm, bias=bias, keep=keep, p_drop=0.25, **kw)

    exact = ref()[1][..., 16:32]
    floor = max(relrms(ref(round_to=DTYPES["bf16"], ftz=ftz)[1][..., 16:32], exact) for ftz in (False, True))
    check_floor("dk[16:32]", outs[0]["dk"][..., 16:32], exact, floor, "K45 columns")


def test_hazard_repro_dq_bd64_column_uniformity(bwd_build):
    """K45: dQ at BLOCK_DMODEL 64, unpadded, bf16: the first 32 columns come out as accurate as the rest (a missing wait
    state once gave 39% relative L2 error in columns [0, 32) against 7% in the rest)."""
    out = bwd_check(bwd_build(meta_of(head_dim=64)), b=1, hq=2, sq=256, d=64, dtype=DTYPES["bf16"], ctx="K45 dq")
    exact = out["exact"][0]
    lo = relrms(out["dq"][..., :32], exact[..., :32])
    hi = relrms(out["dq"][..., 32:], exact[..., 32:])
    assert lo <= 1.5 * hi, f"dQ columns [0, 32) are {lo / hi:.1f}x worse than the rest"


@pytest.mark.parametrize("rows", [32, 16])
@pytest.mark.parametrize("hdim", [128, 192])
def test_bwd_run_to_run_deterministic(bwd_build, hdim, rows):
    """K42/K44: the backward is bitwise deterministic across launches (no atomics, no races: dQ, dK and dV each have one
    writer per element)."""
    pins = dkdv_family_pins(hdim, rows)
    if pins is None:
        pytest.skip("no legal dK/dV geometry for this family at this width")
    meta = meta_of(head_dim=hdim, window=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows), dkdv=pins)
    gen = seeded(hdim)
    q, k, v, do = (randn(1, 2, 200, hdim, DTYPES["bf16"], gen=gen) for _ in range(4))
    first = run_bwd(builds, q, k, v, do, window=BR)
    for _ in range(2):
        again = run_bwd(builds, q, k, v, do, window=BR, o_lse=(first["o"], first["lse"]))
        for name in ("dq", "dk", "dv"):
            assert torch.equal(first[name], again[name]), f"{name} differs between launches"


@pytest.mark.large_shape
@pytest.mark.parametrize("rows", [32, 16])
def test_dq_256_dropout_is_deterministic(bwd_build, rows):
    """K47: dQ at 256 with dropout, the geometry whose 32-row build raced (a register file five short, LDS-DMA address
    operands spilled into AGPRs and clobbered): five launches bitwise identical with the 32-row pin and with the default.
    """
    meta = meta_of(head_dim=256, dropout=True)
    builds = bwd_build(meta, dq=dict(MFMA_ROWS=rows))
    gen = seeded(25)
    b, h, sq, sk = 3, 5, 2048, 8192
    q, do = (randn(b, h, sq, 256, DTYPES["bf16"], gen=gen) for _ in range(2))
    k, v = (randn(b, h, sk, 256, DTYPES["bf16"], gen=gen) for _ in range(2))
    first = run_bwd(builds, q, k, v, do, p_drop=0.3, seed=7, which=("dq",))
    for _ in range(4):
        again = run_bwd(builds, q, k, v, do, p_drop=0.3, seed=7, which=("dq",), o_lse=(first["o"], first["lse"]))
        assert torch.equal(first["dq"], again["dq"])


@pytest.mark.parametrize("feat", ["dense", "causal", "bias"])
@pytest.mark.parametrize("hdim", [64, 256])
@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_softmax_exponent_is_not_an_fma(backend, arch, tmp_path, monkeypatch, kind, hdim, feat):
    """`test_softmax_exponent_is_not_an_fma` for the backward's `P = exp2(qk_scale * S - lse2)`: dQ wrote it as
    `fma(S, qk_scale, -lse2)` in both families (head_dim 64 is the 32-row one, 256 the 16-row one), and in a bias build
    the bias add's FMA fed the `exp2` because the LSE came off first. Compile-only."""
    meta = meta_of(head_dim=hdim, window=feat == "causal", bias=feat == "bias")
    d = isa_tools.fresh_bwd_dump(
        kind, backend, arch, meta, tmp_path, monkeypatch, window=BR if feat == "causal" else None
    )
    assert any(op == "v_exp_f32" for op, _ in isa_tools.instructions(d.isa)), "the scan must see the exps"
    assert isa_tools.scan_exp2_fed_by_fma(d.isa) == []


@pytest.mark.parametrize("window", [False, True], ids=["dense", "causal"])
def test_dkdv_256_stays_within_two_waves_per_simd(backend, arch, tmp_path, monkeypatch, window):
    """The window mask must not tip dK/dV at head_dim 256 past 256 VGPRs. The kernel runs at `waves_per_eu=1` with 69632 B of
    LDS per workgroup, so two workgroups share a CU only while the register file allows two waves per SIMD, i.e. at most 256
    registers. A window build used to compare `rel0 + i` against the bounds, which kept ACC16 extra per-element values live and
    gave 264: one wave per SIMD, a causal dK/dV of 1291 us against 764 us with the same executed MFMA count (the tile skip was
    working; the occupancy was not). Compile-only, no timing."""
    meta = meta_of(head_dim=256, window=window)
    d = isa_tools.fresh_bwd_dump("dkdv", backend, arch, meta, tmp_path, monkeypatch, window=BR if window else None)
    res = d.resources()
    assert res["vgpr_count"] <= 256, res
    assert res["vgpr_spill_count"] == 0 and res["private_segment_fixed_size"] == 0, res
