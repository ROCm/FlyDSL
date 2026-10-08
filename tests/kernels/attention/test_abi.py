# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-only ABI and build-identity tests of the attention builders (ABI-06, 08, 09, 10, 19..22, 24..26, forward part).

Every test compiles a **fresh** builder with the IR dump on and reads what the compiler produced (MLIR, LLVM IR, final ISA,
the code object's metadata); nothing is launched. They guard the contract AOTriton relies on: one uniquely named kernel
per builder, no hidden kernel arguments at the defaults, a cache key that distinguishes every build, and knobs that take
effect.

C13: the backend is not bit-reproducible (rebuilding AOTriton's 216 forward kernels at unchanged settings gave 2 that
differ), so every "identical" or "differs" assertion first builds the same configuration twice; if that control differs
the case xfails with the reason instead of asserting.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from kernels.attention import flash_attn_gfx950 as fwd_module
from kernels.attention import flash_attn_gfx950_config as cfg
from tests.kernels.attention import isa_tools
from tests.kernels.attention.attn_testlib import WINDOW_BOTRIGHT, bwd_compile_inputs, compile_inputs, meta_of

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

_n = iter(range(10**6))


def _dump(backend, arch, meta, tmp_path, monkeypatch, **kw):
    return isa_tools.fresh_fwd_dump(backend, arch, meta, tmp_path / f"d{next(_n)}", monkeypatch, **kw)


def _same_isa(a, b):
    return a.isa == b.isa


def _digest(dump):
    return hashlib.sha256(dump.isa.encode()).hexdigest()[:16]


def _control(backend, arch, meta, tmp_path, monkeypatch, **kw):
    """Two builds of the same configuration; xfail if the backend itself is not reproducible here (C13)."""
    a = _dump(backend, arch, meta, tmp_path, monkeypatch, **kw)
    b = _dump(backend, arch, meta, tmp_path, monkeypatch, **kw)
    if not _same_isa(a, b):
        pytest.xfail("C13: backend nondeterminism (the A/A control differs)")
    return a


# ---------------------------------------------------------------------------
# ABI-07: the optional inputs, folded away, leave the forward as it was
# ---------------------------------------------------------------------------

_BASELINE = json.loads((Path(__file__).parent / "data" / "fwd_isa_baseline.json").read_text())


@pytest.mark.parametrize("name", sorted(_BASELINE["builds"]))
def test_folded_features_leave_abi_and_isa(backend, arch, tmp_path, monkeypatch, name):
    """ABI-07: with ALiBi, sink, split-K, paged and XCD swizzle all off the kernarg block and the **instruction stream**
    equal what the forward compiled to before those features existed (`data/fwd_isa_baseline.json`, recorded at the
    forward-only commit). A byte-identity gate needs the same-input control (C13): if the rebuild itself differs the
    case xfails; if the toolchain is not the one the baseline was recorded with it skips."""
    import flydsl

    if flydsl.__version__ != _BASELINE["flydsl"]:
        pytest.skip(f"the baseline was recorded with flydsl {_BASELINE['flydsl']}, this is {flydsl.__version__}")
    want = _BASELINE["builds"][name]
    meta = meta_of(**_BASELINE["configs"][name])
    d = _dump(backend, arch, meta, tmp_path, monkeypatch)
    assert d.metadata[".kernarg_segment_size"] == want["kernarg"] and d.hidden_args == []
    sha, count = isa_tools.isa_fingerprint(d.isa)
    if sha != want["sha256"]:
        control = _dump(backend, arch, meta, tmp_path, monkeypatch)
        if isa_tools.isa_fingerprint(control.isa)[0] != sha:
            pytest.xfail("C13: backend nondeterminism (the A/A control differs)")
        pytest.fail(f"{name}: the folded ISA moved: {count} instructions against the baseline's {want['instructions']}")


# ---------------------------------------------------------------------------
# ABI-06, ABI-08, ABI-09
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feat", ["dense", "window", "bias", "dropout"])
def test_no_hidden_kernargs_at_default_knobs(backend, arch, tmp_path, monkeypatch, feat):
    """K49/C11: a `gpu.grid_dim` read appends the 256-byte hidden-argument block (`.kernarg_segment_size` 296 -> 552 once
    the first port's LPT was made live), which trips AOTriton's `--verify`. The default builds, LPT on for windows
    included, have none and a 296-byte block."""
    meta = meta_of(head_dim=64, window=feat == "window", bias=feat == "bias", dropout=feat == "dropout")
    d = _dump(backend, arch, meta, tmp_path, monkeypatch)
    assert d.hidden_args == []
    assert (
        d.metadata[".kernarg_segment_size"] == sum(s for _, s in d.kernel_args) + 4
    )  # one 4-byte hole before the strides


def test_one_uniquely_named_kernel_per_builder(backend, arch, tmp_path, monkeypatch):
    """A03/A05: the builder's launcher closure holds exactly one kernel function, named `FWD_KERNEL_NAME` (AOTriton locates
    the kernel by uniqueness), with `known_block_size` set; the code object exports that symbol."""
    from flydsl.compiler.kernel_function import KernelFunction

    meta = meta_of(head_dim=128)
    knobs = backend.fwd_knobs(arch).resolve(meta)
    fn = backend.build_fwd(meta, knobs)
    cells = [c.cell_contents for c in (fn.launcher.func.__closure__ or ())]
    kernels = [c for c in cells if isinstance(c, KernelFunction)]
    assert len(kernels) == 1 and kernels[0]._func.__name__ == cfg.FWD_KERNEL_NAME
    assert list(kernels[0]._known_block_size) == [fn.traits.BLOCK_SIZE, 1, 1]
    d = _dump(backend, arch, meta, tmp_path, monkeypatch)
    assert d.metadata[".name"].startswith(cfg.FWD_KERNEL_NAME)
    assert d.metadata[".max_flat_workgroup_size"] == fn.traits.BLOCK_SIZE
    assert d.metadata[".reqd_workgroup_size"] == [fn.traits.BLOCK_SIZE, 1, 1]


@pytest.mark.parametrize("hdim", [128, 512])
def test_constexpr_synthesised_as_zero(backend, arch, tmp_path, monkeypatch, hdim):
    """A01: `flyc_compile` synthesises `0` for every `Constexpr` launcher parameter. Compiling with every one of them 0
    must still build the knob's `BLOCK_DMODEL`-wide tile (a knob taken as a launcher argument would build a 0-wide tile)
    with the kernarg list unchanged."""
    meta = meta_of(head_dim=hdim)
    knobs = backend.fwd_knobs(arch).resolve(meta)
    fn = backend.build_fwd(meta, knobs)
    args, kw = compile_inputs(meta)
    packed, _ = fn.host_args(*args, **kw)
    # At the defaults the trailing `block_table_stride` and the two folded slots are already the `0` the driver
    # synthesises; `Max_seqlen_*` and `Window_*` are real Int32 kernargs there (Leading_upper_snake_case, plan 4.8), not constexpr.
    assert packed[-1] == 0 and packed[6:8] == (0, 0)
    d = isa_tools.compile_dump(lambda: fn.compile(*args, **kw), tmp_path / "z", monkeypatch)
    assert d.metadata[".group_segment_fixed_size"] == fn.traits.LDS_KV_TOTAL_SIZE * fn.traits.BF16_BYTES
    assert len(d.kernel_args) == 43


# ---------------------------------------------------------------------------
# ABI-10: compile hints ride on the launcher
# ---------------------------------------------------------------------------


def test_compile_hints_ride_on_the_launcher(backend, arch, tmp_path, monkeypatch):
    """K52/§5.2: the hints (`fast_fp_math`, `unsafe_fp_math`, two `llvm_options`) are attached to the launcher
    (`flyc.compile[hints](launch)`), so a direct `jf(*args)` compiles like the JIT path. Without them the build loses the
    `fast` flags; with them a direct call and `.compile` produce the same binary."""
    meta = meta_of(head_dim=64)
    hinted = _control(backend, arch, meta, tmp_path, monkeypatch)
    assert hinted.llvm_ir.count(" fast ") > 100

    # A direct call of the launcher (what AOTriton's driver does) compiles the same binary as `.compile`.
    knobs = backend.fwd_knobs(arch).resolve(meta)
    fn = backend.build_fwd(meta, knobs)
    args, kw = compile_inputs(meta)
    packed, _ = fn.host_args(*args, **kw)

    def direct():
        import flydsl.expr as fx

        fn.launcher(*packed, fx.Stream(None))
        torch.cuda.synchronize()

    via_call = isa_tools.compile_dump(direct, tmp_path / "direct", monkeypatch)
    assert via_call.isa == hinted.isa, "a direct jf(*args) call must compile the same binary as the JIT path"

    monkeypatch.setattr(fwd_module, "_COMPILE_HINTS", {})
    bare = _dump(backend, arch, meta, tmp_path, monkeypatch)
    assert (
        bare.llvm_ir.count(" fast ") < hinted.llvm_ir.count(" fast ") // 2
    ), "the hints must be what adds the fast flags"


def test_llvm_options_reach_codegen(backend, arch, tmp_path, monkeypatch):
    """The compile hints' `llvm_options` are real `cl::opt`s the backend parses (an unknown one raises, so a silently dead
    knob cannot hide), and `enable-post-misched=false` visibly changes the schedule. `lsr-drop-solution` was dropped from the
    hints (plan Q20): the backend accepts it but it left the ISA byte-identical on 11 forward builds, so it only perturbed
    the cache key; it must not creep back."""
    meta = meta_of(head_dim=64)
    base = dict(fwd_module._COMPILE_HINTS)
    opts = dict(base["llvm_options"])

    def isa_with(options):
        monkeypatch.setattr(fwd_module, "_COMPILE_HINTS", {**base, "llvm_options": options})
        return _dump(backend, arch, meta, tmp_path, monkeypatch).isa

    assert set(opts) == {"enable-post-misched"}
    full = isa_with(opts)
    assert isa_with({}) != full
    with pytest.raises(RuntimeError, match="Unknown LLVM option"):
        isa_with({**opts, "no-such-option-xyz": True})


# ---------------------------------------------------------------------------
# ABI-19, ABI-20, ABI-21
# ---------------------------------------------------------------------------


def test_lpt_tile_order_changes_the_isa(backend, arch, tmp_path, monkeypatch):
    """K49: the first port reversed `q_block_idx` after `q_start` had been computed, so the ISA with LPT on was
    **byte-identical** to off. Live, it differs (after an identical control) and neither has hidden kernargs."""
    meta = meta_of(head_dim=64, window=True)
    on = _control(backend, arch, meta, tmp_path, monkeypatch, LPT_TILE_ORDER=True)
    off = _dump(backend, arch, meta, tmp_path, monkeypatch, LPT_TILE_ORDER=False)
    assert on.isa != off.isa
    assert on.hidden_args == off.hidden_args == []
    assert on.metadata[".kernarg_segment_size"] == off.metadata[".kernarg_segment_size"] == 296


def test_static_window_drops_the_left_bound_compare(backend, arch, tmp_path, monkeypatch):
    """§4.2: a baked **unbounded** left edge compiles the left-bound compare and the dead-tile skip out (measured 15% fewer
    instructions), while a baked finite bound keeps them; the two runtime window slots fold away (8 kernarg bytes)."""
    meta = meta_of(head_dim=64, window=True)
    br = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
    runtime = _control(backend, arch, meta, tmp_path, monkeypatch, window=br)
    static = _dump(backend, arch, meta, tmp_path, monkeypatch, window=br, STATIC_WINDOW=True)
    finite = _dump(backend, arch, meta, tmp_path, monkeypatch, window=(127, 0), STATIC_WINDOW=True)
    n = lambda d: sum(isa_tools.isa_stats(d.isa).values())  # noqa: E731
    assert n(static) < 0.95 * n(runtime), (n(static), n(runtime))
    assert n(finite) > n(static)
    assert static.metadata[".kernarg_segment_size"] == runtime.metadata[".kernarg_segment_size"] - 8


def test_static_seqlen_drops_the_mask_all_tiles_walk(backend, arch, tmp_path, monkeypatch):
    """STATIC_SEQLEN (plan 4.2, r13): with `sq == sk` baked, a baked causal window resolves `CROSS_SEQLEN` statically, so the
    `MASK_ALL_TILES` walk (a mask on every loop tile) drops out of the ISA; with `sk % BLOCK_N == 0` the KV-tail mask goes
    too. `Max_seqlen_q/k` are Leading_upper_snake_case: `Constexpr` here, a real Int32 kernarg at the defaults."""
    br = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
    meta = meta_of(head_dim=64, window=True)
    window_only = _control(backend, arch, meta, tmp_path, monkeypatch, window=br, STATIC_WINDOW=True)
    static = _dump(backend, arch, meta, tmp_path, monkeypatch, window=br, STATIC_WINDOW=True, STATIC_SEQLEN=True)
    n = lambda d: sum(isa_tools.isa_stats(d.isa).values())  # noqa: E731
    assert n(static) < 0.9 * n(window_only), (n(static), n(window_only))
    # `Max_seqlen_q/k` (slots 12 and 13) are `Constexpr` in this build, so exactly those two kernargs fold away.
    assert static.kernel_args == window_only.kernel_args[:12] + window_only.kernel_args[14:]
    assert static.hidden_args == [] and window_only.kernel_args[12:14] == [("by_value", 4)] * 2


def test_static_seqlen_leaves_the_default_abi_alone(backend, arch, tmp_path, monkeypatch):
    """At default knobs the kernarg block is unchanged by the knob's existence (only the `def` list grows)."""
    d = _dump(backend, arch, meta_of(head_dim=64), tmp_path, monkeypatch)
    assert d.metadata[".kernarg_segment_size"] == 296


def test_return_lse_modes(backend, arch, tmp_path, monkeypatch):
    """A12/K24: `"never"` drops the LSE store; `"always"` drops the null-pointer branch; `"runtime"` stays within 1% of
    `"always"`. The LSE kernarg exists in every mode, so the mode never changes the ABI."""
    meta = meta_of(head_dim=64)
    dumps = {m: _dump(backend, arch, meta, tmp_path, monkeypatch, RETURN_LSE=m) for m in ("runtime", "always", "never")}
    n = {m: sum(isa_tools.isa_stats(d.isa).values()) for m, d in dumps.items()}
    assert n["never"] < n["always"] <= n["runtime"] <= n["always"] * 1.01, n
    assert dumps["runtime"].kernel_args == dumps["always"].kernel_args == dumps["never"].kernel_args


# ---------------------------------------------------------------------------
# ABI-22: distinct builds are distinct binaries (and never a cache hit)
# ---------------------------------------------------------------------------


def test_distinct_builds_are_distinct_binaries(backend, arch, tmp_path, monkeypatch):
    """K53/C09: with the JIT disk cache on in a temp dir, build the bias-free version first, then each variant: every
    build compiles (a cache hit writes no dump and would fail `compile_dump`) to a different binary. The old cache tag
    omitted 46 of 93 trait fields, so a bias build received the bias-free binary."""
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    variants = [
        ("base", meta_of(head_dim=128), {}, None),
        ("bias", meta_of(head_dim=128, bias=True), {}, None),
        ("dropout", meta_of(head_dim=128, dropout=True), {}, None),
        ("window", meta_of(head_dim=128, window=True), {}, (WINDOW_BOTRIGHT,) * 2),
        ("static_window", meta_of(head_dim=128, window=True), {"STATIC_WINDOW": True}, (WINDOW_BOTRIGHT,) * 2),
        ("lse_always", meta_of(head_dim=128), {"RETURN_LSE": "always"}, None),
        ("lse_never", meta_of(head_dim=128), {"RETURN_LSE": "never"}, None),
        ("daz_off", meta_of(head_dim=128), {"daz": False}, None),
        ("geom_4w", meta_of(head_dim=128), dict(num_warps=4, BLOCK_M=128, BLOCK_N=64, HEAD_DIM_GRANULE=64), None),
        ("geom_8w128", meta_of(head_dim=128), dict(num_warps=8, BLOCK_M=128, BLOCK_N=64, HEAD_DIM_GRANULE=64), None),
    ]
    seen = {}
    for name, meta, pins, window in variants:
        d = _dump(backend, arch, meta, tmp_path, monkeypatch, window=window, **pins)
        h = _digest(d)
        assert h not in seen, f"{name} compiled to the same binary as {seen[h]}"
        seen[h] = name


# ---------------------------------------------------------------------------
# ABI-24 (large), ABI-25, ABI-26
# ---------------------------------------------------------------------------


@pytest.mark.large_shape
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("feat", ["dense", "window", "bias", "dropout", "bias_dropout"])
@pytest.mark.parametrize("hdim", list(cfg.LADDER) + [17, 100, 129, 300])
def test_compile_matrix(backend, arch, tmp_path, monkeypatch, hdim, feat, dtype_str):
    """A13: every rung and padded width x {dense, window, bias, dropout, bias+dropout} x dtype compiles (some PADDED_HEAD
    variants once failed to compile while their unpadded twins did not)."""
    meta = meta_of(
        dtype_str=dtype_str, head_dim=hdim, window=feat == "window", bias="bias" in feat, dropout="dropout" in feat
    )
    d = _dump(backend, arch, meta, tmp_path, monkeypatch, window=(WINDOW_BOTRIGHT,) * 2 if feat == "window" else None)
    assert d.hidden_args == []


def _origin(dump):
    return dump.mlir("origin")


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_fast_math_exemptions_survive(backend, arch, tmp_path, monkeypatch, hdim):
    """K59/K06/K07/C06: under the ambient `fast` compile hint LLVM may delete a -inf mask or fold `log(0)` to poison. The
    bias add (`S += bias * log2e`) and the masked-row LSE `log` carry `contract|reassoc` only (never `ninf`/`nnan`/`fast`),
    and the masked-row LSE select tests the bit pattern of `l` with an integer compare."""
    bias = _origin(_dump(backend, arch, meta_of(head_dim=hdim, bias=True), tmp_path, monkeypatch))
    plain = _origin(_dump(backend, arch, meta_of(head_dim=hdim), tmp_path, monkeypatch))
    import re

    def log2e_ops(text, flag):
        return len(re.findall(r"arith\.(?:mulf|addf) [^\n]*fastmath<" + flag + ">", text))

    assert log2e_ops(bias, "reassoc,contract") >= 64, "the bias add and its log2e scale carry the safe flags"
    assert log2e_ops(plain, "reassoc,contract") <= 4
    for text in (bias, plain):
        assert not re.findall(r"math\.log [^\n]*fastmath<fast>", text), "the LSE log must not carry `fast`"
        assert re.search(r"arith\.cmpi ne, [^\n]*: i32", text), "the masked-row LSE select must be an integer compare"


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_traced_conditions_stay_runtime(backend, arch, tmp_path, monkeypatch, hdim):
    """K54/C07/C08: a Python `if` on a runtime value outside a traced region is decided at **trace time** (silently), and a
    `for ... init=` loop in a plain function is skipped. The kernel's `scf.if`/`scf.for` counts must stay at or above the
    golden taken from the first working port, so a branch or loop folded at trace time shows up here."""
    golden = {64: (16, 1), 384: (12, 1)}[hdim]
    text = _origin(_dump(backend, arch, meta_of(head_dim=hdim), tmp_path, monkeypatch))
    n_if, n_for = text.count("scf.if"), text.count("scf.for")
    assert n_if >= golden[0] and n_for >= golden[1], (n_if, n_for, golden)


# ---------------------------------------------------------------------------
# XCD swizzle: a compile-time no-op where it does not apply, a visible change where it does
# ---------------------------------------------------------------------------


def test_xcd_swizzle_is_a_noop_where_it_does_not_apply(backend, arch, tmp_path, monkeypatch):
    """Causal builds (a causal mask makes block i's work grow with i) never swizzle, so the knob compiles to the same ISA
    as off. A dense build that does swizzle differs and reads `grid_dim`, which appends the hidden-argument block: a
    feature knob, off for AOT, so exempt from the no-hidden-kernargs rule (plan 1.4)."""
    causal = meta_of(head_dim=64, window=True)
    off = _control(backend, arch, causal, tmp_path, monkeypatch)
    assert _same_isa(off, _dump(backend, arch, causal, tmp_path, monkeypatch, XCD_SWIZZLE=True))
    dense = meta_of(head_dim=64)
    plain, swizzled = _dump(backend, arch, dense, tmp_path, monkeypatch), _dump(
        backend, arch, dense, tmp_path, monkeypatch, XCD_SWIZZLE=True
    )
    assert plain.hidden_args == [] and swizzled.hidden_args != []
    assert not _same_isa(plain, swizzled)


# ---------------------------------------------------------------------------
# The backward builders (dQ and dK/dV): the same contract, forward's tests' twins
# ---------------------------------------------------------------------------

BWD = {"dq": cfg.DQ_KERNEL_NAME, "dkdv": cfg.DKDV_KERNEL_NAME}

# The kernel `def` parameter lists AOTriton must declare, in order (the declared order is frozen and load-bearing). The
# tensor group is `q k v b do <outputs> lse delta`, AOTriton's; none of these is folded, so the kernarg block is the list.
_BWD_COMMON = (
    "seqinfo_q0 seqinfo_q1 seqinfo_k0 seqinfo_k1 varlen_bits num_seqlens Max_seqlen_q Max_seqlen_k Window_left Window_right "
    "philox_seed_ptr philox_offset1 philox_offset2 idropout_p dropout_scale num_head_q num_head_k hdim_qk hdim_vo sm_scale "
    "stride_q_batch stride_q_head stride_q_seq stride_k_batch stride_k_head stride_k_seq stride_v_batch stride_v_head "
    "stride_v_seq stride_do_batch stride_do_head stride_do_seq"
)
BWD_DEF_PARAMS = {
    "dq": (
        "Q K V B DO DQ DB LSE Delta " + _BWD_COMMON + " stride_dq_batch stride_dq_head stride_dq_seq stride_b_batch "
        "stride_b_head stride_b_seq_q stride_db_batch stride_db_head stride_db_seq_q"
    ).split(),
    "dkdv": (
        "Q K V B DO DK DV LSE Delta " + _BWD_COMMON + " stride_dk_batch stride_dk_head stride_dk_seq stride_dv_batch "
        "stride_dv_head stride_dv_seq stride_b_batch stride_b_head stride_b_seq_q"
    ).split(),
}
BWD_BUILDERS = {"dq": "build_flash_attn_gfx950_dq", "dkdv": "build_flash_attn_gfx950_dkdv"}
BWD_FILES = {"dq": "flash_attn_gfx950_dq.py", "dkdv": "flash_attn_gfx950_dkdv.py"}


@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_def_parameter_golden_list(kind):
    """ABI-04 for the backward: the kernel `def` parameter list equals a checked-in golden."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "kernels" / "attention"
    tree = ast.parse((root / BWD_FILES[kind]).read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == BWD_BUILDERS[kind])
    kernel = next(n for n in ast.walk(build) if isinstance(n, ast.FunctionDef) and n.name == BWD[kind])
    assert [a.arg for a in kernel.args.args] == BWD_DEF_PARAMS[kind]


@pytest.mark.parametrize("feat", ["dense", "window", "bias", "dropout"])
@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_no_hidden_kernargs_and_one_named_kernel(backend, arch, tmp_path, monkeypatch, kind, feat):
    """ABI-05/06 for the backward: at the defaults the explicit kernarg list is the whole `def` (50 arguments, 352 bytes,
    nothing folded), there are **no hidden arguments** (a `gpu.grid_dim` read would append the 256-byte block), the code
    object exports the one uniquely named kernel, and `known_block_size` is set."""
    meta = meta_of(head_dim=64, window=feat == "window", bias=feat == "bias", dropout=feat == "dropout")
    d = isa_tools.fresh_bwd_dump(kind, backend, arch, meta, tmp_path / f"{kind}{feat}", monkeypatch)
    assert d.hidden_args == []
    assert len(d.kernel_args) == len(BWD_DEF_PARAMS[kind])
    assert d.metadata[".kernarg_segment_size"] == 352
    assert d.metadata[".name"].startswith(BWD[kind])
    knobs = getattr(backend, f"{kind}_knobs")(arch).resolve(meta)
    traits = getattr(backend, f"{kind}_traits")(meta, knobs)
    assert d.metadata[".max_flat_workgroup_size"] == traits.BLOCK_SIZE
    assert d.metadata[".reqd_workgroup_size"] == [traits.BLOCK_SIZE, 1, 1]


@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_builder_holds_exactly_one_kernel(backend, arch, kind):
    from flydsl.compiler.kernel_function import KernelFunction

    meta = meta_of(head_dim=64)
    fn = getattr(backend, f"build_{kind}")(meta, getattr(backend, f"{kind}_knobs")(arch).resolve(meta))
    cells = [c.cell_contents for c in (fn.launcher.func.__closure__ or ())]
    kernels = [c for c in cells if isinstance(c, KernelFunction)]
    assert len(kernels) == 1 and kernels[0]._func.__name__ == BWD[kind]


@pytest.mark.parametrize("daz,mode", [(True, 0), (False, 3)])
@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_daz_reaches_the_hardware_mode(backend, arch, tmp_path, monkeypatch, kind, daz, mode):
    """ABI-11 for the backward: `llvm.denormal_fpenv` moves `.amdhsa_float_denorm_mode_32` (0 flush, 3 IEEE); the old
    `denormal-fp-math-f32` passthrough in these launchers never did."""
    d = isa_tools.fresh_bwd_dump(kind, backend, arch, meta_of(head_dim=64), tmp_path, monkeypatch, daz=daz)
    assert d.denorm_mode_32 == mode


@pytest.mark.parametrize("kind", ["dq", "dkdv"])
def test_bwd_compile_hints_ride_on_the_launcher(backend, arch, tmp_path, monkeypatch, kind):
    """ABI-10 for the backward: a direct `jf(*args)` of the launcher (what AOTriton's driver does) compiles the same binary as
    `.compile`, because the forward's hints (`enable-post-misched=false`, ...) are attached to the launcher itself. (The
    backward bodies spell their fast-math flags explicitly, so unlike the forward the flags do not depend on the hints.)
    """
    import flydsl.expr as fx

    meta = meta_of(head_dim=64)
    via_compile = isa_tools.fresh_bwd_dump(kind, backend, arch, meta, tmp_path / "c", monkeypatch)
    knobs = getattr(backend, f"{kind}_knobs")(arch).resolve(meta)
    fn = getattr(backend, f"build_{kind}")(meta, knobs)
    args, kw = bwd_compile_inputs(kind, meta)
    packed, _ = fn.host_args(*args, **kw)

    def direct():
        fn.launcher(*packed, fx.Stream(None))
        torch.cuda.synchronize()

    via_call = isa_tools.compile_dump(direct, tmp_path / "d", monkeypatch)
    assert via_call.isa == via_compile.isa, "a direct jf(*args) call must compile the same binary as the JIT path"
