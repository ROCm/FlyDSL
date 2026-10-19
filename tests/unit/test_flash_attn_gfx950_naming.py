# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CFG-16: the naming policy of the gfx950 attention kernels, as an AST lint. No GPU.

lower_snake means a runtime value or an input description; UPPER_CASE means baked into the kernel;
Leading_upper_snake_case (clang-tidy's `readability-identifier-naming` style name, AOTriton's `constexpr_or_i32`) means a
value that is constexpr in a JIT build but must be a real argument in an AOT build.

| Identifier | Rule |
|---|---|
| kernel/launcher parameter that is an unconditional `fx.Constexpr` | UPPER_CASE |
| `Int32` / `Int64` / `Float32` parameter | lower_snake |
| parameter named in Leading_upper_snake_case | ONE parameter whose annotation `*_ANN` is `fx.Constexpr if KNOB else <runtime scalar>`, never a pointer; the knob is a `STATIC_*` one, off by default, so at the default knobs (every AOT build) it is a real `Int32` kernarg |
| lower_snake parameter with a per-build annotation (`workspace block_table block_table_stride alibi_slopes alibi_stride_b sink`) | only for folding an absent input away (`<runtime> if PRESENT else fx.Constexpr`), never for a baked value |
| `Pointer` parameter | one of the math operand names (`Q K V O B LSE DO DQ DK DV DB Delta`) or lower_snake |
| trait fields, and knob fields baked as a `const_expr` | UPPER_CASE |
| metadata fields, hint fields, and the knobs that are compiler or launch attributes | lower_snake |
| function locals (builders, kernel bodies, launchers, host wrappers) | no leading underscore (bare `_`, intentionally unused bindings and a keepalive allowlist excepted) |

Module-level private helpers keep Python's single-underscore "not API" prefix. Kernel-body locals beyond these
mechanical checks (e.g. UPPER for a `const_expr` value) remain a review rule.
"""

import ast
import dataclasses
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

from kernels.attention import flash_attn_gfx950_config as cfg  # noqa: E402

ATTN = Path(__file__).resolve().parents[2] / "kernels" / "attention"

UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOWER = re.compile(r"^[a-z][a-z0-9_]*$")
LEADING_UPPER = re.compile(r"^[A-Z][a-z][a-z0-9_]*$")
FOLD_PARAMS = {"workspace", "block_table", "block_table_stride", "alibi_slopes", "alibi_stride_b", "sink"}
OPERAND_NAMES = {"Q", "K", "V", "O", "B", "LSE", "DO", "DQ", "DK", "DV", "DB", "Delta"}
COPT_KNOBS = {"num_warps", "waves_per_eu", "daz"}
KEEPALIVE = {"_dp_keepalive"}
LOCAL_UNDERSCORE = re.compile(r"^_[^_]")


# ---------------------------------------------------------------------------
# The rules, over ASTs
# ---------------------------------------------------------------------------


def _annotation_kind(ann):
    """'constexpr' | 'runtime' | 'pointer' | 'per_build' | None for a parameter annotation."""
    if ann is None:
        return None
    text = ast.unparse(ann)
    if isinstance(ann, ast.Name) and ann.id.endswith("_ANN"):
        return "per_build"
    if "Constexpr" in text:
        return "constexpr"
    last = text.split(".")[-1].split("[")[0]
    if last in ("Int32", "Int64", "Float32", "Int16", "Boolean", "Index", "Stream"):
        return "runtime"
    if last == "Pointer":
        return "pointer"
    return None


def _branch_kind(node):
    """'constexpr' | 'runtime' | 'pointer' | None for one branch of a per-build annotation."""
    text = ast.unparse(node)
    if "Constexpr" in text:
        return "constexpr"
    last = text.split(".")[-1]
    if last in ("Int32", "Int64", "Float32", "Int16", "Boolean", "Index"):
        return "runtime"
    if last in ("Pointer", "Tensor"):
        return "pointer"
    return None


def _ann_defs(tree):
    """`NAME_ANN = a if cond else b` assignments anywhere in `tree`: name -> (kind(a), kind(b), cond text)."""
    defs = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.endswith("_ANN")
            and isinstance(node.value, ast.IfExp)
        ):
            v = node.value
            defs[node.targets[0].id] = (_branch_kind(v.body), _branch_kind(v.orelse), ast.unparse(v.test))
    return defs


def check_params(tree):
    """Violations of the parameter rules over every `@flyc.kernel` / `@flyc.jit` def and nested def in `tree`."""
    bad = []
    defs = _ann_defs(tree)
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        decos = {ast.unparse(d).split("(")[0] for d in fn.decorator_list}
        if not decos & {"flyc.kernel", "flyc.jit"}:
            continue
        for arg in fn.args.args + fn.args.kwonlyargs:
            kind = _annotation_kind(arg.annotation)
            name = arg.arg
            if kind == "per_build":
                body, orelse, cond = defs.get(arg.annotation.id, (None, None, ""))
                if LEADING_UPPER.match(name):
                    if not (body == "constexpr" and orelse == "runtime" and cond.startswith("STATIC_")):
                        bad.append(
                            f"{fn.name}({name}): a Leading_upper_snake_case parameter needs an annotation "
                            "`fx.Constexpr if STATIC_*` else a runtime scalar"
                        )
                elif name in FOLD_PARAMS:
                    if orelse != "constexpr" or body not in ("runtime", "pointer"):
                        bad.append(f"{fn.name}({name}): a lower_snake per-build parameter only folds an input away")
                else:
                    bad.append(
                        f"{fn.name}({name}): a per-build annotation is for Leading_upper_snake_case or {sorted(FOLD_PARAMS)}"
                    )
            elif LEADING_UPPER.match(name) and name not in OPERAND_NAMES:
                bad.append(
                    f"{fn.name}({name}): a Leading_upper_snake_case parameter needs a per-build `*_ANN` annotation"
                )
            elif kind == "constexpr" and not UPPER.match(name):
                bad.append(f"{fn.name}({name}): an fx.Constexpr parameter must be UPPER_CASE")
            elif kind == "runtime" and not LOWER.match(name):
                bad.append(f"{fn.name}({name}): a runtime parameter must be lower_snake")
            elif kind == "pointer" and not (name in OPERAND_NAMES or LOWER.match(name)):
                bad.append(f"{fn.name}({name}): a Pointer parameter must be a math operand name or lower_snake")
    return bad


def _stored_names(fn):
    """Names bound in `fn`'s own scope (not nested functions'), with whether each is ever read in `fn`."""
    stores, loads = {}, set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                stores.setdefault(node.id, node.lineno)
            else:
                loads.add(node.id)
        elif isinstance(node, ast.arg):
            stores.setdefault(node.arg, node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            stores.setdefault(node.name, node.lineno)  # a nested def or class is a local too
    return stores, loads


def check_locals(tree):
    """Function locals (including parameters) in builder functions must not match `^_[^_]`."""
    bad = []
    for top in tree.body:
        if not (isinstance(top, ast.FunctionDef) and top.name.startswith("build_")):
            continue
        # the whole builder, nested kernels and launchers included
        stores, loads = _stored_names(top)
        for name, line in stores.items():
            if LOCAL_UNDERSCORE.match(name) and name not in KEEPALIVE and name in loads:
                bad.append(f"{top.name}:{line}: local {name!r} has a leading underscore")
    return bad


def check_dataclass_fields(cls, rule, exceptions=()):
    return [f.name for f in dataclasses.fields(cls) if not (rule.match(f.name) or f.name in exceptions)]


# ---------------------------------------------------------------------------
# The rules on synthetic snippets (the lint has to be able to fail)
# ---------------------------------------------------------------------------

BAD_KERNEL = """
@flyc.kernel(known_block_size=[256, 1, 1])
def kernel(
    Q: fx.Pointer,
    hdimQK: fx.Int32,
    WindowLeft: fx.Int32,
    window_right: fx.Constexpr[int],
    workspace: WS_ANN,
    Alibi: ALIBI_ANN,
    MixedCase: fx.Pointer,
    Max_seqlen_q: fx.Int32,
    Window_left: WL_BAD_ANN,
    Window_right: WL_ANN,
    stray: STRAY_ANN,
    baked: BAKED_ANN,
    BLOCK_X: fx.Constexpr[int],
    sm_scale: fx.Float32,
):
    pass
"""
BAD_ANNS = """
WS_ANN = fx.Tensor if WS_RUNTIME else fx.Constexpr
ALIBI_ANN = fx.Pointer if STATIC_ALIBI else fx.Constexpr
WL_BAD_ANN = fx.Constexpr if WINDOW_RUNTIME else fx.Int32
WL_ANN = fx.Constexpr if STATIC_WINDOW else fx.Int32
STRAY_ANN = fx.Int32 if X else fx.Constexpr
BAKED_ANN = fx.Constexpr if X else fx.Int32
"""

BAD_BUILDER = """
def build_x(meta):
    _cache_tag = 1
    CACHE_TAG = _cache_tag
    _unused = 2
    _dp_keepalive = 3
    _ = CACHE_TAG
    def inner(_arg):
        return _arg
    def _helper():
        return 1
    _helper()
"""


def test_lint_rejects_bad_parameters():
    bad = check_params(ast.parse(BAD_KERNEL + BAD_ANNS))
    flagged = {b.split("(")[1].split(")")[0] for b in bad}
    # `workspace` folds an absent input away (fine); `Window_right` is a well-formed Leading_upper_snake_case pair member;
    # `Alibi` (not an allowed name here: a pointer under a STATIC_* knob), `Max_seqlen_q` (no per-build annotation),
    # `Window_left` (the knob is not a STATIC_* one), `stray` (not an allowed fold name) and `baked` (a lower_snake name
    # may not carry a baked value) are flagged.
    assert flagged == {
        "hdimQK",
        "WindowLeft",
        "window_right",
        "Alibi",
        "MixedCase",
        "Max_seqlen_q",
        "Window_left",
        "stray",
        "baked",
    }, bad


def test_lint_rejects_leading_underscore_locals():
    bad = check_locals(ast.parse(BAD_BUILDER))
    flagged = {b.split("'")[1] for b in bad}
    # `_unused` is never read (ruff's dummy-variable convention) and `_dp_keepalive` is allowlisted.
    assert flagged == {"_cache_tag", "_arg", "_helper"}, bad


def test_lint_field_rules_can_fail():
    @dataclasses.dataclass
    class Mixed:
        BLOCK_M: int = 0
        blockN: int = 0
        daz: bool = True

    assert check_dataclass_fields(Mixed, UPPER, COPT_KNOBS) == ["blockN"]


# ---------------------------------------------------------------------------
# The rules on the real modules
# ---------------------------------------------------------------------------


def test_metadata_and_hint_fields_are_lower_snake():
    for cls in (cfg.FmhaInputMetadata, cfg.FmhaHints):
        assert not check_dataclass_fields(cls, LOWER), cls


@pytest.mark.parametrize("cls", [cfg.Gfx950Traits, cfg.Gfx950DqTraits, cfg.Gfx950DkdvTraits])
def test_trait_fields_are_upper_case(cls):
    assert not check_dataclass_fields(cls, UPPER), cls


@pytest.mark.parametrize("cls", [cfg.Gfx950FwdKnobs, cfg.Gfx950DqKnobs, cfg.Gfx950DkdvKnobs])
def test_knob_fields_are_upper_case_except_the_copt_allowlist(cls):
    """UPPER for knobs baked as a `const_expr`; lower for the compiler/launch attributes."""
    assert not check_dataclass_fields(cls, UPPER, COPT_KNOBS), cls
    names = {f.name for f in dataclasses.fields(cls)}
    assert {"num_warps", "waves_per_eu", "daz"} <= names
    # ... and the allowlist is exactly the copt knobs: nothing else is lower_snake.
    assert {n for n in names if LOWER.match(n)} == COPT_KNOBS


def _new_kernel_sources():
    for path in sorted(ATTN.glob("flash_attn_gfx950*.py")):
        if path.name == "flash_attn_gfx950_config.py":
            continue
        yield path.name, ast.parse(path.read_text())


@pytest.mark.parametrize(
    "cls,make",
    [(cfg.Gfx950FwdKnobs, cfg.fwd_knobs), (cfg.Gfx950DqKnobs, cfg.dq_knobs), (cfg.Gfx950DkdvKnobs, cfg.dkdv_knobs)],
)
def test_static_knobs_are_off_at_the_defaults(cls, make):
    """A Leading_upper_snake_case parameter is `fx.Int32` (a real kernarg) at the default knobs: every `STATIC_*` knob
    defaults to False, so AOT builds (which use the defaults) keep the kernarg."""
    meta = cfg.FmhaInputMetadata(dtype_str="bf16", head_dim=64)
    knobs = make("gfx950").resolve(meta)
    statics = [f.name for f in dataclasses.fields(cls) if f.name.startswith("STATIC_")]
    assert all(getattr(knobs, n) is False for n in statics), statics


def test_kernel_and_launcher_parameters_follow_the_policy():
    for name, tree in _new_kernel_sources():
        assert not check_params(tree), name


def test_function_locals_have_no_leading_underscore():
    for name, tree in _new_kernel_sources():
        assert not check_locals(tree), name
