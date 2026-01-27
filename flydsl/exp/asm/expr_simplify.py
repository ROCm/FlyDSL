# Copyright 2025 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Algebraic simplification using declarative rewrite rules with singledispatch."""

import os
import sympy
from sympy import Wild, floor, Mod, Integer, Symbol, Add, Mul, Pow, Rational, S
from functools import singledispatch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

DEBUG = os.environ.get("WAVE_EXPR_SIMPLIFY_LOG", "0") == "1"

# --- Type Dispatch Helpers ---


def is_floor(e):
    return getattr(e, "func", None) == floor


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


# --- Singledispatch: get_max_value ---


@singledispatch
def get_max_value(expr, bounds: Dict) -> Optional[int]:
    """Get maximum value of expression given bounds. Default: unknown."""
    return None


@get_max_value.register(Integer)
def _(expr, bounds):
    return int(expr)


@get_max_value.register(Symbol)
def _(expr, bounds):
    for sym, (_, hi) in bounds.items():
        if str(sym) == str(expr):
            return hi
    return None


@get_max_value.register(Mod)
def _(expr, bounds):
    return int(expr.args[1]) - 1 if isinstance(expr.args[1], Integer) else None


@get_max_value.register(Add)
def _(expr, bounds):
    vals = [get_max_value(a, bounds) for a in expr.args]
    return sum(vals) if all(v is not None for v in vals) else None


@get_max_value.register(Mul)
def _(expr, bounds):
    vals = [get_max_value(a, bounds) for a in expr.args]
    if any(v is None for v in vals):
        return None
    result = 1
    for v in vals:
        result *= v
    return result


# --- Singledispatch: simplify_expr ---


@singledispatch
def simplify_expr(expr, bounds: Dict):
    """Simplify a single expression based on its type. Default: return as-is."""
    return expr


@simplify_expr.register(Mul)
def simplify_mul(expr, bounds):
    """x * 2^a * 2^b → x * 2^(a+b)."""
    shift, other = 0, []
    for f in expr.args:
        if isinstance(f, Integer) and is_pow2(int(f)):
            shift += int(f).bit_length() - 1
        elif isinstance(f, Pow) and f.args[0] == 2 and isinstance(f.args[1], Integer):
            shift += int(f.args[1])
        else:
            other.append(f)
    if shift > 0 and other:
        return (Mul(*other) if len(other) > 1 else other[0]) * (2**shift)
    return expr


@simplify_expr.register(Add)
def simplify_add(expr, bounds):
    """Simplify Add expressions: floor difference + linear floor to mod."""
    expr = _simplify_floor_difference(expr, bounds)
    expr = _simplify_linear_floor_to_mod(expr, bounds)
    return expr


# floor is a function, not a class - handle via is_floor check in main simplify

# --- Wild Symbols ---

_x, _y = Wild("x"), Wild("y")
_n = Wild("n", properties=[lambda k: isinstance(k, Integer) and k > 0])
_m = Wild("m", properties=[lambda k: isinstance(k, Integer) and k > 0])
_a, _b = Wild("a"), Wild("b")


@dataclass
class RewriteRule:
    name: str
    pattern: sympy.Expr
    replacement: sympy.Expr
    condition: Optional[Callable[[Dict, Dict], bool]] = None


REWRITE_RULES = [
    RewriteRule("floor_mod_identity", floor(_x / _n) * _n + Mod(_x, _n), _x),
    RewriteRule(
        "linear_floor_to_mod", _a * _x + (-_a * _n) * floor(_x / _n), _a * Mod(_x, _n)
    ),
    RewriteRule("nested_mod", Mod(Mod(_x, _n * _m), _n), Mod(_x, _n)),
]

# --- Pattern Matching ---


def try_match(expr, pattern) -> Optional[Dict]:
    """Match expr against pattern, with subset matching for Add/Mul."""
    match = expr.match(pattern)
    if match is not None:
        return match
    if isinstance(expr, Add) and isinstance(pattern, Add):
        return _match_subset(list(expr.args), list(pattern.args), {})
    if isinstance(expr, Mul) and isinstance(pattern, Mul):
        return _match_subset(list(expr.args), list(pattern.args), {})
    return None


def _match_subset(expr_terms, pattern_terms, bindings) -> Optional[Dict]:
    if not pattern_terms:
        return bindings
    for i, et in enumerate(expr_terms):
        match = et.match(pattern_terms[0])
        if match and all(bindings.get(k, v) == v for k, v in match.items()):
            result = _match_subset(
                expr_terms[:i] + expr_terms[i + 1 :],
                pattern_terms[1:],
                {**bindings, **match},
            )
            if result is not None:
                return result
    return None


def apply_rule(expr, rule, bounds) -> Tuple[sympy.Expr, bool]:
    """Try to apply a single rule."""
    match = try_match(expr, rule.pattern)
    if match is None or (rule.condition and not rule.condition(match, bounds)):
        return expr, False
    try:
        result = rule.replacement.xreplace(match)
        if isinstance(expr, Add):
            matched = set()
            pargs = (
                rule.pattern.args if isinstance(rule.pattern, Add) else [rule.pattern]
            )
            for pt in pargs:
                for et in expr.args:
                    if et.match(pt.xreplace(match)) is not None:
                        matched.add(et)
                        break
            remaining = [t for t in expr.args if t not in matched]
            if remaining:
                result = result + Add(*remaining)
        return result, True
    except:
        return expr, False


def apply_all_rules(expr, bounds) -> sympy.Expr:
    """Apply all rewrite rules until fixpoint."""
    changed = True
    while changed:
        changed = False
        for rule in REWRITE_RULES:
            expr, applied = apply_rule(expr, rule, bounds)
            if applied:
                changed = True
                if DEBUG:
                    print(f"[{rule.name}] Applied")
                break
    return expr


# --- Floor-specific Rules ---


def _simplify_redundant_floor(expr, bounds):
    """floor(x/n) → 0 when max(x) < n."""
    if not is_floor(expr):
        return expr
    arg = expr.args[0]
    if not isinstance(arg, Mul):
        return expr
    x_parts, n = [], None
    for f in arg.args:
        if isinstance(f, Pow) and f.args[1] == -1:
            n = f.args[0]
        elif isinstance(f, Rational) and not isinstance(f, Integer):
            n = Integer(f.q)
            if f.p != 1:
                x_parts.append(Integer(f.p))
        else:
            x_parts.append(f)
    if n and x_parts:
        x = Mul(*x_parts) if len(x_parts) > 1 else x_parts[0]
        max_x = get_max_value(x, bounds)
        if max_x is not None and max_x < int(n):
            return S.Zero
    return expr


def _simplify_floor_difference(expr, bounds):
    """Combine floor(x/c) terms when one divisor divides the other."""
    if not isinstance(expr, Add):
        return expr
    floors, other = {}, []
    for term in expr.args:
        info = _extract_floor_term(term)
        if info:
            key = (info[1], info[2])
            floors[key] = floors.get(key, S.Zero) + info[0]
        else:
            other.append(term)
    if len(floors) < 2:
        return expr
    keys = list(floors.keys())
    for i, (b1, d1) in enumerate(keys):
        for j, (b2, d2) in enumerate(keys):
            if (
                i >= j
                or b1 != b2
                or not isinstance(d1, Integer)
                or not isinstance(d2, Integer)
            ):
                continue
            large, small = (d1, d2) if int(d1) > int(d2) else (d2, d1)
            if int(large) % int(small) != 0:
                continue
            k = int(large) // int(small)
            floors[(b1, large)] = floors[(b1, small)] * k + floors[(b1, large)]
            other.append(floors[(b1, small)] * floor(Mod(b1, large) / small))
            del floors[(b1, small)]
            result = list(other)
            for (base, div), coeff in floors.items():
                if coeff != 0:
                    result.append(
                        coeff * floor(base / div) if coeff != 1 else floor(base / div)
                    )
            return _simplify_floor_difference(
                result[0] if len(result) == 1 else Add(*result) if result else S.Zero,
                bounds,
            )
    return expr


def _simplify_linear_floor_to_mod(expr, bounds):
    """a*x - a*n*floor(x/n) → a*Mod(x,n)."""
    if not isinstance(expr, Add):
        return expr
    linear, floors, other = {}, {}, []
    for term in expr.args:
        if isinstance(term, Symbol):
            linear[term] = linear.get(term, 0) + 1
        elif isinstance(term, Mul):
            sym, coeff, ok = None, S.One, True
            for f in term.args:
                if isinstance(f, Symbol):
                    if sym is None:
                        sym = f
                    else:
                        ok = False
                        break
                elif isinstance(f, (Integer, Rational)):
                    coeff *= f
                else:
                    ok = False
                    break
            if ok and sym:
                linear[sym] = linear.get(sym, 0) + coeff
                continue
        info = _extract_floor_term(term)
        if info and isinstance(info[1], Symbol):
            floors[(info[1], info[2])] = floors.get((info[1], info[2]), 0) + info[0]
            continue
        other.append(term)
    used_lin, used_fl, new_terms = set(), set(), list(other)
    for sym, a in linear.items():
        if sym in used_lin:
            continue
        for (fsym, n), b in floors.items():
            if (
                fsym == sym
                and (fsym, n) not in used_fl
                and isinstance(n, Integer)
                and b == -a * int(n)
            ):
                new_terms.append(a * Mod(sym, n))
                used_lin.add(sym)
                used_fl.add((fsym, n))
                break
    if not used_lin:
        return expr
    for sym, c in linear.items():
        if sym not in used_lin:
            new_terms.append(c * sym if c != 1 else sym)
    for (sym, n), c in floors.items():
        if (sym, n) not in used_fl:
            new_terms.append(c * floor(sym / n) if c != 1 else floor(sym / n))
    return (
        new_terms[0]
        if len(new_terms) == 1
        else Add(*new_terms) if new_terms else S.Zero
    )


# --- Helpers ---


def _extract_floor_term(term):
    """Extract (coeff, base, divisor) from c*floor(x/n)."""
    if is_floor(term):
        parts = _extract_floor_parts(term)
        return (S.One, parts[0], parts[1]) if parts else None
    if isinstance(term, Mul):
        coeff, fl = S.One, None
        for f in term.args:
            if is_floor(f):
                fl = f
            elif isinstance(f, (Integer, Rational)):
                coeff *= f
            else:
                return None
        if fl:
            parts = _extract_floor_parts(fl)
            return (coeff, parts[0], parts[1]) if parts else None
    return None


def _extract_floor_parts(floor_expr):
    """Extract (base, divisor) from floor(base/divisor)."""
    if not is_floor(floor_expr):
        return None
    arg = floor_expr.args[0]
    if not isinstance(arg, Mul):
        return None
    divisor, base = None, []
    for f in arg.args:
        if isinstance(f, Pow) and f.args[1] == -1:
            divisor = f.args[0]
        elif isinstance(f, Rational) and not isinstance(f, Integer):
            divisor = Integer(f.q)
            if f.p != 1:
                base.append(Integer(f.p))
        else:
            base.append(f)
    if divisor and base:
        return (Mul(*base) if len(base) > 1 else base[0], divisor)
    return None


def get_default_bounds():
    mk = lambda n: Symbol(n, nonnegative=True, integer=True)
    return {mk("tid_x"): (0, 1023), mk("tid_y"): (0, 1023), mk("tid_z"): (0, 1023)}


# --- Singledispatch: combine_like_terms ---


@singledispatch
def combine_like_terms(expr):
    """Combine like terms. Default: return as-is."""
    return expr


@combine_like_terms.register(Add)
def _(expr):
    """Combine like terms in Add: 3*x + 5*x → 8*x."""
    groups, consts = {}, S.Zero
    for term in expr.args:
        if isinstance(term, Integer):
            consts += term
            continue
        if isinstance(term, Mul):
            coeff, base = S.One, []
            for f in term.args:
                if isinstance(f, (Integer, Rational)):
                    coeff *= f
                else:
                    base.append(f)
            if base:
                groups.setdefault(tuple(sorted(str(f) for f in base)), []).append(
                    (coeff, base)
                )
                continue
        groups.setdefault((str(term),), []).append((S.One, [term]))
    result = []
    for terms in groups.values():
        total = sum(t[0] for t in terms)
        if total == 0:
            continue
        base_expr = Mul(*terms[0][1]) if len(terms[0][1]) > 1 else terms[0][1][0]
        result.append(total * base_expr if total != 1 else base_expr)
    if consts != 0:
        result.append(consts)
    return result[0] if len(result) == 1 else Add(*result) if result else S.Zero


# --- Main Entry Point ---


def _apply_all_simplifications(expr, bounds):
    """Apply all rules to a single expression."""
    expr = apply_all_rules(expr, bounds)
    expr = _simplify_redundant_floor(expr, bounds)
    expr = simplify_expr(expr, bounds)  # Dispatches based on type
    return expr


def simplify_for_emission(expr, bounds=None):
    """Simplify expression for optimal assembly emission."""
    if bounds is None:
        bounds = get_default_bounds()
    original = expr
    expr = _apply_all_simplifications(expr, bounds)
    # Simplify subexpressions
    if (
        hasattr(expr, "args")
        and expr.args
        and not isinstance(expr, (Symbol, Integer, Rational))
    ):
        new_args = [_apply_all_simplifications(arg, bounds) for arg in expr.args]
        if new_args != list(expr.args):
            try:
                expr = expr.func(*new_args)
            except:
                pass
    expr = combine_like_terms(expr)  # Dispatches based on type
    if DEBUG and expr != original:
        print(f"[SIMPLIFY] {original} → {expr}")
    return expr


@dataclass
class SimplifyStats:
    total: int = 0
    simplified: int = 0

    def record(self, before, after):
        self.total += 1
        if before != after:
            self.simplified += 1

    def summary(self):
        return (
            f"Simplified {self.simplified}/{self.total}"
            if self.total
            else "No expressions"
        )
