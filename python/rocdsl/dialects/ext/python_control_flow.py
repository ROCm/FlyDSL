"""Python control-flow lowering helpers.

RocDSL executes Python code to build MLIR IR. A plain Python loop like:

  for i in range(10):
      ...

is executed by the Python interpreter, effectively *unrolling* IR construction.

Sometimes we want loops written with Python syntax to become IR loops (`scf.for`)
instead of Python unrolling. This module provides an opt-in AST rewrite that
transforms:

  for i in range(bound):
      body(i)

into:

  with scf.range_(...) as i:   # emits scf.for
      body(i)

To explicitly request Python unrolling (compile-time loop expansion), use
`range_constexpr(...)` in the loop iterable:

  for i in range_constexpr(10):
      ...

Limitations (current):
  - `break` / `continue` inside rewritten loops are not supported.
  - `for ... else:` is not supported.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class LoweringOptions:
    scf_alias: str = "__rocdsl_scf"
    value_alias: str = "__rocdsl_Value"
    is_value_fn: str = "__rocdsl_is_value"
    lower_builtin_range: bool = True


class _BreakContinueChecker(ast.NodeVisitor):
    def __init__(self):
        self.has_break_or_continue = False

    def visit_Break(self, node):  # noqa: N802
        self.has_break_or_continue = True

    def visit_Continue(self, node):  # noqa: N802
        self.has_break_or_continue = True


class _RangeForLowerer(ast.NodeTransformer):
    def __init__(self, opts: LoweringOptions):
        super().__init__()
        self.opts = opts
        self._tmp_id = 0

    def _fresh(self, prefix: str) -> str:
        self._tmp_id += 1
        return f"__rocdsl_{prefix}_{self._tmp_id}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        node = self.generic_visit(node)
        # Drop decorators: caller will re-wrap.
        node.decorator_list = []

        # Inject imports and helpers at function entry so the transformed code is self-contained.
        injected = [
            ast.ImportFrom(
                module="rocdsl.dialects.ext",
                names=[ast.alias(name="scf", asname=self.opts.scf_alias)],
                level=0,
            ),
            ast.ImportFrom(
                module="rocdsl.dialects.ext.python_control_flow",
                names=[ast.alias(name="range_constexpr", asname="range_constexpr")],
                level=0,
            ),
        ]

        node.body = injected + node.body
        return ast.fix_missing_locations(node)

    def visit_For(self, node: ast.For):  # noqa: N802
        node = self.generic_visit(node)

        if node.orelse:
            raise NotImplementedError("Lowering does not support `for ... else:`")

        # Only handle builtin `range(...)` for now.
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)):
            return node
        func_id = node.iter.func.id
        if func_id == "range_constexpr":
            # Explicit opt-out: keep Python unrolling semantics.
            node.iter.func.id = "range"
            return node
        if func_id != "range" or not self.opts.lower_builtin_range:
            return node

        # Reject break/continue (they won't map to scf semantics in this simple rewrite).
        chk = _BreakContinueChecker()
        for stmt in node.body:
            chk.visit(stmt)
        if chk.has_break_or_continue:
            raise NotImplementedError("break/continue inside lowered range-loops is not supported yet")

        # Evaluate range args once into temps.
        args = list(node.iter.args)
        if not (1 <= len(args) <= 3):
            return node

        tmp_names = [self._fresh(f"range_arg{i}") for i in range(len(args))]
        assigns = [
            ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=a)
            for n, a in zip(tmp_names, args)
        ]

        # with scf.range_(args...) as <target>: body
        with_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                attr="range_",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=n, ctx=ast.Load()) for n in tmp_names],
            keywords=[],
        )
        with_stmt = ast.With(
            items=[ast.withitem(context_expr=with_call, optional_vars=node.target)],
            body=node.body,
            type_comment=None,
        )

        lowered = assigns + [with_stmt]
        for n in lowered:
            ast.copy_location(n, node)
        return lowered


def range_constexpr(*args):
    """Marker iterable to keep Python unrolling semantics for a `for` loop.

    This is intended to be used in code that otherwise enables range-loop lowering:

      for i in range_constexpr(4):
          ...
    """
    return range(*args)


def lower_range_for_loops(fn: Callable, *, options: Optional[LoweringOptions] = None) -> Callable:
    """Return a new function with range-loop lowering applied."""
    if options is None:
        options = LoweringOptions()

    try:
        src = inspect.getsource(fn)
    except OSError as e:  # pragma: no cover
        raise RuntimeError(f"Cannot get source for function {fn.__name__}: {e}") from e

    src = textwrap.dedent(src)
    mod = ast.parse(src)

    func_def = None
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            func_def = node
            break
    if func_def is None:
        raise RuntimeError(f"Could not find function definition for {fn.__name__} in source")

    transformer = _RangeForLowerer(options)
    new_def = transformer.visit(func_def)
    new_mod = ast.Module(body=[new_def], type_ignores=[])
    new_mod = ast.fix_missing_locations(new_mod)

    # Build globals: keep original globals and materialize closure vars as globals.
    g: Dict = dict(fn.__globals__)
    if fn.__closure__ and fn.__code__.co_freevars:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                g[name] = cell.cell_contents
            except ValueError:
                pass

    compiled = compile(new_mod, filename=fn.__code__.co_filename, mode="exec")
    l: Dict = {}
    exec(compiled, g, l)  # noqa: S102 (intentional)
    new_fn = l[fn.__name__]

    # Preserve a few useful attributes.
    for attr in ("__defaults__", "__kwdefaults__", "__annotations__", "__doc__", "__module__", "__qualname__"):
        try:
            setattr(new_fn, attr, getattr(fn, attr))
        except Exception:
            pass
    if hasattr(fn, "__type_params__"):
        setattr(new_fn, "__type_params__", getattr(fn, "__type_params__"))

    return new_fn


