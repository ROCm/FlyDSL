# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ast
import contextlib
import difflib
import functools
import inspect
import types
import warnings
from textwrap import dedent
from typing import List

from .._mlir import ir
from .._mlir.dialects import arith, scf
from ..expr import const_expr
from ..expr.meta import capture_user_location
from ..expr.typing import as_dsl_value, as_ir_value
from ..utils import env, log


def _set_lineno(node, n=1):
    for child in ast.walk(node):
        child.lineno = n
        child.end_lineno = n
    return node


def _locate_block_args(block, loc):
    """Give a region block's arguments (e.g. scf.for iv / iter_args) *loc*."""
    for arg in block.arguments:
        ir.BlockArgument(arg).set_location(loc)


def _find_func_in_code_object(co, func_name):
    for c in co.co_consts:
        if type(c) is types.CodeType:
            if c.co_name == func_name:
                return c
            else:
                f = _find_func_in_code_object(c, func_name)
                if f is not None:
                    return f


def _is_constexpr(node):
    if not isinstance(node, ast.Call):
        return False
    target = node.func
    target_name = getattr(target, "id", None) or getattr(target, "attr", None)
    return target_name == "const_expr"


def _unwrap_constexpr(node):
    if _is_constexpr(node):
        return node.args[0] if node.args else node
    return node


def _collect_assigned_vars(body_stmts, active_symbols, orelse_stmts=None, test_expr=None):
    write_args = []
    invoked_args = []

    def add_unique(items, name):
        if isinstance(name, str) and name not in items:
            items.append(name)

    def in_active_symbols(name):
        return any(name in symbol_scope for symbol_scope in active_symbols)

    class RegionAnalyzer(ast.NodeVisitor):
        force_store = False

        @staticmethod
        def _get_call_base(func_node):
            if isinstance(func_node, ast.Attribute):
                if isinstance(func_node.value, ast.Attribute):
                    return RegionAnalyzer._get_call_base(func_node.value)
                if isinstance(func_node.value, ast.Name):
                    return func_node.value.id
            return None

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store) or self.force_store:
                add_unique(write_args, node.id)

        def visit_Subscript(self, node):
            if isinstance(node.ctx, ast.Store):
                self.force_store = True
                self.visit(node.value)
                self.force_store = False
                self.visit(node.slice)
            else:
                self.generic_visit(node)

        def visit_Assign(self, node):
            self.force_store = True
            for target in node.targets:
                self.visit(target)
            self.force_store = False
            self.visit(node.value)

        def visit_AugAssign(self, node):
            self.force_store = True
            self.visit(node.target)
            self.force_store = False
            self.visit(node.value)

        def visit_Call(self, node):
            base_name = RegionAnalyzer._get_call_base(node.func)
            if base_name is not None and base_name != "self":
                add_unique(invoked_args, base_name)

            self.generic_visit(node)

    analyzer = RegionAnalyzer()
    analyzer.visit(ast.Module(body=body_stmts, type_ignores=[]))
    if orelse_stmts:
        analyzer.visit(ast.Module(body=orelse_stmts, type_ignores=[]))
    if test_expr is not None:
        analyzer.visit(ast.Module(body=[ast.Expr(value=test_expr)], type_ignores=[]))

    invoked_args = [name for name in invoked_args if name not in write_args]
    write_args = [name for name in write_args if in_active_symbols(name)]
    invoked_args = [name for name in invoked_args if in_active_symbols(name)]
    return write_args + invoked_args


def _check_local_var(name, local_vars):
    if name not in local_vars:
        warnings.warn(
            f"Variable '{name}' is assigned inside a control flow body (while/if/for) "
            f"but is not a local variable in the current scope. "
            f"It may be a closure variable captured from an outer function. "
            f"Fix: pass it as a function parameter or assign it to a local variable "
            f"before the control flow statement.",
            stacklevel=2,
        )
        return None
    return local_vars[name]


def _state_value_expr(name):
    return ast.Call(
        func=ast.Name(id="__check_local_var", ctx=ast.Load()),
        args=[
            ast.Constant(value=name),
            ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
        ],
        keywords=[],
    )


class ASTRewriter:
    transformers: List = []
    rewrite_globals: dict = {}

    @classmethod
    def register(cls, transformer):
        cls.transformers.append(transformer)
        if hasattr(transformer, "rewrite_globals"):
            cls.rewrite_globals.update(transformer.rewrite_globals())
        return transformer

    @classmethod
    def transform(cls, f):
        if not cls.transformers:
            return f

        f_src = dedent(inspect.getsource(f))
        module = ast.parse(f_src)
        assert isinstance(module.body[0], ast.FunctionDef), f"unexpected ast node {module.body[0]}"

        context = types.SimpleNamespace(python_globals=f.__globals__)
        context.filename = f.__code__.co_filename
        for transformer_ctor in cls.transformers:
            orig_code = ast.unparse(module) if env.debug.ast_diff else None
            func_node = module.body[0]
            rewriter = transformer_ctor(context=context, first_lineno=f.__code__.co_firstlineno - 1)
            func_node = rewriter.visit(func_node)
            if env.debug.ast_diff:
                new_code = ast.unparse(func_node)
                diff = list(
                    difflib.unified_diff(
                        orig_code.splitlines(),
                        new_code.splitlines(),
                        lineterm="",
                    )
                )
                log().info("[%s diff]\n%s", rewriter.__class__.__name__, "\n".join(diff))
            module.body[0] = func_node

        log().info("[final transformed code]\n\n%s", ast.unparse(module))

        if f.__closure__:
            enclosing_mod = ast.FunctionDef(
                name="enclosing_mod",
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[],
                decorator_list=[],
                type_params=[],
            )
            for var in f.__code__.co_freevars:
                enclosing_mod.body.append(
                    ast.Assign(
                        targets=[ast.Name(var, ctx=ast.Store())],
                        value=ast.Constant(None, kind="None"),
                    )
                )
            enclosing_mod = _set_lineno(enclosing_mod, module.body[0].lineno)
            enclosing_mod = ast.fix_missing_locations(enclosing_mod)
            enclosing_mod.body.extend(module.body)
            module.body = [enclosing_mod]

        module = ast.fix_missing_locations(module)
        module = ast.increment_lineno(module, f.__code__.co_firstlineno - 1)
        module_code_o = compile(module, f.__code__.co_filename, "exec")
        new_f_code_o = _find_func_in_code_object(module_code_o, f.__name__)
        if new_f_code_o is None:
            log().warning("could not find rewritten function %s in code object", f.__name__)
            return f

        for name, val in cls.rewrite_globals.items():
            f.__globals__[name] = val

        # AST transforms may remove free-variable references, e.g.
        # const_expr(True) is unpacked to True, dropping const_expr
        # from co_freevars.  This makes co_freevars shorter than
        # __closure__, so direct f.__code__ assignment raises ValueError.
        # Rebuild the function with a matching (smaller) closure.
        if f.__closure__ and new_f_code_o.co_freevars != f.__code__.co_freevars:
            old_cells = {name: cell for name, cell in zip(f.__code__.co_freevars, f.__closure__)}
            new_closure = []
            for var in new_f_code_o.co_freevars:
                if var in old_cells:
                    new_closure.append(old_cells[var])
                else:
                    raise RuntimeError(
                        f"AST rewriter produced free variable {var!r} that " f"is not in the original closure"
                    )
            new_f = types.FunctionType(
                new_f_code_o,
                f.__globals__,
                f.__name__,
                f.__defaults__,
                tuple(new_closure),
            )
            new_f.__kwdefaults__ = f.__kwdefaults__
            functools.update_wrapper(new_f, f)
            return new_f

        f.__code__ = new_f_code_o
        return f


_ASTREWRITE_MARKER = "_flydsl_ast_rewriter_generated_"


class SymbolScopeTracker:
    def __init__(self):
        self.scopes = []
        self.callables = []

    def record_symbol(self, name: str):
        if not self.scopes:
            return
        if name == "_":
            return
        self.scopes[-1].add(name)

    def record_callable(self, name: str):
        if not self.callables:
            return
        self.callables[-1].add(name)

    def snapshot_symbol_scopes(self):
        return self.scopes.copy()

    def snapshot_callable_scopes(self):
        return self.callables.copy()

    @contextlib.contextmanager
    def function_scope(self):
        self.scopes.append(set())
        self.callables.append(set())
        try:
            yield
        finally:
            self.scopes.pop()
            self.callables.pop()

    @contextlib.contextmanager
    def control_flow_scope(self):
        self.scopes.append(set())
        try:
            yield
        finally:
            self.scopes.pop()


class Transformer(ast.NodeTransformer):
    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno
        self.symbol_scopes = SymbolScopeTracker()

    def _record_target_symbols(self, target):
        if isinstance(target, ast.Name):
            self.symbol_scopes.record_symbol(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self._record_target_symbols(t)
        elif isinstance(target, ast.Starred):
            self._record_target_symbols(target.value)

    def _visit_stmt_block(self, stmts):
        new_stmts = []
        for stmt in stmts:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_stmts.extend(transformed)
            else:
                new_stmts.append(transformed)
        return new_stmts

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if getattr(node, _ASTREWRITE_MARKER, None) == type(self).__name__:
            return node

        with self.symbol_scopes.function_scope():
            for arg in node.args.posonlyargs:
                self.symbol_scopes.record_symbol(arg.arg)
            for arg in node.args.args:
                self.symbol_scopes.record_symbol(arg.arg)
            for arg in node.args.kwonlyargs:
                self.symbol_scopes.record_symbol(arg.arg)
            node = self.generic_visit(node)

        return node

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._record_target_symbols(target)
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._record_target_symbols(node.target)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._record_target_symbols(node.target)
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        with self.symbol_scopes.control_flow_scope():
            self._record_target_symbols(node.target)
            node.iter = self.visit(node.iter)
            with self.symbol_scopes.control_flow_scope():
                node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                with self.symbol_scopes.control_flow_scope():
                    node.orelse = self._visit_stmt_block(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        with self.symbol_scopes.control_flow_scope():
            node.test = self.visit(node.test)
            with self.symbol_scopes.control_flow_scope():
                node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                with self.symbol_scopes.control_flow_scope():
                    node.orelse = self._visit_stmt_block(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        with self.symbol_scopes.control_flow_scope():
            node.test = self.visit(node.test)
            with self.symbol_scopes.control_flow_scope():
                node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                with self.symbol_scopes.control_flow_scope():
                    node.orelse = self._visit_stmt_block(node.orelse)
        return node

    def visit_With(self, node: ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                self._record_target_symbols(item.optional_vars)
        return self.generic_visit(node)


@ASTRewriter.register
class RewriteBoolOps(Transformer):
    @staticmethod
    def dsl_and_(lhs, rhs):
        if hasattr(lhs, "__fly_and__"):
            return lhs.__fly_and__(rhs)
        if hasattr(rhs, "__fly_and__"):
            return rhs.__fly_and__(lhs)
        return lhs and rhs

    @staticmethod
    def dsl_or_(lhs, rhs):
        if hasattr(lhs, "__fly_or__"):
            return lhs.__fly_or__(rhs)
        if hasattr(rhs, "__fly_or__"):
            return rhs.__fly_or__(lhs)
        return lhs or rhs

    @staticmethod
    def dsl_not_(x):
        if hasattr(x, "__fly_not__"):
            return x.__fly_not__()
        return not x

    @classmethod
    def rewrite_globals(cls):
        return {
            "dsl_and_": cls.dsl_and_,
            "dsl_or_": cls.dsl_or_,
            "dsl_not_": cls.dsl_not_,
        }

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)

        _BOOL_OP_MAP = {ast.And: "dsl_and_", ast.Or: "dsl_or_"}
        handler = _BOOL_OP_MAP.get(type(node.op))
        if handler is None:
            return node

        def _should_skip(operand):
            bail_val = ast.Constant(value=(type(node.op) is ast.Or))
            return ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Call(
                        func=ast.Name(id="isinstance", ctx=ast.Load()),
                        args=[operand, ast.Name(id="bool", ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Compare(left=operand, ops=[ast.Eq()], comparators=[bail_val]),
                ],
            )

        result = node.values[0]
        for rhs in node.values[1:]:
            result = ast.IfExp(
                test=_should_skip(result),
                body=result,
                orelse=ast.Call(
                    func=ast.Name(handler, ctx=ast.Load()),
                    args=[result, rhs],
                    keywords=[],
                ),
            )

        return ast.copy_location(ast.fix_missing_locations(result), node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Not):
            return node
        replacement = ast.Call(
            func=ast.Name("dsl_not_", ctx=ast.Load()),
            args=[node.operand],
            keywords=[],
        )
        return ast.copy_location(ast.fix_missing_locations(replacement), node)


@ASTRewriter.register
class ReplaceIfWithDispatch(Transformer):
    _counter = 0

    @staticmethod
    def _is_dynamic(cond):
        if isinstance(cond, ir.Value):
            return True
        if hasattr(cond, "value") and isinstance(cond.value, ir.Value):
            return True
        return False

    @staticmethod
    def _to_i1(cond):
        return as_ir_value(cond)

    @staticmethod
    def _normalize_named_values(names, values, names_label="names", values_label="values"):
        names = tuple(names or ())
        values = tuple(values or ())
        if len(names) != len(values):
            raise ValueError(
                f"{names_label} and {values_label} must have the same length, " f"got {len(names)} and {len(values)}"
            )
        return names, values

    @staticmethod
    def _normalize_branch_result(branch_result, state_names, state_map, branch_label):
        if not state_names:
            return []

        if isinstance(branch_result, dict):
            result_map = dict(branch_result)
        elif branch_result is None:
            result_map = {}
        elif len(state_names) == 1 and not isinstance(branch_result, (list, tuple)):
            result_map = {state_names[0]: branch_result}
        elif isinstance(branch_result, (list, tuple)) and len(branch_result) == len(state_names):
            result_map = dict(zip(state_names, branch_result))
        else:
            raise TypeError(
                f"{branch_label} must return dict/tuple/list for stateful dispatch; got {type(branch_result).__name__}"
            )

        values = []
        for name in state_names:
            if name in result_map:
                values.append(result_map[name])
            elif name in state_map:
                values.append(state_map[name])
            else:
                raise NameError(
                    f"variable '{name}' is not available before if/else and is not assigned in {branch_label}"
                )
        return values

    @staticmethod
    def _unwrap_mlir_values(values, state_names, branch_label):
        raw_values = []
        for name, value in zip(state_names, values):
            raw = as_ir_value(value)
            if not isinstance(raw, ir.Value):
                raise TypeError(
                    f"if/else variable '{name}' in {branch_label} is {type(raw).__name__}, "
                    "not an MLIR Value. Only MLIR Values can be yielded from dynamic if/else branches."
                )
            raw_values.append(raw)
        return raw_values

    @staticmethod
    def _pack_dispatch_results(results, state_values):
        if not results:
            return None
        wrapped = [as_dsl_value(v, exemplar) for v, exemplar in zip(results, state_values)]
        if len(wrapped) == 1:
            return wrapped[0]
        return tuple(wrapped)

    @staticmethod
    def _collect_result_dict(result_names, local_vars):
        return {name: local_vars[name] for name in result_names}

    @staticmethod
    def _pack_named_values(names, values):
        if not names:
            return None
        if len(names) == 1:
            return values[0]
        return tuple(values)

    @staticmethod
    def _call_branch(fn, result_names, state_values):
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        pos_params = [
            p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs or len(pos_params) >= len(state_values) + 1:
            return fn(result_names, *state_values)
        return fn(*state_values)

    @staticmethod
    def scf_if_dispatch(
        cond,
        then_fn,
        else_fn=None,
        *,
        result_names=(),
        result_values=(),
        state_names=(),
        state_values=(),
        auto_else=False,
    ):
        # Backward compatibility: old call-sites pass state_* only.
        if not result_names and state_names:
            result_names = state_names
        if not result_values and state_values:
            result_values = state_values
        result_names, result_values = ReplaceIfWithDispatch._normalize_named_values(
            result_names, result_values, "result_names", "result_values"
        )
        result_map = {name: value for name, value in zip(result_names, result_values)}

        if not ReplaceIfWithDispatch._is_dynamic(cond):
            taken = then_fn if cond else else_fn
            if taken is None:
                return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)
            result = ReplaceIfWithDispatch._call_branch(taken, result_names, result_values)
            if not result_names:
                return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)
            partial_values = ReplaceIfWithDispatch._normalize_branch_result(
                result, result_names, result_map, "selected branch"
            )
            return ReplaceIfWithDispatch._pack_named_values(result_names, partial_values)

        cond_i1 = ReplaceIfWithDispatch._to_i1(cond)
        if not isinstance(cond_i1, ir.Value):
            raise TypeError(f"dynamic if condition must lower to ir.Value, got {type(cond_i1).__name__}")

        none_vars = [name for name, value in zip(result_names, result_values) if as_ir_value(value) is None]
        if none_vars:
            raise TypeError(
                f"Variable(s) {none_vars} initialized as None before a dynamic "
                f"if/else (scf.if). Branches assign to these variables, but "
                f"None cannot be yielded as an scf.if result. Initialize them "
                f"with a value whose type matches the branch assignment (e.g. "
                f"fx.Int32(0), fx.Float32(0.0)), or wrap the condition with "
                f"const_expr() if it is a compile-time constant."
            )

        if not result_names:
            has_else = else_fn is not None
            if_op = scf.IfOp(cond_i1, [], has_else=has_else, loc=capture_user_location())
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                ReplaceIfWithDispatch._call_branch(then_fn, result_names, result_values)
                scf.YieldOp([], loc=capture_user_location())
            if has_else:
                if len(if_op.regions[1].blocks) == 0:
                    if_op.regions[1].blocks.append(*[])
                with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                    ReplaceIfWithDispatch._call_branch(else_fn, result_names, result_values)
                    scf.YieldOp([], loc=capture_user_location())
            return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)

        if else_fn is None:
            else_fn = lambda *args: {}

        state_raw = []
        for name, value in zip(result_names, result_values):
            raw = as_ir_value(value)
            if not isinstance(raw, ir.Value):
                raise TypeError(
                    f"state variable '{name}' is {type(raw).__name__}, not an MLIR Value; "
                    "stateful dynamic if requires MLIR-backed values."
                )
            state_raw.append(raw)

        result_types = [v.type for v in state_raw]
        if_op = scf.IfOp(cond_i1, result_types, has_else=True, loc=capture_user_location())

        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            then_result = ReplaceIfWithDispatch._call_branch(then_fn, result_names, result_values)
            then_values = ReplaceIfWithDispatch._normalize_branch_result(
                then_result, result_names, result_map, "then-branch"
            )
            then_raw = ReplaceIfWithDispatch._unwrap_mlir_values(then_values, result_names, "then-branch")
            for name, expect_ty, got in zip(result_names, result_types, then_raw):
                if got.type != expect_ty:
                    raise TypeError(
                        f"if/else variable '{name}' type mismatch in then-branch: "
                        f"expected {expect_ty}, got {got.type}"
                    )
            scf.YieldOp(then_raw, loc=capture_user_location())

        if len(if_op.regions[1].blocks) == 0:
            if_op.regions[1].blocks.append(*[])
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            else_result = ReplaceIfWithDispatch._call_branch(else_fn, result_names, result_values)
            else_values = ReplaceIfWithDispatch._normalize_branch_result(
                else_result, result_names, result_map, "else-branch"
            )
            else_raw = ReplaceIfWithDispatch._unwrap_mlir_values(else_values, result_names, "else-branch")
            for name, expect_ty, got in zip(result_names, result_types, else_raw):
                if got.type != expect_ty:
                    raise TypeError(
                        f"if/else variable '{name}' type mismatch in else-branch: "
                        f"expected {expect_ty}, got {got.type}"
                    )
            scf.YieldOp(else_raw, loc=capture_user_location())

        wrapped = ReplaceIfWithDispatch._pack_dispatch_results(list(if_op.results), result_values)
        if len(result_names) == 1:
            final_values = [wrapped]
        else:
            final_values = list(wrapped)
        return ReplaceIfWithDispatch._pack_named_values(result_names, final_values)

    @classmethod
    def rewrite_globals(cls):
        return {
            "const_expr": const_expr,
            "__check_local_var": _check_local_var,
            "scf_if_dispatch": cls.scf_if_dispatch,
            "scf_ifexp_dispatch": cls.scf_ifexp_dispatch,
            "scf_if_collect_results": cls._collect_result_dict,
        }

    def visit_If(self, node: ast.If) -> List[ast.AST]:
        active_symbols_before_if = self.symbol_scopes.snapshot_symbol_scopes()
        if _is_constexpr(node.test):
            node.test = _unwrap_constexpr(node.test)
            node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                node.orelse = self._visit_stmt_block(node.orelse)
            return node
        with self.symbol_scopes.control_flow_scope():
            node.test = self.visit(node.test)
            with self.symbol_scopes.control_flow_scope():
                node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                with self.symbol_scopes.control_flow_scope():
                    node.orelse = self._visit_stmt_block(node.orelse)
            uid = ReplaceIfWithDispatch._counter
            ReplaceIfWithDispatch._counter += 1

            then_name = f"__then_{uid}"
            result_names = _collect_assigned_vars(node.body, active_symbols_before_if, node.orelse)

            fn_args = [ast.arg(arg="__ret_names", annotation=None)] + [
                ast.arg(arg=v, annotation=None) for v in result_names
            ]

            def _state_return_node():
                return ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="scf_if_collect_results", ctx=ast.Load()),
                        args=[
                            ast.Name(id="__ret_names", ctx=ast.Load()),
                            ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                        ],
                        keywords=[],
                    )
                )

            then_func = ast.FunctionDef(
                name=then_name,
                args=ast.arguments(posonlyargs=[], args=fn_args, kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=list(node.body) + ([_state_return_node()] if result_names else []),
                decorator_list=[],
                type_params=[],
            )
            setattr(then_func, _ASTREWRITE_MARKER, type(self).__name__)
            then_func = ast.copy_location(then_func, node)
            then_func = ast.fix_missing_locations(then_func)

            dispatch_args = [node.test, ast.Name(then_name, ctx=ast.Load())]
            dispatch_keywords = []
            if result_names:
                dispatch_keywords.extend(
                    [
                        ast.keyword(
                            arg="result_names",
                            value=ast.Tuple(elts=[ast.Constant(value=v) for v in result_names], ctx=ast.Load()),
                        ),
                        ast.keyword(
                            arg="result_values",
                            value=ast.Tuple(
                                elts=[_state_value_expr(v) for v in result_names],
                                ctx=ast.Load(),
                            ),
                        ),
                    ]
                )
            result = [then_func]

            else_name = None
            synthesized_else = False
            if node.orelse:
                else_name = f"__else_{uid}"
                else_func = ast.FunctionDef(
                    name=else_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="__ret_names", annotation=None)]
                        + [ast.arg(arg=v, annotation=None) for v in result_names],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=list(node.orelse) + ([_state_return_node()] if result_names else []),
                    decorator_list=[],
                    type_params=[],
                )
                setattr(else_func, _ASTREWRITE_MARKER, type(self).__name__)
                else_func = ast.copy_location(else_func, node)
                else_func = ast.fix_missing_locations(else_func)
                dispatch_args.append(ast.Name(else_name, ctx=ast.Load()))
                result.append(else_func)
            elif result_names:
                else_name = f"__else_{uid}"
                synthesized_else = True
                else_func = ast.FunctionDef(
                    name=else_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="__ret_names", annotation=None)]
                        + [ast.arg(arg=v, annotation=None) for v in result_names],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[_state_return_node()],
                    decorator_list=[],
                    type_params=[],
                )
                setattr(else_func, _ASTREWRITE_MARKER, type(self).__name__)
                else_func = ast.copy_location(else_func, node)
                else_func = ast.fix_missing_locations(else_func)
                dispatch_args.append(ast.Name(else_name, ctx=ast.Load()))
                result.append(else_func)

            if synthesized_else:
                dispatch_keywords.append(ast.keyword(arg="auto_else", value=ast.Constant(value=True)))

            dispatch_value = ast.Call(
                func=ast.Name("scf_if_dispatch", ctx=ast.Load()),
                args=dispatch_args,
                keywords=dispatch_keywords,
            )
            if result_names and else_name is not None:
                if len(result_names) == 1:
                    target = ast.Name(id=result_names[0], ctx=ast.Store())
                else:
                    target = ast.Tuple(elts=[ast.Name(id=v, ctx=ast.Store()) for v in result_names], ctx=ast.Store())
                dispatch_stmt = ast.Assign(targets=[target], value=dispatch_value)
            else:
                dispatch_stmt = ast.Expr(value=dispatch_value)
            dispatch_stmt = ast.copy_location(dispatch_stmt, node)
            dispatch_stmt = ast.fix_missing_locations(dispatch_stmt)
            result.append(dispatch_stmt)

            return result

    def visit_IfExp(self, node: ast.IfExp):
        node = self.generic_visit(node)
        empty_args = ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        return ast.copy_location(
            ast.Call(
                func=ast.Name("scf_ifexp_dispatch", ctx=ast.Load()),
                args=[
                    node.test,
                    ast.Lambda(args=empty_args, body=node.body),
                    ast.Lambda(args=empty_args, body=node.orelse),
                ],
                keywords=[],
            ),
            node,
        )

    @staticmethod
    def scf_ifexp_dispatch(cond, then_fn, else_fn):
        if not ReplaceIfWithDispatch._is_dynamic(cond):
            return then_fn() if cond else else_fn()

        cond_i1 = ReplaceIfWithDispatch._to_i1(cond)
        if not isinstance(cond_i1, ir.Value):
            raise TypeError(f"dynamic ifexp condition must lower to ir.Value, got {type(cond_i1).__name__}")

        sandbox = scf.ExecuteRegionOp(result=[])
        sandbox.region.blocks.append()
        with ir.InsertionPoint(sandbox.region.blocks[0]):
            probe_then = then_fn()
            probe_then_raw = as_ir_value(probe_then)
            probe_else = else_fn()
            probe_else_raw = as_ir_value(probe_else)
            if not isinstance(probe_then_raw, ir.Value):
                raise TypeError(
                    f"dynamic ifexp then-branch must produce an MLIR Value, " f"got {type(probe_then_raw).__name__}"
                )
            if not isinstance(probe_else_raw, ir.Value):
                raise TypeError(
                    f"dynamic ifexp else-branch must produce an MLIR Value, " f"got {type(probe_else_raw).__name__}"
                )
            if probe_then_raw.type != probe_else_raw.type:
                raise TypeError(
                    f"dynamic ifexp type mismatch: "
                    f"then-branch produces {probe_then_raw.type}, "
                    f"else-branch produces {probe_else_raw.type}"
                )
            yield_type = probe_then_raw.type

        op = scf.IfOp(cond_i1, [yield_type], has_else=True, loc=capture_user_location())
        with ir.InsertionPoint(op.regions[0].blocks[0]):
            scf.YieldOp([as_ir_value(then_fn())], loc=capture_user_location())
        if len(op.regions[1].blocks) == 0:
            op.regions[1].blocks.append()
        with ir.InsertionPoint(op.regions[1].blocks[0]):
            scf.YieldOp([as_ir_value(else_fn())], loc=capture_user_location())

        sandbox.operation.erase()
        return as_dsl_value(op.results[0], probe_then)


@ASTRewriter.register
class InsertEmptyYieldForSCFFor(Transformer):
    _counter = 0

    @staticmethod
    def _to_index(val):
        loc = capture_user_location()
        if isinstance(val, ir.Value):
            if val.type == ir.IndexType.get():
                return val
            return arith.IndexCastOp(ir.IndexType.get(), val, loc=loc).result
        if hasattr(val, "ir_value"):
            raw = val.ir_value()
            if isinstance(raw, ir.Value) and raw.type != ir.IndexType.get():
                return arith.IndexCastOp(ir.IndexType.get(), raw, loc=loc).result
            return raw
        if isinstance(val, int) and not isinstance(val, bool):
            return arith.ConstantOp(ir.IndexType.get(), val, loc=loc).result
        raise TypeError(f"_to_index expected ir.Value, object with ir_value(), or int; got {type(val).__name__}")

    @staticmethod
    def scf_range(start, stop=None, step=None, *, init=None):
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1
        start_val = InsertEmptyYieldForSCFFor._to_index(start)
        stop_val = InsertEmptyYieldForSCFFor._to_index(stop)
        step_val = InsertEmptyYieldForSCFFor._to_index(step)
        if init is not None:
            init = [as_ir_value(v) for v in init]
            loc = capture_user_location()
            for_op = scf.ForOp(start_val, stop_val, step_val, init, loc=loc)
            _locate_block_args(for_op.body, loc)
            with ir.InsertionPoint(for_op.body):
                yield for_op.induction_variable, list(for_op.inner_iter_args)
        else:
            loc = capture_user_location()
            for_op = scf.ForOp(start_val, stop_val, step_val, loc=loc)
            _locate_block_args(for_op.body, loc)
            with ir.InsertionPoint(for_op.body):
                yield for_op.induction_variable

    @staticmethod
    def scf_for_dispatch(start, stop, step, body_fn, *, result_names=(), result_values=()):
        start_val = as_ir_value(start)
        stop_val = as_ir_value(stop)
        step_val = as_ir_value(step)

        i32_ty = ir.IntegerType.get_signless(32)
        idx_ty = ir.IndexType.get()
        bounds = [("start", start_val), ("stop", stop_val), ("step", step_val)]
        for i, (name, val) in enumerate(bounds):
            if not isinstance(val, ir.Value):
                raise TypeError(f"for-loop {name} must be i32, got {type(val).__name__}")
            if val.type == idx_ty:
                log().warning("for-loop %s is index type, consider using fx.Int32 instead", name)
                bounds[i] = (name, arith.IndexCastOp(i32_ty, val, loc=capture_user_location()).result)
            elif val.type != i32_ty:
                raise TypeError(f"for-loop {name} must be i32, got {val.type}")
        start_val, stop_val, step_val = bounds[0][1], bounds[1][1], bounds[2][1]

        result_names = tuple(result_names)
        result_values = tuple(result_values)
        result_map = {name: value for name, value in zip(result_names, result_values)}

        none_vars = [name for name, value in zip(result_names, result_values) if as_ir_value(value) is None]
        if none_vars:
            raise TypeError(
                f"Variable(s) {none_vars} initialized as None before a dynamic "
                f"for loop (scf.for). The loop body assigns to these variables, "
                f"but None cannot be yielded as an scf.for iter_arg. Initialize "
                f"them with a value whose type matches the loop body assignment "
                f"(e.g. fx.Int32(0), fx.Float32(0.0))."
            )

        if not result_names:
            loc = capture_user_location()
            for_op = scf.ForOp(start_val, stop_val, step_val, loc=loc)
            _locate_block_args(for_op.body, loc)
            with ir.InsertionPoint(for_op.body):
                iv = for_op.induction_variable
                body_fn(iv, result_names)
                scf.YieldOp([], loc=capture_user_location())
            return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)

        state_raw = []
        for name, value in zip(result_names, result_values):
            raw = as_ir_value(value)
            if not isinstance(raw, ir.Value):
                raise TypeError(
                    f"for-loop variable '{name}' is {type(raw).__name__}, not an MLIR Value; "
                    "only MLIR-backed values can be loop-carried in dynamic for loops."
                )
            state_raw.append(raw)

        loc = capture_user_location()
        for_op = scf.ForOp(start_val, stop_val, step_val, state_raw, loc=loc)
        _locate_block_args(for_op.body, loc)

        with ir.InsertionPoint(for_op.body):
            iv = for_op.induction_variable
            inner_args = [as_dsl_value(a, ex) for a, ex in zip(for_op.inner_iter_args, result_values)]

            body_result = body_fn(iv, result_names, *inner_args)

            body_values = ReplaceIfWithDispatch._normalize_branch_result(
                body_result, result_names, result_map, "for-body"
            )
            body_raw = ReplaceIfWithDispatch._unwrap_mlir_values(body_values, result_names, "for-body")
            result_types = [v.type for v in state_raw]
            for name, expect_ty, got in zip(result_names, result_types, body_raw):
                if got.type != expect_ty:
                    raise TypeError(
                        f"for-loop variable '{name}' type mismatch: " f"expected {expect_ty}, got {got.type}"
                    )
            scf.YieldOp(body_raw, loc=capture_user_location())

        wrapped = ReplaceIfWithDispatch._pack_dispatch_results(list(for_op.results), result_values)
        if len(result_names) == 1:
            final_values = [wrapped]
        else:
            final_values = list(wrapped)
        return ReplaceIfWithDispatch._pack_named_values(result_names, final_values)

    @classmethod
    def rewrite_globals(cls):
        return {
            "scf_range": cls.scf_range,
            "scf_for_dispatch": cls.scf_for_dispatch,
            "scf_for_collect_results": ReplaceIfWithDispatch._collect_result_dict,
        }

    @staticmethod
    def _is_yield(stmt):
        return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield)) or (
            isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield)
        )

    @staticmethod
    def _iter_call_name(iter_node):
        if not isinstance(iter_node, ast.Call):
            return None
        target = iter_node.func
        return getattr(target, "id", None) or getattr(target, "attr", None)

    @classmethod
    def _is_range_constexpr(cls, iter_node):
        return cls._iter_call_name(iter_node) == "range_constexpr"

    @classmethod
    def _is_range(cls, iter_node):
        return cls._iter_call_name(iter_node) == "range"

    @staticmethod
    def _has_init_kwarg(iter_node):
        if not isinstance(iter_node, ast.Call):
            return False
        return any(kw.arg == "init" for kw in iter_node.keywords)

    @staticmethod
    def _extract_range_args(iter_node):
        args = iter_node.args
        if len(args) == 1:
            return ast.Constant(value=0), args[0], ast.Constant(value=1)
        elif len(args) == 2:
            return args[0], args[1], ast.Constant(value=1)
        else:
            return args[0], args[1], args[2]

    def _transform_for_auto(self, node):
        start, stop, step = self._extract_range_args(node.iter)

        if isinstance(node.target, ast.Name):
            iv_name = node.target.id
        else:
            raise TypeError(
                "auto for-loop iter_args inference only supports a single loop variable; "
                f"got {ast.dump(node.target)}. Use range(..., init=[...]) for tuple unpacking."
            )

        active_symbols_before_for = self.symbol_scopes.snapshot_symbol_scopes()

        iv_active_before = any(iv_name in scope for scope in active_symbols_before_for)

        uid = InsertEmptyYieldForSCFFor._counter
        InsertEmptyYieldForSCFFor._counter += 1

        loop_carried_var_name = None
        if iv_active_before:
            loop_carried_var_name = f"__loop_carried_{uid}"

        with self.symbol_scopes.control_flow_scope():
            self.symbol_scopes.record_symbol(iv_name)
            node.body = self._visit_stmt_block(node.body)

        if loop_carried_var_name:
            node.body.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=loop_carried_var_name, ctx=ast.Store())],
                        value=ast.Name(id=iv_name, ctx=ast.Load()),
                    ),
                    node,
                )
            )
            active_symbols_before_for = list(active_symbols_before_for) + [{loop_carried_var_name}]

        result_names = _collect_assigned_vars(node.body, active_symbols_before_for)
        result_names = [n for n in result_names if n != iv_name]

        body_name = f"__for_body_{uid}"
        fn_args = (
            [ast.arg(arg=iv_name, annotation=None)]
            + [ast.arg(arg="__ret_names", annotation=None)]
            + [ast.arg(arg=v, annotation=None) for v in result_names]
        )

        if result_names:
            body_return = ast.Return(
                value=ast.Call(
                    func=ast.Name(id="scf_for_collect_results", ctx=ast.Load()),
                    args=[
                        ast.Name(id="__ret_names", ctx=ast.Load()),
                        ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                    ],
                    keywords=[],
                )
            )
            body_stmts = list(node.body) + [body_return]
        else:
            body_stmts = list(node.body)

        body_func = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[],
                args=fn_args,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body_stmts,
            decorator_list=[],
            type_params=[],
        )
        body_func = ast.copy_location(body_func, node)
        body_func = ast.fix_missing_locations(body_func)

        dispatch_args = [start, stop, step, ast.Name(body_name, ctx=ast.Load())]
        dispatch_keywords = []
        if result_names:
            dispatch_keywords.extend(
                [
                    ast.keyword(
                        arg="result_names",
                        value=ast.Tuple(
                            elts=[ast.Constant(value=v) for v in result_names],
                            ctx=ast.Load(),
                        ),
                    ),
                    ast.keyword(
                        arg="result_values",
                        value=ast.Tuple(
                            elts=[_state_value_expr(v) for v in result_names],
                            ctx=ast.Load(),
                        ),
                    ),
                ]
            )

        dispatch_value = ast.Call(
            func=ast.Name("scf_for_dispatch", ctx=ast.Load()),
            args=dispatch_args,
            keywords=dispatch_keywords,
        )

        if result_names:
            if len(result_names) == 1:
                target = ast.Name(id=result_names[0], ctx=ast.Store())
            else:
                target = ast.Tuple(
                    elts=[ast.Name(id=v, ctx=ast.Store()) for v in result_names],
                    ctx=ast.Store(),
                )
            dispatch_stmt = ast.Assign(targets=[target], value=dispatch_value)
        else:
            dispatch_stmt = ast.Expr(value=dispatch_value)

        dispatch_stmt = ast.copy_location(dispatch_stmt, node)
        dispatch_stmt = ast.fix_missing_locations(dispatch_stmt)

        pre_stmts = []
        post_stmts = []
        if loop_carried_var_name:
            pre_stmt = ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id=loop_carried_var_name, ctx=ast.Store())],
                    value=ast.Name(id=iv_name, ctx=ast.Load()),
                ),
                node,
            )
            ast.fix_missing_locations(pre_stmt)
            pre_stmts.append(pre_stmt)

            post_stmt = ast.copy_location(
                ast.Assign(
                    targets=[ast.Name(id=iv_name, ctx=ast.Store())],
                    value=ast.Name(id=loop_carried_var_name, ctx=ast.Load()),
                ),
                node,
            )
            ast.fix_missing_locations(post_stmt)
            post_stmts.append(post_stmt)

        return pre_stmts + [body_func, dispatch_stmt] + post_stmts

    def visit_For(self, node: ast.For) -> ast.For:
        if self._is_range_constexpr(node.iter):
            node.iter.func = ast.Name(id="range", ctx=ast.Load())
            node = self.generic_visit(node)
            node = ast.fix_missing_locations(node)
            return node

        if self._is_range(node.iter) and not self._has_init_kwarg(node.iter):
            return self._transform_for_auto(node)

        if self._is_range(node.iter):
            node.iter.func = ast.Name(id="scf_range", ctx=ast.Load())
        line = ast.dump(node.iter)
        if "for_" in line or "scf.for_" in line or "scf_range" in line:
            with self.symbol_scopes.control_flow_scope():
                node.iter = self.visit(node.iter)
                with self.symbol_scopes.control_flow_scope():
                    if isinstance(node.target, ast.Name):
                        self.symbol_scopes.record_symbol(node.target.id)
                    node.body = self._visit_stmt_block(node.body)
                if node.orelse:
                    with self.symbol_scopes.control_flow_scope():
                        node.orelse = self._visit_stmt_block(node.orelse)
                new_yield = ast.Expr(ast.Yield(value=None))
                if node.body and not self._is_yield(node.body[-1]):
                    last_statement = node.body[-1]
                    assert (
                        last_statement.end_lineno is not None
                    ), f"last_statement {ast.unparse(last_statement)} must have end_lineno"
                    new_yield = ast.fix_missing_locations(_set_lineno(new_yield, last_statement.end_lineno))
                    node.body.append(new_yield)
                node = ast.fix_missing_locations(node)
        return node


@ASTRewriter.register
class ReplaceYieldWithSCFYield(Transformer):
    @staticmethod
    def scf_yield_(*args):
        if len(args) == 1 and isinstance(args[0], (list, ir.OpResultList)):
            args = list(args[0])
        processed = []
        for a in args:
            if isinstance(a, ir.Value):
                processed.append(a)
            elif hasattr(a, "ir_value"):
                processed.append(a.ir_value())
            else:
                processed.append(a)
        scf.YieldOp(processed, loc=capture_user_location())
        parent_op = ir.InsertionPoint.current.block.owner
        if hasattr(parent_op, "results") and len(parent_op.results):
            results = list(parent_op.results)
            if len(results) > 1:
                return results
            return results[0]

    @classmethod
    def rewrite_globals(cls):
        return {
            "scf_yield_": cls.scf_yield_,
        }

    def visit_Yield(self, node: ast.Yield) -> ast.Expr:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        call = ast.copy_location(ast.Call(func=ast.Name("scf_yield_", ctx=ast.Load()), args=args, keywords=[]), node)
        call = ast.fix_missing_locations(call)
        return call


@ASTRewriter.register
class CanonicalizeWhile(Transformer):
    _counter = 0

    @staticmethod
    def scf_while_dispatch(before_fn, after_fn, *, result_names=(), result_values=()):
        result_names, result_values = ReplaceIfWithDispatch._normalize_named_values(
            result_names, result_values, "result_names", "result_values"
        )
        result_map = {name: value for name, value in zip(result_names, result_values)}

        none_vars = [name for name, value in zip(result_names, result_values) if as_ir_value(value) is None]
        if none_vars:
            raise TypeError(
                f"Variable(s) {none_vars} initialized as None before a dynamic "
                f"while loop (scf.while). Branches assign to these variables, but "
                f"None cannot be yielded as an scf.while result. Initialize them "
                f"with a value whose type matches the loop body assignment (e.g. "
                f"fx.Int32(0), fx.Float32(0.0)), or wrap the condition with "
                f"const_expr() if it is a compile-time constant."
            )

        state_raw = []
        for name, value in zip(result_names, result_values):
            raw = as_ir_value(value)
            if not isinstance(raw, ir.Value):
                raise TypeError(
                    f"while-loop variable '{name}' is {type(raw).__name__}, not an MLIR Value; "
                    "only MLIR-backed values can be loop-carried in dynamic while loops."
                )
            state_raw.append(raw)

        result_types = [v.type for v in state_raw]
        loc = capture_user_location()
        while_op = scf.WhileOp(result_types, state_raw, loc=loc)
        # Give the loop-carried block arguments the user location.
        arg_locs = [loc] * len(result_types)
        while_op.regions[0].blocks.append(*result_types, arg_locs=arg_locs)
        while_op.regions[1].blocks.append(*result_types, arg_locs=arg_locs)

        with ir.InsertionPoint(while_op.regions[0].blocks[0]):
            before_args = list(while_op.regions[0].blocks[0].arguments)
            wrapped_before = [as_dsl_value(a, ex) for a, ex in zip(before_args, result_values)] if result_names else []
            before_cond = ReplaceIfWithDispatch._call_branch(before_fn, result_names, wrapped_before)
            cond_i1 = ReplaceIfWithDispatch._to_i1(before_cond)
            if not isinstance(cond_i1, ir.Value):
                raise TypeError(f"dynamic while condition must lower to ir.Value, got {type(cond_i1).__name__}")
            scf.ConditionOp(cond_i1, before_args, loc=capture_user_location())

        with ir.InsertionPoint(while_op.regions[1].blocks[0]):
            after_args = list(while_op.regions[1].blocks[0].arguments)
            wrapped_after = [as_dsl_value(a, ex) for a, ex in zip(after_args, result_values)] if result_names else []
            body_result = ReplaceIfWithDispatch._call_branch(after_fn, result_names, wrapped_after)
            if result_names:
                body_values = ReplaceIfWithDispatch._normalize_branch_result(
                    body_result, result_names, result_map, "while-body"
                )
                body_raw = ReplaceIfWithDispatch._unwrap_mlir_values(body_values, result_names, "while-body")
                for name, expect_ty, got in zip(result_names, result_types, body_raw):
                    if got.type != expect_ty:
                        raise TypeError(
                            f"while-loop variable '{name}' type mismatch: expected {expect_ty}, got {got.type}"
                        )
                scf.YieldOp(body_raw, loc=capture_user_location())
            else:
                scf.YieldOp([], loc=capture_user_location())

        if not result_names:
            return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)

        wrapped = ReplaceIfWithDispatch._pack_dispatch_results(list(while_op.results), result_values)
        if len(result_names) == 1:
            final_values = [wrapped]
        else:
            final_values = list(wrapped)
        return ReplaceIfWithDispatch._pack_named_values(result_names, final_values)

    @classmethod
    def rewrite_globals(cls):
        return {
            "scf_while_dispatch": cls.scf_while_dispatch,
            "scf_while_collect_results": ReplaceIfWithDispatch._collect_result_dict,
        }

    def visit_While(self, node: ast.While) -> List[ast.AST]:
        if _is_constexpr(node.test):
            node.test = _unwrap_constexpr(node.test)
            node = super().visit_While(node)
            return node

        if node.orelse:
            raise NotImplementedError("while...else is not supported in dynamic while loops (scf.while)")

        active_symbols_before_while = self.symbol_scopes.snapshot_symbol_scopes()

        with self.symbol_scopes.control_flow_scope():
            node.test = self.visit(node.test)
            with self.symbol_scopes.control_flow_scope():
                node.body = self._visit_stmt_block(node.body)

            uid = CanonicalizeWhile._counter
            CanonicalizeWhile._counter += 1

            result_names = _collect_assigned_vars(node.body, active_symbols_before_while, test_expr=node.test)

            fn_args = [ast.arg(arg="__ret_names", annotation=None)] + [
                ast.arg(arg=v, annotation=None) for v in result_names
            ]

            def _state_return_node():
                return ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="scf_while_collect_results", ctx=ast.Load()),
                        args=[
                            ast.Name(id="__ret_names", ctx=ast.Load()),
                            ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                        ],
                        keywords=[],
                    )
                )

            before_name = f"__while_before_{uid}"
            before_func = ast.FunctionDef(
                name=before_name,
                args=ast.arguments(posonlyargs=[], args=fn_args, kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[ast.Return(value=node.test)],
                decorator_list=[],
                type_params=[],
            )
            setattr(before_func, _ASTREWRITE_MARKER, type(self).__name__)
            before_func = ast.copy_location(before_func, node)
            before_func = ast.fix_missing_locations(before_func)

            after_name = f"__while_after_{uid}"
            after_stmts = list(node.body) + ([_state_return_node()] if result_names else [])
            after_func = ast.FunctionDef(
                name=after_name,
                args=ast.arguments(posonlyargs=[], args=fn_args, kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=after_stmts,
                decorator_list=[],
                type_params=[],
            )
            setattr(after_func, _ASTREWRITE_MARKER, type(self).__name__)
            after_func = ast.copy_location(after_func, node)
            after_func = ast.fix_missing_locations(after_func)

            dispatch_args = [ast.Name(before_name, ctx=ast.Load()), ast.Name(after_name, ctx=ast.Load())]
            dispatch_keywords = []
            if result_names:
                dispatch_keywords.extend(
                    [
                        ast.keyword(
                            arg="result_names",
                            value=ast.Tuple(
                                elts=[ast.Constant(value=v) for v in result_names],
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.keyword(
                            arg="result_values",
                            value=ast.Tuple(
                                elts=[_state_value_expr(v) for v in result_names],
                                ctx=ast.Load(),
                            ),
                        ),
                    ]
                )

            dispatch_value = ast.Call(
                func=ast.Name("scf_while_dispatch", ctx=ast.Load()),
                args=dispatch_args,
                keywords=dispatch_keywords,
            )

            if result_names:
                if len(result_names) == 1:
                    target = ast.Name(id=result_names[0], ctx=ast.Store())
                else:
                    target = ast.Tuple(
                        elts=[ast.Name(id=v, ctx=ast.Store()) for v in result_names],
                        ctx=ast.Store(),
                    )
                dispatch_stmt = ast.Assign(targets=[target], value=dispatch_value)
            else:
                dispatch_stmt = ast.Expr(value=dispatch_value)
            dispatch_stmt = ast.copy_location(dispatch_stmt, node)
            dispatch_stmt = ast.fix_missing_locations(dispatch_stmt)

            return [before_func, after_func, dispatch_stmt]
