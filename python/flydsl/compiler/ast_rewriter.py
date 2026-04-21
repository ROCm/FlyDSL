# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ast
import contextlib
import difflib
import inspect
import types
from textwrap import dedent
from typing import List

from .._mlir import ir
from .._mlir.dialects import arith, scf
from ..expr import const_expr
from ..expr.numeric import _unwrap_value, _wrap_like
from ..utils import env, log


def _set_lineno(node, n=1):
    for child in ast.walk(node):
        child.lineno = n
        child.end_lineno = n
    return node


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

        context = types.SimpleNamespace()
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

        f.__code__ = new_f_code_o

        for name, val in cls.rewrite_globals.items():
            f.__globals__[name] = val

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
        if getattr(node, _ASTREWRITE_MARKER, False):
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
        self._record_target_symbols(node.target)
        node.iter = self.visit(node.iter)
        with self.symbol_scopes.control_flow_scope():
            node.body = self._visit_stmt_block(node.body)
        if node.orelse:
            with self.symbol_scopes.control_flow_scope():
                node.orelse = self._visit_stmt_block(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.test = self.visit(node.test)
        with self.symbol_scopes.control_flow_scope():
            node.body = self._visit_stmt_block(node.body)
        if node.orelse:
            with self.symbol_scopes.control_flow_scope():
                node.orelse = self._visit_stmt_block(node.orelse)
        return node

    def visit_While(self, node: ast.While):
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
        return _unwrap_value(cond)

    @staticmethod
    def _normalize_named_values(names, values, names_label="names", values_label="values"):
        names = tuple(names or ())
        values = tuple(values or ())
        if len(names) != len(values):
            raise ValueError(
                f"{names_label} and {values_label} must have the same length, "
                f"got {len(names)} and {len(values)}"
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
            raw = _unwrap_value(value)
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
        wrapped = [_wrap_like(v, exemplar) for v, exemplar in zip(results, state_values)]
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
    def _merge_partial_results(base_names, base_values, part_names, part_values):
        merged = {name: value for name, value in zip(base_names, base_values)}
        merged.update({name: value for name, value in zip(part_names, part_values)})
        return [merged[name] for name in base_names]

    @staticmethod
    def _call_branch(fn, result_names, state_values):
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        pos_params = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
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
        # Only variables with an incoming value can be scf.if results/yields.
        effective_result_pairs = [
            (name, value)
            for name, value in zip(result_names, result_values)
            if _unwrap_value(value) is not None
        ]
        effective_result_names = tuple(name for name, _ in effective_result_pairs)
        effective_result_values = tuple(value for _, value in effective_result_pairs)
        effective_result_map = {name: value for name, value in effective_result_pairs}

        if not ReplaceIfWithDispatch._is_dynamic(cond):
            taken = then_fn if cond else else_fn
            if taken is None:
                return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)
            result = ReplaceIfWithDispatch._call_branch(taken, effective_result_names, result_values)
            if not effective_result_names:
                return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)
            partial_values = ReplaceIfWithDispatch._normalize_branch_result(
                result, effective_result_names, effective_result_map, "selected branch"
            )
            merged_values = ReplaceIfWithDispatch._merge_partial_results(
                result_names, result_values, effective_result_names, partial_values
            )
            return ReplaceIfWithDispatch._pack_named_values(result_names, merged_values)

        cond_i1 = ReplaceIfWithDispatch._to_i1(cond)
        if not isinstance(cond_i1, ir.Value):
            raise TypeError(f"dynamic if condition must lower to ir.Value, got {type(cond_i1).__name__}")

        if not effective_result_names:
            has_else = else_fn is not None
            if_op = scf.IfOp(cond_i1, [], has_else=has_else, loc=ir.Location.unknown())
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                ReplaceIfWithDispatch._call_branch(then_fn, effective_result_names, result_values)
                scf.YieldOp([])
            if has_else:
                if len(if_op.regions[1].blocks) == 0:
                    if_op.regions[1].blocks.append(*[])
                with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                    ReplaceIfWithDispatch._call_branch(else_fn, effective_result_names, result_values)
                    scf.YieldOp([])
            return ReplaceIfWithDispatch._pack_named_values(result_names, result_values)

        if else_fn is None:
            else_fn = lambda *args: {}

        state_raw = []
        for name, value in zip(effective_result_names, effective_result_values):
            raw = _unwrap_value(value)
            if not isinstance(raw, ir.Value):
                raise TypeError(
                    f"state variable '{name}' is {type(raw).__name__}, not an MLIR Value; "
                    "stateful dynamic if requires MLIR-backed values."
                )
            state_raw.append(raw)

        result_types = [v.type for v in state_raw]
        if_op = scf.IfOp(cond_i1, result_types, has_else=True, loc=ir.Location.unknown())

        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            then_result = ReplaceIfWithDispatch._call_branch(then_fn, effective_result_names, result_values)
            then_values = ReplaceIfWithDispatch._normalize_branch_result(
                then_result, effective_result_names, effective_result_map, "then-branch"
            )
            then_raw = ReplaceIfWithDispatch._unwrap_mlir_values(then_values, effective_result_names, "then-branch")
            for name, expect_ty, got in zip(effective_result_names, result_types, then_raw):
                if got.type != expect_ty:
                    raise TypeError(
                        f"if/else variable '{name}' type mismatch in then-branch: "
                        f"expected {expect_ty}, got {got.type}"
                    )
            scf.YieldOp(then_raw)

        if len(if_op.regions[1].blocks) == 0:
            if_op.regions[1].blocks.append(*[])
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            else_result = ReplaceIfWithDispatch._call_branch(else_fn, effective_result_names, result_values)
            else_values = ReplaceIfWithDispatch._normalize_branch_result(
                else_result, effective_result_names, effective_result_map, "else-branch"
            )
            else_raw = ReplaceIfWithDispatch._unwrap_mlir_values(else_values, effective_result_names, "else-branch")
            for name, expect_ty, got in zip(effective_result_names, result_types, else_raw):
                if got.type != expect_ty:
                    raise TypeError(
                        f"if/else variable '{name}' type mismatch in else-branch: "
                        f"expected {expect_ty}, got {got.type}"
                    )
            scf.YieldOp(else_raw)

        partial_wrapped = ReplaceIfWithDispatch._pack_dispatch_results(
            list(if_op.results), effective_result_values
        )
        if len(effective_result_names) == 1:
            partial_values = [partial_wrapped]
        else:
            partial_values = list(partial_wrapped)
        merged_values = ReplaceIfWithDispatch._merge_partial_results(
            result_names, result_values, effective_result_names, partial_values
        )
        return ReplaceIfWithDispatch._pack_named_values(result_names, merged_values)

    @classmethod
    def rewrite_globals(cls):
        return {
            "const_expr": const_expr,
            "scf_if_dispatch": cls.scf_if_dispatch,
            "scf_if_collect_results": cls._collect_result_dict,
        }

    _REWRITE_HELPER_NAMES = {
        "const_expr",
        "type",
        "bool",
        "isinstance",
        "hasattr",
    }

    @staticmethod
    def _could_be_dynamic(test_node):
        """Check if an if-condition AST could produce an MLIR Value at runtime.

        Layer-by-layer recursive check:
        1) classify current node if possible,
        2) otherwise recurse into direct children,
        3) unresolved nodes default to static (no forced rewrite).
        """
        def _is_literal_expr(node):
            if isinstance(node, ast.Constant):
                return True
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                return all(_is_literal_expr(e) for e in node.elts)
            if isinstance(node, ast.Dict):
                return all(
                    (k is None or _is_literal_expr(k)) and _is_literal_expr(v)
                    for k, v in zip(node.keys, node.values)
                )
            return False

        def _try_static_value(node):
            if not _is_literal_expr(node):
                return False, None
            if isinstance(node, ast.Constant):
                return True, node.value
            try:
                return True, ast.literal_eval(node)
            except Exception as e:
                log().error(
                    "[FlyDSL ast_rewriter] literal_eval failed: "
                    f"node={ast.dump(node, include_attributes=False)}, err={e!r}"
                )
                return False, None

        def _eval_static_compare_pair(lhs, op, rhs):
            op_text_map = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Is: "is",
                ast.IsNot: "is not",
                ast.In: "in",
                ast.NotIn: "not in",
            }
            try:
                op_text = op_text_map.get(type(op))
                if op_text is None:
                    return None
                return eval(
                    f"lhs_val {op_text} rhs_val",
                    {"__builtins__": {}},
                    {"lhs_val": lhs, "rhs_val": rhs},
                )
            except Exception as e:
                log().error(
                    "[FlyDSL ast_rewriter] static compare eval failed: "
                    f"op={type(op).__name__}, lhs={lhs!r}, rhs={rhs!r}, err={e!r}"
                )
                return None

        def _visit(node):
            if _is_literal_expr(node):
                return False
            if isinstance(node, ast.Compare):
                compare_parts = [node.left, *node.comparators]
                for i, op in enumerate(node.ops):
                    lhs_node = compare_parts[i]
                    rhs_node = compare_parts[i + 1]
                    lhs_ok, lhs_val = _try_static_value(lhs_node)
                    rhs_ok, rhs_val = _try_static_value(rhs_node)
                    if lhs_ok and rhs_ok:
                        pair_result = _eval_static_compare_pair(lhs_val, op, rhs_val)
                        if pair_result is False:
                            return False
                return any(_visit(part) for part in compare_parts)
            if isinstance(node, ast.Call):
                func = node.func
                if not (isinstance(func, ast.Name) and func.id in ReplaceIfWithDispatch._REWRITE_HELPER_NAMES):
                    return True
            if isinstance(node, ast.Name):
                return True

            for child in ast.iter_child_nodes(node):
                if _visit(child):
                    return True

            # If this expression cannot be proven dynamic from itself or children,
            # keep it static to avoid over-rewriting unrelated Python control flow.
            return False

        return _visit(test_node)

    @staticmethod
    def _collect_assigned_vars(node: ast.If, active_symbols):
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
        analyzer.visit(ast.Module(body=node.body, type_ignores=[]))
        if node.orelse:
            analyzer.visit(ast.Module(body=node.orelse, type_ignores=[]))

        invoked_args = [name for name in invoked_args if name not in write_args]
        write_args = [name for name in write_args if in_active_symbols(name)]
        invoked_args = [name for name in invoked_args if in_active_symbols(name)]
        return write_args + invoked_args

    @staticmethod
    def _state_value_expr(name):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                attr="get",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=name), ast.Constant(value=None)],
            keywords=[],
        )

    def visit_If(self, node: ast.If) -> List[ast.AST]:
        active_symbols_before_if = self.symbol_scopes.snapshot_symbol_scopes()
        if _is_constexpr(node.test):
            node.test = _unwrap_constexpr(node.test)
            node.body = self._visit_stmt_block(node.body)
            if node.orelse:
                node.orelse = self._visit_stmt_block(node.orelse)
            return node
        if not self._could_be_dynamic(node.test):
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
            result_names = self._collect_assigned_vars(node, active_symbols_before_if)

            fn_args = [ast.arg(arg="__ret_names", annotation=None)] + [ast.arg(arg=v, annotation=None) for v in result_names]

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
            setattr(then_func, _ASTREWRITE_MARKER, True)
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
                                elts=[self._state_value_expr(v) for v in result_names],
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
                        args=[ast.arg(arg="__ret_names", annotation=None)] + [ast.arg(arg=v, annotation=None) for v in result_names],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=list(node.orelse) + ([_state_return_node()] if result_names else []),
                    decorator_list=[],
                    type_params=[],
                )
                setattr(else_func, _ASTREWRITE_MARKER, True)
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
                        args=[ast.arg(arg="__ret_names", annotation=None)] + [ast.arg(arg=v, annotation=None) for v in result_names],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[_state_return_node()],
                    decorator_list=[],
                    type_params=[],
                )
                setattr(else_func, _ASTREWRITE_MARKER, True)
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


@ASTRewriter.register
class InsertEmptyYieldForSCFFor(Transformer):
    @staticmethod
    def _to_index(val):
        if isinstance(val, ir.Value):
            if val.type == ir.IndexType.get():
                return val
            return arith.IndexCastOp(ir.IndexType.get(), val).result
        if hasattr(val, "ir_value"):
            raw = val.ir_value()
            if isinstance(raw, ir.Value) and raw.type != ir.IndexType.get():
                return arith.IndexCastOp(ir.IndexType.get(), raw).result
            return raw
        if isinstance(val, int) and not isinstance(val, bool):
            return arith.ConstantOp(ir.IndexType.get(), val).result
        raise TypeError(
            f"_to_index expected ir.Value, object with ir_value(), or int; got {type(val).__name__}"
        )

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
            for_op = scf.ForOp(start_val, stop_val, step_val, list(init))
            with ir.InsertionPoint(for_op.body):
                yield for_op.induction_variable, list(for_op.inner_iter_args)
        else:
            for_op = scf.ForOp(start_val, stop_val, step_val)
            with ir.InsertionPoint(for_op.body):
                yield for_op.induction_variable

    @classmethod
    def rewrite_globals(cls):
        return {
            "scf_range": cls.scf_range,
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

    def visit_For(self, node: ast.For) -> ast.For:
        if self._is_range_constexpr(node.iter):
            node.iter.func = ast.Name(id="range", ctx=ast.Load())
            node = self.generic_visit(node)
            node = ast.fix_missing_locations(node)
            return node
        if self._is_range(node.iter):
            node.iter.func = ast.Name(id="scf_range", ctx=ast.Load())
        line = ast.dump(node.iter)
        if "for_" in line or "scf.for_" in line or "scf_range" in line:
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
                assert last_statement.end_lineno is not None, (
                    f"last_statement {ast.unparse(last_statement)} must have end_lineno"
                )
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
        scf.YieldOp(processed)
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
    @staticmethod
    def scf_while_init(cond, *, loc=None, ip=None):
        if loc is None:
            loc = ir.Location.unknown()

        def wrapper():
            nonlocal ip
            inits = list(cond.owner.operands)
            result_types = [i.type for i in inits]
            while_op = scf.WhileOp(result_types, inits, loc=loc, ip=ip)
            while_op.regions[0].blocks.append(*[i.type for i in inits])
            before = while_op.regions[0].blocks[0]
            while_op.regions[1].blocks.append(*[i.type for i in inits])
            after = while_op.regions[1].blocks[0]
            with ir.InsertionPoint(before) as ip:
                cond_op = scf.ConditionOp(cond, list(before.arguments))
                cond.owner.move_before(cond_op)
            with ir.InsertionPoint(after):
                yield inits

        if hasattr(CanonicalizeWhile.scf_while_init, "wrapper"):
            next(CanonicalizeWhile.scf_while_init.wrapper, False)
            del CanonicalizeWhile.scf_while_init.wrapper
            return False
        else:
            CanonicalizeWhile.scf_while_init.wrapper = wrapper()
            return next(CanonicalizeWhile.scf_while_init.wrapper)

    @staticmethod
    def scf_while_gen(cond, *, loc=None, ip=None):
        yield CanonicalizeWhile.scf_while_init(cond, loc=loc, ip=ip)
        yield CanonicalizeWhile.scf_while_init(cond, loc=loc, ip=ip)

    @classmethod
    def rewrite_globals(cls):
        return {
            "scf_while_gen": cls.scf_while_gen,
            "scf_while_init": cls.scf_while_init,
        }

    def visit_While(self, node: ast.While) -> List[ast.AST]:
        if _is_constexpr(node.test):
            node.test = _unwrap_constexpr(node.test)
            node = super().visit_While(node)
            return node
        with self.symbol_scopes.control_flow_scope():
            node = super().visit_While(node)
            if isinstance(node.test, ast.NamedExpr):
                test = node.test.value
            else:
                test = node.test
            w = ast.Call(func=ast.Name("scf_while_gen", ctx=ast.Load()), args=[test], keywords=[])
            w = ast.copy_location(w, node)
            assign = ast.Assign(
                targets=[ast.Name(f"w_{node.lineno}", ctx=ast.Store())],
                value=w,
            )
            assign = ast.fix_missing_locations(ast.copy_location(assign, node))

            next_ = ast.Call(
                func=ast.Name("next", ctx=ast.Load()),
                args=[
                    ast.Name(f"w_{node.lineno}", ctx=ast.Load()),
                    ast.Constant(False, kind="bool"),
                ],
                keywords=[],
            )
            next_ = ast.fix_missing_locations(ast.copy_location(next_, node))
            if isinstance(node.test, ast.NamedExpr):
                node.test.value = next_
            else:
                new_test = ast.NamedExpr(target=ast.Name(f"__init__{node.lineno}", ctx=ast.Store()), value=next_)
                new_test = ast.copy_location(new_test, node)
                node.test = new_test

            node = ast.fix_missing_locations(node)
            assign = ast.fix_missing_locations(assign)
            return [assign, node]
