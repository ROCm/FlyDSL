import ast
import difflib
import inspect
import types
from textwrap import dedent
from typing import List

from .._mlir import ir
from .._mlir.dialects import arith, scf
from ..expr import const_expr
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
    func = node.func
    if isinstance(func, ast.Name) and func.id == "const_expr":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "const_expr":
        return True
    return False


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
            func_node = rewriter.generic_visit(func_node)
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


class Transformer(ast.NodeTransformer):
    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if getattr(node, _ASTREWRITE_MARKER, False):
            return node
        node = self.generic_visit(node)
        return node


@ASTRewriter.register
class RewriteBoolOps(Transformer):
    @staticmethod
    def dsl_and_(lhs, rhs):
        if hasattr(lhs, "__dsl_and__"):
            return lhs.__dsl_and__(rhs)
        if hasattr(rhs, "__dsl_and__"):
            return rhs.__dsl_and__(lhs)
        return lhs and rhs

    @staticmethod
    def dsl_or_(lhs, rhs):
        if hasattr(lhs, "__dsl_or__"):
            return lhs.__dsl_or__(rhs)
        if hasattr(rhs, "__dsl_or__"):
            return rhs.__dsl_or__(lhs)
        return lhs or rhs

    @staticmethod
    def dsl_not_(x):
        if hasattr(x, "__dsl_not__"):
            return x.__dsl_not__()
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

        if isinstance(node.op, ast.And):
            func_name = "dsl_and_"
            short_circuit_value = ast.Constant(value=False)
        elif isinstance(node.op, ast.Or):
            func_name = "dsl_or_"
            short_circuit_value = ast.Constant(value=True)
        else:
            return node

        def short_circuit_eval(value, sc_value):
            return ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Call(
                            func=ast.Name(id="type", ctx=ast.Load()),
                            args=[value],
                            keywords=[],
                        ),
                        ops=[ast.Eq()],
                        comparators=[ast.Name(id="bool", ctx=ast.Load())],
                    ),
                    ast.Compare(
                        left=value,
                        ops=[ast.Eq()],
                        comparators=[sc_value],
                    ),
                ],
            )

        lhs = node.values[0]
        for i in range(1, len(node.values)):
            test = short_circuit_eval(lhs, short_circuit_value)
            lhs = ast.IfExp(
                test=test,
                body=lhs,
                orelse=ast.Call(func=ast.Name(func_name, ctx=ast.Load()), args=[lhs, node.values[i]], keywords=[]),
            )

        return ast.copy_location(ast.fix_missing_locations(lhs), node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            call = ast.Call(func=ast.Name("dsl_not_", ctx=ast.Load()), args=[node.operand], keywords=[])
            return ast.copy_location(ast.fix_missing_locations(call), node)
        return node


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
        if hasattr(cond, "ir_value"):
            return cond.ir_value()
        return cond

    @staticmethod
    def scf_if_dispatch(cond, then_fn, else_fn=None):
        if not ReplaceIfWithDispatch._is_dynamic(cond):
            # compile-time evaluation
            if cond:
                then_fn()
            elif else_fn is not None:
                else_fn()
            return

        has_else = else_fn is not None
        loc = ir.Location.unknown()
        if_op = scf.IfOp(ReplaceIfWithDispatch._to_i1(cond), [], hasElse=has_else, loc=loc)
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            then_fn()
            scf.YieldOp([])
        if has_else:
            if len(if_op.regions[1].blocks) == 0:
                if_op.regions[1].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                else_fn()
                scf.YieldOp([])

    @classmethod
    def rewrite_globals(cls):
        return {
            "const_expr": const_expr,
            "scf_if_dispatch": cls.scf_if_dispatch,
        }

    _REWRITE_HELPER_NAMES = {"dsl_not_", "dsl_and_", "dsl_or_",
                              "scf_if_dispatch", "const_expr", "type",
                              "bool", "isinstance", "hasattr"}

    @staticmethod
    def _could_be_dynamic(test_node):
        """Check if an if-condition AST could produce an MLIR Value at runtime.

        Calls to RewriteBoolOps helpers (dsl_not_, dsl_and_, dsl_or_) and
        Python builtins are NOT considered dynamic — they just wrap Python-level
        boolean logic. Only calls to user/MLIR functions can produce Values.
        """
        for child in ast.walk(test_node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id in ReplaceIfWithDispatch._REWRITE_HELPER_NAMES:
                    continue
                return True
        return False

    def visit_If(self, node: ast.If) -> List[ast.AST]:
        if _is_constexpr(node.test):
            node.test = _unwrap_constexpr(node.test)
            node = self.generic_visit(node)
            return node
        if not self._could_be_dynamic(node.test):
            node = self.generic_visit(node)
            return node
        node = self.generic_visit(node)
        uid = ReplaceIfWithDispatch._counter
        ReplaceIfWithDispatch._counter += 1

        then_name = f"__then_{uid}"
        then_func = ast.FunctionDef(
            name=then_name,
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=node.body,
            decorator_list=[],
            type_params=[],
        )
        setattr(then_func, _ASTREWRITE_MARKER, True)
        then_func = ast.copy_location(then_func, node)
        then_func = ast.fix_missing_locations(then_func)

        dispatch_args = [node.test, ast.Name(then_name, ctx=ast.Load())]
        result = [then_func]

        if node.orelse:
            else_name = f"__else_{uid}"
            else_func = ast.FunctionDef(
                name=else_name,
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=node.orelse,
                decorator_list=[],
                type_params=[],
            )
            setattr(else_func, _ASTREWRITE_MARKER, True)
            else_func = ast.copy_location(else_func, node)
            else_func = ast.fix_missing_locations(else_func)
            dispatch_args.append(ast.Name(else_name, ctx=ast.Load()))
            result.append(else_func)

        dispatch_call = ast.Expr(
            value=ast.Call(func=ast.Name("scf_if_dispatch", ctx=ast.Load()), args=dispatch_args, keywords=[])
        )
        dispatch_call = ast.copy_location(dispatch_call, node)
        dispatch_call = ast.fix_missing_locations(dispatch_call)
        result.append(dispatch_call)

        return result


@ASTRewriter.register
class InsertEmptyYieldForSCFFor(Transformer):
    @staticmethod
    def _to_index(val):
        if isinstance(val, ir.Value):
            return val
        if hasattr(val, "ir_value"):
            return val.ir_value()
        return arith.ConstantOp(ir.IndexType.get(), val).result

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
    def _is_range_constexpr(iter_node):
        if not isinstance(iter_node, ast.Call):
            return False
        func = iter_node.func
        if isinstance(func, ast.Name) and func.id == "range_constexpr":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "range_constexpr":
            return True
        return False

    @staticmethod
    def _is_range(iter_node):
        if not isinstance(iter_node, ast.Call):
            return False
        func = iter_node.func
        return isinstance(func, ast.Name) and func.id == "range"

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
            node = self.generic_visit(node)
            new_yield = ast.Expr(ast.Yield(value=None))
            if not self._is_yield(node.body[-1]):
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
            node = self.generic_visit(node)
            return node
        node = self.generic_visit(node)
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
            new_test = ast.NamedExpr(target=ast.Name(f"__init__{node.lineno}"), value=next_)
            new_test = ast.copy_location(new_test, node)
            node.test = new_test

        node = ast.fix_missing_locations(node)
        assign = ast.fix_missing_locations(assign)
        return [assign, node]
