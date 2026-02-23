import ast
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Union, Optional, Sequence

from bytecode import ConcreteBytecode

from .arith import constant as _ext_arith_constant, index_cast
from .gpu import get_device_mapping_array_attr
from ...ast.canonicalize import BytecodePatcher, Canonicalizer, StrictTransformer
from ...ast.util import ast_call, set_lineno, append_hidden_node
from ...meta import region_op
from ...util import get_user_code_loc, region_adder
from ....dialects._ods_common import (
    _cext,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type

# gotta come first
from ....dialects.scf import *
from ....dialects.scf import _Dialect, yield_ as yield__
from ....ir import (
    Attribute,
    IndexType,
    InsertionPoint,
    OpResultList,
    OpView,
    OpaqueType,
    Operation,
    Value,
    _denseI64ArrayAttr,
)

logger = logging.getLogger(__name__)

opaque = lambda dialect_namespace, buffer: OpaqueType.get(dialect_namespace, buffer)


def canonicalize_start_stop_step(start, stop, step):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    type = IndexType.get()
    maybe_types = {p.type for p in params if isinstance(p, Value)}
    if maybe_types:
        if len(maybe_types) > 1:
            raise ValueError(
                f"all {start=} and {stop=} and {step=} ir.Value objects must have the same type"
            )
        type = maybe_types.pop()

    for i, p in enumerate(params):
        if isinstance(p, int):
            p = _ext_arith_constant(p, type=type)
        assert isinstance(p, Value)
        params[i] = p

    return params[0], params[1], params[2]


def _build_for(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    start, stop, step = canonicalize_start_stop_step(start, stop, step)
    return ForOp(start, stop, step, iter_args, loc=loc, ip=ip)


def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()

    for_op = _build_for(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, for_op.results
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv


def placeholder_opaque_t():
    return opaque("scf", "placeholder")


for__ = region_op(_build_for, terminator=yield__)


def _parfor(op_ctor):
    def _base(
        lower_bounds, upper_bounds=None, steps=None, *, loc=None, ip=None, **kwargs
    ):
        if loc is None:
            loc = get_user_code_loc()

        if upper_bounds is None:
            upper_bounds = lower_bounds
            lower_bounds = [0] * len(upper_bounds)
        if steps is None:
            steps = [1] * len(lower_bounds)

        params = [lower_bounds, upper_bounds, steps]
        for i, p in enumerate(params):
            for j, pp in enumerate(p):
                if isinstance(pp, int):
                    pp = _ext_arith_constant(pp, index=True)
                if not _is_index_type(pp.type):
                    pp = index_cast(pp)
                p[j] = pp
            params[i] = p

        if loc is None:
            loc = get_user_code_loc()

        return op_ctor(*params, loc=loc, ip=ip, **kwargs)

    return _base


@region_op
def in_parallel():
    return InParallelOp()


def in_parallel_(parallel_insert_slice=None):
    if isinstance(parallel_insert_slice, (tuple, list)):
        assert (
            len(parallel_insert_slice) <= 1
        ), "expected at most one parallel_insert_slice op"
        if len(parallel_insert_slice) == 1:
            parallel_insert_slice = parallel_insert_slice[0]
        else:
            parallel_insert_slice = None

    @in_parallel
    def foo():
        if parallel_insert_slice is not None:
            parallel_insert_slice()
        return


def parallel_insert_slice(
    source,
    dest,
    static_offsets=None,
    static_sizes=None,
    static_strides=None,
    offsets=None,
    sizes=None,
    strides=None,
):
    from . import tensor

    @in_parallel
    def foo():
        tensor.parallel_insert_slice(
            source,
            dest,
            offsets,
            sizes,
            strides,
            static_offsets,
            static_sizes,
            static_strides,
        )


forall_ = region_op(_parfor(ForallOp), terminator=in_parallel_)


def _parfor_cm(op_ctor):
    def _base(*args, **kwargs):
        for_op = _parfor(op_ctor)(*args, **kwargs)
        block = for_op.regions[0].blocks[0]
        block_args = tuple(block.arguments)
        with InsertionPoint(block):
            yield block_args

    return _base


forall = _parfor_cm(ForallOp)


class ParallelOp(ParallelOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        inits: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        *,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        if inits is None:
            inits = []
        results = [i.type for i in inits]
        iv_types = [IndexType.get()] * len(lower_bounds)
        super().__init__(
            results,
            lower_bounds,
            upper_bounds,
            steps,
            inits,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*iv_types)

    @property
    def body(self):
        return self.regions[0].blocks[0]

    @property
    def induction_variables(self):
        return self.body.arguments


parange_ = region_op(
    _parfor(ParallelOp), terminator=lambda xs: reduce_return(xs[0]) if xs else None
)
parange = _parfor_cm(ParallelOp)


def while___(cond: Value, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()

    def wrapper():
        nonlocal ip
        inits = list(cond.owner.operands)
        results_ = [i.type for i in inits]
        while_op = WhileOp(results_, inits, loc=loc, ip=ip)
        while_op.regions[0].blocks.append(*[i.type for i in inits])
        before = while_op.regions[0].blocks[0]
        while_op.regions[1].blocks.append(*[i.type for i in inits])
        after = while_op.regions[1].blocks[0]
        with InsertionPoint(before) as ip:
            cond_ = condition(cond, list(before.arguments))
            cond.owner.move_before(cond_)
        with InsertionPoint(after):
            yield inits

    if hasattr(while___, "wrapper"):
        # needed in order to exit the `after` insertion point
        next(while___.wrapper, False)
        del while___.wrapper
        return False
    else:
        while___.wrapper = wrapper()
        # enter `after` insertion point
        return next(while___.wrapper)


def while__(cond: Value, *, loc=None, ip=None):
    yield while___(cond, loc=loc, ip=ip)
    yield while___(cond, loc=loc, ip=ip)


class ReduceOp(ReduceOp):
    def __init__(self, operands, num_reductions, *, loc=None, ip=None):
        super().__init__(operands, num_reductions, loc=loc, ip=ip)
        for i in range(num_reductions):
            self.regions[i].blocks.append(operands[i].type, operands[i].type)


def reduce_(*operands, num_reductions=1, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return ReduceOp(operands, num_reductions, loc=loc, ip=ip)


reduce = region_op(reduce_, terminator=lambda xs: reduce_return(*xs))


@region_adder(terminator=lambda xs: reduce_return(*xs))
def another_reduce(reduce_op):
    for r in reduce_op.regions:
        if len(r.blocks[0].operations) == 0:
            return r


def yield_(*args, results_=None):
    if len(args):
        assert results_ is None, "must provide results_ or args"
    if results_ is not None:
        args = results_
    if len(args) == 1 and isinstance(args[0], (list, OpResultList)):
        args = list(args[0])
    y = yield__(args)
    parent_op = y.operation.parent.opview
    if len(parent_op.results):
        results = get_op_result_or_op_results(parent_op)
        assert (
            isinstance(results, (OpResultList, Value))
            or isinstance(results, list)
            and all(isinstance(r, Value) for r in results)
        ), f"api has changed: {results=}"
        if isinstance(results, Value):
            results = [results]
        unpacked_args = args
        if any(isinstance(a, OpResultList) for a in unpacked_args):
            assert len(unpacked_args) == 1
            unpacked_args = list(unpacked_args[0])

        for i, r in enumerate(results):
            if r.type == placeholder_opaque_t():
                r.set_type(unpacked_args[i].type)

        if len(results) > 1:
            return results
        return results[0]


def _if(cond, results=None, *, has_else=False, loc=None, ip=None):
    if results is None:
        results = []
    if results:
        has_else = True
    if loc is None:
        loc = get_user_code_loc()
    return IfOp(cond, results, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)


@contextmanager
def if_ctx_manager(cond, results=None, *, has_else=False, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if_op = _if(cond, results, has_else=has_else, loc=loc, ip=ip)
    with InsertionPoint(if_op.regions[0].blocks[0]):
        yield if_op


@contextmanager
def else_ctx_manager(if_op):
    if len(if_op.regions[1].blocks) == 0:
        if_op.regions[1].blocks.append(*[])
    with InsertionPoint(if_op.regions[1].blocks[0]):
        yield


@region_adder(terminator=yield__)
def else_(ifop):
    return ifop.regions[1]


def is_yield_(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    )


def is_yield(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Yield)
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Yield)
    )


class InsertEmptyYield(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        updated_node = self.generic_visit(updated_node)

        new_yield = ast.Expr(ast.Yield(value=None))
        if not is_yield(updated_node.body[-1]):
            updated_node.body = append_hidden_node(
                updated_node.body, deepcopy(new_yield)
            )
        if updated_node.orelse and not is_yield(updated_node.orelse[-1]):
            updated_node.orelse = append_hidden_node(
                updated_node.orelse, deepcopy(new_yield)
            )

        updated_node = ast.fix_missing_locations(updated_node)
        return updated_node

    def visit_For(self, updated_node: ast.For) -> ast.For:
        # TODO(max): this isn't robust at all...
        line = ast.dump(updated_node.iter.func)
        if "range_" in line or "for_" in line:
            updated_node = self.generic_visit(updated_node)
            new_yield = ast.Expr(ast.Yield(value=None))
            if not is_yield(updated_node.body[-1]):
                updated_node.body = append_hidden_node(updated_node.body, new_yield)
            updated_node = ast.fix_missing_locations(updated_node)
        return updated_node


def forward_yield_from_nested_if(node_body):
    last_statement = node_body[0].body[-1]
    if isinstance(last_statement.targets[0], ast.Tuple):
        res = ast.Tuple(
            [ast.Name(t.id, ast.Load()) for t in last_statement.targets[0].elts],
            ast.Load(),
        )
        targets = [
            ast.Tuple(
                [ast.Name(t.id, ast.Store()) for t in last_statement.targets[0].elts],
                ast.Store(),
            )
        ]
    else:
        res = ast.Name(last_statement.targets[0].id, ast.Load())
        targets = [ast.Name(last_statement.targets[0].id, ast.Store())]
    forwarding_yield = ast.Assign(
        targets=targets,
        value=ast.Yield(res),
    )
    return append_hidden_node(node_body, forwarding_yield)


class CanonicalizeElIfs(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        # postorder
        updated_node = self.generic_visit(updated_node)
        needs_forward = lambda body: (
            body
            and isinstance(body[0], ast.If)
            and is_yield(body[0].body[-1])
            and not is_yield(body[-1])
        )
        if needs_forward(updated_node.body):
            updated_node.body = forward_yield_from_nested_if(updated_node.body)

        if needs_forward(updated_node.orelse):
            updated_node.orelse = forward_yield_from_nested_if(updated_node.orelse)
        updated_node = ast.fix_missing_locations(updated_node)
        return updated_node


class CanonicalizeWhile(StrictTransformer):
    def visit_While(self, updated_node: ast.While) -> List[ast.AST]:
        # postorder
        updated_node = self.generic_visit(updated_node)
        if isinstance(updated_node.test, ast.NamedExpr):
            test = updated_node.test.value
        else:
            test = updated_node.test
        w = ast_call(while__.__name__, [test])
        w = ast.copy_location(w, updated_node)
        assign = ast.Assign(
            targets=[ast.Name(f"w_{updated_node.lineno}", ctx=ast.Store())],
            value=w,
        )
        assign = ast.fix_missing_locations(ast.copy_location(assign, updated_node))

        next_ = ast_call(
            next.__name__,
            [
                ast.Name(f"w_{updated_node.lineno}", ctx=ast.Load()),
                ast.Constant(False, kind="bool"),
            ],
        )
        next_ = ast.fix_missing_locations(ast.copy_location(next_, updated_node))
        if isinstance(updated_node.test, ast.NamedExpr):
            updated_node.test.value = next_
        else:
            new_test = ast.NamedExpr(
                target=ast.Name(f"__init__{updated_node.lineno}"), value=next_
            )
            new_test = ast.copy_location(new_test, updated_node)
            updated_node.test = new_test

        updated_node = ast.fix_missing_locations(updated_node)
        assign = ast.fix_missing_locations(assign)

        return [assign, updated_node]


class ReplaceYieldWithSCFYield(StrictTransformer):
    def visit_Yield(self, node: ast.Yield) -> ast.Expr:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        call = ast.copy_location(ast_call(yield_.__name__, args), node)
        call = ast.fix_missing_locations(call)
        return call


class ReplaceIfWithWith(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> Union[ast.With, List[ast.With]]:
        is_elif = (
            len(updated_node.orelse) >= 1
            and isinstance(updated_node.orelse[0], ast.If)
            and updated_node.body[-1].end_lineno + 1 == updated_node.orelse[0].lineno
        )

        updated_node = self.generic_visit(updated_node)
        last_statement = updated_node.body[-1]
        assert is_yield_(last_statement) or is_yield(
            last_statement
        ), f"{last_statement=}"

        test = updated_node.test
        num_results = max(
            len(last_statement.value.args),
            # if lhs of assign is a tuple unpacking
            (
                len(last_statement.targets[0].elts)
                if isinstance(last_statement, ast.Assign)
                and isinstance(last_statement.targets[0], ast.Tuple)
                else 0
            ),
        )
        results = [ast_call(placeholder_opaque_t.__name__) for _ in range(num_results)]
        results = ast.fix_missing_locations(
            ast.copy_location(ast.Tuple(results, ctx=ast.Load()), test)
        )

        if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Store())
        withitem = ast.withitem(
            context_expr=ast_call(if_ctx_manager.__name__, args=[test, results]),
            optional_vars=if_op_name,
        )
        then_with = ast.With(items=[withitem])
        then_with = ast.copy_location(then_with, updated_node)
        then_with = ast.fix_missing_locations(then_with)
        then_with.body = updated_node.body

        if updated_node.orelse:
            if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Load())
            withitem = ast.withitem(
                context_expr=ast_call(else_ctx_manager.__name__, args=[if_op_name]),
                optional_vars=None,
            )
            else_with = ast.With(items=[withitem])
            if is_elif:
                else_with = ast.copy_location(else_with, updated_node.orelse[0])
            else:
                else_with = set_lineno(else_with, updated_node.orelse[0].lineno - 1)
            else_with = ast.fix_missing_locations(else_with)
            else_with.body = updated_node.orelse
            return [then_with, else_with]
        else:
            return then_with


class RemoveJumpsAndInsertGlobals(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
        # TODO(max): this is bad and should be in the closure rather than as a global
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[if_ctx_manager.__name__] = if_ctx_manager
        f.__globals__[else_ctx_manager.__name__] = else_ctx_manager
        f.__globals__[placeholder_opaque_t.__name__] = placeholder_opaque_t
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
        CanonicalizeWhile,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()

execute_region = region_op(ExecuteRegionOp)
