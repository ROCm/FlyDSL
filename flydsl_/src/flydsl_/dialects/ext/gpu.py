import inspect
from functools import partial
import sys
from pathlib import Path
from functools import wraps
from typing import Any, List, Optional, Tuple, Union, Callable


import _mlir.ir as ir
from _mlir.dialects.arith import AffineConstantExpr
from _mlir.extras import types as T
from _mlir.extras.meta import (
    region_op,
)

from .func import FuncBase, make_maybe_no_args_decorator, get_user_code_loc
from .python_control_flow import lower_range_for_loops

# from ...util import (
#     ModuleMeta,
#     # _get_previous_frame_idents,
#     # make_maybe_no_args_decorator,
# )

from _mlir.dialects._gpu_ops_gen import _Dialect
from _mlir.dialects._ods_common import (
    _cext,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from _mlir.dialects import gpu
from _mlir.dialects._gpu_ops_gen import _Dialect
from _mlir.dialects._gpu_ops_gen import *

from _mlir.ir import (
    ArrayAttr,
    AttrBuilder,
    Attribute,
    Context,
    InsertionPoint,
    ShapedType,
    Type,
    UnitAttr,
    Value,
)

from ._loc import maybe_default_loc

# -----------------------------------------------------------------------------
# Debug location defaults for common GPU builtins
# -----------------------------------------------------------------------------
#
# The ODS-generated helpers (imported via `from _gpu_ops_gen import *`) accept an
# optional `loc=` but default to inheriting `Location.current`. During kernel
# emission we often have a single outer `with Location(...)` active, which makes
# large blocks of IR appear to come from a single source line (`loc(#loc)`).
#
# For debug/dump workflows, it is much more useful if these common builtins pick
# up the user call-site location automatically.
_ods_block_id = block_id
_ods_thread_id = thread_id
_ods_block_dim = block_dim
_ods_grid_dim = grid_dim


def block_id(dimension, *, upper_bound=None, results=None, loc=None, ip=None):
    return _ods_block_id(dimension, upper_bound=upper_bound, results=results, loc=maybe_default_loc(loc), ip=ip)


def thread_id(dimension, *, upper_bound=None, results=None, loc=None, ip=None):
    return _ods_thread_id(dimension, upper_bound=upper_bound, results=results, loc=maybe_default_loc(loc), ip=ip)


def block_dim(dimension, *, upper_bound=None, results=None, loc=None, ip=None):
    return _ods_block_dim(dimension, upper_bound=upper_bound, results=results, loc=maybe_default_loc(loc), ip=ip)


def grid_dim(dimension, *, upper_bound=None, results=None, loc=None, ip=None):
    return _ods_grid_dim(dimension, upper_bound=upper_bound, results=results, loc=maybe_default_loc(loc), ip=ip)


def _as_mlir_value(x, *, index: bool = False, loc=None, ip=None):
    """Best-effort normalize common value-like inputs to an MLIR Value.

    Supports:
    - `ir.Value` / `OpResult`
    - FLIR wrappers with `.value` (e.g. ArithValue)
    - Python `int` (materialized as index const when `index=True`)
    """
    if x is None:
        return None
    # Unwrap ArithValue-like wrappers (avoid importing flir/arith to prevent cycles).
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            pass
    if isinstance(x, int):
        if index:
            from . import arith as _arith_ext
            return _arith_ext.index(x, loc=loc, ip=ip).value
        raise TypeError("expected MLIR Value, got int (use index=True to materialize an index const)")
    return x


def find_ops(op, pred: Callable[[ir.OpView, ir.Operation, ir.Module], bool], single=False):
    if isinstance(op, (ir.OpView, ir.Module)):
        op = op.operation

    matching = []

    def find(op: ir.Operation):
        if single and len(matching):
            return
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if pred(o):
                        matching.append(o)
                    find(o)

    find(op)
    if single and matching:
        matching = matching[0]
    return matching


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def set_container_module(module):
    module.operation.attributes["gpu.container_module"] = UnitAttr.get()
    return module


def get_device_mapping_array_attr(mapping: List[Attribute], context: Optional[Context] = None) -> ArrayAttr:
    if context is None:
        context = Context.current
    if isinstance(mapping, ArrayAttr):
        return mapping

    return ArrayAttr.get(mapping, context=context)


def gpu_attr(mnemonic, attr_value):
    return Attribute.parse(f"#gpu.{mnemonic}<{attr_value}>")


def thread_attr(thread):
    return gpu_attr("thread", thread)


def block_attr(block):
    return gpu_attr("block", block)


def warp_attr(warp):
    return gpu_attr("warp", warp)


def warpgroup_attr(warpgroup):
    return gpu_attr("warpgroup", warpgroup)


def address_space_attr(address_space: gpu.AddressSpace):
    return gpu_attr("address_space", address_space)


_int = int


def smem_space(int=False):
    a = gpu.AddressSpace.Workgroup
    if int:
        return _int(a)

    return address_space_attr(a)


lds_space = smem_space


class ModuleMeta(type):
    def __new__(cls, name, bases, classdict, **kwargs):
        ip = classdict.pop("ip")
        new = super().__new__(cls, name, bases, classdict)
        for k, v in classdict.items():
            if callable(v):
                v.qualname = name
        ip.__exit__(None, None, None)
        return new


@_cext.register_operation(_Dialect, replace=True)
class GPUModuleOp(GPUModuleOp):
    def __init__(self, sym_name, targets: Optional[List[Attribute]] = None, *, loc=None, ip=None):
        if loc is None:
            loc = get_user_code_loc()
        if targets is not None:
            for i, t in enumerate(targets):
                if isinstance(t, str):
                    targets[i] = Attribute.parse(t)
        _ods_context = get_default_loc_context(loc)
        sym_name = (
            sym_name
            if (issubclass(type(sym_name), Attribute) or not AttrBuilder.contains("SymbolNameAttr"))
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        super().__init__(
            sym_name=sym_name,
            targets=None if targets is None else ArrayAttr.get(targets),
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append()

    @property
    def body(self):
        return self.regions[0].blocks[0]


module = region_op(GPUModuleOp)


class GPUModuleMeta(ModuleMeta):
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = get_user_code_loc()
        targets = kwargs.pop("targets", None)
        gpu_module_op = GPUModuleOp(
            sym_name=name,
            targets=targets,
            ip=kwargs.pop("ip", None),
            loc=loc,
        )
        ip = InsertionPoint(gpu_module_op.body)
        ip.__enter__()
        return {"ip": ip, "gpu_module_op": gpu_module_op}


@_cext.register_operation(_Dialect, replace=True)
class GPUFuncOp(GPUFuncOp):
    def __init__(
        self,
        sym_name,
        function_type,
        *,
        sym_visibility=None,
        arg_attrs=None,
        res_attrs=None,
        workgroup_attrib_attrs=None,
        private_attrib_attrs=None,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        super().__init__(
            function_type=function_type,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
            workgroup_attrib_attrs=workgroup_attrib_attrs,
            private_attrib_attrs=private_attrib_attrs,
            loc=loc,
            ip=ip,
        )
        self.operation.attributes["gpu.kernel"] = UnitAttr.get()
        _ods_context = get_default_loc_context(loc)
        self.operation.attributes["sym_name"] = (
            sym_name
            if (issubclass(type(sym_name), Attribute) or not AttrBuilder.contains("SymbolNameAttr"))
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        if sym_visibility is not None:
            self.operation.attributes["sym_visibility"] = (
                sym_visibility
                if (issubclass(type(sym_visibility), Attribute) or not AttrBuilder.contains("StrAttr"))
                else AttrBuilder.get("StrAttr")(sym_visibility, context=_ods_context)
            )


@_cext.register_operation(_Dialect, replace=True)
class LaunchOp(LaunchOp):
    def __init__(
        self,
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        _ods_context = get_default_loc_context(loc)
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size

        super().__init__(
            async_token,
            async_dependencies,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*[T.index() for _ in range(12)])


def launch_(
    grid_size: Tuple[Any, Any, Any],
    block_size: Tuple[Any, Any, Any],
    async_dependencies=None,
    dynamic_shared_memory_size: Optional[Value] = None,
    *,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    for size in [grid_size, block_size]:
        for i, s in enumerate(size):
            if isinstance(s, int):
                size[i] = constant(T.i32(), s)
    launch_op = LaunchOp(
        grid_size,
        block_size,
        async_dependencies,
        dynamic_shared_memory_size,
        loc=loc,
        ip=ip,
    )
    return launch_op


launch = region_op(launch_, terminator=lambda *_args: TerminatorOp())


@_cext.register_operation(_Dialect, replace=True)
class LaunchFuncOp(LaunchFuncOp):
    def __init__(
        self,
        kernel: List[str],
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        kernel_operands: List[Value] = None,
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        async_object=None,
        *,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        _ods_context = get_default_loc_context(loc)
        if async_dependencies is None:
            async_dependencies = []
        else:
            async_dependencies = [_as_mlir_value(v, loc=loc, ip=ip) for v in async_dependencies]
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size

        # gpu.launch_func operands are index-typed; unwrap ArithValue and materialize ints as index consts.
        grid_size_x = _as_mlir_value(grid_size_x, index=True, loc=loc, ip=ip)
        grid_size_y = _as_mlir_value(grid_size_y, index=True, loc=loc, ip=ip)
        grid_size_z = _as_mlir_value(grid_size_z, index=True, loc=loc, ip=ip)
        block_size_x = _as_mlir_value(block_size_x, index=True, loc=loc, ip=ip)
        block_size_y = _as_mlir_value(block_size_y, index=True, loc=loc, ip=ip)
        block_size_z = _as_mlir_value(block_size_z, index=True, loc=loc, ip=ip)

        if kernel_operands is None:
            kernel_operands = []
        else:
            # Some call sites may accidentally pass a single Value; normalize.
            if isinstance(kernel_operands, Value):
                kernel_operands = [kernel_operands]
            kernel_operands = [_as_mlir_value(v, loc=loc, ip=ip) for v in list(kernel_operands)]
        dynamic_shared_memory_size = _as_mlir_value(dynamic_shared_memory_size, loc=loc, ip=ip)
        async_object = _as_mlir_value(async_object, loc=loc, ip=ip)

        super().__init__(
            async_token,
            async_dependencies,
            kernel,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            kernel_operands,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            asyncObject=async_object,
            loc=loc,
            ip=ip,
        )


class GPUFunc(FuncBase):
    def __call__(
        self,
        *kernel_operands: List[Value],
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        stream=None,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        # Normalize sizes/operands to Values (unwrap ArithValue; materialize ints as index consts).
        grid_size = tuple(_as_mlir_value(s, index=True, loc=loc, ip=ip) for s in grid_size)
        block_size = tuple(_as_mlir_value(s, index=True, loc=loc, ip=ip) for s in block_size)
        kernel_operands = [_as_mlir_value(v, loc=loc, ip=ip) for v in kernel_operands]
        if async_dependencies is None:
            async_dependencies = []
        else:
            async_dependencies = [_as_mlir_value(v, loc=loc, ip=ip) for v in async_dependencies]
        dynamic_shared_memory_size = _as_mlir_value(dynamic_shared_memory_size, loc=loc, ip=ip)
        return get_op_result_or_op_results(
            LaunchFuncOp(
                ([self.qualname, self.func_name] if self.qualname is not None else [self.func_name]),
                grid_size,
                block_size,
                kernel_operands,
                async_dependencies,
                dynamic_shared_memory_size,
                async_object=stream,
                loc=loc,
            )
        )


def _get_previous_frame_idents(val, previous_frame):
    return [var_name for var_name, var_val in previous_frame.f_locals.items() if var_val is val]


class Grid:
    def __init__(self, func_):
        self.func = func_

    def __getitem__(self, item):
        previous_frame = inspect.currentframe().f_back
        var_names = [_get_previous_frame_idents(arg, previous_frame) for arg in item]
        kwargs = {}
        for i, it in enumerate(item):
            assert len(var_names[i]) == 1, "expected unique kwarg"
            k = var_names[i][0]
            kwargs[k] = it

        return partial(self.func, **kwargs)

    @property
    def qualname(self):
        return self.func.qualname

    @qualname.setter
    def qualname(self, v):
        self.func.qualname = v

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    emit=False,
    generics=None,
    loc=None,
    ip=None,
    emit_grid=False,
    lower_range_loops: bool = False,
) -> Grid:
    if loc is None:
        loc = get_user_code_loc()
    if generics is None and hasattr(f, "__type_params__") and f.__type_params__:
        generics = f.__type_params__
    if lower_range_loops:
        f = lower_range_for_loops(f)
    func_ = GPUFunc(
        body_builder=f,
        func_op_ctor=GPUFuncOp,
        return_op_ctor=ReturnOp,
        call_op_ctor=LaunchFuncOp,
        sym_visibility=sym_visibility,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        generics=generics,
        loc=loc,
        ip=ip,
    )
    func_.__name__ = f.__name__
    if emit:
        func_.emit()
    if emit_grid:
        func_ = Grid(func_)
    return func_


@make_maybe_no_args_decorator
def kernel(
    f,
    *,
    emit_grid: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> Grid:
    """
    Enables lowering of `for i in range(...)` into `scf.for`.
    """
    return func(
        f,
        emit=True,
        emit_grid=emit_grid,
        lower_range_loops=True,
        loc=loc,
        ip=ip,
        **kwargs,
    )


def all_reduce__(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return AllReduceOp(value, op=op, uniform=uniform, loc=loc, ip=ip)


def all_reduce_(value: Value, *, op=None, uniform=None, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return get_op_result_or_op_results(all_reduce__(value, op=op, uniform=uniform, loc=loc, ip=ip))


all_reduce = region_op(all_reduce__, terminator=YieldOp)


_alloc = alloc


def alloc(
    sizes: Union[int, Value],
    element_type: Type = None,
    async_dependencies=None,
    dynamic_sizes=None,
    symbol_operands=None,
    host_shared=None,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if symbol_operands is None:
        symbol_operands = []
    if dynamic_sizes is None:
        dynamic_sizes = []
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()

    memref_shape = []
    for s in sizes:
        if isinstance(s, int):
            memref_shape.append(s)
        else:
            memref_shape.append(ShapedType.get_dynamic_size())
            dynamic_sizes.append(s)
    memref = T.memref(*memref_shape, element_type)
    return _alloc(
        memref,
        async_token,
        async_dependencies,
        dynamic_sizes,
        symbol_operands,
        host_shared=host_shared,
        loc=loc,
        ip=ip,
    )


_dealloc = dealloc


def dealloc(memref, async_dependencies=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()
    return _dealloc(async_token, async_dependencies, memref, loc=loc, ip=ip)


_memcpy = memcpy


def memcpy(dst, src, async_dependencies=None, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    if async_dependencies is None:
        async_dependencies = []
    async_token = None
    if len(async_dependencies):
        async_token = gpu_async_token()
    return _memcpy(
        async_token,
        async_dependencies,
        dst,
        src,
        loc=loc,
        ip=ip,
    )


def get_compile_object_bytes(compiled_module):
    binary = find_ops(compiled_module, lambda o: isinstance(o, BinaryOp), single=True)
    objects = list(map(gpu.ObjectAttr, binary.objects))
    return objects[-1].object


_printf = printf


def printf(format, *args, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _printf(format=format, args=args, loc=loc, ip=ip)


_dynamic_shared_memory = dynamic_shared_memory


def dynamic_shared_memory(*, int=False, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return _dynamic_shared_memory(
        T.memref(
            ShapedType.get_dynamic_size(),
            element_type=T.i8(),
            memory_space=smem_space(int),
        ),
        loc=loc,
        ip=ip,
    )


def barrier(*, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()

    return BarrierOp(loc=loc, ip=ip)
