import inspect
from functools import partial
import sys
from pathlib import Path
from functools import wraps
from typing import Any, List, Optional, Tuple, Union, Callable
from typing import Optional, List, Union, TypeVar

# Import from flydsl._mlir for types
from flydsl._mlir import ir as flydsl_ir
from flydsl._mlir.extras import types as T
from flydsl._mlir.interop import to_upstream, to_upstream_types

# Import from upstream MLIR for dialect operations (still needed)
from mlir.dialects._func_ops_gen import FuncOp
from mlir.extras.meta import region_op, op_region_builder


from mlir.dialects._ods_common import (
    _cext,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from mlir.dialects._gpu_ops_gen import _Dialect
from mlir.dialects._gpu_ops_gen import *
from mlir.dialects._gpu_enum_gen import *


from mlir.ir import (
    ArrayAttr,
    AttrBuilder,
    Attribute,
    Context,
    InsertionPoint,
    ShapedType,
    Type as UpstreamType,
    UnitAttr,
    Value,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    OpView,
    Operation,
    OpResultList,
    TypeAttr,
    register_attribute_builder,
)

# Also import flydsl Type for type checking
from flydsl._mlir import Type as FlyDSLType

_block_id = block_id
_thread_id = thread_id
_block_dim = block_dim
_grid_dim = grid_dim


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class block_idx:
    @classproperty
    def x(cls):
        return _block_id("x")

    @classproperty
    def y(cls):
        return _block_id("y")

    @classproperty
    def z(cls):
        return _block_id("z")


class block_dim:
    @classproperty
    def x(cls):
        return _block_dim("x")

    @classproperty
    def y(cls):
        return _block_dim("y")

    @classproperty
    def z(cls):
        return _block_dim("z")


class thread_idx:
    @classproperty
    def x(cls):
        return _thread_id("x")

    @classproperty
    def y(cls):
        return _thread_id("y")

    @classproperty
    def z(cls):
        return _thread_id("z")


class grid_dim:
    @classproperty
    def x(cls):
        return _grid_dim("x")

    @classproperty
    def y(cls):
        return _grid_dim("y")

    @classproperty
    def z(cls):
        return _grid_dim("z")


def gpu_attr(mnemonic, attr_value):
    return Attribute.parse(f"#gpu.{mnemonic}<{attr_value}>")


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
    def __init__(
        self, sym_name, targets: Optional[List[Attribute]] = None, *, loc=None, ip=None
    ):
        if targets is None:
            targets = []
        for i, t in enumerate(targets):
            if isinstance(t, str):
                targets[i] = Attribute.parse(t)
        _ods_context = get_default_loc_context(loc)
        sym_name = (
            sym_name
            if (
                issubclass(type(sym_name), Attribute)
                or not AttrBuilder.contains("SymbolNameAttr")
            )
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        super().__init__(sym_name=sym_name, targets=ArrayAttr.get(targets), ip=ip)
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
            if (
                issubclass(type(sym_name), Attribute)
                or not AttrBuilder.contains("SymbolNameAttr")
            )
            else AttrBuilder.get("SymbolNameAttr")(sym_name, context=_ods_context)
        )
        if sym_visibility is not None:
            self.operation.attributes["sym_visibility"] = (
                sym_visibility
                if (
                    issubclass(type(sym_visibility), Attribute)
                    or not AttrBuilder.contains("StrAttr")
                )
                else AttrBuilder.get("StrAttr")(sym_visibility, context=_ods_context)
            )


def isalambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def prep_func_types(sig, return_types):
    assert not (
        not sig.return_annotation is inspect.Signature.empty and len(return_types) > 0
    ), f"func can use return annotation or explicit return_types but not both"
    return_types = (
        sig.return_annotation
        if not sig.return_annotation is inspect.Signature.empty
        else return_types
    )
    if not isinstance(return_types, (tuple, list)):
        return_types = [return_types]
    return_types = list(return_types)
    assert all(
        isinstance(r, (str, UpstreamType, FlyDSLType, TypeVar)) or isalambda(r) for r in return_types
    ), f"all return types must be ..._mlir types or strings or TypeVars or lambdas {return_types=}"

    input_types = [
        p.annotation
        for p in sig.parameters.values()
        if not p.annotation is inspect.Signature.empty
    ]
    
    def is_valid_type(t):
        """Check if t is a valid type annotation for GPU functions."""
        # Direct type instances
        if isinstance(t, (str, UpstreamType, FlyDSLType, TypeVar)):
            return True
        # Lambda functions
        if isalambda(t):
            return True
        # Type factory functions (like T.i32, T.f16)
        if callable(t) and hasattr(t, '__name__'):
            return True
        # Classes that are types
        if isinstance(t, type):
            return True
        return False
    
    assert all(
        is_valid_type(r) for r in input_types
    ), f"all input types must be ..._mlir types or strings or TypeVars or lambdas or callables {input_types=}"
    user_loc = None
    # If ir.Context is none (like for deferred func emit)
    if user_loc is None:
        user_locs = None
    else:
        user_locs = [user_loc] * len(sig.parameters)
    return input_types, return_types, user_locs


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
        _ods_context = get_default_loc_context(loc)
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        grid_size_x, grid_size_y, grid_size_z = grid_size
        block_size_x, block_size_y, block_size_z = block_size
        
        # CRITICAL: Convert flydsl types to upstream types
        kernel = to_upstream(kernel)
        grid_size_x = to_upstream(grid_size_x)
        grid_size_y = to_upstream(grid_size_y)
        grid_size_z = to_upstream(grid_size_z)
        block_size_x = to_upstream(block_size_x)
        block_size_y = to_upstream(block_size_y)
        block_size_z = to_upstream(block_size_z)
        if kernel_operands:
            kernel_operands = [to_upstream(op) for op in kernel_operands]

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


class GPUFunc:
    def __init__(
        self,
        body_builder,
        func_op_ctor,
        return_op_ctor,
        call_op_ctor,
        *,
        return_types=None,
        sym_visibility=None,
        sym_name=None,
        arg_attrs=None,
        res_attrs=None,
        func_attrs=None,
        function_type=None,
        generics: List[Union[TypeVar]] = None,
        qualname=None,
        loc=None,
        ip=None,
    ):
        assert inspect.isfunction(body_builder), body_builder
        assert inspect.isclass(func_op_ctor), func_op_ctor
        if return_op_ctor is not None:
            assert inspect.isclass(return_op_ctor), return_op_ctor
        assert inspect.isclass(call_op_ctor), call_op_ctor

        self.body_builder = body_builder
        if sym_name is None:
            sym_name = self.body_builder.__name__
        self.func_name = sym_name
        self.func_op_ctor = func_op_ctor
        self.return_op_ctor = return_op_ctor
        self.call_op_ctor = call_op_ctor
        self.arg_attrs = arg_attrs
        self.res_attrs = res_attrs
        self.generics = generics
        self.loc = loc
        self.ip = ip
        self._func_op = None
        # in case this function lives inside a class
        self.qualname = qualname

        self.sym_visibility = sym_visibility
        self.func_attrs = func_attrs
        if self.func_attrs is None:
            self.func_attrs = {}
        self.function_type = function_type

        if return_types is None:
            return_types = []
        sig = inspect.signature(self.body_builder)
        self.input_types, self.return_types, self.arg_locs = prep_func_types(
            sig, return_types
        )

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def emit(self, *call_args, decl=False, force=False):
        if self._func_op is None or decl or force:
            if self.function_type is None:
                if len(call_args) == 0:
                    input_types = self.input_types[:]
                    locals = {"T": T}
                    for i, v in enumerate(input_types):
                        if isinstance(v, TypeVar):
                            v = v.__name__
                        if isinstance(v, str):
                            input_types[i] = Type(
                                eval(v, self.body_builder.__globals__, locals)
                            )
                        elif isalambda(v):
                            input_types[i] = v()
                else:
                    input_types = [a.type for a in call_args]

                # CRITICAL: Convert flydsl types to upstream types for FunctionType
                upstream_input_types = to_upstream_types(input_types)
                upstream_return_types = to_upstream_types(self.return_types)
                function_type = TypeAttr.get(
                    FunctionType.get(
                        inputs=upstream_input_types,
                        results=upstream_return_types,
                    )
                )
            else:
                input_types = self.function_type.inputs
                function_type = TypeAttr.get(self.function_type)

            self._func_op = self.func_op_ctor(
                self.func_name,
                function_type,
                sym_visibility=self.sym_visibility,
                arg_attrs=self.arg_attrs,
                res_attrs=self.res_attrs,
                loc=self.loc,
                ip=self.ip or InsertionPoint.current,
            )
            if isinstance(self._func_op, FuncOp):
                self._func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            for k, v in self.func_attrs.items():
                self._func_op.attributes[k] = v

            # CRITICAL: Convert flydsl types to upstream types for blocks.append
            append_input_types = to_upstream_types(input_types)
            self._func_op.regions[0].blocks.append(*append_input_types, arg_locs=self.arg_locs)
            builder_wrapper = op_region_builder(
                self._func_op, self._func_op.regions[0], terminator=self.return_op_ctor
            )

            return_types = []

            def grab_results(*args):
                nonlocal return_types
                results = self.body_builder(*args)
                if isinstance(results, (tuple, list, OpResultList)):
                    return_types.extend([r.type for r in results])
                elif results is not None:
                    return_types.append(results.type)
                return results

            if self.function_type is None:
                builder_wrapper(grab_results)
                # CRITICAL: Convert flydsl types to upstream types for FunctionType
                upstream_input_types2 = to_upstream_types(input_types)
                upstream_return_types2 = to_upstream_types(return_types)
                function_type = FunctionType.get(
                    inputs=upstream_input_types2, results=upstream_return_types2
                )
                self._func_op.attributes["function_type"] = TypeAttr.get(function_type)
            else:
                builder_wrapper(self.body_builder)

        return self._func_op


def gpu_func(
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
):
    if generics is None and hasattr(f, "__type_params__") and f.__type_params__:
        generics = f.__type_params__
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
    return func_
