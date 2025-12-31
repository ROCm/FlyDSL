from __future__ import annotations

import inspect
from typing import Optional

from _mlir import ir
from _mlir.dialects import func as mlir_func

from pyflir.compiler.context import ensure_flir_python_extensions
from pyflir.dialects.ext import gpu
from pyflir.dialects.ext.func import get_user_code_loc
from pyflir.dialects.ext.func import prep_func_types
from pyflir.dialects.ext.python_control_flow import lower_range_for_loops


def _bind_method_builder(fn, instance_self, *, lower_range_loops: bool):
    """Return a plain Python function suitable for mlir-extras FuncBase/GPUFunc.

    - Ensures `inspect.isfunction(...)` is true (mlir-extras asserts this).
    - If `fn` is a method (`self` parameter), binds `instance_self` so `self` is
      NOT materialized as an MLIR function argument.
    - Optionally applies range-loop lowering prior to binding.
    """
    if lower_range_loops:
        try:
            f = lower_range_for_loops(fn)
        except Exception:
            # In dynamic contexts (e.g. interactive/closure-defined functions),
            # `inspect.getsource` can fail. Fall back to the original function.
            f = fn
    else:
        f = fn
    sig = inspect.signature(f)
    params = list(sig.parameters.values())
    drop_self = bool(params) and params[0].name == "self"
    if drop_self:
        sig = sig.replace(parameters=params[1:])

    def body_builder(*args, **kwargs):
        if drop_self:
            return f(instance_self, *args, **kwargs)
        return f(*args, **kwargs)

    # Make introspection see the original argument list (minus `self`).
    body_builder.__signature__ = sig  # type: ignore[attr-defined]
    try:
        anns = dict(getattr(f, "__annotations__", {}))
        if drop_self:
            anns.pop("self", None)
        body_builder.__annotations__ = anns
    except Exception:
        pass
    for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
        try:
            setattr(body_builder, attr, getattr(fn, attr))
        except Exception:
            pass
    # Preserve the original globals for later type materialization (string annotations).
    # We can't change `__globals__` of a Python function, so we stash it on the wrapper.
    try:
        body_builder.__flir_orig_globals__ = getattr(f, "__globals__", getattr(fn, "__globals__", {}))
    except Exception:
        pass
    return body_builder


def _unwrap_return_value(v):
    """Best-effort unwrap of return values for `flir.jit` builders.

    `mlir_func.FuncOp.from_py_func` requires returned operands to be `ir.Value`
    (or a sequence of `ir.Value`). In our Python front-end, many builders return
    lightweight wrappers (e.g. `arith.ArithValue`). This helper peels those
    wrappers so tests/examples don't need to sprinkle `arith.as_value(...)`.
    """
    if v is None:
        return None

    # Already an MLIR Value.
    if isinstance(v, ir.Value):
        return v

    # Common wrapper pattern: `._value` holds the underlying `ir.Value`.
    try:
        internal = object.__getattribute__(v, "_value")
        return _unwrap_return_value(internal)
    except Exception:
        pass

    # MLIR OpView results often expose `.value` as a property.
    try:
        if hasattr(v, "value") and callable(getattr(type(v).value, "fget", None)):
            return _unwrap_return_value(v.value)
    except Exception:
        pass

    # Fallback for wrappers that store `_value` as a normal attribute.
    try:
        if hasattr(v, "_value"):
            return _unwrap_return_value(v._value)
    except Exception:
        pass

    # If a plain Python int is returned, materialize it as an index constant.
    # (This matches existing behavior in `pyflir.dialects.ext.flir._unwrap_value`.)
    if isinstance(v, int):
        from _mlir.dialects import arith as mlir_arith
        from _mlir.ir import IndexType, IntegerAttr

        try:
            loc = get_user_code_loc()
        except Exception:
            loc = None
        if loc is None:
            loc = ir.Location.unknown()
        op = mlir_arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), v), loc=loc)
        return op.result

    return v


def _unwrap_returns(results):
    if results is None:
        return None
    if isinstance(results, (tuple, list)):
        return [_unwrap_return_value(v) for v in results]
    return _unwrap_return_value(results)


def _wrap_body_builder_unwrap_returns(body_builder):
    """Wrap a body builder to ensure all returned values are `ir.Value`."""

    def _wrapped(*args, **kwargs):
        return _unwrap_returns(body_builder(*args, **kwargs))

    # Preserve metadata for better debug locations / type materialization.
    try:
        _wrapped.__name__ = getattr(body_builder, "__name__", _wrapped.__name__)
        _wrapped.__qualname__ = getattr(body_builder, "__qualname__", _wrapped.__qualname__)
        _wrapped.__doc__ = getattr(body_builder, "__doc__", _wrapped.__doc__)
        _wrapped.__module__ = getattr(body_builder, "__module__", _wrapped.__module__)
    except Exception:
        pass
    try:
        _wrapped.__signature__ = inspect.signature(body_builder)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _wrapped.__annotations__ = dict(getattr(body_builder, "__annotations__", {}))
    except Exception:
        pass
    try:
        _wrapped.__flir_orig_globals__ = getattr(body_builder, "__flir_orig_globals__", getattr(body_builder, "__globals__", {}))
    except Exception:
        pass
    return _wrapped


class MlirModule:
    GPU_MODULE_NAME = "kernels"
    GPU_MODULE_TARGETS = None
    ALLOW_UNREGISTERED_DIALECTS = True

    cls_kernel_fn = []
    cls_jit_fn = []
    cls_kernel_sym = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Each MlirModule subclass owns a dedicated MLIR context.
        # We intentionally avoid process-global `Context.__enter__()` at import time:
        # some environments crash during interpreter shutdown if a context is left
        # on the thread-local stack.
        cls._context = ir.Context()
        cls._context.allow_unregistered_dialects = getattr(
            cls, "ALLOW_UNREGISTERED_DIALECTS", False
        )
        ensure_flir_python_extensions(cls._context)
        # Default location used when builders don't pass an explicit `loc=...`.
        # Prefer a file/line location pointing at user code (tests/examples) so IR dumps
        # are actionable. Fall back to `unknown` in dynamic/interactive contexts.
        with cls._context:
            try:
                cls._location = get_user_code_loc() or ir.Location.unknown(cls._context)
            except Exception:
                cls._location = ir.Location.unknown(cls._context)

        with cls._context, cls._location:
            # Initialize MLIR module for this subclass.
            cls.module = ir.Module.create()
            gpu.set_container_module(cls.module)

            # Create gpu.module container.
            with ir.InsertionPoint(cls.module.body):
                cls.gpu_module = gpu.GPUModuleOp(cls.GPU_MODULE_NAME)

        # Collect descriptor wrappers for this class (not inherited).
        temp_kernel_fn = []
        temp_jit_fn = []
        temp_kernel_sym = {}
        for name, value in cls.__dict__.items():
            if isinstance(value, _KernelDescriptor) and hasattr(value, "_wrapper"):
                temp_kernel_fn.append(value._wrapper)
                temp_kernel_sym[name] = name
            elif isinstance(value, _JitDescriptor) and hasattr(value, "_wrapper"):
                temp_jit_fn.append(value._wrapper)

        cls.cls_kernel_fn = temp_kernel_fn
        cls.cls_jit_fn = temp_jit_fn
        cls.cls_kernel_sym = temp_kernel_sym

    def __init__(self):
        self.kernel_func_op = {}
        with self._context, self._location:
            # Create a fresh module per instance so tests/examples can instantiate
            # the same MlirModule subclass multiple times without symbol redefinition.
            self.module = ir.Module.create()
            gpu.set_container_module(self.module)

            # Create gpu.module container.
            with ir.InsertionPoint(self.module.body):
                self.gpu_module = gpu.GPUModuleOp(
                    self.GPU_MODULE_NAME,
                    targets=getattr(self, "GPU_MODULE_TARGETS", None),
                )

            # Optional hook: allow subclasses (e.g. shared-memory allocators) to
            # insert ops into the gpu.module body before kernels are emitted.
            init_gpu_module = getattr(self, "init_gpu_module", None)
            if callable(init_gpu_module):
                with ir.InsertionPoint.at_block_begin(self.gpu_module.body):
                    init_gpu_module()

            # Emit host-side jit functions first.
            for fn in self.cls_jit_fn:
                fn(self)
            # Emit GPU kernels.
            for fn in self.cls_kernel_fn:
                fn(self)

    def __repr__(self):
        return str(self.module)

    def __getattr__(self, name: str):
        if name in self.cls_kernel_sym:
            with self._context, self._location:
                return ir.SymbolRefAttr.get([self.GPU_MODULE_NAME, self.cls_kernel_sym[name]])
        raise AttributeError(f"{name} not found in kernel functions.")

    @classmethod
    def kernel(cls, fn):
        """Alternative classmethod decorator (non-descriptor), for completeness."""

        def wrapper(self, *args, **kwargs):
            with self._context, self._location:
                with ir.InsertionPoint.at_block_begin(self.gpu_module.body):
                    # Emit as gpu.func in the gpu.module region.
                    body_builder = _bind_method_builder(fn, self, lower_range_loops=True)
                    k = gpu.func(emit=True, lower_range_loops=False)(body_builder)
                    # Ensure launch path qualifies as `@kernels::fn` when used as GPUFunc callable.
                    try:
                        k.qualname = self.GPU_MODULE_NAME
                    except Exception:
                        pass
                    self.kernel_func_op[fn.__name__] = k

        cls.cls_kernel_fn.append(wrapper)
        cls.cls_kernel_sym[fn.__name__] = fn.__name__
        return fn

    @classmethod
    def jit(cls, fn):
        """Alternative classmethod decorator (non-descriptor), for completeness."""

        def wrapper(self):
            with self._context, self._location:
                with ir.InsertionPoint.at_block_begin(self.module.body):
                    body_builder = _bind_method_builder(fn, self, lower_range_loops=False)
                    body_builder = _wrap_body_builder_unwrap_returns(body_builder)
                    sig = inspect.signature(body_builder)
                    input_types, _, _ = prep_func_types(sig, [])
                    # `prep_func_types` may return lambdas/strings. Materialize them
                    # into concrete MLIR types inside the active Context.
                    locals_ = {"T": gpu.T} if hasattr(gpu, "T") else {}
                    materialized_inputs = []
                    for t in input_types:
                        if isinstance(t, str):
                            materialized_inputs.append(ir.Type(eval(t, fn.__globals__, locals_)))
                        elif callable(t) and getattr(t, "__name__", None) == (lambda: 0).__name__:
                            materialized_inputs.append(t())
                        else:
                            materialized_inputs.append(t)
                    mlir_func.FuncOp.from_py_func(*materialized_inputs)(body_builder)

        cls.cls_jit_fn.append(wrapper)
        return fn


class _KernelDescriptor:
    """Descriptor that registers kernel to the owning MlirModule subclass."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        try:
            if issubclass(owner, MlirModule):
                fn = self.fn

                def wrapper(instance_self, *args, **kwargs):
                    with instance_self._context, instance_self._location:
                        with ir.InsertionPoint.at_block_begin(instance_self.gpu_module.body):
                            body_builder = _bind_method_builder(fn, instance_self, lower_range_loops=True)
                            k = gpu.func(emit=True, lower_range_loops=False)(body_builder)
                            try:
                                k.qualname = instance_self.GPU_MODULE_NAME
                            except Exception:
                                pass
                            instance_self.kernel_func_op[fn.__name__] = k

                self._wrapper = wrapper
                self._name = name
        except TypeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


class _JitDescriptor:
    """Descriptor that registers a host-side jit function to MlirModule."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        try:
            if issubclass(owner, MlirModule):
                fn = self.fn

                def wrapper(instance_self):
                    with instance_self._context, instance_self._location:
                        with ir.InsertionPoint.at_block_begin(instance_self.module.body):
                            body_builder = _bind_method_builder(fn, instance_self, lower_range_loops=False)
                            body_builder = _wrap_body_builder_unwrap_returns(body_builder)
                            sig = inspect.signature(body_builder)
                            input_types, _, _ = prep_func_types(sig, [])
                            # `prep_func_types` may return lambdas/strings. Materialize them
                            # into concrete MLIR types inside the active Context.
                            locals_ = {"T": gpu.T} if hasattr(gpu, "T") else {}
                            materialized_inputs = []
                            for t in input_types:
                                if isinstance(t, str):
                                    materialized_inputs.append(ir.Type(eval(t, fn.__globals__, locals_)))
                                elif callable(t) and getattr(t, "__name__", None) == (lambda: 0).__name__:
                                    materialized_inputs.append(t())
                                else:
                                    materialized_inputs.append(t)
                            wrapped = mlir_func.FuncOp.from_py_func(*materialized_inputs)(body_builder)
                            # Match ExecutionEngine packed-args calling convention.
                            try:
                                wrapped.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                            except Exception:
                                pass

                self._wrapper = wrapper
        except TypeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


def _looks_like_method(fn) -> bool:
    # Prefer a semantic check over `__qualname__` heuristics: nested free
    # functions (e.g. tests defining helpers inside test bodies) also contain
    # dots in `__qualname__` ("...<locals>...") but are NOT methods.
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        return bool(params) and params[0].name == "self"
    except Exception:
        return False


def kernel(fn=None, *, emit: bool = True):
    """Kernel decorator.

    - **Inside `MlirModule` subclasses**: use as `@flir.kernel` on methods.
    - **Elsewhere** (tests/examples): use as a drop-in replacement for
      `@gpu.func(emit=True, lower_range_loops=True)`.
    """
    if fn is None:
        return lambda real_fn: kernel(real_fn, emit=emit)

    # Class-method form (descriptor).
    if _looks_like_method(fn):
        return _KernelDescriptor(fn)

    # Free-function form (emit a gpu.func into the current insertion point).
    return gpu.func(emit=emit, lower_range_loops=True)(fn)


def _materialize_types(fn, types):
    # `prep_func_types` may return lambdas/strings. Materialize them into MLIR
    # types inside the active Context.
    locals_ = {"T": gpu.T} if hasattr(gpu, "T") else {}
    materialized = []
    for t in types:
        if isinstance(t, str):
            materialized.append(ir.Type(eval(t, fn.__globals__, locals_)))
        elif callable(t) and getattr(t, "__name__", None) == (lambda: 0).__name__:
            materialized.append(t())
        else:
            materialized.append(t)
    return materialized


def jit(*arg_types, emit: bool = True):
    """Jit decorator.

    - **Inside `MlirModule` subclasses**: use as `@flir.jit` on methods.
    - **Elsewhere** (tests/examples): drop-in replacement for
      `@func.FuncOp.from_py_func(...)` with either:
        - `@flir.jit` (types from annotations), or
        - `@flir.jit(type0, type1, ...)` (explicit types).
    """

    # Called as `@flir.jit` (no args): arg_types will be (fn,)
    if len(arg_types) == 1 and callable(arg_types[0]) and not isinstance(arg_types[0], (ir.Type, str)):
        fn = arg_types[0]

        if _looks_like_method(fn):
            return _JitDescriptor(fn)

        if not emit:
            raise ValueError("flir.jit(emit=False) is only supported on MlirModule methods")

        sig = inspect.signature(fn)
        input_types, _, _ = prep_func_types(sig, [])
        materialized_inputs = _materialize_types(fn, input_types)
        return mlir_func.FuncOp.from_py_func(*materialized_inputs)(_wrap_body_builder_unwrap_returns(fn))

    # Called as `@flir.jit(type0, type1, ...)`
    def _decorator(fn):
        if _looks_like_method(fn):
            # Keep MlirModule method behavior annotation-driven (no explicit type list).
            return _JitDescriptor(fn)
        if not emit:
            raise ValueError("flir.jit(emit=False) is only supported on MlirModule methods")
        materialized_inputs = _materialize_types(fn, list(arg_types))
        return mlir_func.FuncOp.from_py_func(*materialized_inputs)(_wrap_body_builder_unwrap_returns(fn))

    return _decorator




