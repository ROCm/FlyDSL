import copy
import inspect
import sys
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
from functools import update_wrapper
from typing import Optional, List, Union, TypeVar


import _mlir.ir as ir
from _mlir.extras.meta import op_region_builder
from _mlir.extras import types as T

from _mlir.dialects._ods_common import get_op_result_or_op_results
from _mlir.dialects.func import *
from _mlir.ir import (
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    OpView,
    Operation,
    OpResultList,
    Type,
    TypeAttr,
    Value,
)


# from ...ast.util import copy_func, get_user_code_loc, make_maybe_no_args_decorator
# from ...ast.py_type import PyTypeVarObject


_call = call


def call(
    callee_or_results: Union[FuncOp, List[Type]],
    arguments_or_callee: Union[List[Value], FlatSymbolRefAttr, str],
    arguments: Optional[list] = None,
    *,
    call_op_ctor=CallOp.__base__,
    loc=None,
    ip=None,
):
    if loc is None:
        loc = get_user_code_loc()
    if isinstance(callee_or_results, FuncOp.__base__):
        if not isinstance(arguments_or_callee, (list, tuple)):
            raise ValueError(
                "when constructing a call to a function, expected "
                + "the second argument to be a list of call arguments, "
                + f"got {type(arguments_or_callee)}"
            )
        if arguments is not None:
            raise ValueError("unexpected third argument when constructing a call" + "to a function")
        if not all(isinstance(a, (Value, Operation, OpView)) for a in arguments_or_callee):
            raise ValueError(f"{arguments_or_callee} must all be Value, Operation, or OpView")

        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results.function_type.value.results,
                FlatSymbolRefAttr.get(callee_or_results.sym_name.value),
                arguments_or_callee,
                loc=loc,
                ip=ip,
            )
        )

    if isinstance(arguments_or_callee, list):
        raise ValueError(
            "when constructing a call to a function by name, "
            + "expected the second argument to be a string or a "
            + f"FlatSymbolRefAttr, got {type(arguments_or_callee)}"
        )

    if isinstance(arguments_or_callee, FlatSymbolRefAttr):
        return get_op_result_or_op_results(
            call_op_ctor(callee_or_results, arguments_or_callee, arguments, loc=loc, ip=ip)
        )
    elif isinstance(arguments_or_callee, str):
        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results,
                FlatSymbolRefAttr.get(arguments_or_callee),
                arguments,
                loc=loc,
                ip=ip,
            )
        )
    else:
        raise ValueError(f"unexpected type {callee_or_results=}")


def isalambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def prep_func_types(sig, return_types):
    assert not (
        not sig.return_annotation is inspect.Signature.empty and len(return_types) > 0
    ), f"func can use return annotation or explicit return_types but not both"
    return_types = sig.return_annotation if not sig.return_annotation is inspect.Signature.empty else return_types
    if not isinstance(return_types, (tuple, list)):
        return_types = [return_types]
    return_types = list(return_types)
    assert all(
        isinstance(r, (str, Type, TypeVar)) or isalambda(r) for r in return_types
    ), f"all return types must be mlir types or strings or TypeVars or lambdas {return_types=}"

    input_types = [p.annotation for p in sig.parameters.values() if not p.annotation is inspect.Signature.empty]
    assert all(
        isinstance(r, (str, Type, TypeVar)) or isalambda(r) for r in input_types
    ), f"all input types must be mlir types or strings or TypeVars or lambdas {input_types=}"
    user_loc = get_user_code_loc()
    # If ir.Context is none (like for deferred func emit)
    if user_loc is None:
        user_locs = None
    else:
        user_locs = [user_loc] * len(sig.parameters)
    return input_types, return_types, user_locs


def is_relative_to(self, other):
    return other == self or other in self.parents


def get_user_code_loc(user_base: Optional[Path] = None):
    from _mlir import extras
    import sysconfig
    import importlib
    import linecache

    if Context.current is None:
        return

    mlir_extras_root_path = Path(extras.__path__[0])
    # Standard library root (e.g. /usr/lib/python3.10). These frames are never "user code".
    try:
        stdlib_root = Path(sysconfig.get_paths().get("stdlib", ""))
    except Exception:
        stdlib_root = Path("")

    def _pkg_root(pkg_name: str) -> Optional[Path]:
        try:
            m = importlib.import_module(pkg_name)
            p = getattr(m, "__file__", None)
            if not p:
                return None
            return Path(p).resolve().parent
        except Exception:
            return None

    skip_roots = [mlir_extras_root_path, Path(sys.prefix)]
    if str(stdlib_root):
        skip_roots.append(stdlib_root)

    prev_frame = inspect.currentframe().f_back
    # Back-compat: user_base historically was a *file* path used to skip frames in
    # that exact file. Keep accepting it.
    user_base_file = Path(user_base).resolve() if user_base is not None else None

    def _should_skip(filename: str) -> bool:
        try:
            p = Path(filename).resolve()
        except Exception:
            return True
        if user_base_file is not None and p == user_base_file:
            return True
        return any(is_relative_to(p, r) for r in skip_roots)

    # Collect candidate "user" frames and prefer a non-<module> frame when available.
    candidates = []
    while prev_frame:
        if not _should_skip(prev_frame.f_code.co_filename):
            candidates.append(prev_frame)
        prev_frame = prev_frame.f_back

    if not candidates:
        return ir.Location.unknown()

    best = None

    # Prefer a frame that appears to be a direct DSL call site in user code.
    # This yields much more intuitive #loc line numbers than picking the first
    # non-stdlib frame, especially when decorators/tracing introduce helper frames.
    for f in candidates:
        try:
            if f.f_code.co_name == "<module>":
                continue
            filename = f.f_code.co_filename
            lineno = int(getattr(f, "f_lineno", 0) or 0)
            if not filename or lineno <= 0:
                continue
            src = (linecache.getline(filename, lineno) or "").strip()
            if not src:
                continue
            if any(tok in src for tok in ("rocir.", "arith.", "scf.", "gpu.", "memref.", "vector.")):
                best = f
                break
        except Exception:
            continue

    # Fallback: first non-<module> user frame if we didn't find a DSL-like line.
    if best is None:
        for f in candidates:
            try:
                if f.f_code.co_name != "<module>":
                    best = f
                    break
            except Exception:
                continue
    if best is None:
        best = candidates[0]

    frame_info = inspect.getframeinfo(best)
    if sys.version_info.minor >= 11:
        col = 0
        line = frame_info.lineno
        try:
            pos = getattr(frame_info, "positions", None)
            if pos is not None:
                line = int(getattr(pos, "lineno", line) or line)
                col = int(getattr(pos, "col_offset", 0) or 0)
        except Exception:
            pass
        return ir.Location.file(frame_info.filename, line, col)
    else:
        # On Python < 3.11 we don't have reliable column offsets. Use the frame's
        # current line number (f_lineno) to better match multi-line call sites.
        return ir.Location.file(frame_info.filename, getattr(best, "f_lineno", frame_info.lineno), col=0)


def make_maybe_no_args_decorator(decorator):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(decorator)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_dec


@dataclass
class ReifiedTypeParams:
    name: str
    val: object


class FuncBase:
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
        generics: List[Union[TypeVar, ReifiedTypeParams]] = None,
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
        self.input_types, self.return_types, self.arg_locs = prep_func_types(sig, return_types)

        if self._is_decl():
            assert len(self.input_types) == len(sig.parameters), f"func decl needs all input types annotated"
            self.sym_visibility = "private"
            self.emit()

    def _is_decl(self):
        # magic constant found from looking at the code for an empty fn
        if sys.version_info.minor == 14:
            return self.body_builder.__code__.co_code == b"\x80\x00R\x00#\x00"
        elif sys.version_info.minor == 13:
            return self.body_builder.__code__.co_code == b"\x95\x00g\x00"
        elif sys.version_info.minor == 12:
            return self.body_builder.__code__.co_code == b"\x97\x00y\x00"
        elif sys.version_info.minor == 11:
            return self.body_builder.__code__.co_code == b"\x97\x00d\x00S\x00"
        elif sys.version_info.minor in {8, 9, 10}:
            return self.body_builder.__code__.co_code == b"d\x00S\x00"
        else:
            raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def emit(self, *call_args, decl=False, force=False) -> FuncOp:
        if self._func_op is None or decl or force:
            if self.function_type is None:
                if len(call_args) == 0:
                    input_types = self.input_types[:]
                    locals = {"T": T}
                    if self.generics is not None:
                        for t in self.generics:
                            if not isinstance(t, ReifiedTypeParams):
                                raise RuntimeError(f"{t=} must reified")
                            locals[t.name] = t.val
                    for i, v in enumerate(input_types):
                        if isinstance(v, TypeVar):
                            v = v.__name__
                        if isinstance(v, str):
                            g = getattr(self.body_builder, "__flir_orig_globals__", self.body_builder.__globals__)
                            input_types[i] = Type(eval(v, g, locals))
                        elif isalambda(v):
                            input_types[i] = v()
                else:
                    input_types = [a.type for a in call_args]

                function_type = TypeAttr.get(
                    FunctionType.get(
                        inputs=input_types,
                        results=self.return_types,
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
            for k, v in self.func_attrs.items():
                self._func_op.attributes[k] = v
            if self._is_decl() or decl:
                return self._func_op

            self._func_op.regions[0].blocks.append(*input_types, arg_locs=self.arg_locs)
            builder_wrapper = op_region_builder(self._func_op, self._func_op.regions[0], terminator=self.return_op_ctor)

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
                function_type = FunctionType.get(inputs=input_types, results=return_types)
                self._func_op.attributes["function_type"] = TypeAttr.get(function_type)
            else:
                builder_wrapper(self.body_builder)

        return self._func_op

    def __call__(self, *call_args):
        return call(self.emit(*call_args), call_args)

    def __getitem__(self, item):
        if self.generics is None:
            raise RuntimeError("using a generic call requires the func be generic (i.e., have type_params)")
        # this also copies the function so that the original body_builder remains "generic" (via its closure)
        body_builder = copy_func(self.body_builder)
        reified_type_params = []
        # dumb but whatever
        already_reified_type_params = {}
        generics = copy.deepcopy(self.generics)
        for i, t in enumerate(generics):
            if sys.version_info >= (3, 12):
                type_var_bound = PyTypeVarObject.from_object(t).bound
            else:
                type_var_bound = t.__bound__
            if type_var_bound:
                # before 3.12 typevar was just a python class
                # https://github.com/python/cpython/blob/3.11/Lib/typing.py#L966
                if sys.version_info < (3, 12):
                    type_var_bound = lambda: type_var_bound
                else:
                    type_var_bound = type_var_bound.contents.into_object()
                    cvrs = inspect.getclosurevars(type_var_bound).nonlocals
                    if len(cvrs):
                        for k, v in cvrs.items():
                            if not isinstance(v, TypeVar):
                                continue
                            if k not in already_reified_type_params:
                                raise RuntimeError(f"typevar {k} not reified prior to evaluating dependent typevar {t}")
                            cvrs[k] = already_reified_type_params[k]
                        type_var_bound = copy_func(type_var_bound, cvrs)
                r = ReifiedTypeParams(t.__name__, type_var_bound())
            else:
                r = ReifiedTypeParams(t.__name__, item[i])

            reified_type_params.append(r)
            already_reified_type_params[r.name] = r.val

            if t.__name__ in body_builder.__globals__:
                body_builder.__globals__[t.__name__] = r.val
            if r.name in body_builder.__code__.co_freevars:
                free_i = body_builder.__code__.co_freevars.index(r.name)
                assert body_builder.__closure__[free_i].cell_contents == t, "typevars don't match"
                body_builder.__closure__[free_i].cell_contents = r.val

        return FuncBase(
            body_builder,
            self.func_op_ctor,
            self.return_op_ctor,
            self.call_op_ctor,
            return_types=self.return_types,
            sym_visibility=self.sym_visibility,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            func_attrs=self.func_attrs,
            generics=reified_type_params,
            qualname=self.qualname,
            loc=self.loc,
            ip=self.ip,
        )


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    sym_name=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    function_type=None,
    emit=False,
    generics=None,
    loc=None,
    ip=None,
) -> FuncBase:
    if loc is None:
        loc = get_user_code_loc()
    if generics is None and hasattr(f, "__type_params__") and f.__type_params__:
        generics = f.__type_params__
    func_ = FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp.__base__,
        sym_visibility=sym_visibility,
        sym_name=sym_name,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        function_type=function_type,
        generics=generics,
        loc=loc,
        ip=ip,
    )
    func_ = update_wrapper(func_, f)
    if emit:
        func_.emit()
    return func_
