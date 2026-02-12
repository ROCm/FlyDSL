# Proxy: re-export _flir_ir._mlir so LLVM wrappers' `from ._mlir_libs._mlir import X` works.
# _ods_common.py does `_cext.ir.Location` etc., so submodules must be accessible as attrs.
from .._flir_ir._mlir import *
from .._flir_ir._mlir import register_type_caster, register_value_caster, globals
from . import ir, rewrite, passmanager
