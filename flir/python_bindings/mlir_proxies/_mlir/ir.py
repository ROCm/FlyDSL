# Proxy: re-export _flir_ir._mlir.ir so LLVM wrappers' `from ._mlir_libs._mlir.ir import *` works.
from .._flir_ir._mlir.ir import *
from .._flir_ir._mlir.ir import _GlobalDebug
