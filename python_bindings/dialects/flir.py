"""Flir dialect Python bindings.

This file is intentionally small: the actual operation/enum definitions are
auto-generated into `_flir_ops_gen.py` and `_flir_enum_gen.py` under the embedded
`_mlir.dialects` package.

Some build layouts symlink `_mlir.dialects.flir` to this file; if it's missing,
imports like `from _mlir.dialects import flir` will fail and break GPU tests.
"""

from ._flir_ops_gen import *  # noqa: F401,F403
from ._flir_enum_gen import *  # noqa: F401,F403


