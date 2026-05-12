# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
from .typing import *
from .primitive import *
from .gpu import *
from .derived import *
from .struct import *

from . import utils as utils
from . import arith as arith
from . import gpu as gpu
from . import math as math
from . import vector as vector

_OPTIONAL_ROCDL_MODULES = {
    "buffer_ops": ".buffer_ops",
    "rocdl": ".rocdl",
    "tdm_ops": ".rocdl.tdm_ops",
}


def _is_missing_optional_rocdl(exc: ImportError) -> bool:
    missing_name = getattr(exc, "name", None) or ""
    message = str(exc).lower()
    return missing_name.startswith("flydsl._mlir") and "rocdl" in f"{missing_name.lower()} {message}"


try:
    from . import buffer_ops as buffer_ops
    from . import rocdl as rocdl
    from .rocdl import tdm_ops as tdm_ops
except (ImportError, ModuleNotFoundError) as exc:
    if not _is_missing_optional_rocdl(exc):
        raise


def __getattr__(name: str):
    module_name = _OPTIONAL_ROCDL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    return module
