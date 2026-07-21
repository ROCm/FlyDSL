# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
from .typing import *
from .enum import *
from .primitive import *
from .gpu import *
from .derived import *
from .struct import *
from .arith import *
from .math import *

from . import utils as utils
from . import arith as arith
from . import gpu as gpu
from . import math as math
from . import vector as vector

_LAZY_MODULES = {
    "buffer_ops": ".buffer_ops",
    "rocdl": ".rocdl",
    "tdm_ops": ".rocdl.tdm_ops",
}


def __getattr__(name: str):
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    return module
