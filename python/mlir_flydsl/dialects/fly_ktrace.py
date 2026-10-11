# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# isort: skip_file
# ruff: noqa: F401,F403
# No enum bindings and no nanobind extension: the dialect has neither. The token
# type is reached by parsing it, the way the frontend builds it.
from ._fly_ktrace_ops_gen import *
from ._fly_ktrace_ops_gen import _Dialect
