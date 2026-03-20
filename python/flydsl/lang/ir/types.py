# Copyright (c) 2025 FlyDSL Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward-compat shim — canonical location is now flydsl.expr.typing."""

from flydsl.expr.typing import Types, T, default_f8_type  # noqa: F401

memref = None
try:
    from flydsl._mlir.extras.types import memref  # noqa: F401
except ImportError:
    pass

__all__ = ["Types", "T", "default_f8_type", "memref"]
