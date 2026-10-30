# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""An inline struct type gets the same name in every process, and a different one for a different layout.

The name reaches the JIT disk-cache key through the closures that capture the type, so a name that changed
with the process's hash seed made a kernel miss the cache in every new process, and two layouts sharing a name
would share a compiled kernel.
"""

import inspect
import os
import subprocess
import sys

import pytest

import flydsl.expr as fx

pytestmark = [pytest.mark.l0_backend_agnostic]


def _print_name():
    import flydsl.expr as fx

    print(fx.Struct["count" : fx.Int32, "slots" : fx.Array[fx.Float32, 4]].__name__)


def _name_with_hash_seed(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    source = f"{inspect.getsource(_print_name)}\n_print_name()\n"
    out = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_inline_struct_name_does_not_depend_on_the_hash_seed():
    names = {_name_with_hash_seed(seed) for seed in (1, 2, 3)}

    assert len(names) == 1
    assert next(iter(names)).startswith("_DslStruct_")


def _make_item(field_type):
    @fx.struct
    class Item:
        value: field_type

    return Item


def test_inline_structs_over_different_layouts_get_different_names():
    # Array[Item, 4] reads the same for every class named Item, whatever its fields.
    narrow = fx.Struct["data" : fx.Array[_make_item(fx.Int32), 4]]
    wide = fx.Struct["data" : fx.Array[_make_item(fx.Int64), 4]]

    assert narrow.__name__ != wide.__name__
