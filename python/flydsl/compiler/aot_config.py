# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Linker configuration for objects exported by ``export_to_c``.

Usage::

    python -m flydsl.compiler.aot_config --libdir
    python -m flydsl.compiler.aot_config --ldflags
    python -m flydsl.compiler.aot_config --libs

Output is shell-quoted so it can be spliced into a build command, e.g.
``cc -shared k.o $(python -m flydsl.compiler.aot_config --ldflags --libs)``.
"""

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Optional

from ..runtime.libraries import find_runtime_libraries


def _libdirs(backend: Optional[str]) -> List[str]:
    dirs: List[str] = []
    for lib in find_runtime_libraries(backend):
        d = str(Path(lib.path).parent)
        if d not in dirs:
            dirs.append(d)
    return dirs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flydsl.compiler.aot_config", description=__doc__.split("\n")[0])
    parser.add_argument("--libdir", action="store_true", help="directories containing the runtime libraries")
    parser.add_argument("--ldflags", action="store_true", help="-L and -rpath flags for the runtime libraries")
    parser.add_argument("--libs", action="store_true", help="-l flags for the runtime libraries")
    parser.add_argument("--backend", default=None, help="compile backend (default: FLYDSL_COMPILE_BACKEND)")
    args = parser.parse_args(argv)
    if not (args.libdir or args.ldflags or args.libs):
        parser.error("specify at least one of --libdir, --ldflags, --libs")

    out: List[str] = []
    if args.libdir:
        out += _libdirs(args.backend)
    if args.ldflags:
        for d in _libdirs(args.backend):
            out += [f"-L{d}", f"-Wl,-rpath,{d}"]
    if args.libs:
        out += [f"-l:{lib.soname}" for lib in find_runtime_libraries(args.backend)]
    print(" ".join(shlex.quote(item) for item in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
