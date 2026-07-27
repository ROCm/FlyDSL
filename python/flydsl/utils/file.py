# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Optional


@contextmanager
def atomic_write(
    path: Path,
    *,
    mode: str,
    encoding: Optional[str] = None,
) -> Iterator[IO]:
    """Write a file through a same-directory temporary and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            tmp_path = Path(output.name)
            yield output
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
