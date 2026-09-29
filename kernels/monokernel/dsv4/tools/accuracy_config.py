# SPDX-License-Identifier: Apache-2.0
"""Generate an explicit stable/RNE ATOM comparison profile from merged CSVs.

This never edits AITER's defaults. Only the bs1/seq1..4 router and shared
GEMM1 shapes are replaced; the compressor fix belongs to the layer adapter.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

CK_RNE = "a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1"


def generate(bf16_source, shared_source, output_dir, tp):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"profile": "stable-rne", "tp": tp, "files": {}}
    for kind, source in (("bf16", bf16_source), ("shared", shared_source)):
        with source.open() as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        n = 384 if kind == "bf16" else 6144 // tp
        keys = {"gfx": "gfx950", "cu_num": "256", "N": str(n), "K": "7168"}
        if kind == "bf16":
            keys.update(
                bias="False", dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB="False", bpreshuffle="False"
            )

        def matches(row):
            return all(row.get(k) == v for k, v in keys.items())

        templates = [r for r in rows if matches(r) and r["M"] in ("1", "16")]
        if not templates:
            raise ValueError(f"missing source shape {keys} in {source}")
        template = templates[0]
        if kind == "shared":
            candidates = [r for r in rows if r.get("libtype") == "ck" and r.get("kernelName") == CK_RNE]
            if not candidates:
                raise ValueError(f"verified CK RNE kernel is absent from {source}")
            template = dict(template, **{k: candidates[0][k] for k in ("libtype", "kernelName", "kernelId", "splitK")})
        else:
            template = dict(template, libtype="torch", solidx="0", splitK="1", kernelName="native")
        rows = [r for r in rows if not (matches(r) and r["M"] in ("1", "2", "3", "4"))]
        rows.extend(dict(template, M=str(m), us="0") for m in (1, 2, 3, 4))
        path = output_dir / f"{kind}.csv"
        with path.open("w") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest["files"][kind] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "overridden_shape": dict(keys, M=[1, 2, 3, 4]),
            "kernel": template["kernelName"],
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def configure(directory, tp):
    """Select and fingerprint this comparison profile before importing AITER."""
    import os

    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["profile"] != "stable-rne" or manifest["tp"] != tp:
        raise ValueError("stable/RNE profile TP does not match the requested TP")
    for kind, env in (("bf16", "AITER_CONFIG_GEMM_BF16"), ("shared", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE")):
        path = directory / f"{kind}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][kind]["sha256"]:
            raise ValueError(f"comparison profile checksum mismatch: {path}")
        os.environ[env] = str(path.resolve())
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16-source", type=Path, required=True)
    p.add_argument("--shared-source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tp", type=int, choices=(4, 8), required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.bf16_source, a.shared_source, a.output_dir, a.tp), indent=2))


if __name__ == "__main__":
    main()
