# SPDX-License-Identifier: Apache-2.0
"""Check a shared DSV4-Pro checkpoint without loading weights or importing Torch."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def inspect_checkpoint(root):
    root = Path(root).resolve()
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    required = {
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "hidden_size": 7168,
        "moe_intermediate_size": 3072,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "head_dim": 512,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "o_groups": 16,
        "hc_mult": 4,
        "scoring_func": "sqrtsoftplus",
    }
    mismatch = {k: (expected, config.get(k)) for k, expected in required.items() if config.get(k) != expected}
    if mismatch:
        raise ValueError(f"checkpoint is not the supported DSV4-Pro configuration: {mismatch}")
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    shards, tensor_count, total_bytes, tensor_bytes = [], 0, 0, 0
    seen = set()
    for name in sorted(set(weight_map.values())):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"shard escapes checkpoint directory: {name}")
        size = path.stat().st_size
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > min(size - 8, 100_000_000):
                raise ValueError(f"invalid safetensors header: {name}")
            header_bytes = f.read(length)
        header = json.loads(header_bytes)
        end = 0
        for tensor, info in header.items():
            if tensor == "__metadata__":
                continue
            if weight_map.get(tensor) != name or tensor in seen:
                raise ValueError(f"index/header mismatch: {tensor} in {name}")
            first, last = info["data_offsets"]
            if not 0 <= first <= last <= size - 8 - length:
                raise ValueError(f"invalid tensor extent: {tensor}")
            end = max(end, last)
            tensor_bytes += last - first
            tensor_count += 1
            seen.add(tensor)
        if end + length + 8 != size:
            raise ValueError(f"truncated shard or unexpected trailing bytes: {name}")
        total_bytes += size
        shards.append({"name": name, "bytes": size, "header_sha256": hashlib.sha256(header_bytes).hexdigest()})
    if seen != set(weight_map):
        raise ValueError(f"missing tensors: {sorted(set(weight_map) - seen)[:10]}")
    if tensor_bytes != index["metadata"]["total_size"]:
        raise ValueError("tensor byte count does not match index metadata")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing {name}")
    return {
        "model": "DeepSeek-V4-Pro",
        "path": str(root),
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "file_bytes": total_bytes,
        "tensor_bytes": tensor_bytes,
        "shards": shards,
        "validation": "configuration, index, all shard headers and file lengths; no full weight checksum",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    report = inspect_checkpoint(args.checkpoint)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("config", "shards")}, indent=2))


if __name__ == "__main__":
    main()
