"""Collective autotuning for MegaMoE v2 stage1."""

import fcntl
import json
import os

import torch
import torch.distributed as dist

from flydsl.autotune import Autotuner, Config, do_bench

_SHAPES = (
    (32, 128, 2),
    (32, 128, 4),
    (32, 256, 2),
    (32, 256, 4),
    (32, 256, 8),
    (32, 512, 4),
    (32, 512, 8),
    (64, 128, 2),
    (64, 128, 4),
    (64, 256, 4),
    (64, 256, 8),
    (64, 512, 8),
    (128, 256, 4),
    (128, 256, 8),
    (128, 512, 8),
)
_ANCHORS = {
    (32, 128, 4),
    (32, 256, 4),
    (64, 256, 4),
    (128, 256, 4),
    (128, 512, 8),
}
_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16)
_DISPATCH_CU_VALUES = (8, 16, 24, 32, 48, 64, 96, 128)
_CALIBRATED_VARIANTS = {
    (32, 256, 4): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 16,
            "pipe_weights": False,
            "mfma_amajor": False,
            "active_expert_producer": True,
            "use_tile_resource": False,
        },
    ),
    (32, 512, 8): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": False,
            "mfma_amajor": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": False,
            "mfma_amajor": False,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": True,
            "mfma_amajor": True,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 128,
            "pipe_weights": True,
            "mfma_amajor": True,
            "async_a_copy": True,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": True,
            "mfma_amajor": True,
            "cooperative_payload_copy": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": True,
            "mfma_amajor": True,
            "async_a_copy": True,
            "cooperative_payload_copy": False,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 192,
            "pipe_weights": False,
            "mfma_amajor": False,
            "cooperative_payload_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 16,
            "pipe_weights": False,
            "mfma_amajor": False,
            "active_expert_producer": True,
            "use_tile_resource": False,
        },
    ),
    (64, 512, 8): (
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 192,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 128,
            "use_tile_resource": False,
        },
    ),
    (128, 512, 8): (
        {
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": False,
        },
        {
            "grid_mult": 1,
            "num_dispatch_cu": 32,
            "async_a_copy": True,
            "use_tile_resource": True,
        },
        {
            "grid_mult": 3,
            "num_dispatch_cu": 32,
            "use_tile_resource": True,
        },
    ),
}


def _candidate_variants(shape):
    variants = [{}, *_CALIBRATED_VARIANTS.get(shape, ())]
    if shape[0] == 32:
        variants.append({"mfma_amajor": True})
    if shape == (128, 512, 8):
        variants.append({"async_a_copy": True})
    if shape not in _ANCHORS:
        return variants
    variants += [{"grid_mult": value} for value in _GRID_MULT_VALUES if value != 4]
    variants += [{"num_dispatch_cu": value} for value in _DISPATCH_CU_VALUES if value != 64]
    variants += [
        {"pipe_weights": False, "mfma_amajor": False},
        {"mfma_amajor": False},
        {"swizzle_a": False},
        {"use_tile_resource": False},
        {"waves_per_eu_hint": 1},
        {"b_nt": 0},  # cached B-load (L2 reuse); best at large bs
        {"b_nt": 3},  # streamed B-load (bypass); best at small/decode bs
    ]
    return variants


def get_stage1_autotune_configs(dispatch_cu=None, grid_mult=None, tile_m_values=(32,)):
    tile_m_values = {int(value) for value in tile_m_values}
    configs = []
    seen = set()
    for sort_block_m, tile_n, num_waves in _SHAPES:
        if sort_block_m not in tile_m_values:
            continue
        base = dict(
            sort_block_m=sort_block_m,
            tile_n=tile_n,
            tile_k=256,
            num_waves=num_waves,
            grid_mult=4,
            pipe_weights=True,
            mfma_amajor=sort_block_m >= 64,
            swizzle_a=True,
            async_a_copy=False,
            active_expert_producer=False,
            cooperative_payload_copy=False,
            num_dispatch_cu=64,
            use_tile_resource=True,
            waves_per_eu_hint=2,
            b_nt=-1,  # -1 = per-bucket default policy (stream<=512, cached>=1024)
        )
        for update in _candidate_variants((sort_block_m, tile_n, num_waves)):
            values = dict(base, **update)
            if dispatch_cu is not None:
                values["num_dispatch_cu"] = int(dispatch_cu)
            if grid_mult is not None:
                values["grid_mult"] = int(grid_mult)
            signature = tuple(sorted(values.items()))
            if signature not in seen:
                configs.append(Config(**values))
                seen.add(signature)
    return configs


def prune_stage1_autotune_configs(configs, sig_args):
    """Prune invalid and batch-irrelevant configs before collective compilation."""
    tokens = int(sig_args["tune_tokens"])
    model_dim = int(sig_args["model_dim"])
    inter_dim = int(sig_args["inter_dim"])
    num_cu = int(sig_args["num_cu"])
    fuse_npes = int(sig_args["fuse_npes"])
    fuse_topk = int(sig_args["fuse_topk"])
    fuse_cap = int(sig_args["fuse_cap"])
    fuse_mtpr = int(sig_args["fuse_mtpr"])
    experts_per_rank = int(sig_args["experts_per_rank"])
    out = []
    for config in configs:
        values = config.kwargs
        block_m = int(values["sort_block_m"])
        tile_n = int(values["tile_n"])
        tile_k = int(values["tile_k"])
        num_waves = int(values["num_waves"])
        grid_mult = int(values["grid_mult"])
        dispatch_cu = int(values["num_dispatch_cu"])
        b_nt = int(values["b_nt"])
        use_tile_resource = bool(values["use_tile_resource"])
        direct_fixed_slot = (
            fuse_npes == 8
            and experts_per_rank == 48
            and fuse_mtpr <= 256
            and fuse_cap == ((fuse_npes * fuse_mtpr + block_m - 1) // block_m) * block_m
        )
        if direct_fixed_slot and (values["active_expert_producer"] or values["cooperative_payload_copy"]):
            continue
        num_acc_n = tile_n // num_waves // 16
        m_repeat = block_m // 16
        lds_pool = max(2 * block_m * tile_k, 2 * block_m * tile_n)
        lds_scale = block_m * (model_dim // 32)
        max_rows = fuse_npes * fuse_mtpr * fuse_topk + experts_per_rank * block_m
        payload_bytes = max_rows * model_dim
        output_bytes = max_rows * inter_dim
        if (
            model_dim % tile_k
            or (2 * inter_dim) % tile_n
            or tile_n % (num_waves * 32)
            or block_m % 32
            or m_repeat * num_acc_n * 4 > 256
            or lds_pool + lds_scale > 160 * 1024
            or not 0 < dispatch_cu <= num_cu
            or (payload_bytes >= 1 << 32 and not use_tile_resource)
            or (output_bytes >= 1 << 32 and not use_tile_resource)
        ):
            continue
        if tokens <= 64:
            # The production default uses M32 here, but an explicit joint SBM sweep must also be
            # able to evaluate M64/M128 instead of being pruned to an empty candidate set.
            keep = tile_n <= 512 and grid_mult <= 4 and dispatch_cu >= 32 and not (block_m > 32 and b_nt == 0)
        elif tokens <= 1024:
            keep = tile_n <= 512 and grid_mult <= 8 and dispatch_cu >= 24
        else:
            # Large compact batches normally use M64/M128. A forced M32 joint sweep is still
            # useful at bucket boundaries, but only its known-progress wide-N/high-wave family is
            # safe; narrower M32 candidates have stalled this workload on gfx950.
            keep = (
                tile_n == 512
                and num_waves == 8
                and (
                    (block_m == 32 and grid_mult == 1 and dispatch_cu >= 32)
                    or (block_m >= 64 and grid_mult <= 3 and dispatch_cu <= 96)
                )
            )
        if keep:
            out.append(config)
    if not out:
        raise ValueError(f"no valid stage1 configs for tokens={tokens}")
    return out


def collective_bench(fn, warmup, rep, quantiles=None):
    elapsed = do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    if not dist.is_initialized():
        return elapsed
    value = torch.tensor(float(elapsed), dtype=torch.float32, device=torch.cuda.current_device())
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


class CollectiveAutotuner(Autotuner):
    @staticmethod
    def _config_signature(config):
        return tuple(sorted(config.to_dict().items()))

    def _is_allowed_config(self, config):
        allowed = getattr(self, "_allowed_config_signatures", None)
        if allowed is None:
            allowed = {self._config_signature(candidate) for candidate in self.configs}
            self._allowed_config_signatures = allowed
        return self._config_signature(config) in allowed

    def _load_disk_cache(self):
        super()._load_disk_cache()
        self.cache = {key: config for key, config in self.cache.items() if self._is_allowed_config(config)}

    def __call__(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        if not dist.is_initialized():
            result = super().__call__(*args, **kwargs)
            self.last_config = self.cache[key]
            return result
        ready = getattr(self, "_collective_ready", set())
        if key in ready:
            if key in self.cache and self._is_allowed_config(self.cache[key]):
                self.last_config = self.cache[key]
                return self._run_config(self.cache[key], args, kwargs)
            ready.discard(key)
            self.cache.pop(key, None)
        payload = [self.cache[key].to_dict() if dist.get_rank() == 0 and key in self.cache else None]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is not None:
            config = Config.from_dict(payload[0])
            if self._is_allowed_config(config):
                self.cache[key] = config
                ready.add(key)
                self._collective_ready = ready
                self.last_config = config
                return self._run_config(config, args, kwargs)
        self.cache.pop(key, None)
        result = super().__call__(*args, **kwargs)
        ready.add(key)
        self._collective_ready = ready
        self.last_config = self.cache[key]
        return result

    def _save_disk_cache(self):
        distributed = dist.is_initialized()
        error = None
        if not distributed or dist.get_rank() == 0:
            try:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                lock_path = self._cache_file.with_suffix(".lock")
                with lock_path.open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    data = {}
                    if self._cache_file.exists():
                        try:
                            data = json.loads(self._cache_file.read_text())
                        except (OSError, ValueError):
                            data = {}
                    for key, config in self.cache.items():
                        if self._is_allowed_config(config):
                            data[json.dumps(list(key))] = config.to_dict()
                    tmp = self._cache_file.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(data, indent=2))
                    os.replace(tmp, self._cache_file)
            except Exception as exc:
                error = exc
        if distributed:
            dist.barrier()
        if error is not None:
            raise error
