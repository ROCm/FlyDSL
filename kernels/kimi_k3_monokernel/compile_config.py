"""Host-only, immutable specialization choices for the complete K3 kernel.

The resolved object is part of the cached builder's arguments. No configuration
field is read by a running GPU kernel. Resource/occupancy checks are still
required for every newly compiled specialization before executing it.
"""
from dataclasses import dataclass

from kernels.kimi_k3_monokernel.shapes import resolve_sequence_shape


def _input_k_partition(samples: int) -> tuple[int, int]:
    # Native FP32 GEMM reduction order for K=7168, N=6284. The larger
    # token counts select different native kernels; retain their partition
    # boundaries so BF16 rounding does not amplify through the KDA chain.
    if samples > 48:
        return (5, 64)
    if samples > 32:
        return (10, 32)
    if samples == 25:
        return (7, 64)
    return ((10, 32) if samples < 8 else (10, 64) if samples <= 16 else
            (7, 64) if samples <= 24 else (6, 64) if samples == 28 else (1, 256))


@dataclass(frozen=True)
class KimiK3CompileConfig:
    path: str = 'auto'
    staged_samples: int | None = None
    input_row_groups: int | None = None
    grid_blocks: int | None = None
    arithmetic: str = 'auto'
    latent_arithmetic: str = 'auto'
    input_schedule: str = 'auto'
    gate_schedule: str = 'auto'
    route_publication: str = 'auto'
    output_prefetch_units: int | None = None
    down_task_mapping: str = 'auto'
    input_mfma: str = 'auto'
    gate_input_arithmetic: str = 'auto'
    ug_arithmetic: str = 'auto'
    input_fp32_repair: bool | None = None
    router_guard_ulp: int | None = None
    router_guard_lanes: int | None = None
    pre_attn_res: str = 'auto'
    weight_pool: str = 'auto'
    mtp_prepare: str = 'auto'

    def resolve(self, samples: int, mtp: bool, seq_len: int | None = None):
        batch, seq = resolve_sequence_shape(samples, mtp, seq_len)
        if self.path not in {'auto', 'small_batch', 'general'}:
            raise ValueError('path must be auto, small_batch, or general')
        small = batch in {1, 2, 4} and seq <= 4
        path = ('small_batch' if small else 'general') if self.path == 'auto' else self.path
        if path == 'small_batch' and not small:
            raise ValueError('small_batch specializes batch 1, 2, and 4 with seq <= 4')
        staged = self.staged_samples
        if staged is None:
            staged = 2 if samples == 24 or (batch, seq) == (7, 4) else max(n for n in range(1, min(samples, 4) + 1) if samples % n == 0)
            if seq > 4 and staged == 3:
                # Three staged rows spill with the long-K, >28-token path.
                # Keep the established short-sequence configurations intact.
                staged = 2 if samples % 2 == 0 else 1
            if seq > 4 and samples == 28:
                # The four-row S7 recurrence allocates private memory; use
                # the same two-row input staging as the validated B7/S4 path.
                staged = 2
            if seq > 4 and samples in {36, 40, 48, 56}:
                # Two rows keep the native-partition input projection free
                # of private allocations at these larger token counts.
                staged = 2
        if type(staged) is not int or staged not in {1, 2, 3, 4} or samples % staged:
            raise ValueError('staged_samples must divide samples and be in [1, 4]')
        if self.arithmetic not in {'auto', 'legacy', 'native_fp32'}:
            raise ValueError('arithmetic must be auto, legacy, or native_fp32')
        arithmetic = ('legacy' if samples <= 4 and not (batch == 1 and seq == 4) else 'native_fp32') if self.arithmetic == 'auto' else self.arithmetic
        if self.latent_arithmetic not in {'auto', 'scaled_mxfp8', 'decoded_bf16'}:
            raise ValueError('latent_arithmetic must be auto, scaled_mxfp8, or decoded_bf16')
        latent_arithmetic = ('decoded_bf16' if samples == 1 or (batch == 2 and seq == 1) else 'scaled_mxfp8') if self.latent_arithmetic == 'auto' else self.latent_arithmetic
        row_groups = self.input_row_groups
        if row_groups is None:
            row_groups = (8 if samples == 32 else 2) if arithmetic == 'native_fp32' else (2 if samples <= 4 else 4)
            if batch == 1 and seq == 4 and arithmetic == 'native_fp32':
                row_groups = 1
            if seq > 4 and samples == 30 and arithmetic == 'native_fp32':
                # Distribute the long-K input projection across eight groups
                # to keep the 30-token polling specialization scratch-free.
                row_groups = 8
            if seq > 4 and samples == 25 and arithmetic == 'native_fp32':
                row_groups = 1
        if type(row_groups) is not int or row_groups not in {1, 2, 4, 8}:
            raise ValueError('input_row_groups must be one of 1, 2, 4, 8')
        parts, _ = _input_k_partition(samples)
        if arithmetic == 'native_fp32' and row_groups * parts * staged * 16 > 2048:
            raise ValueError('native partials exceed the fixed reduction LDS capacity')
        grid = self.grid_blocks
        if grid is None:
            grid = 512 if path == 'small_batch' and batch == 1 and seq == 4 else 256
        if type(grid) is not int or grid not in {256, 448, 512}:
            raise ValueError('grid_blocks must be one of 256, 448, 512')
        if self.input_schedule not in {'auto', 'cta', 'flat'}:
            raise ValueError('input_schedule must be auto, cta, or flat')
        flat_eligible = (batch == 1 and seq == 4 and arithmetic == 'native_fp32'
                         and staged == 4 and row_groups == 1 and grid == 512)
        schedule = ('flat' if flat_eligible else 'cta') if self.input_schedule == 'auto' else self.input_schedule
        if schedule == 'flat' and not flat_eligible:
            raise ValueError('flat input requires B1/S4 native_fp32, staged_samples=4, input_row_groups=1, grid_blocks=512')
        if self.gate_schedule not in {'auto', 'serial', 'overlap'}:
            raise ValueError('invalid gate_schedule')
        gate_schedule = ('overlap' if batch == 1 and seq in (4,) else 'serial') if self.gate_schedule == 'auto' else self.gate_schedule
        if gate_schedule == 'overlap' and not (batch == 1 and seq in (4,)):
            raise ValueError('gate_schedule optimization is unsupported for this shape')
        if self.route_publication not in {'auto', 'stream', 'wave'}:
            raise ValueError('invalid route_publication')
        route_publication = ('wave' if batch == 1 and seq in (4,) else 'stream') if self.route_publication == 'auto' else self.route_publication
        if route_publication == 'wave' and not (batch == 1 and seq in (1, 4)):
            raise ValueError('route_publication optimization is unsupported for this shape')
        prefetch_default = (6 if seq == 1 else 6 if seq == 4 else 0) if batch == 1 else 0
        output_prefetch_units = prefetch_default if self.output_prefetch_units is None else self.output_prefetch_units
        if type(output_prefetch_units) is not int or output_prefetch_units not in {0, 1, 6}:
            raise ValueError('output_prefetch_units must be None, 0, 1, or 6')
        if output_prefetch_units and not (batch == 1 and seq in (1, 4)):
            raise ValueError('output prefetch is supported only for B1/S1 and B1/S4')
        if self.down_task_mapping not in {'auto', 'linear', 'ready64'}:
            raise ValueError('down_task_mapping must be auto, linear, or ready64')
        down_ready_eligible = batch == 1 and seq == 4 and grid == 512 and staged == 4
        down_task_mapping = ('ready64' if down_ready_eligible else 'linear') if self.down_task_mapping == 'auto' else self.down_task_mapping
        if down_task_mapping == 'ready64' and not down_ready_eligible:
            raise ValueError('ready64 requires B1/S4, grid_blocks=512, staged_samples=4')
        if self.input_mfma not in {'auto', 'f32', 'bf16_k16'}:
            raise ValueError('input_mfma must be auto, f32, or bf16_k16')
        k16_eligible = path == 'small_batch' and flat_eligible and schedule == 'flat'
        # Keep the validated single-batch/decode path. Explicit exact pre also
        # retains the FP32 input projection needed at the B1/S4 rounding boundary.
        exact_pre = self.pre_attn_res in {'exact2', 'parallel4'} or (
            self.pre_attn_res == 'auto' and batch > 1 and seq > 1)
        input_mfma = ('bf16_k16' if k16_eligible and not exact_pre else 'f32') if self.input_mfma == 'auto' else self.input_mfma
        if input_mfma == 'bf16_k16' and not k16_eligible:
            raise ValueError('bf16_k16 input requires the B1/S4 small_batch flat input specialization')
        if self.gate_input_arithmetic not in {'auto', 'original', 'fp32_parts', 'fp32_distributed', 'k16_distributed', 'k16_pair_local', 'k16_pair_dense', 'k16_pair_half'}:
            raise ValueError('gate_input_arithmetic must be auto, original, fp32_parts, fp32_distributed, k16_distributed, k16_pair_local, k16_pair_dense, or k16_pair_half')
        gate_input_eligible = path == 'small_batch' and batch == 1 and seq == 1 and staged == 1 and arithmetic == 'legacy'
        gate_input_arithmetic = (('k16_pair_half' if row_groups == 2 and grid == 256 else 'fp32_parts') if gate_input_eligible else 'original') if self.gate_input_arithmetic == 'auto' else self.gate_input_arithmetic
        if gate_input_arithmetic in {'fp32_parts', 'fp32_distributed', 'k16_distributed', 'k16_pair_local', 'k16_pair_dense', 'k16_pair_half'} and not gate_input_eligible:
            raise ValueError('fp32_parts gate input requires B1/S1 small_batch legacy staged1')
        if gate_input_arithmetic in {'fp32_distributed', 'k16_distributed', 'k16_pair_local', 'k16_pair_dense', 'k16_pair_half'} and (row_groups != 2 or grid != 256):
            raise ValueError('fp32_distributed gate input requires row_groups=2 and grid256')
        if self.ug_arithmetic not in {'auto', 'bf16', 'native_split'}:
            raise ValueError('ug_arithmetic must be auto, bf16, or native_split')
        ug_eligible = path == 'small_batch' and batch == 1 and seq == 4 and staged == 4
        ug_arithmetic = ('native_split' if ug_eligible else 'bf16') if self.ug_arithmetic == 'auto' else self.ug_arithmetic
        if ug_arithmetic == 'native_split' and not ug_eligible:
            raise ValueError('native_split UG requires B1/S4 small_batch with staged_samples=4')
        repair_eligible = path == 'small_batch' and batch == 2 and seq == 2 and arithmetic == 'legacy'
        if self.input_fp32_repair is not None and type(self.input_fp32_repair) is not bool:
            raise ValueError('input_fp32_repair must be None or bool')
        repair = repair_eligible if self.input_fp32_repair is None else self.input_fp32_repair
        if repair and not repair_eligible:
            raise ValueError('input_fp32_repair requires B2/S2 small_batch legacy arithmetic')
        router_guard_eligible = (batch, seq) in {(2, 3), (5, 1)}
        router_guard = (64 if router_guard_eligible else 0) if self.router_guard_ulp is None else self.router_guard_ulp
        if type(router_guard) is not int or router_guard not in {0, 16, 32, 64, 128}:
            raise ValueError('router_guard_ulp must be None, 0, 16, 32, 64, or 128')
        if router_guard and not router_guard_eligible:
            raise ValueError('router guard requires B2/S3 or B5/S1')
        guard_lanes = (8 if router_guard else 1) if self.router_guard_lanes is None else self.router_guard_lanes
        if type(guard_lanes) is not int or guard_lanes not in {1, 8}:
            raise ValueError('router_guard_lanes must be None, 1, or 8')
        if guard_lanes != 1 and not router_guard:
            raise ValueError('cooperative router guard requires an enabled guard')
        if self.pre_attn_res not in {'auto', 'chunked', 'exact2', 'parallel4'}:
            raise ValueError('pre_attn_res must be auto, chunked, or exact2')
        pre_attn_res = ('chunked' if (batch == 1 and seq <= 4) or seq == 1 else 'parallel4') if self.pre_attn_res == 'auto' else self.pre_attn_res
        if pre_attn_res == 'parallel4' and samples < 4:
            raise ValueError('parallel4 requires at least four samples')
        weight_pool_eligible = batch == 1 and seq in (1,4)
        if self.weight_pool not in {'auto','separate','dense','dense_expert'}:
            raise ValueError('invalid weight_pool mode')
        weight_pool = (('dense_expert' if seq == 1 else 'dense') if weight_pool_eligible else 'separate') if self.weight_pool == 'auto' else self.weight_pool
        if weight_pool != 'separate' and not weight_pool_eligible:
            raise ValueError('weight pools require B1/S1 or B1/S4')
        if self.mtp_prepare not in {'auto','replicated','cta'}:
            raise ValueError('invalid mtp_prepare mode')
        local_prepare_eligible = path == 'small_batch' and batch == 1 and seq == 4 and schedule == 'flat'
        mtp_prepare = ('cta' if local_prepare_eligible else 'replicated') if self.mtp_prepare == 'auto' else self.mtp_prepare
        if mtp_prepare == 'cta' and not local_prepare_eligible:
            raise ValueError('CTA-local preparation requires B1/S4 flat small_batch')
        return ResolvedKimiK3Config(path, batch, seq, staged, row_groups, grid, arithmetic, latent_arithmetic, schedule, gate_schedule, route_publication, output_prefetch_units, down_task_mapping, input_mfma, gate_input_arithmetic, ug_arithmetic, repair, router_guard, guard_lanes, pre_attn_res, weight_pool, mtp_prepare)


@dataclass(frozen=True)
class ResolvedKimiK3Config:
    path: str
    batch: int
    seq: int
    staged_samples: int
    input_row_groups: int
    grid_blocks: int
    arithmetic: str
    latent_arithmetic: str
    input_schedule: str
    gate_schedule: str
    route_publication: str
    output_prefetch_units: int
    down_task_mapping: str
    input_mfma: str
    gate_input_arithmetic: str
    ug_arithmetic: str
    input_fp32_repair: bool
    router_guard_ulp: int
    router_guard_lanes: int
    pre_attn_res: str
    weight_pool: str
    mtp_prepare: str

    @property
    def samples(self):
        return self.batch * self.seq

    @property
    def packed_scale_high_rows(self):
        # Every small_batch specialization fits in the first 16 packed rows.
        return self.path == 'general' and self.samples > 16

    @property
    def selector_waves(self):
        # Shared output has staged_samples * 32 float slots; routing needs
        # 16 slots per active wave. Do not grow LDS for uneven batch shapes.
        return min(8, self.samples, self.staged_samples * 2)

    @property
    def input_k_partition(self):
        # Native FP32 accumulation order for the fixed K=7168, N=6284
        # projection. Each pair is (partitions, K quantum); all integer
        # shapes in the public B/S range were independently profiled.
        return _input_k_partition(self.samples)

    @property
    def cache_suffix(self):
        return (f'{self.path}_b{self.batch}_s{self.seq}'
                f'_m{self.staged_samples}_i{self.input_row_groups}'
                + (f'_g{self.grid_blocks}' if self.grid_blocks != 256 else '')
                + ('_native_fp32' if self.arithmetic == 'native_fp32' else '')
                + ('_latent_bf16' if self.latent_arithmetic == 'decoded_bf16' else '')
                + ('_inputflat' if self.input_schedule == 'flat' else '')
                + ('_gateoverlap' if self.gate_schedule == 'overlap' else '')
                + ('_routewave' if self.route_publication == 'wave' else '')
                + (f'_outpf{self.output_prefetch_units}' if self.output_prefetch_units else '')
                + ('_downready64' if self.down_task_mapping == 'ready64' else '')
                + ('_inputmfma_k16' if self.input_mfma == 'bf16_k16' else '')
                + ('_inputrepair_f32' if self.input_fp32_repair else '')
                + (f'_routerguard{self.router_guard_ulp}' if self.router_guard_ulp else '')
                + ('_guard8lane' if self.router_guard_lanes == 8 else '')
                + ('_preexact2' if self.pre_attn_res == 'exact2' else '')
                + ('_preparallel4' if self.pre_attn_res == 'parallel4' else '')
                + (f'_pool_{self.weight_pool}' if self.weight_pool != 'separate' else ''))
