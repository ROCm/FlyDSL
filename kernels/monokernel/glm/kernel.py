# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 indexed decode MonoKernel: one persistent launch per TP rank.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole decoder-layer body for this rank's TP shard.  The indexed path adds the
selection refresh to the same resident grid instead of consuming externally
prepared sparse indices::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> index K/Q/W projection -> index K norm/RoPE/cache
      -> index score -> exact radix top-2048
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks, and each resident CTA walks its
assigned tasks in stage order. Native launches above twelve rows use separate
scratch and peer arenas per tile. The general path executes tiles sequentially;
a tagged grid barrier completes each tile's cache writes before the next tile.

Eligible native TP4 launches assign one row tile to each power-of-two CTA
cohort derived from the row-tile count and the 256-CTA grid. The released
forty-row TP8 path retains four cohorts of 64 CTAs. Task ownership uses each
cohort's local CTA index. All CTAs finish their front-stage tasks and join one
tagged grid barrier. At 4K capacity, the indexed path publishes current cache
rows at this boundary before causal index selection.

External sparse indices can reference slots belonging to later tiles. Where
the original row tile has one ordered cache publisher, cohorts preserve the
persistent cache until every reader finishes. They select the latest incoming
row visible at each original tile boundary from tagged scratch. A second grid
barrier precedes final cache stores; the last incoming row owns each slot.
ATOM external-index paths with multiple cache publishers keep the serial-tile
schedule and select the final row as the publisher for each physical slot.

Indexed cohorts normalize input once per row. Eligible TP4 gate/up compacts
selected tokens across cohorts and distributes expert/token-chunk/output tasks
over the grid. Each task reuses one activation tile for 64 indexed or 128
external-index outputs, preserving the original eight-output MFMA and reduction.
Staging and live LDS capacity determine whether global projections and gate/up
can run; otherwise their per-cohort bodies remain. TP8 retains its existing
expert-union gate/up path. Every schedule preserves each task's arithmetic and
requires all 256 CTAs to be co-resident.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: weights are host-packed (``pack_fp8`` /
``pack_bf16``) so one wave loads 16 rows x 64 k as one contiguous 1 KB, FP8 is
widened exactly to bf16 and fed to ``mfma_f32_16x16x32_bf16`` with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: the general path pushes tagged partials to every peer, which sums
in rank order. TP4/T12 native ATOM FFN without an indexer uses a tile owner:
BF16 partials go to that owner, whose original-order FP32 sum is broadcast.
Each rank adds its local residual before BF16 output; both paths preserve the
same numerical order and epoch protocol.

TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import HIDDEN, INTER, SOFTMAX_SCALE, AttentionWeight, KvCacheLayout
from kernels.monokernel.glm.attention import attention
from kernels.monokernel.glm.cache import cache_commit
from kernels.monokernel.glm.context import (
    ExecutionContext,
    ParallelContext,
    StageContext,
    ThreadContext,
    TokenMetadata,
    Workspace,
    stage_source_key,
)
from kernels.monokernel.glm.indexer import indexer
from kernels.monokernel.glm.interfaces import (
    AttentionInput,
    AttentionOutput,
    AttentionWeights,
    IndexerInput,
    IndexerOutput,
    IndexQuery,
    IndexWeights,
    KvCache,
    LinearWeight,
    Mailbox,
    MlaQuery,
    MoeOutput,
    MoeStage1Input,
    MoeStage1Output,
    MoeStage2Input,
    PreattnInput,
    PreattnOutput,
    PreattnWeights,
    QuantizedActivation,
    Rope,
    RouterInput,
    RouterOutput,
)
from kernels.monokernel.glm.layout import BLOCKS
from kernels.monokernel.glm.moe import moe_stage1, moe_stage2, router
from kernels.monokernel.glm.plan import build_plan
from kernels.monokernel.glm.preattn import preattn
from kernels.monokernel.glm.tools import build_tools
from kernels.monokernel.layout import LAYER_SLOTS, THREADS
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import uniform as _uniform


def build_glm5_monokernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    index_request_width: int = 1,
    index_page_size: int = 16,
    index_block_table_stride: int | None = None,
    expert_mxfp4: bool = False,
    atom_experts: bool = False,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    kv_cache_dtype: str = "bf16",
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    native_fp4_mfma: bool = False,
    moe_activation_dtype: str = "fp8",
    rope_dtype: str = "fp32",
    indexer_cp: bool = False,
    indexer_quant: bool = False,
    _partition: str = "full",
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer."""
    partition_options = dict(locals())
    assert _partition in ("full", "front", "index", "back")
    if _partition != "full":
        assert indexer_quant and with_indexer and indexer_cp
    split_launch = indexer_quant and indexer_cp and _partition == "full"
    if split_launch:
        front_kernel = build_glm5_monokernel(**dict(partition_options, _partition="front"))
        index_kernel = build_glm5_monokernel(**dict(partition_options, _partition="index"))
        back_kernel = build_glm5_monokernel(**dict(partition_options, _partition="back"))
    implementation_key = stage_source_key()
    plan = build_plan(**partition_options)
    COHORT_ROW_TILES = plan.COHORT_ROW_TILES
    DNW_OFF = plan.DNW_OFF
    G = plan.G
    KEYS_OFF = plan.KEYS_OFF
    KT_OFF = plan.KT_OFF
    MISC_OFF = plan.MISC_OFF
    OUT_OFF = plan.OUT_OFF
    PT_OFF = plan.PT_OFF
    RED_OFF = plan.RED_OFF
    ROW_TILES = plan.ROW_TILES
    S = plan.S
    TILE_PEER_BYTES = plan.TILE_PEER_BYTES
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    W = plan.W
    WORK_WORDS = plan.WORK_WORDS
    _partition = plan.partition
    index_grid_scale = plan.index_grid_scale
    topk = plan.topk
    use_atom_kv_cache = plan.use_atom_kv_cache
    with_indexer = plan.with_indexer

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_layer_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        index_params: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
    ):
        # A host-only assertion retains the source digest in the kernel closure.
        assert implementation_key
        tid = fx.thread_idx.x
        full_bid = fx.block_idx.x
        bid = full_bid % G if const_expr(COHORT_ROW_TILES) else full_bid
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        pl = xs  # score Q is dead before split probabilities are written
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        full_h, full_out = h_in, x_out
        full_positions, full_slots, full_indptr = (
            positions,
            slot_mapping,
            sparse_kv_indptr,
        )
        full_scratch, full_sym, full_indices = scratch, sym, indices

        def run_row_tile(row_tile):
            row_offset = fx.Int32(row_tile) * S
            peer_offset = fx.Int64(row_tile) * TILE_PEER_BYTES
            h_in = full_h + fx.Int64(row_offset) * (HIDDEN * 2)
            x_out = full_out + fx.Int64(row_offset) * (HIDDEN * 2)
            positions = full_positions + fx.Int64(row_offset) * 8
            slot_mapping = full_slots + fx.Int64(row_offset) * 8
            sparse_kv_indptr = full_indptr + fx.Int64(row_offset) * 4
            scratch = full_scratch + fx.Int64(row_tile) * TILE_SCRATCH_BYTES
            sym = full_sym + peer_offset
            indices = full_indices
            if const_expr(not with_indexer and not use_atom_kv_cache):
                indices = indices + fx.Int64(row_offset) * (topk * 4)
            r_h = _rsrc(h_in)
            # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
            # device counter bumped once per decode step (graph friendly); ``layer``
            # makes it unique per layer within the step.
            step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
            tag = step_value * LAYER_SLOTS + layer + 1
            peer_slot = (step_value * launches_per_step + layer) & 1
            # In fused mode cur_pos carries the per-request page table.  ATOM cache
            # rows and selected attention keys both use physical page slots.
            if const_expr(with_indexer and use_atom_kv_cache):
                pos0 = fx.Int32(0)
            else:
                pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32)) + row_offset
            r_peers = _rsrc(peers)
            # One wave sends to one peer, so retain only that wave's destination.
            pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
            peer_dst = ((fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))) + peer_offset

            execution = ExecutionContext(
                thread=ThreadContext(
                    block_id=bid,
                    global_block_id=full_bid,
                    thread_id=tid,
                    lane_id=lane,
                    wave_id=wave,
                    tile_id=row_tile,
                    row_offset=row_offset,
                    epoch=tag,
                ),
                workspace=Workspace(
                    scratch=scratch,
                    all_scratch=full_scratch,
                    lds=lds,
                    activations=xs,
                    reduction=red,
                    output=outs,
                    routing_ids=keys,
                    routing_weights=dnw,
                    temporary=misc,
                    kv_tile=ktile,
                    rope_tile=petile,
                    probabilities=pl,
                    vector_type=v4f,
                ),
                parallel=ParallelContext(
                    rank=rank,
                    peer_table=r_peers,
                    peer_destination=peer_dst,
                    peer_offset=peer_offset,
                    peer_slot=peer_slot,
                    buffer=sym,
                    all_buffer=full_sym,
                ),
                tokens=TokenMetadata(
                    block_tables=cur_pos,
                    positions=positions,
                    slot_mapping=slot_mapping,
                    kv_indptr=sparse_kv_indptr,
                    all_positions=full_positions,
                    all_slots=full_slots,
                    all_indptr=full_indptr,
                    first_position=pos0,
                ),
            )
            index_weights = IndexWeights(index_params)
            ctx = StageContext(plan, execution, build_tools(plan, execution, index_weights))
            put, get = ctx.ops.memory.put, ctx.ops.memory.get

            # Named tensor contracts expose the producer/consumer edges. These
            # descriptors create no buffers or instructions; tagged views are
            # resolved only at the original load/store sites.
            rope = Rope(rope_cos, rope_sin)
            cache = KvCache(kv_cache, pe_cache)
            query = MlaQuery(latent=Mailbox("q_lat"), rope=Mailbox("q_pe"))
            index_query = IndexQuery(
                kv_buffer=indices,
                q=Mailbox("index_q"),
                q_quant=Mailbox("index_q_quant"),
                weights=Mailbox("index_w"),
            )
            pre_input = PreattnInput(
                hidden_states=r_h,
                weights=PreattnWeights(
                    input_norm=g_in,
                    query_norm=g_q,
                    kv_norm=g_kv,
                    qkv_proj=LinearWeight(w_qkv_a, s_qkv_a),
                    q_proj=LinearWeight(w_q_b, s_q_b),
                    q_absorb=LinearWeight(w_uk, s_uk),
                    index=index_weights,
                ),
                rope=rope,
            )
            pre_output = PreattnOutput(q=query, index=index_query, cache=cache)
            selected = IndexerOutput(kv_indices=Mailbox("indices"), ready=Mailbox("indices_ready"))
            attn_input = AttentionInput(
                kv_indices=selected.kv_indices if with_indexer else indices,
                indices_ready=selected.ready,
                q=pre_output.q,
                cache=pre_output.cache,
                residual=r_h,
                weights=AttentionWeights(
                    v_proj=LinearWeight(w_uv, s_uv),
                    out_proj=LinearWeight(w_o, s_o),
                ),
            )
            attn_output = AttentionOutput(o=Mailbox("o"), hidden_states=Mailbox("a"))
            router_input = RouterInput(hidden_states=attn_output.hidden_states, norm_weight=g_post, gate=w_r)
            router_output = RouterOutput(
                gating_output=Mailbox("scores"),
                hidden_states=QuantizedActivation(
                    data=Mailbox("xq"),
                    scale=Mailbox("xqs"),
                    dequant=Mailbox("xqd"),
                ),
            )
            expert_input = MoeStage1Input(
                hidden_states=attn_output.hidden_states,
                gating_output=router_output.gating_output,
                activation=router_output.hidden_states,
                e_score_correction_bias=bias,
                norm_weight=g_post,
                w1=LinearWeight(w_ug, s_ug),
            )
            expert_output = MoeStage1Output(
                intermediate=Mailbox("mid"),
                topk_ids=Mailbox("sel"),
                topk_weights=Mailbox("prob"),
            )
            down_input = MoeStage2Input(
                hidden_states=expert_output.intermediate,
                topk_ids=expert_output.topk_ids,
                topk_weights=expert_output.topk_weights,
                gating_output=router_output.gating_output,
                e_score_correction_bias=bias,
                residual=attn_output.hidden_states,
                w2=LinearWeight(w_dn, s_dn),
            )
            layer_output = MoeOutput(out=x_out, all_out=full_out)

            # These Python helpers emit IR into the same kernel; they do not launch kernels.
            if const_expr(_partition in ("full", "front")):
                pre = preattn(pre_input, pre_output, ctx)
                # Input RMSNorm, QKV/index K projections, normalized query and cache publication.
                pre.rmsnorm_qkv()
                pre.rmsnorm_q()
                pre.rope_cache()
                # Query/RoPE and index-query preparation share the projection operands.
                pre.q_proj()
                pre.index_quant()
                pre.q_absorb()
                if const_expr(COHORT_ROW_TILES and not plan.FRONT_PAYLOAD_RETIRE and _partition == "full"):
                    # Every cohort finishes its front tasks before cross-tile reads.
                    # Indexed mode publishes cache stores; external-index mode keeps
                    # persistent cache intact and reads current versions from scratch.
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                    gpu.barrier()
                    cohort_arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES)
                    if tid == 0:
                        put(cohort_arrivals, full_bid, fx.Int32(1))
                    if full_bid == 0:
                        if tid < BLOCKS:
                            get(cohort_arrivals, tid)
                            fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                        gpu.barrier()
                        if tid == 0:
                            put(cohort_arrivals, BLOCKS, fx.Int32(1))
                    if tid == 0:
                        get(cohort_arrivals, BLOCKS)
                    gpu.barrier()
                    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
            if const_expr(_partition in ("full", "index")):
                indexer(IndexerInput(index_query, rope, index_weights), selected, ctx)
            if const_expr(_partition in ("full", "back")):
                attn = attention(attn_input, attn_output, ctx)
                # Sparse QK/softmax/PV, reader retirement, split merge and W_UV.
                attn.mla_decode()
                attn.mla_reduce()
                # W_o, TP attention reduction and residual publish the MoE input.
                attn.output_projection()
                # Router and quantized gate/up/down retain their task and mailbox dependencies.
                router(router_input, router_output, ctx)
                moe_stage1(expert_input, expert_output, ctx)
                moe_stage2(down_input, layer_output, ctx)
                cache_commit(cache, ctx)

            if const_expr(ROW_TILES > 1 and not COHORT_ROW_TILES):
                if row_tile + 1 < ROW_TILES:
                    # Every wave publishes its cache/output stores before its CTA arrives.
                    # Separate tile payloads and barrier slots avoid overwriting readers.
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                    gpu.barrier()
                    arrivals = full_scratch + fx.Int64(ROW_TILES * TILE_SCRATCH_BYTES)
                    arrivals = arrivals + fx.Int64(row_tile) * ((G + 1) * 8)
                    if tid == 0:
                        put(arrivals, bid, fx.Int32(1))
                    if bid == 0:
                        if tid < G:
                            get(arrivals, tid)
                            fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                        gpu.barrier()
                        if tid == 0:
                            put(arrivals, G, fx.Int32(1))
                    if tid == 0:
                        get(arrivals, G)
                    gpu.barrier()
                    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")

        if const_expr(COHORT_ROW_TILES):
            run_row_tile(full_bid // G)
        elif const_expr(ROW_TILES == 1):
            run_row_tile(0)
        elif const_expr(ROW_TILES <= 4):
            # Bound code growth while exposing constant row offsets/lifetimes.
            for row_tile in range_constexpr(ROW_TILES):
                run_row_tile(row_tile)
        else:
            for row_tile in range(ROW_TILES):
                run_row_tile(row_tile)

    if _partition != "full":
        return glm5_layer_kernel

    @flyc.jit
    def launch(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        index_params: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        arguments = (
            h_in,
            x_out,
            cur_pos,
            positions,
            slot_mapping,
            sparse_kv_indptr,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            index_params,
            step,
            rank,
            layer,
        )
        if const_expr(split_launch):
            front_kernel(*arguments).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)
            index_kernel(*arguments).launch(grid=(BLOCKS * index_grid_scale,), block=(THREADS,), stream=stream)
            back_kernel(*arguments).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)
        else:
            glm5_layer_kernel(*arguments).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
