# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Assemble shared compile-time tools for every GLM stage and quant format."""

import flydsl.compiler as flyc
from kernels.monokernel.glm.context import DeviceTools
from kernels.monokernel.glm.tools.communication import build_communication
from kernels.monokernel.glm.tools.gemm import build_gemm
from kernels.monokernel.glm.tools.memory import build_memory
from kernels.monokernel.glm.tools.normalization import build_normalization
from kernels.monokernel.glm.tools.quantization import build_quantization
from kernels.monokernel.glm.tools.reductions import build_reductions
from kernels.monokernel.glm.tools.routing import build_routing


@flyc.jit
def build_tools(plan, state):
    memory = build_memory(plan, state)
    reductions = build_reductions(plan, state, memory)
    gemm = build_gemm(plan, state, memory)
    routing = build_routing(plan, state, memory)
    normalization = build_normalization(plan, state, memory, reductions)
    quantization = build_quantization(plan, state, memory, reductions)
    communication = build_communication(plan, state, memory, reductions)
    return DeviceTools(
        quant_moe=quantization.quant_moe,
        ld_f32=memory.ld_f32,
        ld_bf16=memory.ld_bf16,
        ld_rope=memory.ld_rope,
        quant_index_pair=quantization.quant_index_pair,
        row_index_bounds=memory.row_index_bounds,
        row_active=memory.row_active,
        row_position=memory.row_position,
        row_slot=memory.row_slot,
        row_writes_cache=memory.row_writes_cache,
        index_request_start=memory.index_request_start,
        index_page_slot=memory.index_page_slot,
        lds_ld=memory.lds_ld,
        lds_st=memory.lds_st,
        bf16_pair=memory.bf16_pair,
        bf16_round=memory.bf16_round,
        index_arg=memory.index_arg,
        mb=memory.mb,
        cp_local_rank=memory.cp_local_rank,
        cp_peer=memory.cp_peer,
        cp_buffer=memory.cp_buffer,
        put=memory.put,
        put2=memory.put2,
        put_bf=memory.put_bf,
        bf2_f32=memory.bf2_f32,
        poll=memory.poll,
        hint_wait=memory.hint_wait,
        pre_poll=memory.pre_poll,
        get=memory.get,
        getf=memory.getf,
        getf_many=memory.getf_many,
        get2_many=memory.get2_many,
        get2=memory.get2,
        get_bf2_many=memory.get_bf2_many,
        wave_sum=reductions.wave_sum,
        wave_max=reductions.wave_max,
        block_sums=reductions.block_sums,
        block_sums_f64=reductions.block_sums_f64,
        block_maxs=reductions.block_maxs,
        block_sum=reductions.block_sum,
        unit_fp8=gemm.unit_fp8,
        unit_fp8x2=gemm.unit_fp8x2,
        unit_f8f8=gemm.unit_f8f8,
        unit_ptpc=gemm.unit_ptpc,
        unit_ptpc64=gemm.unit_ptpc64,
        unit_mxfp4=gemm.unit_mxfp4,
        unit_mxfp4_bf16=gemm.unit_mxfp4_bf16,
        unit_native_mxfp4=gemm.unit_native_mxfp4,
        lds_mxfp8=gemm.lds_mxfp8,
        unit_bf16=gemm.unit_bf16,
        unit_attention=gemm.unit_attention,
        mma_units=gemm.mma_units,
        run_units=gemm.run_units,
        reduce_rows=gemm.reduce_rows,
        emit_out=gemm.emit_out,
        pick=memory.pick,
        attention_b_word=gemm.attention_b_word,
        emit_ptpc=gemm.emit_ptpc,
        emit_ptpc_scalar=gemm.emit_ptpc_scalar,
        stage_x_rmsnorm_ptpc=normalization.stage_x_rmsnorm_ptpc,
        stage_x_pairs_ptpc=normalization.stage_x_pairs_ptpc,
        stage_x_rmsnorm=normalization.stage_x_rmsnorm,
        load_x_rmsnorm=normalization.load_x_rmsnorm,
        stage_x_pairs=normalization.stage_x_pairs,
        quant_scaled=quantization.quant_scaled,
        quant_mxfp8=quantization.quant_mxfp8,
        pack_moe4=quantization.pack_moe4,
        publish_moe_quant=quantization.publish_moe_quant,
        moe_roundtrip=quantization.moe_roundtrip,
        stage_xq=quantization.stage_xq,
        st_f8=quantization.st_f8,
        st_mxfp8=quantization.st_mxfp8,
        st_ptpc=quantization.st_ptpc,
        load_bias=routing.load_bias,
        route_top8=routing.route_top8,
        ffn_owner_reduce=communication.ffn_owner_reduce,
        o_owner_reduce=communication.o_owner_reduce,
        peer_reduce=communication.peer_reduce,
        dcp_merge_latent=communication.dcp_merge_latent,
        start=memory.start,
        n_sel=memory.n_sel,
        dn_route=routing.dn_route,
    )
