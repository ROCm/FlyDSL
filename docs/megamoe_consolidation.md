# MegaMoE consolidation (dispatch buffers merged into the combine op)

面向迁移到其它环境的说明文档：记录本次把 MegaMoE 相关 host 文件聚拢成一个 op 的
改动、完成度、以及在目标环境（gfx950 / MI355X，8 卡 EP）上**必须做的验证**。

## 目标

把分散的 MegaMoE host 文件聚拢：dispatch 的 expert-major symmetric buffer 归并进通信
算子 `FlyDSLDispatchCombineIntraNodeOp`（新增 `enable_group_major` 布局），`total_recv`
与 combine 侧统一为同一 buffer（去掉 Plan-A 跨 op 桥接）；stage-1（dispatch⊕GEMM1）与
stage-2（GEMM2+combine）逻辑合并进 `fused_moe_stage1_stage2.py` 的 `MegaMoE`；量化纳入
op（`forward(x_bf16,...)` 内部量化 → bf16 输出）。

## 文件改动总览

修改：
- `kernels/dispatch_combine_intranode_op.py`
  - `FlyDSLDispatchCombineConfig` 新增 `enable_group_major` + `gm_data_type / gm_unit_size /
    gm_scale_dim / gm_scale_type_size / gm_scheme / gm_compact`。
  - 把 `FlyDSLDispatchGroupMajorOp` 类**逐字节原样移入**本文件（helper 改名 `_gm_is_fp4 /
    _gm_row_bytes / _gm_row_view` 避免与既有符号冲突）。
  - `FlyDSLDispatchCombineIntraNodeOp.__init__` 末尾：当 `enable_group_major=True` 时创建
    `self._gm = FlyDSLDispatchGroupMajorOp(...)`，并 `self._gm.total_recv = self.total_recv`
    统一计数。`enable_group_major=False`（默认）路径**完全不变**。
- `kernels/fused_moe_stage1_stage2.py`（由 361 行扩到 ~1160 行）
  - 并入 megastage1/megagemm2 调优表助手（`_mega_*` / `_detect_gpu_model_name`）。
  - 内联 `compile_fused_moe_gemm2_combine`（薄 builder，转调 `compile_mixed_moe_gemm2` +
    `fused_p2p_scatter`）。
  - 并入 `FlyDSLMoeGemm2CombineOp`（stage-2 gemm2+combine op；新增 back-compat `force_mode`）。
  - 并入 `FusedMoEMegaStage1`（stage-1 驱动），改为**使用 `comm_op._gm`**，不再自建 dispatch op。
  - 新增 `_resolve_stage1_config()`：把 tile / compact / xcd 解析前置（供 comm op 建 `_gm`）。
  - `MegaMoE`：新增 `enable_fused_stage1 / enable_fused_stage2`（init 固定，非融合暂占位）；
    `forward(x_bf16, wts, topk_ids)` 内部量化为主入口，`forward_prequant(x_q, scales, ...)`
    为已量化快路径；`stage2_mode` 保留为 back-compat。

删除（内容已并入上面两文件）：
- `kernels/ep_dispatch_groupmajor_op.py`
- `kernels/fused_moe_megakernel.py`
- `kernels/mixed_moe_gemm2_combine_fused.py`
- `kernels/mixed_moe_gemm2_combine_fused_op.py`

调用方更新：
- `tests/kernels/bench_moe_intranode_stage1_groupgemm.py`：`FlyDSLMoeGemm2CombineOp` import 改指
  `kernels.fused_moe_stage1_stage2`；megav1 调用 `moe.forward` → `moe.forward_prequant`。
- `tests/kernels/test_profiler_moe_gemm2_combine.py`：`FlyDSLMoeGemm2CombineOp` import 路径更新。

## 对象关系（改动后）

```
MegaMoE (fused_moe_stage1_stage2.py)                 # 唯一对外 op
  ├── FlyDSLDispatchCombineIntraNodeOp (comb_op)     # 唯一 symmetric buffer owner
  │      └── comb_op._gm = FlyDSLDispatchGroupMajorOp # group-major dispatch buffers
  ├── FusedMoEMegaStage1 (stage1, 用 comb_op._gm)     # compile_fused_moe_gemm1
  └── FlyDSLMoeGemm2CombineOp (stage2)                # compile_fused_moe_gemm2_combine
```

## 行为保持（关键，用于判断迁移是否等价）

- **total_recv 统一是行为等价的**：原代码本就把 `comb_op.total_recv` 作为 `total_recv_buf`
  零拷贝传给 stage1；现在把 `_gm.total_recv` 别名到同一 buffer，disp 表用 `op.total_recv`
  ——同一块内存。
- **symmetric heap 分配顺序不变**：原来 [combine buffers] → [group-major buffers]；现在
  comb op 先分配 combine，再在 `__init__` 末尾建 `_gm`，顺序一致（mori 对称堆一致性关键）。
- gemm2 ghost-gate 边界仍重指到常量 `max_recv`（`MegaMoE._gemm2_gate_bound` + repoint
  `comb_op._fx_out_total_recv`），语义不变。
- `enable_group_major=False` 路径与改动前逐字节一致（保护 `test_profiler_dispatch_combine`
  等既有使用方）。

## 计划完成度

已完成（本期）：
- [x] comm op 增加 `group_major` 布局，移入 `FlyDSLDispatchGroupMajorOp`
- [x] 调优表助手搬入 `fused_moe_stage1_stage2.py`
- [x] stage-1 合并（使用 `comm_op._gm`）
- [x] stage-2（gemm2+combine + builder）合并
- [x] `total_recv` 统一（去 Plan-A 桥接）+ 统一 alloc
- [x] 删除 4 个文件
- [x] 更新 bench/test import 与调用
- [x] 全 `kernels/` `python -m compileall` 通过；无残留失效 import

尚未做 / 后续：
- [ ] **端到端精度验证**（需 8 卡 gfx950，见下）——本次开发机为 gfx942 且无容器访问，未跑。
- [ ] 非融合分支（`enable_fused_stage1/2=False`）仅占位 `NotImplementedError`，未接线。
- [ ] dispatch 布局**运行时切换** + payload/scale "保留大块+双 view" 复用：本期按 init 固定
      mode、只实际使用 group-major；运行时切换留待后续。
- [ ] INTERLEAVE gate_mode：需 `compile_fused_moe_gemm1(gate_mode=INTERLEAVE)`（A8W4 加
      `a_scale_one=True`）+ `shuffle_weight_a16w4(gate_up=True)` + scale 交织 + facade XCD-guard
      gx 公式改 interleave 版；未做。

## 迁移到目标环境后必须做的验证

开发环境无法跑（gfx942 + 无容器）。到 gfx950 / 8 卡环境后，在容器内 `FlyDSL` 目录：

1. 标准 dispatch+combine 零回归（`enable_group_major=False` 路径）：
   跑 `tests/kernels/test_profiler_dispatch_combine.py`（按其实际入口 / torchrun）。
2. megav1 端到端正确性 gate（融合路径，fixedslot 与 compact 两路）：
   ```
   torchrun --standalone --nproc_per_node=8 \
     tests/kernels/bench_moe_intranode_stage1_groupgemm.py \
     --network v4_flash --quant a8w4 --bs-list 64,2048,4096,8192
   ```
   期望每个 shape 打印 `[FULL-E2E] ... -> PASS (all 8 ranks)`。
   精度判据：mega 输出相对 fp32 torch oracle 的 relL2 落在量化地板内
   （fp4≈0.32 / fp8≈0.25），且与 ATOM-fp8 基线一致。

## 头号风险（若 bench FAIL，优先排查）

融合 dispatch prologue 与原 `FlyDSLDispatchGroupMajorOp` 的 buffer 名/字节布局**强耦合**。
移入 comm op 后，重点核对 `comb_op._gm` 的构造参数是否与原 `FusedMoEMegaStage1` 一一对应：
`gm_unit_size`（== stage1 tile_m）、`gm_data_type`（a8w4→fp8 / a4w4→fp4）、`gm_scale_dim`
（== model_dim//32）、`gm_compact`、`gm_scheme="fixedslot"`；以及
`FusedMoEMegaStage1._build_disp_table` 里各 `op.*` 指针（`gb1/running/done2/ll_count/
p2p_* /srcmap_em/tile_row_base/num_valid/total_recv/dest_ctr/recv_num` 及 compact 扩展项）。
