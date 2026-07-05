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
- `kernels/mega_moe.py`（合并文件；原名 `fused_moe_stage1_stage2.py`，后重命名 + 英文注释精简）
  - 并入 megastage1/megagemm2 调优表助手（`_mega_*` / `_detect_gpu_model_name`）。
  - 内联 `compile_fused_moe_gemm2_combine`（薄 builder，转调 `compile_mixed_moe_gemm2` +
    `fused_p2p_scatter`）。
  - 并入 `MegaMoeStage2`（stage-2 gemm2+combine op；原 `FlyDSLMoeGemm2CombineOp`；back-compat `force_mode`）。
  - 并入 `MegaMoeStage1`（stage-1 驱动；原 `FusedMoEMegaStage1`），改为**使用 `comm_op._gm`**，不再自建 dispatch op。
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
- `tests/kernels/bench_moe_intranode_stage1_groupgemm.py`：`MegaMoE` / `MegaMoeStage2` import 改指
  `kernels.mega_moe`；megav1 调用 `moe.forward` → `moe.forward_prequant`。
- `tests/kernels/test_profiler_moe_gemm2_combine.py`：`MegaMoeStage2` import 路径更新。

## 对象关系（改动后）

```
MegaMoE (mega_moe.py)                                # 唯一对外 op
  ├── FlyDSLDispatchCombineIntraNodeOp (comb_op)     # 唯一 symmetric buffer owner
  │      └── comb_op._gm = FlyDSLDispatchGroupMajorOp # group-major dispatch buffers
  ├── MegaMoeStage1 (stage1, 用 comb_op._gm)          # compile_fused_moe_gemm1
  └── MegaMoeStage2 (stage2)                          # compile_fused_moe_gemm2_combine
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
- [x] **端到端精度验证**（8 卡 gfx950 / MI355X，容器 `yanbo_discom`）——全 PASS，见「验证结果」。

尚未做 / 后续：
- [ ] 非融合分支（`enable_fused_stage1/2=False`）仅占位 `NotImplementedError`，未接线。
- [ ] dispatch 布局**运行时切换** + payload/scale "保留大块+双 view" 复用：本期按 init 固定
      mode、只实际使用 group-major；运行时切换留待后续。
- [ ] INTERLEAVE gate_mode：需 `compile_fused_moe_gemm1(gate_mode=INTERLEAVE)`（A8W4 加
      `a_scale_one=True`）+ `shuffle_weight_a16w4(gate_up=True)` + scale 交织 + facade XCD-guard
      gx 公式改 interleave 版；未做。

## 验证结果（已在 8 卡 gfx950 / MI355X 完成）

环境：容器 `yanbo_discom`（ROCm 7.0.2 / HIP 7.0），`/home/yashao/FlyDSL`，8×MI355X（gfx950）。

### 环境准备（迁移到新机时的必做步骤）

1. **重建 FlyDSL 原生扩展**：`mega_moe_v1` 分支的 C++ 绑定（`FlyExtension.cpp` 的 `_Basis`）比镜像旧构建新，
   直接 import 会报 `cannot import name '_Basis'`。需重建：
   ```
   bash scripts/build.sh -j64 && pip install -e .
   ```
   （`build.sh` 会移除 editable 的 `python/flydsl/_mlir` 软链，构建后 `pip install -e .` 重建软链。）
2. **aiter 版本**：bench 与 MegaMoE 依赖 `aiter.moe_sorting_fwd / mxfp4_moe_sort_hip` 与
   `aiter.ops.quant.per_1x32_mx_quant_hip`。这些符号在 aiter `main` 上**已被移除**；需用带这些符号的版本
   （本次用 ATOM 分支 `ep_fix`：`ROCm/aiter` base `ef114b0d4` + cherry-pick PR#3377「flydsl moe EP reduce
   masked gather」）。该 aiter 为 rocm7.2/triton3.6 而构建，在 discom（triton 3.4）需设
   `AITER_USE_SYSTEM_TRITON=1`（把 gluon 的 triton 版本硬错误降级为警告；gluon attention kernel 与 MoE 无关）。
3. **对称堆**：bs≥4096 的 group-major buffer 超出默认 4GB 静态堆，需 `MORI_SHMEM_HEAP_SIZE=40G`。

### 1. 标准 dispatch+combine 零回归（`enable_group_major=False`）—— PASS

```
AITER_USE_SYSTEM_TRITON=1 MORI_SHMEM_HEAP_SIZE=40G \
  python tests/kernels/test_profiler_dispatch_combine.py --ci-sweep
```
结果：`>>> ALL PASS (accuracy across 19 cases) <<<`（19/19 verify PASS + profile ok，含 bf16 / fp8 /
fp4 / std_moe / zero_copy / recv_cap / mixed-dispatch 等；每 case 均 `ALL PASS (global across 8 ranks)`）。

### 2. megav1 端到端正确性 gate（融合路径，fixedslot 与 compact 由 buffer 大小自动选择）—— PASS

```
AITER_USE_SYSTEM_TRITON=1 MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
  tests/kernels/bench_moe_intranode_stage1_groupgemm.py \
  --network v4_flash --quant a8w4 --bs-list 64,2048,4096,8192
```
每个 shape 均 `[FULL-E2E] ... -> PASS (all 8 ranks)`：

| quant | bs   | mega relL2 | atom-fp8 baseline | mega-vs-baseline | floor | e2e speedup |
|-------|------|-----------|-------------------|------------------|-------|-------------|
| a8w4  | 64   | 0.2077    | 0.2077            | 4.2e-5           | 0.25  | 1.26x       |
| a8w4  | 2048 | 0.2074    | 0.2074            | 4.3e-5           | 0.25  | 1.16x       |
| a8w4  | 4096 | 0.2073    | 0.2073            | 4.1e-5           | 0.25  | 1.05x       |
| a8w4  | 8192 | 0.2073    | 0.2073            | 4.1e-5           | 0.25  | 1.09x       |
| a4w4  | 64   | 0.2948    | 0.3009            | 8.6e-2           | 0.32  | —           |
| a4w4  | 2048 | 0.2945    | 0.3002            | 8.6e-2           | 0.32  | —           |
| a4w4  | 4096 | 0.2943    | 0.3001            | 8.6e-2           | 0.32  | —           |
| a4w4  | 8192 | 0.2943    | 0.3000            | 8.6e-2           | 0.32  | —           |

a8w4：mega 与 baseline 逐比特级一致（差 ~4e-5），均在地板 0.25 内。
a4w4（fp4）：mega（~0.294）优于 baseline（~0.300），均在地板 0.32 内；mega-vs-baseline ~8.6e-2 属 fp4
E2M1 粗步长下两次合法量化的正常发散（gate 用「mega ≤ baseline+0.02」判定，符合文档预期）。

## 头号风险（若 bench FAIL，优先排查）

## 头号风险（若 bench FAIL，优先排查）

融合 dispatch prologue 与原 `FlyDSLDispatchGroupMajorOp` 的 buffer 名/字节布局**强耦合**。
移入 comm op 后，重点核对 `comb_op._gm` 的构造参数是否与原 `FusedMoEMegaStage1` 一一对应：
`gm_unit_size`（== stage1 tile_m）、`gm_data_type`（a8w4→fp8 / a4w4→fp4）、`gm_scale_dim`
（== model_dim//32）、`gm_compact`、`gm_scheme="fixedslot"`；以及
`FusedMoEMegaStage1._build_disp_table` 里各 `op.*` 指针（`gb1/running/done2/ll_count/
p2p_* /srcmap_em/tile_row_base/num_valid/total_recv/dest_ctr/recv_num` 及 compact 扩展项）。
