# megaMoE 合并落地方案（ghu_moe_stage1 → dev/universe_yadai）

> 目标：把 `ghu_moe_stage1` 的 **megaMoE 特性 + 完整 EP 层** 合入当前分支 `dev/universe_yadai`，
> 同时**不回退** dev 分支已有的 MoE/PA 优化。
> 范围决策（已确认）：仅 megaMoE 特性、包含完整 EP 层（含 `flydsl.expr.extern`）。

---

## 0. 关键结论与策略建议（先读这一段）

| 事实 | 数据 |
|---|---|
| 共同祖先 | `8eec98bc` |
| dev 领先 | **4 commit**，改动集中在 kernels + 2 个核心文件（`buffer_ops.py`、`rocdl/__init__.py`），均为**加法式** |
| ghu 领先 | **165 commit**，其中 **`python/flydsl/` 核心库改了 43 个文件，+7974/−3682**（`jit_function.py` 1212、`typing.py` 1010、`primitive.py` 980、新增 `struct.py`、删除旧 `rocdl.py`…） |

**核心判断：完整 EP 层 ≠ 可独立摘出的"几个新文件"。**
EP 走 `mori.ir.flydsl` 的 wait 原语 → 经 `flydsl.expr.extern.ExternFunction` → 真正链接进二进制依赖
`compiler/extern_link.py` + `external_llvm.py` + `link_utils.py`（均为 ghu 新增），而这套链接钩子**深度嵌在
ghu 重写过的 `jit_function.py` / `kernel_function.py` / `meta.py` 中**（`dsl_loc_tracing` 在 dev 上根本不存在）。
因此完整 EP 层与 ghu 的 compiler 核心代际**不可分离**。

### 策略建议：反转合并方向（强烈推荐）
- ❌ 不要：以 dev 为底，把 ghu 的 43 文件核心重写 + megaMoE "移植"进来 —— 工作量无界、风险极高。
- ✅ 推荐：**以 `ghu_moe_stage1` 的树为底**（它已含可工作的 core+EP+megaMoE），
  **把 dev 的 4 个 commit 的优化重新 apply 到其上**。dev 优化已被完整盘点（见 §4），范围有界、可逐项 grep 验证。
- 最终分支名仍可叫 `dev/universe_yadai`（或新建 `dev/universe_yadai_megamoe`）；"方向反转"只是底座选择，不改变最终归属。

> 若坚持以 dev 为底（正向合并），则 §5 的 `moe_gemm_2stage.py` 28-hunk 仍需手工解，且 §3 的核心库需整块拖入——
> 等价于把 ghu compiler 代际搬过来，反而更难。本方案两种方向都给出，但默认按"反转方向"组织执行步骤（§7）。

---

## 1. 文件分类总览

### A. 直接取 ghu 版（megaMoE 新文件，dev 无对应，无冲突）
生产文件：
- `kernels/fused_moe_stage1_stage2.py`（顶层 `MegaMoE`）
- `kernels/fused_moe_megakernel.py`（`FusedMoEMegaStage1`）
- `kernels/fused_moe_gemm_2stage.py`（`compile_fused_moe_gemm1`，是 `mixed_moe_gemm_2stage` 的 **fork**）
- `kernels/ep_dispatch_groupmajor_op.py`（`FlyDSLDispatchGroupMajorOp`）
- `kernels/dispatch_combine_intranode_kernel.py` / `dispatch_combine_intranode_op.py`
- `kernels/mixed_moe_gemm2_combine_fused.py` / `mixed_moe_gemm2_combine_fused_op.py`
- `kernels/moe_common.py`（`GateMode` enum）
- `kernels/moe_sorting_kernel.py`（baseline sorter；megaMoE 主路径未用，可选，见 §6）
- `kernels/tuning_configs/flydsl_gfx950_mi355x_MegaStage1_ep8.json`
- `kernels/tuning_configs/flydsl_gfx950_mi355x_MegaGemm2_ep8.json`
- （若存在）`kernels/tuning_configs/*IntraNode_ep8.json`
- `docs/moe_stage1_mega.md`（权威设计文档，保留）

### B. 核心库闭包（EP/extern 必需，dev 上缺失或最小）—— 见 §3
- 新增：`python/flydsl/expr/extern.py`、`compiler/extern_link.py`、`external_llvm.py`、`link_utils.py`
- 取 ghu 版：`compiler/jit_function.py`、`kernel_function.py`、`compiler/protocol.py`、`expr/meta.py`
- 连带（jit 子系统依赖）：`compiler/jit_argument.py`、`jit_executor.py`、`backends/*`、`expr/__init__.py`

### C. 真冲突文件（双方都改，需手工/逐 hunk）—— 见 §5
- `kernels/moe_gemm_2stage.py`（**重度，28 hunk**）
- `python/flydsl/expr/buffer_ops.py`（轻度，1 hunk，加法）
- `python/flydsl/expr/rocdl/__init__.py`（轻度，1 hunk，加法）
- `tests/kernels/test_moe_gemm.py`（轻度，9 hunk，参数并集）

### D. ghu-only 共享改动（megaMoE 依赖，随底座一起进，无 dev 风险）
- `kernels/mixed_moe_gemm_2stage.py`（gemm2 fused 复用 `compile_mixed_moe_gemm2`；dev 仅注释了 2 处 `s_setprio`，须保留）
- `kernels/mfma_preshuffle_pipeline.py`（fused gemm1 用其 preshuffle helpers；dev 加了 `make_preshuffle_b_layout_packed4bit` 须保留）
- `kernels/mfma_epilogues.py`、`kernels/layout_utils.py`、`kernels/kernels_common.py`、`kernels/topk_gating_softmax_kernel.py`

### E. 明确**不合**（实验/脚手架）
- `tmp_test/`（整目录）、`kernels/tmp_mega_*.py`（符号链接）、`kernels/megamoe_exp.py`
- `docs/moe_stage1_mega_notes.md`（中文实验日志）

### F. dev 独有、保持不动（ghu 也未触及或我们不取 ghu 版）
- `kernels/a8w4_moe_gemm_2stage.py`（dev 新文件，无冲突，保留）
- `kernels/pa_decode_swa.py`（dev 新文件；注意 ghu 也有同名文件 → add/add，**取 dev 版**，见 §5.5）
- ghu-only 的 gfx1250/blockscale 重写（`moe_blockscale_2stage`、`*_gfx1250.py`、`silu_and_mul_fq`、`preshuffle_gemm`）：
  **决策已定（2026-06-30）：逐个回退成 dev 版**。这 12 个文件 dev==base（dev 从未改过），
  所以"回退成 dev 版" = checkout merge-base 版本 = 把 ghu 对它们的全部改动 revert 掉。**细化见新增 §9。**

---

## 2. megaMoE 依赖闭包图（精简）

```
MegaMoE (fused_moe_stage1_stage2)
├── FusedMoEMegaStage1 (fused_moe_megakernel)        [mori.shmem + dist]
│   ├── FlyDSLDispatchGroupMajorOp (ep_dispatch_groupmajor_op)  [mori.shmem]
│   └── compile_fused_moe_gemm1 (fused_moe_gemm_2stage)         [mori.ir.flydsl]
│         └─ helpers: mfma_preshuffle_pipeline / mfma_epilogues / layout_utils / kernels_common
├── FlyDSLMoeGemm2CombineOp (mixed_moe_gemm2_combine_fused_op)
│   ├── FlyDSLDispatchCombineIntraNodeOp (dispatch_combine_intranode_op)  [mori.shmem]
│   └── compile_fused_moe_gemm2_combine (mixed_moe_gemm2_combine_fused)
│         └─ compile_mixed_moe_gemm2  ← kernels/mixed_moe_gemm_2stage.py [ghu 版]
└── (sort 融合进 dispatch；不依赖 moe_sorting_kernel)

外部硬依赖：mori (shmem + ir.flydsl)、软依赖：aiter
in-kernel extern → flydsl.expr.extern.ExternFunction → compiler/{extern_link,external_llvm,link_utils}
                  → 钩子嵌于 jit_function.py / kernel_function.py
```

环境已确认：`mori`、`aiter` 已安装；当前 dev 上 `mori.shmem` 可导入，但 `mori.ir.flydsl` 因缺
`flydsl.expr.extern.ExternFunction` **导入失败** → §3 解决。

---

## 3. 核心库（compiler/expr）落地细节

> 这是本次合并最难、最危险的部分。按"反转方向"，这些**直接是 ghu 版**，
> 只需在其上**回填 dev 的 2 处加法**（§4.2、§4.3）。按"正向"则需整块移植，强烈不建议。

| 文件 | 状态 | 处理 |
|---|---|---|
| `python/flydsl/expr/extern.py` | ghu 新增 | 取 ghu。仅依赖 `_mlir.ir`、`llvm`、`meta.dsl_loc_tracing` |
| `python/flydsl/expr/meta.py` | ghu 改 206 行（含 `dsl_loc_tracing`） | 取 ghu（dev 未改） |
| `python/flydsl/compiler/extern_link.py` | ghu 新增 | 取 ghu |
| `python/flydsl/compiler/external_llvm.py` | ghu 新增（`external_llvm_fingerprint`、`run_external_binary_codegen`） | 取 ghu |
| `python/flydsl/compiler/link_utils.py` | ghu 新增 | 取 ghu |
| `python/flydsl/compiler/jit_function.py` | ghu 重写 1212 行；含 extern-link 钩子（行 38/400/869/911/1181/1406/1512-1558） | 取 ghu（dev 未改） |
| `python/flydsl/compiler/kernel_function.py` | ghu 改 325 行；`CompilationContext` 承载 extern module_init | 取 ghu（dev 未改） |
| `python/flydsl/compiler/protocol.py`、`jit_argument.py`、`jit_executor.py`、`backends/*` | ghu 重写 | 取 ghu（jit 子系统整体一致性需要） |
| `python/flydsl/expr/__init__.py` | ghu 改 28 行（导出 extern 等） | 取 ghu |
| 其余 expr 核心（`typing/primitive/struct/numeric/math/gpu/vector/derived/utils/arith`、`rocdl/*`、`smem_allocator`、`runtime/*`） | ghu 大改 | 反转方向下=ghu 版底座；正向下需评估 dev 优化兼容性（见 §6 风险） |

**dsl_loc_tracing 验证**：合并后 `python -c "from flydsl.expr.meta import dsl_loc_tracing"` 必须成功，
否则 `extern.py` 及全仓被 `@dsl_loc_tracing` 装饰的 op 都会 import 失败。

---

## 4. dev 优化保护清单（必须在合并后仍存在 —— 逐项 grep 验证）

> 这是"不回退"的验收标准。按反转方向，这些需**重新 apply 到 ghu 底座**；逐项可 grep。

### 4.1 `kernels/moe_gemm_2stage.py`（核心，见 §5.1 逐 hunk）
- LDS ping/pong 物理双 buffer：`allocator_pong / allocator_ping`、`global_sym_name="smem0"/"smem1"`、
  `lds_x_pong / lds_x_ping`、`_per_buf_pong_bytes / _per_buf_ping_bytes`
- 异步拷贝：`use_async_copy`、`dma_x_tile_to_lds`、`prefetch_x_to_lds`、`rocdl.raw_ptr_buffer_load_lds`、
  `rocdl.s_waitcnt(8)` / `s_waitcnt(0)`、cache-key `_async` tag
- gfx950 K64 MFMA 调度 + `hot_loop_scheduler` / `_build_scheduler`、`num_a_async_loads`
- 占用控制：`waves_per_eu`（`compile_moe_gemm1/2`、`compile_moe_gemm2_ex` 新增 kwarg，默认值勿翻）
- helper 签名：`store_x_tile_to_lds(vec, lds_buf, ...)`、`lds_load_packs_k64(..., lds_buf)`、`compute_tile(..., lds_buf, ...)`

### 4.2 `python/flydsl/expr/buffer_ops.py`（加法，§5.2）
- `extract_aligned_ptr(tensor, address_space=1)`
- `extract_workgroup_aligned_ptr(memref, address_space=3)`（异步 DMA provenance 关键，回退**不报错只掉性能**）

### 4.3 `python/flydsl/expr/rocdl/__init__.py`（加法，§5.3）
- `mfma_i32_16x16x64_i8(result_type, operands, *, loc, ip)`

### 4.4 `kernels/mfma_preshuffle_pipeline.py`
- `make_preshuffle_b_layout_packed4bit(...)` + `PreshuffleBLayout`（a8w4smooth 依赖）

### 4.5 `kernels/mixed_moe_gemm_2stage.py`
- 2 处 `rocdl.s_setprio(1)/(0)` **保持注释掉**（dev 的有意调优，别在解冲突时"手贱修回来"）

### 4.6 dev 新文件（原样保留）
- `kernels/a8w4_moe_gemm_2stage.py`、`kernels/pa_decode_swa.py`、`tests/kernels/test_pa_swa.py`、
  `tests/kernels/test_a8w4_moe_gemm_2stage.py`、`tests/utils.py` 的 `shuffle_weight(interleave_k64=...)`

> ⚠️ **隐性回退风险**：异步路径是"separate smem0/smem1 + provenance 指针 + s_waitcnt 同步"三件套，
> 缺任一项不会报错、只会让 AA 重新插入 wait → **功能测试发现不了，只能靠 benchmark**（`bench_moe.sh`/`bench_moe_prefill.sh`）。

---

## 5. 真冲突文件逐文件解决

### 5.1 `kernels/moe_gemm_2stage.py` —— 重度，28 hunk（唯一高危）
- 现象：双方都重写了核心 compute loop（gemm1 区 ~L293–1629、gemm2 区 ~L2367–3633，故 hunk 近似成对出现）。
  - dev 侧：异步 `buffer_load_lds` DMA、ping/pong 独立 memref、gfx950 16x16x64 I8、`hot_loop_scheduler`、provenance LDS 指针。
  - ghu 侧：寄存器 load + `store_x_tile_to_lds` 前导、较简调度 hint、`int4_bf16_single_field`、`persist_m`、reduce-mode。
- 解决原则：
  - 以 **dev 的流水骨架为基准**（异步 + 独立 ping/pong + 16x16x64 + scheduler 全保留）。
  - 把 ghu 的 MoE 特性**叠加**上去：`int4_bf16_single_field`、`persist_m`、reduce-mode 分支。
  - load/sched/LDS-alloc 三个区域的任何 hunk **不得取 ghu 侧**。
- 反转方向下等价操作：在 ghu 的 `moe_gemm_2stage.py` 上重放 dev 的 28-hunk 优化（推荐用 dev commit 的 patch 作参照逐段 apply）。
- 验收：解完后 grep 确认 §4.1 全部符号在 `compile_moe_gemm1` 和 `compile_moe_gemm2` 两处都在。

### 5.2 `python/flydsl/expr/buffer_ops.py` —— 1 hunk，加法
- dev 插入 2 个函数的位置 与 ghu 插入 `@dsl_loc_tracing` 装饰相邻 → **两边都留**（dev 的 2 函数 + ghu 的装饰/重构）。

### 5.3 `python/flydsl/expr/rocdl/__init__.py` —— 1 hunk，加法
- dev 的 `mfma_i32_16x16x64_i8` 紧邻 ghu 重新装饰的 `mfma_f32_16x16x32_f16` → **两个函数都留**。

### 5.4 `tests/kernels/test_moe_gemm.py` —— 9 hunk，参数并集
- dev 加 `waves_per_eu` / `tile_m2`；ghu 加 `stage1_persist_m` / `stage2_persist_m` + `_make_reduce_mode_compile_fn`。
- 每个签名/调用点取**参数并集**即可。

### 5.5 `kernels/pa_decode_swa.py` —— add/add 冲突（与 MoE 无关）
- 双方都新增了同名文件。本次合并**取 dev 版**（dev 的 PS-SWA 实现，§4.6），除非确认 ghu 版更新（需 diff 比对，见 §8）。

---

## 6. 兼容性风险（反转方向下需重点验证）

1. **dev 优化 × ghu 演进核心**：dev 的 ping/pong 用 `SmemAllocator`，而 ghu 改了 `smem_allocator.py`（+117 行）。
   重放 dev 优化时须对齐 ghu 的 `SmemAllocator` API（`_align`、`finalize`、`global_sym_name` 等）。
2. **`extract_workgroup_aligned_ptr` 依赖的 `fly` 方言绑定**：检查 `lib/Bindings/Python/FlyExtension.cpp` /
   `extract_aligned_pointer_as_index` 在合并后底座可用（dev 与 ghu 的 C++ 绑定是否一致）。
3. **`mfma_i32_16x16x64_i8` ODS**：依赖底层 MLIR/ROCDL 已注册该 op；ghu 底座（embedded MLIR）须支持 gfx950。
4. **disk-cache × extern**：ghu 的 jit_function 对 extern-linked kernel 跳过 disk cache（`_extern_linkage_keys`）。
   迭代 megaMoE 时按 CLAUDE.md 设 `FLYDSL_RUNTIME_ENABLE_CACHE=0`。
5. **`kernels/__init__.py` 不可自动 import mori 依赖模块**（否则无 mori 环境集体 import 失败）。当前 `__init__` 不自动导入，保持。

---

## 7. 执行步骤（反转方向，推荐）

> 在 worktree 中操作，避免污染当前分支。

1. **建底座**：从 `ghu_moe_stage1` 建工作分支（含全部 core+EP+megaMoE）。
2. **删脚手架**：移除 §1.E（`tmp_test/`、`tmp_mega_*` 链接、`megamoe_exp.py`、`*_notes.md`）。
3. **回填 dev 加法核心**（§4.2/4.3/4.4）：在 ghu 的 `buffer_ops.py`、`rocdl/__init__.py`、`mfma_preshuffle_pipeline.py`
   上加回 dev 的新函数；`mixed_moe_gemm_2stage.py` 确认 `s_setprio` 注释态。
4. **重放 moe_gemm_2stage 优化**（§5.1）：以 dev commit 为 patch 参照，将 28-hunk 优化逐段 apply 到 ghu 版。
5. **带回 dev 新文件**（§4.6）：`a8w4_moe_gemm_2stage.py`、`pa_decode_swa.py` 及对应测试、`tests/utils.py` 改动。
6. **测试并集**（§5.4）。
7. **范围确认**：gfx1250/blockscale 等 ghu-only 文件保留（反转底座自带）—— 与 §8 未决项一并确认。

### 验证清单
- 导入：`PYTHONPATH=./ python -c "from flydsl.expr.meta import dsl_loc_tracing; from flydsl.expr.extern import ExternFunction; import mori.ir.flydsl; print('extern OK')"`
- dev 优化 grep（§4.1–4.4 符号全部命中，且分布在 gemm1/gemm2 两处）。
- 功能：`PYTHONPATH=./ pytest tests/kernels/test_moe_gemm.py tests/kernels/test_a8w4_moe_gemm_2stage.py tests/kernels/test_pa_swa.py`
- 性能（异步路径未回退，关键）：`FLYDSL_MOE_USE_ASYNC_COPY=1` 跑 `bench_moe.sh` / `bench_moe_prefill.sh`，对比 dev 基线。
- megaMoE：在多卡 EP8 环境跑 `MegaMoE.forward` / `forward_bf16` 冒烟（需 mori 运行时）。

---

## 8. 未决项（需你拍板，影响最终范围）

1. **底座方向**：确认采用"反转方向"（以 ghu 为底回填 dev）。否则我按正向，但 §3 核心库需整块拖入、风险显著更高。
2. ~~ghu-only gfx1250/blockscale 重写~~ **已决：逐个回退成 dev 版**。细化与残留风险见 §9。
   ⚠️ 代价提醒：这等于**丢弃 ghu 在 gfx1250/blockscale 上的全部优化**（dev 无对应优化，回退后是 base 老代码）。
   若 MI450/gfx1250 是目标平台，需确认能接受性能/特性回退。
3. **`moe_sorting_kernel.py`**：megaMoE 主路径不用它（sort 融进 dispatch）。是否一并带入（供 baseline/对比）？
4. **`pa_decode_swa.py`**：dev 与 ghu 均有同名新文件，默认取 dev 版；是否需要先 diff 两版确认？
5. **C++ 绑定 / embedded MLIR**：ghu 底座的 `lib/` 与 embedded MLIR 是否需配套重新 `build_llvm.sh`+`build.sh`？
   （§6.2/6.3 依赖）

---

## 9. gfx1250/blockscale 回退细化（决策 #2，2026-06-30 落实）

> 决策：在 ghu 底座上，把下列 12 个文件回退成 dev 版。已逐项核查三方状态与依赖闭包。

### 9.1 回退清单（全部 dev==base，即 checkout merge-base 版本）
| 文件 | ghu vs base | 性质 |
|---|---|---|
| `kernels/moe_blockscale_2stage.py` | +1003/−1084 | 近乎重写 |
| `kernels/moe_gemm_2stage_mxscale_gfx1250.py` | +1536/−368 | 重写 |
| `kernels/gemm_fp8fp4_gfx1250.py` | +1796/−792 | 重写 |
| `kernels/preshuffle_gemm.py` | +467/−235 | 重写 |
| `kernels/moe_gemm_2stage_common_gfx1250.py` | +511/−104 | 重写 |
| `kernels/silu_and_mul_fq.py` | +361/−255 | 重写 |
| `kernels/wmma_gemm_gfx1250.py` | +234/−247 | 重写 |
| `kernels/blockscale_preshuffle_gemm.py` | +152/−132 | 改写 |
| `kernels/moe_gemm_2stage_wmma_gfx1250.py` | +96/−60 | 改写 |
| `kernels/preshuffle_gemm_v2.py` | +51/−46 | 改写 |
| `kernels/gemm_common_gfx1250.py` | +26/−20 | 小改 |
| `kernels/rdna_fp8_preshuffle_gemm.py` | +13/−6 | 小改 |

> 12 个全部 `dev vs base = 无变更`，所以"取 dev 版"在 git 操作上就是 `git checkout <merge-base> -- <file>`（或从 dev 分支 checkout）。

### 9.2 三条关键结论（已验证）
1. **megaMoE 闭包不依赖这 12 个文件**（grep 闭包 import 全空）→ 回退不会打断 megaMoE。
2. **必须整簇原子回退,不可逐个**：这 12 个文件互相 import（如 `moe_blockscale_2stage` →
   `gemm_common_gfx1250` / `moe_gemm_2stage_common_gfx1250`,后两者也在回退集内）。
   只回退一部分会造成"base 调用方 × ghu 被调方"签名错配。**12 个一起回退,簇内自洽。**
3. **簇→ghu 核心库的 import 面在 ghu 底座上仍可解析**（之前 grep `__init__.py` 报的"缺失"是假阳性）：
   - `const_expr / range_constexpr / idx2crd`：来自 ghu `expr/__init__.py` 的 `from .primitive import *`（primitive `__all__` 已含）。
   - dtype（`Float16/Float8E4M3FNUZ/Int8/...`）：ghu `typing.py` 的 `__all__` 仍导出。
   - legacy `buffer_ops.create_buffer_resource / buffer_load / buffer_store`：ghu 仍保留（394/432/528 行）。
   - `rocdl.sched_barrier / s_waitcnt / s_wait_dscnt / sched_*`：来自 `from ..._mlir.dialects.rocdl import *`（生成的 MLIR 绑定）,
     当前已 build 的 `_mlir` 中 `hasattr` 实测为 True。**grep Python 源码找不到属正常**。

### 9.3 残留风险（grep 到此为止,需 trace/编译验证）
- **簇→保留的 ghu helper 的签名兼容**:base 版 12 文件还 import 了**保留为 ghu 版**的共享 helper——
  `mfma_preshuffle_pipeline`（ghu +105/−33）、`mfma_epilogues`（−47）、`pipeline_utils`（−18）、
  `kernels_common`、`layout_utils`。这些 helper 的**符号名在 ghu 版均存在**
  （`make_tail_plan`、`tdm_epilogue_fence_threshold_bytes`、`mfma_epilog`、`c_shuffle_epilog`、
  `default_epilog`、`get_warp_size` 全部命中）,**但函数体/签名 ghu 改过**。
  → 名级可过,**形参/语义级兼容只能靠实际 trace 这些 kernel 验证**,grep 验证不了。
- **簇→§5.1 的 `moe_gemm_2stage`**:部分 gfx1250 文件 `from kernels.moe_gemm_2stage import (...)`。
  §5.1 合并后的 `moe_gemm_2stage` 必须仍导出这些符号(回退后须 grep 确认)。

### 9.4 回退执行（并入 §7 步骤之间,建议放在第 4 步之后）
```bash
# 在 ghu 底座 worktree 中
BASE=$(git merge-base dev/universe_yadai ghu_moe_stage1)   # = 8eec98bc
for f in kernels/moe_blockscale_2stage.py kernels/moe_gemm_2stage_mxscale_gfx1250.py \
         kernels/gemm_fp8fp4_gfx1250.py kernels/preshuffle_gemm.py \
         kernels/moe_gemm_2stage_common_gfx1250.py kernels/silu_and_mul_fq.py \
         kernels/wmma_gemm_gfx1250.py kernels/blockscale_preshuffle_gemm.py \
         kernels/moe_gemm_2stage_wmma_gfx1250.py kernels/preshuffle_gemm_v2.py \
         kernels/gemm_common_gfx1250.py kernels/rdna_fp8_preshuffle_gemm.py; do
  git checkout "$BASE" -- "$f"      # 整簇一起回退
done
```
验证:
- import:`PYTHONPATH=./ python -c "import kernels.moe_blockscale_2stage, kernels.gemm_fp8fp4_gfx1250, kernels.preshuffle_gemm"`（无 ImportError/AttributeError）。
- trace:对每个回退文件的入口跑一次最小 compile（gfx1250/gfx950 各自 arch),确认 §9.3 的 helper 签名兼容。
- 若 trace 报 helper 签名不符 → 该 helper 需要"base 调用约定 ↔ ghu 实现"的小适配垫片,而非再回退 helper(helper 是 megaMoE 共享依赖,不能回退)。

---

## 10. 执行记录与状态（分支 `universe_ghu_tmp`，底座 `ghu_moe_stage1@5b9fef47`）

### 10.1 已完成步骤（均已 import/语法验证）
1. **删脚手架**：`tmp_test/`、`kernels/tmp_mega_*`、`megamoe_exp.py`、`moe_stage1_mega_notes.md`（确认无保留代码依赖后 `git rm`）。
2. **回填 dev 加法核心**：
   - `buffer_ops.py`：`extract_aligned_ptr` / `extract_workgroup_aligned_ptr`（加 `@dsl_loc_tracing` 匹配 ghu 约定）。
   - `rocdl/__init__.py`：`mfma_i32_16x16x64_i8`（**按 ghu 签名适配**，非照搬 dev：`(result_type, operands)` + `@dsl_loc_tracing` + None-guard）。
   - `mfma_preshuffle_pipeline.py`：仅加 `make_preshuffle_b_layout_packed4bit`（`PreshuffleBLayout` ghu 已有且相同，不重复定义）。
   - `mixed_moe_gemm_2stage.py`：2 处 `s_setprio(1)/(0)` 注释掉（§4.5）。
3. **回退 gfx1250/blockscale 12 文件**到 base（§9）。import 全过，**§9.3 残留风险在 import 级已消解**。
4. **PA 簇决策（§8.4 / 用户定:取 dev 版）**：取 dev 整个 PA 簇——`pa_decode_swa` + `pa_decode_fp8` + `test_pa` + `test_pa_swa`。
   原因:dev `pa_decode_swa` 有独有 ps-swa(#468)+ aiter 依赖,与 ghu(已 drop aiter、新签名)双向冲突;dev `pa_decode_fp8` 不依赖
   `pa_decode_swa`/`pa_metadata`,取 dev 簇自洽。ghu `pa_metadata.py` 取后无人 import(孤立无害,保留)。
5. **`moe_gemm_2stage` 决策（用户定:按 1,取 dev 整文件）**：已**严密验证 megaMoE 传递闭包(14 模块)不依赖 `moe_gemm_2stage`**;
   megaMoE 的 gemm2 复用的是 `mixed_moe_gemm_2stage`(独立文件)。dev 版是超集(async 流水 + int4_bf16 特性 + 全 API),
   取 dev 整文件,丢弃 ghu 在此文件的 4 个核心库集成 commit(mark_static/make_rmem_tensor/type-closure/aiter-port)。
6. **测试对齐**:`test_moe_gemm`/`test_moe_reduce`/`test_moe_gemm_{mxscale,wmma}_gfx1250` 取 dev 版(匹配各自 kernel 版本);
   `test_profiler_moe_gemm2_combine` 保留 ghu(测 megaMoE gemm2_combine);`tests/utils.py` 三方干净合并(dev `interleave_k64` + ghu import 风格)。

### 10.4 moe_gemm_2stage #552 移植 + async 路径说明（2026-06-30）
- **同 a8w4 的 #552 问题**:dev 整文件取来的 `moe_gemm_2stage` 主路径 `compile_moe_gemm1/2` 也用裸
  `fx.idx2crd` + 非规范化 `crd2idx`,在 ghu 核心 `fly-layout-lowering` 报 `extsi i64→i32`。
- **修复(镜像 ghu 自己 moe_gemm_2stage 的 #552 写法)**:9 处 idx2crd/crd2idx 的 idx/coord 包 `fx.Int32(...)`
  (`coord_wl/coord_l16/coord_gate/coord_up/coord_w` + 2 处 `idx_a16`),逐字对齐 ghu 版。
- **验证(gfx942)**:`test_moe_gemm.py` fp8-S / fp16-S / int8-S / int4_bf16-S(f16&bf16)**全部通过**,
  `test_moe_reduce` 10/10。default 路径 OK。
- **async 路径(`FLYDSL_MOE_USE_ASYNC_COPY=1`)**:my #552 修复让它越过 layout-lowering,但在 **LLVM 后端**报
  `Do not know how to expand this operator's operand`(`llvm.amdgcn.raw.ptr.buffer.load.lds`,16B s128)。
  根因:async DMA 用 `_dma_bytes=16` 的 16 字节 buffer_load_lds,是 **gfx950(CDNA4)特性**;gfx942 后端无法 lower。
  → async 是 **opt-in 的 gfx950 优化,在 gfx942 上本就不启用**,当前硬件无法验证,**非合并回归**。需 gfx950 环境验证。

### 10.2 验证结果（已 build `_mlir`,gfx942/MI300X 环境）
- ✅ 核心 + extern + **`import mori.ir.flydsl` 成功**(EP 启用的关键链路,之前在 dev 上缺 extern 失败)。
- ✅ 所有改动/回退/新增 kernel **import 全过**(含 dev `moe_gemm_2stage`、回退的 gfx1250 簇、a8w4、PA 簇)。
- ✅ `§4` 不回退 grep 验收全过(async 符号、3 backfill 函数、s_setprio 0激活/2注释、dev 新文件)。
- ✅ **功能测试 `test_moe_reduce` 10/10 通过**(端到端编译+运行 `compile_moe_reduction` 在 ghu 核心上)。
- ⚠️ 环境限制:`fused_moe_stage1_stage2` / `test_moe_gemm` import 失败于 **aiter 环境问题**
  (`per_1x32_mx_quant_hip` 缺失、`module_aiter_core.so` JIT 失败)——文件零改动,**非回归**,原始分支同环境亦失败。
- ⚠️ gfx1250 测试在 gfx942 上正常 skip(arch 门控)。

### 10.3 ✅ a8w4 codegen 回归（已修复，2026-06-30）
- **现象**:`test_a8w4_stage1` 编译期 LLVM 断言 `APInt::sext: Width >= BitWidth`,定位在
  `fly-layout-lowering` pass(gfx942 走 K=32 INT8 路径,与 K64 mfma 无关)。
- **根因**:a8w4(dev kernel)用裸 `fx.idx2crd` + `mfma_preshuffle_pipeline.crd2idx`,这两个**不规范化坐标/索引类型**。
  ghu 核心 #552(type closure)更严格:`tx=gpu.thread_id`(index)经裸 `fx.idx2crd` 变 i64,坐标(i64/index)
  与 i32 layout stride 混用 → `fly-layout-lowering` 生成非法 `extsi i64→i32`。ghu 自己的 kernel 用
  `layout_utils.{crd2idx,idx2crd}`(全程 index 计算 + 末端规范化到 i32)所以不崩——a8w4 是 dev-only,从没接受 #552 适配。
- **修复(仅改 `kernels/a8w4_moe_gemm_2stage.py`,megaMoE 零影响)**:
  - `crd2idx`/`idx2crd` 改从 `kernels.layout_utils` 导入(替换 `mfma_preshuffle_pipeline.crd2idx` 和裸 `fx.idx2crd`)。
  - 7 处 `fx.idx2crd`→`idx2crd`;对应 `fx.get(coord, i)`→`coord[i]`(layout_utils.idx2crd 返回 list)。
- **验证**:`pytest tests/kernels/test_a8w4_moe_gemm_2stage.py` → **5/5 通过**(含参考数值对比)。
- **意义**:这是"dev kernel × ghu 核心"需 #552 适配的范例;若后续发现其它 dev kernel 同类崩,按此法(切 layout_utils 助手)修。
