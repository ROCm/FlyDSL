# MoE Intranode Stage1 A8W4（元数据 + P2P Gather + 融合 GEMM1）

本文档说明本次在 **gfx950（如 MI355X）** 上新增的 **intranode A8W4 stage1** 相关 kernel、**功能与精度测试**覆盖范围，以及 **性能测试**脚本与用法。

---

## 一、新增与相关模块（Kernel）

| 路径 | 作用 |
|------|------|
| `kernels/recv_meta_a8w4.py` | `recv_meta` 行布局、pack/unpack 与排序等主机侧辅助。 |
| `kernels/moe_metadata_dispatch_recv_meta.py` | 仅做 **metadata dispatch** 的 FlyDSL JIT（与 MoE 路由/shmem 拓扑配合）。 |
| `kernels/moe_peer_gather_mxfp8_a8w4.py` | 独立 **MX-FP8 行级 peer gather**（按 `recv_meta` + P2P 指针表从远端 PE 拉行）。 |
| `kernels/moe_fused_dispatch_gather_gemm1_a8w4.py` | **融合 kernel**：可选 **intranode peer gather 前导**（recv_meta + P2P gather + 栅格屏障）+ 与 `mixed_moe_gemm_2stage` 对齐的 **A8W4 stage1 MFMA** 路径。 |
| `kernels/moe_fused_complete_intranode_a8w4.py` | **完整 stage1 编排**：metadata JIT + 上述 fused GEMM1（两段子程序在同一 pipeline 内顺序执行）。 |
| `kernels/moe_intranode_metadata_stage1_a8w4.py` | 对外的薄封装 / 说明入口（与 stage1 命名对齐）。 |

**说明**：`test_moe_gemm.py` 中 `test_moe_stage2_standalone` 的 **large_shape DS/EP prefill** 用例已将 `tile_k` 调整为 **256**，以满足 **a8w4** 路径对 `tile_k ≥ 256` 的约束；decode 等小 `tile_k` 用例对 a8w4 仍会 **skip**（预期行为）。

---

## 二、功能与精度测试（已完成范围）

### 2.1 Intranode 元数据、recv_meta、peer gather、融合编译

- **文件**：`tests/kernels/test_moe_intranode_metadata_stage1_a8w4.py`
- **覆盖要点**：`recv_meta` 往返与参数化、按行排序与 multiset、**peer gather** 恒等/散布精度、**fused intranode** 多组 shape **编译**、MXFP8 peer gather 编译与运行、metadata dispatch **JIT smoke** 等。

```bash
cd /path/to/FlyDSL
python -m pytest tests/kernels/test_moe_intranode_metadata_stage1_a8w4.py -v --tb=short
```

### 2.2 多卡跨 rank 元数据与路由一致性

- **文件**：`tests/kernels/test_moe_intranode_metadata_multigpu.py`（`@pytest.mark.multi_gpu`）
- **要求**：子进程 `torchrun`；**4 卡**需至少 4 张 GPU，**8 卡**需至少 8 张；成功时日志中含跨 rank 校验通过信息。

```bash
python -m pytest tests/kernels/test_moe_intranode_metadata_multigpu.py -m multi_gpu -v --tb=short
```

### 2.3 A8W4 Stage2 单测（与 `test_moe_gemm` 大 shape 对齐）

- **文件**：`tests/kernels/test_moe_gemm.py`，用例：`test_moe_stage2_standalone`，**`-k a8w4`**
- **覆盖**：含 **large_shape** 的 DS/EP prefill（`tile_k=256` 修正后）、FP4-bench-S/M/L；DS/EP **decode** 因 `tile_k`/`tile_m` 与 a8w4 约束不符会 **skip**。

```bash
python -m pytest tests/kernels/test_moe_gemm.py::test_moe_stage2_standalone -k a8w4 -v --tb=short
# 仅 large_shape：
python -m pytest tests/kernels/test_moe_gemm.py::test_moe_stage2_standalone -k a8w4 -m large_shape -v --tb=short
```

### 2.4 一次跑完 intranode + 多卡（推荐）

```bash
python -m pytest \
  tests/kernels/test_moe_intranode_metadata_stage1_a8w4.py \
  tests/kernels/test_moe_intranode_metadata_multigpu.py \
  -v --tb=short
```

**环境**：需 ROCm、`torch` CUDA/ROCm、`mori.shmem`；多卡测试需本机 GPU 数满足标记要求。

---

## 三、性能测试（范围与脚本）

### 3.1 Stage1：baseline vs fused vs complete vs meta-only

- **脚本**：`tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py`
- **指标**：各 rank 本地 **`torch.cuda.Event`** 计时得到 **ms/iter**，再对 **`world_size` 维做 `MAX`**（保守、对齐多卡慢 rank）。
- **路径语义**：
  - **baseline**：`make_metadata_dispatch_recv_meta_jit` + `compile_peer_gather_mxfp8_rows` + `compile_mixed_moe_gemm1`（stage1）。
  - **fused**：metadata + `compile_fused_dispatch_gather_gemm1_a8w4`（`world_size>1` 时打开 intranode peer gather）。
  - **complete**：`compile_complete_intranode_fused_stage1_a8w4`（metadata + fused GEMM1 的另一条 JIT 编排）。
  - **meta_only**：仅 metadata dispatch。

**必须用 `torchrun`**（mori + `torch.distributed`）。

```bash
export MORI_SHMEM_HEAP_SIZE=16G   # 按机器调整

# 单卡
torchrun --standalone --nproc_per_node=1 \
  tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py \
  --tokens 4 --model-dim 256 --inter-dim 128 --experts 8 --topk 2 \
  --tile-m 32 --tile-n 64 --tile-k 256 --iters 20 --warmup 5

# 多卡（每 PE 一张卡）
torchrun --standalone --nproc_per_node=8 \
  tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py \
  --experts 8 --tokens 4 --model-dim 256 --inter-dim 128
```

**可选参数**：

| 参数 | 含义 |
|------|------|
| `--skip-baseline-timing` | 不编译、不测 baseline；JSON 里 baseline 为 `null`，用于 baseline 在 GPU 上 fault 后 **新开进程** 只测 fused/complete/meta。 |
| `--json-out PATH` | rank0 将摘要 JSON 写入文件。 |
| `--trace` | 各 rank 打时间戳日志，便于排查卡死。 |

**诊断**：`FLYDSL_BENCH_FORCE_NO_PEER_GATHER=1` 时 fused/complete 可编译为 **无 in-kernel peer gather**（语义非多卡 MoE 正确路径，仅用于确认 hang 是否在前导 gather）。

### 3.2 全矩阵采集（多 shape × 1/4/8 卡）

- **脚本**：`tests/kernels/collect_stage1_a8w4_perf_matrix.py`
- **行为**：对每个预设场景先 **完整 bench**；若失败或无 JSON，再 **`torchrun --skip-baseline-timing`** 重跑，保证 **fused/complete/meta** 尽量有数；baseline 失败在结果里为空（`null`）。

```bash
export MORI_SHMEM_HEAP_SIZE=16G
export COLLECT_ITERS=6          # 可选
export COLLECT_WARMUP=3         # 可选
export COLLECT_OUT=/tmp/stage1_a8w4_perf_matrix.jsonl
python tests/kernels/collect_stage1_a8w4_perf_matrix.py
```

### 3.3 Peer gather 微基准（单进程）

- **脚本**：`tests/kernels/bench_moe_intranode_metadata_stage1_a8w4.py`
- **指标**：`time_ms`（`perf_counter` + 前后 `cuda.synchronize`）；`eff_GB/s` 为脚本按 `max_recv * model_dim * 2 / dt / 1e9` 估算的 **有效搬运量**（粗算，非硬件计数器）。

```bash
python tests/kernels/bench_moe_intranode_metadata_stage1_a8w4.py \
  --model-dim 7168 --max-recv 4096 --iters 50 --warmup 10
```

### 3.4 参考性能数据（采集快照）

以下为 **`collect_stage1_a8w4_perf_matrix.py`** 一次完整跑出的 **JSONL 汇总**（硬件：**8×MI355X / gfx950**；stage1 bench：**`COLLECT_ITERS=6`、`COLLECT_WARMUP=3`**；时间为 **ms/iter**，多卡为各 rank **`MAX`**）。**baseline 为空** 表示该配置下整段 baseline 未成功计时（常见为多卡 **HIP illegal address**），已用 **`--skip-baseline-timing`** 另起进程补测 **fused / complete / meta**（`mode = skip_baseline_retry`）。**b/f** = baseline ÷ fused（仅 baseline 有值时）。

**`run_tok`**：与 bench JSON 中 `tokens` 一致，即 `max(命令行 --tokens, world_size)`。

| scenario | ws | run_tok | md | baseline | fused | complete | meta | b/f | mode |
|----------|---:|--------:|---:|-----------:|------:|-----------:|-----:|----:|------|
| S_default | 1 | 4 | 256 | 0.2473 | 0.1886 | 0.1675 | 0.1110 | 1.311× | full |
| S_default | 4 | 4 | 256 | 0.4256 | 0.3263 | 0.3355 | 0.2409 | 1.304× | full |
| S_default | 8 | 8 | 256 | 0.5823 | 0.5508 | 0.3958 | 0.3050 | 1.057× | full |
| M_batch | 1 | 64 | 512 | 0.4561 | 0.3072 | 0.2853 | 0.2344 | 1.485× | full |
| M_batch | 4 | 64 | 512 | 0.8965 | 0.6590 | 0.6272 | 0.5504 | 1.360× | full |
| M_batch | 8 | 64 | 512 | 1.0479 | 0.8805 | 0.7198 | 0.6234 | 1.190× | full |
| D_decode | 1 | 1 | 512 | 0.3310 | 0.1837 | 0.1634 | 0.1077 | 1.802× | full |
| D_decode | 4 | 4 | 512 | 0.5014 | 0.3307 | 0.3282 | 0.2411 | 1.516× | full |
| D_decode | 8 | 8 | 512 | 0.6556 | 0.4973 | 0.4416 | 0.3148 | 1.318× | full |
| LARGE_M_FP4benchM | 1 | 512 | 4096 | 4.9221 | 3.1270 | 2.8022 | 2.6845 | 1.574× | full |
| LARGE_M_FP4benchM | 4 | 512 | 4096 |  | 10.5941 | 10.5794 | 10.3380 |  | skip_baseline_retry |
| LARGE_M_FP4benchM | 8 | 512 | 4096 |  | 21.3543 | 20.6777 | 20.6224 |  | skip_baseline_retry |
| LARGE_L_FP4benchL | 1 | 1024 | 4096 | 7.8288 | 5.6585 | 5.6574 | 5.5236 | 1.384× | full |
| LARGE_L_FP4benchL | 4 | 1024 | 4096 |  | 20.8189 | 20.6651 | 20.5497 |  | skip_baseline_retry |
| LARGE_L_FP4benchL | 8 | 1024 | 4096 |  | 40.6795 | 40.5101 | 40.6993 |  | skip_baseline_retry |
| WIDE64 | 1 | 64 | 4096 | 2.6664 | 0.5385 | 0.5136 | 0.4297 | 4.952× | full |
| WIDE64 | 4 | 64 | 4096 | 3.8606 | 1.6112 | 1.5935 | 1.4896 | 2.396× | full |
| WIDE64 | 8 | 64 | 4096 | 5.1600 | 2.8863 | 3.0218 | 2.9372 | 1.788× | full |
| WIDE192 | 1 | 192 | 4096 | 3.3447 | 1.2032 | 1.2062 | 1.1158 | 2.780× | full |
| WIDE192 | 4 | 192 | 4096 | 6.3596 | 4.1786 | 4.1258 | 3.9604 | 1.522× | full |
| WIDE192 | 8 | 192 | 4096 |  | 7.9325 | 7.9325 | 7.8333 |  | skip_baseline_retry |
| WIDE256 | 1 | 256 | 4096 | 3.6622 | 1.5390 | 1.5146 | 1.4412 | 2.380× | full |
| WIDE256 | 4 | 256 | 4096 | 7.7061 | 5.4153 | 5.4501 | 5.3065 | 1.423× | full |
| WIDE256 | 8 | 256 | 4096 |  | 10.3714 | 10.3712 | 10.2564 |  | skip_baseline_retry |
| WIDE384 | 1 | 384 | 4096 | 4.3900 | 2.1956 | 2.2214 | 2.0969 | 1.999× | full |
| WIDE384 | 4 | 384 | 4096 |  | 8.0213 | 7.9302 | 7.8543 |  | skip_baseline_retry |
| WIDE384 | 8 | 384 | 4096 |  | 15.6279 | 15.6153 | 15.5573 |  | skip_baseline_retry |
| WIDE512 | 1 | 512 | 4096 | 5.0081 | 2.8969 | 2.8788 | 2.7697 | 1.729× | full |
| WIDE512 | 4 | 512 | 4096 |  | 10.4562 | 10.4847 | 10.2639 |  | skip_baseline_retry |
| WIDE512 | 8 | 512 | 4096 |  | 20.4549 | 20.4524 | 20.1072 |  | skip_baseline_retry |

原始行存 **`COLLECT_OUT`** 指向的 JSONL（示例：`/tmp/stage1_a8w4_perf_matrix.jsonl`），字段含 `collect_scenario`、`collect_mode`、`skip_baseline_timing` 等，便于复现与对比。

**Peer gather 微基准**（`bench_moe_intranode_metadata_stage1_a8w4.py`，**iters=6、warmup=2**；同机 gfx950）：

| model_dim | max_recv | time_ms/iter | eff_GB/s |
|-----------|----------|--------------|----------|
| 1024 | 512 | 0.1707 | 6.144 |
| 4096 | 2048 | 1.6693 | 10.050 |
| 4096 | 4096 | 3.2950 | 10.183 |
| 7168 | 2048 | 3.2159 | 9.130 |
| 7168 | 4096 | 6.3789 | 9.205 |

---

## 四、注意事项

- 多卡 bench 首次 **FlyDSL JIT** 可能较慢；各 rank JIT 完成后脚本内会 **`dist.barrier`**，避免快慢 rank 交错进入 shmem 屏障。
- 若进程残留占用 GPU，可先结束对应 `torchrun` / bench（例如 `pkill -f bench_moe_intranode_stage1_fused_vs_split_a8w4`），再重跑。
- **baseline** 在部分 **大 M + 多卡** 配置下可能触发 **HIP illegal address**；此时用 **`--skip-baseline-timing`** 或采集脚本中的 **retry** 仍可得到 **新 kernel** 的 fused/complete 数据。
