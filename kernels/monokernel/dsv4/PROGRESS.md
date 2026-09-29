# DSV4-Pro mono-kernel：实现进展、验证与后续目标

更新：2026-09-29。当前版本已完成 **bs1、seq1/2/3/4、TP4/TP8 的单层功能验证**，并接入 ATOM。多 token 阶段调度已通过完整精度矩阵；本轮 36 项精度/性能测试全部通过。当前验收范围是单层 forward，整模型生成和 MTP accept/reject 仍待验证。

## 1. 代码与环境

| 项目 | 版本/位置 |
| --- | --- |
| FlyDSL 实现分支 | [codex/dsv4-a8w4-monokernel](https://github.com/ROCm/FlyDSL/tree/codex/dsv4-a8w4-monokernel) |
| FlyDSL 起点 | K3 PR #1204 head `21a3d1ee6ba2d6c0f33e3feb56e23cbf9e79c085` |
| ATOM 配套分支 | [codex/dsv4-flydsl-monokernel](https://github.com/ROCm/ATOM/tree/codex/dsv4-flydsl-monokernel) |
| ATOM 起点 | `aa0c5c3a131d98124726b41ab62342bb8cafb207` |
| AITER 验证源码 | `5e77166e0c627caa109046f6469de8220efa5ba6` |
| GPU/运行时 | MI355X / gfx950，FlyDSL 0.3.2，Torch 2.12.0 + ROCm 7.14.0 |
| Checkpoint | 原生 `DeepSeek-V4-Pro`，配置由 ATOM `DeepseekV4Args.from_hf_config` 读取 |

模型为 Pro：hidden=7168、expert intermediate=3072、384 routed experts、top6、1 shared expert、route scale=2.5、SwiGLU limit=10，前 3 层使用 hash routing。Routed 权重为 MXFP4/per32 E8M0，activation 为 FP8；shared 保留原生 FP8 权重/128×128 E8M0，router 为 BF16。

## 2. 当前接口与实现

```python
from kernels.monokernel.dsv4 import Dsv4MonoKernel

op = Dsv4MonoKernel(block, samples=4, layer_idx=block.layer_id,
                   rank=rank, npes=tp, group=tp_cpu_group)
state = op.forward(state, positions)
# 同一层的五阶段 MoE 对照：
state = op.forward(state, positions, unfused=True)
```

`block` 为已加载、已准备 mono adapter 的 ATOM `Block`。调用前须设置真实 ATOM forward context、input IDs 和 attention metadata，并绑定原生 cache。对照执行前必须恢复相同的输入和 cache；上面的两次调用只是接口示例。

每层 pipeline 为 **attention mHC → indexer/attention → FFN mHC → MoE**。CSA 的 indexer 在 attention 内部调用，避免重复写 cache。接口返回完整的 delayed `HCState`。每层一个 forward，内部多次 GPU launch；MoE 本身融合 router、top-k/hash、量化、routed/shared GEMM、combine 和本地 TP reduction，为一个 launch。

**五阶段对照是同一 FlyDSL MoE 的分阶段版本，不是 ATOM 默认的原生算子路径。** 它按 router→route/top-k→input quant→up（含激活/中间量化）→down（含 combine/TP）发出 5 次 launch，用于隔离融合和调度的影响。另一个对照 `--atom-module --atom-profile default` 才使用真实 ATOM 的默认 AITER 算子 dispatch；`stable-rne` 是显式替换 router/shared GEMM1 后的 ATOM 对照。性能结果必须注明是哪一类对照。

| 能力 | 当前状态 |
| --- | --- |
| bs1、seq1–4、local TP4/TP8 | 已验证；seq4 对应最多 MTP3 的验证长度 |
| TP1/TP2 | 接口接受用于诊断，未列入本次部署验收 |
| hash/bias、HCA/CSA、CUDA Graph 改输入 replay | 已验证 |
| 显式 `mono_kernel_forward` 的范围拦截 | cache 写入前拒绝范围外调用 |
| 普通 ATOM `Block.forward` | 支持范围内选择 mono，其余走原生 fallback |
| bs2–8、prefill、多请求、EP/DP/PP/PCP/DCP/TBO | 当前 mono 不支持 |
| 整模型生成、MTP accept/reject、长历史 context | 尚未验收 |

开启 `ATOM_DSV4_MONOKERNEL=1` 使用完整层接口；只开启 `ATOM_DSV4_MOE_MONOKERNEL=1` 可单独使用 MoE。两者默认关闭。权重在子模块 shuffle 前准备，并借用 ATOM 的 GU-interleaved routed Parameters，四个 query bucket 共用权重；各 bucket 独立持有 scratch、epoch 和 IPC。释放顺序是先结束所有 graph 使用，再集体 `close()`，最后销毁 TP process group。

### 本轮已落地的精度与调度改动

- Routed SwiGLU 保持 FP32，直接量化为 FP8，使用原生 fused exponent rule，移除额外 BF16 中间舍入；独立 oracle 同步对齐。
- Shared GEMM/activation 保留 BF16 RNE 边界。支持范围内的 main/index compressor 使用局部绑定的稳定 Torch BF16 projection，避免 atomic split-K 漂移传播到压缩 cache。
- 显式 stable/RNE ATOM 对照使用 Torch router、CK RNE shared GEMM1；配置单独生成并记录 SHA，不覆盖默认 AITER CSV。
- seq2–4 的 fused 路径将 router、selector、input quant、routed up、shared up、shared mid quant、down/TP exchange 的任务分别展开到全部 token，再按 CTA stride 分发。仍为 256 CTA×512 线程，无 grid barrier，使用原有 token-tagged scratch。
- seq1 和五阶段 reference 保留串行调度。调度依赖图、unique producer 检查和 24 组无 GPU 编译均通过，随后通过真实 GPU replay 验证。

关键代码：[op.py](op.py)、[moe.py](moe.py)、[moe_kernel.py](moe_kernel.py)、[atom_layer.py](atom_layer.py)。

## 3. 已完成的功能验证

本轮 `stage_batch_v1` 先完成 12 项精度门禁，再完成 24 项配对性能测试，共 **216 份 rank JSON、384 份 launch trace**。每个 shape 有 3 次改输入 replay；ATOM 对照对同一输入重复 3 次；完整层 staged 对照也重复 3 次。

| MoE | TP | Mono 对独立 oracle 最大 NRMSE |
| --- | --- | ---: |
| L0 hash | 4 | 0.378549% |
| L3 bias | 4 | 0.208479% |
| L0 hash | 8 | 0.373755% |
| L3 bias | 8 | 0.191013% |

Mono 与五阶段输出逐位一致；真实 ATOM module 的 captured mono 与 standalone mono 逐位一致；routing IDs exact，概率满足 rtol=2e-5/atol=2e-6。Stable/RNE ATOM 通过同一 1.5% oracle NRMSE 门限。

L0/L2/L3/L4×TP4/TP8 的完整层检查全部通过，bs1/seq1–4 的 **HCState NRMSE=0、cache backing byte 差异=0**。Fixture 使用 position128 的私有初始零 cache。这不能替代真实长历史、整模型或投机接受率验证。

提交源码的基本回归另外通过了 5 项单测，以及 L0/TP4、L3/TP8 的全部 seq1–4 MoE 功能与计时入口检查。回归发现并修复了 standalone MoE benchmark 在默认 stream 上捕获 Graph 的问题；现在由 Torch 选择非默认 capture stream，ATOM 仍使用自己的 collective capture stream。服务器回收时停止了后续完整层重复测试；完整层验收依据为上述已完成的 8 项矩阵。

默认 native ATOM 对照独立保留历史失败：L3/TP4 最大 oracle NRMSE 为 20.319961%，自身重复漂移最大 18.426175%。已定位的敏感边界包括 router BF16 atomic split-K 和 shared GEMM1 的舍入/归约。该结果没有用 stable/RNE 的通过结果覆盖；mono 的生产实现也不依赖测试用的全局 CSV override。

## 4. 测试脚本与重现命令

从 FlyDSL 实现 checkout 执行。以下路径须替换成实际安装位置；两仓使用上面的配套分支。

```bash
export DSV4_FLYDSL_ROOT=/path/to/FlyDSL
export DSV4_ATOM_ROOT=/path/to/ATOM
export DSV4_AITER_ROOT=/path/to/aiter
export DSV4_FLYDSL_RUNTIME=/path/to/flydsl032
export DSV4_MODEL_PATH=/path/to/DeepSeek-V4-Pro
export DSV4_RESULTS_DIR=/path/to/dsv4-results
export PYTHONPATH="$DSV4_FLYDSL_ROOT:$DSV4_ATOM_ROOT:$DSV4_AITER_ROOT:$DSV4_FLYDSL_RUNTIME"
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_V4_USE_TRITON_FUSION=0
export ATOM_DSV4_MONOKERNEL=1
mkdir -p "$DSV4_RESULTS_DIR"
cd "$DSV4_FLYDSL_ROOT"
```

如安装使用 ROCm SDK wheel，须将其目录设为 `ROCM_PATH`，并把 `$ROCM_PATH/lib/llvm/bin` 加入 `PATH`，确保 JIT 可以找到 `lld`。测试前确认所有需要使用的 GPU 空闲，运行中记录占用情况。

### 4.1 Stable/RNE ATOM 对照配置

使用实际 ATOM/AITER 部署选中的 merged CSV，下面是验证环境中的路径示例。TP4/TP8 分别生成 profile。

```bash
for tp in 4 8; do
  python3 -m kernels.monokernel.dsv4.tools.accuracy_config \
    --bf16-source /tmp/aiter_configs/bf16_tuned_gemm.csv \
    --shared-source /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
    --tp "$tp" --output-dir "$DSV4_RESULTS_DIR/stable-rne-tp$tp"
done
```

输出 manifest 保存原始/生成 CSV 的 SHA256。`--atom-profile default` 使用默认原生配置；`stable-rne` 必须显式传入对应 `--atom-config-dir`，两类结果分开报告。

### 4.2 MoE 功能矩阵：mono、五阶段、oracle、真实 ATOM module

```bash
for tp in 4 8; do
  for layer in 0 3; do
    torchrun --standalone --nproc-per-node="$tp" \
      -m kernels.monokernel.dsv4.tools.compare \
      --checkpoint "$DSV4_MODEL_PATH" --tp "$tp" --layer "$layer" \
      --batch-size 1 --seq-lens 1 2 3 4 --replays 3 \
      --atom-module --atom-samples 3 --atom-profile stable-rne \
      --atom-config-dir "$DSV4_RESULTS_DIR/stable-rne-tp$tp" \
      --output "$DSV4_RESULTS_DIR/moe-l$layer-tp$tp.json"
  done
done
```

L0 自动选择 hash，L3 自动选择 bias。`--atom-baseline` 只比较 ATOM operation path；`--atom-module` 还会验证真实模块的 post-load hook、借用权重和 mono dispatch。Checkpoint 模式会拒绝与真实层不符的 routing/shared-format 配置。

### 4.3 完整层功能矩阵：HCState、cache、重复执行

```bash
for tp in 4 8; do
  for layer in 0 2 3 4; do
    torchrun --standalone --nproc-per-node="$tp" \
      -m kernels.monokernel.dsv4.tools.monokernel \
      --checkpoint "$DSV4_MODEL_PATH" --tp "$tp" --layer-idx "$layer" \
      --batch-size 1 --seq-lens 1 2 3 4 --check \
      --replays 3 --baseline-repeats 3 \
      --output "$DSV4_RESULTS_DIR/layer-l$layer-tp$tp.json"
  done
done
```

每个 rank 写独立 `*.rankN.json`。两路均使用 CUDA Graph；每次执行前恢复相同初始 cache。对照为同一套稳定 attention/mHC 加五阶段 FlyDSL MoE。

### 4.4 可选性能和 launch profile

在以上命令添加 `--bench --warmup 5 --repeats 30 --graph-iters 10` 即可比较 mono 与 non-fused GPU 路径。完整层命令额外支持 `--profile-launches`，生成每 rank/shape/path 的 Chrome trace 和实际 kernel 数量。

性能比较使用 `--seed 43`、`44`、`45` 分别运行；每个 seed 取最慢 TP rank 的 median，再取三 seed 的 median。比较实现 A/B 时，按相同 seed、TP、layer 配对执行。使用 GPU events/CUDA Graph 计时，排除加载、packing、cache reset 和 TP setup。功能失败的 case 不计时。性能结果须附占用记录；发生外部 GPU 竞争的区间不作性能验收。

### 4.5 Shape guard、单测与诊断

```bash
# 应返回参数错误（exit 2），在 GPU 初始化前拒绝多请求。
python3 -m kernels.monokernel.dsv4.tools.compare \
  --batch-size 2 --seq-lens 1
python3 -m kernels.monokernel.dsv4.tools.monokernel \
  --checkpoint "$DSV4_MODEL_PATH" --batch-size 1 --seq-lens 5 \
  --output "$DSV4_RESULTS_DIR/unsupported.json"

python3 -m pytest -q tests/unit/test_dsv4_cache_observation.py \
  "$DSV4_ATOM_ROOT/tests/test_dsv4_monokernel_policy.py"
```

已检查两个入口各自对 bs2、seq0、seq5 的拒绝。Policy 单测可在 CPU 环境执行；cache 单测用 CPU tensor 检查真实字节布局，但 native ATOM decoder 的导入需要可见 gfx950 GPU，无对应硬件时显式 skip。

`tools.diagnose_accuracy --help` 提供 router、shared quant/GEMM/activation、routed 中间量化的分段误差诊断；`tools.checkpoint --help` 提供 checkpoint header/index 检查。诊断脚本用于定位失败，不能代替上述功能门禁。

## 5. 验证标准

1. 所有进程 exit 0；每个 rank 的 JSON 完整覆盖请求的 seq，所有 `passed` 为 true。不能只检查 rank0 或只看日志中最后一项。
2. Mono 与五阶段、module mono 与 standalone 要逐位一致；routing IDs exact，概率按固定容差比较。
3. 每个实现独立对 oracle 检查 NRMSE，默认上限 0.015。`--max-nrmse` 可配置并写入 JSON，调整门限必须在报告中说明。
4. 完整层检查 4 个 HCState tensor、压缩 ring 和解码后的 FP8/FP4 cache 数值；数值区域外的 backing bytes 必须完全一致。重复漂移单独记录，不能用对照漂移抵消失败。
5. 固定地址的改输入/hash-token replay 必须产生新输出；shape guard 必须在 cache 写入前生效。生命周期检查包含 IPC 在 TP teardown 前释放。

## 6. 性能基线与 launch 数

本轮三 seed 汇总，单位 µs/完整层 forward。以下均为 **bs1、seq4**，旧版为串行 token 调度的 mono；五阶段对照共用相同的稳定 attention/mHC。

| TP | 层 | 旧 mono | 当前 mono | 延迟下降 | 五阶段/当前 mono |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | L3 HCA | 276.243 | 222.712 | 19.38% | 1.230× |
| 4 | L4 CSA | 348.041 | 295.259 | 15.17% | 1.165× |
| 8 | L3 HCA | 250.960 | 193.130 | 23.04% | 1.295× |
| 8 | L4 CSA | 318.861 | 259.841 | 18.51% | 1.226× |

seq2 的延迟下降为 6.27%–11.64%，seq3 为 12.53%–19.46%；所有 seq2–4 bucket 在三个 seed 中均改善。seq1 继续使用原调度，本轮中位数差异为 -0.19%～+1.08%，不把该变化归因于调度优化。

| 层/TP | 当前 mono launches | 五阶段 launches |
| --- | ---: | ---: |
| HCA/TP4 | 29 | 33 |
| HCA/TP8 | 28 | 32 |
| CSA/TP4 | 48 | 52 |
| CSA/TP8 | 47 | 51 |

MoE 为 1 对 5，每层少 4 次 launch。以上数据不代表默认 native ATOM 整层或整模型服务吞吐。

## 7. 后续目标与验收方式

| 优先级 | 工作 | 验收方式 |
| --- | --- | --- |
| P0 | 完整 ModelRunner 初始化、graph capture、退出；真实生成和 MTP accept/reject | 对同一 checkpoint/请求比较输出与接受行为，反复启动/退出，确认 IPC/cache 生命周期正确 |
| P0 | 非零历史和更长 context；page、compression commit 边界 | 扩展单层 fixture，覆盖多个 cache page、ring wrap、连续 decode，检查未触及字节和逻辑 cache |
| P1 | bs2–8 与不同请求 query length | 明确 request/token 映射，扩展 scratch 与 hash IDs/metadata；先加范围拦截，再覆盖 TP4/8×请求数×query length 的精度矩阵 |
| P1 | GEMM1 的 CTA 内 K-wave 并行、tile 调整 | 参考 AITER 小 M 的 N64/N128 tile、kw2/OPUS kw7；保留 FP32→FP8 和 RNE 边界，逐阶段比较 oracle，再测三 seed |
| P1 | GEMM2 权重/scale 双缓冲和 LDS 等待重叠 | 同时分析 A LDS、B/scale 预取、barrier、VGPR/spill、cache policy；每次仅改变一个因素，保留精度与占用审计 |
| P2 | 减少完整层的 indexer/attention/mHC launch | 从 profile 确认主耗时和可融合边界，保留 cache 单次写入和 delayed HCState 语义，再与当前层接口比较 |

AITER 调优审阅的启发：TP4 小 M GEMM1 多用 FlyDSL，TP8 M1/2 可用 OPUS pair K-wave；M3 通常归入 M4 MoE bucket。OPUS kw7 将 K7168 分成 7×1024，并在 CTA 内 FP32 归约；GEMM2 同时调整权重/scale 预取、A LDS、cache policy 和 reduce/atomic 路径。不能只复制 kernel 名或单独提前一次 load 就假定获得同样收益。上一轮 `down_prefetch_v1` 仅提前 GEMM2 首批 load，没有稳定收益，未合入。

## 8. 结果归档与源码对应

共享任务目录 `dsv4_mono_20260929/` 中保存：

- `watch46_stage_batch_v1/attempt_0002/`：本轮 36 项完整结果，`comparison.json`、两路 `summary.json`、216 份 rank JSON、384 份 trace、源码与占用审计。
- `watch46_stage_batch_v1/attempt_0001/`：被外部 GPU 进程中断的历史尝试，未用于本轮性能结果。
- `watch46_release/`：提交源码的基本回归；5 项单测、2 项全 seq MoE GPU 检查已通过，后续完整层重复测试按服务器回收要求停止。首个 attempt 的 benchmark stream 失败记录保留。
- `production_fix46_summary.json`：早期精度修复矩阵，包括默认 native ATOM 的独立失败记录。
- `aiter_tune_audit_20260929/`：原始/merged CSV、实际 dispatch 解析及源码 SHA。

本轮 GPU 测试调度源码 SHA256 为 `3488b95822a66f94a53ef0d0ec4794d42899f37d1a66a070b8b360780872cb99`；提交前格式整理后的 kernel SHA256 为 `bff0bb267014923860e10a3b6d47ed3e106d68fb5fb9d806e15bbfe85983242b`，两者完整 AST 一致。性能批次每 0.5 秒采样 KFD 与进程归属，最大间隔 0.560 秒，没有观察到外部活跃 GPU 进程；这描述采样范围，不推断采样间隔内不可见的短进程历史。
