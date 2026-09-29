# Kimi K3 MTP3：soft 打点与 ATT 分析

本轮基于 PR #1204 的 `3563ef29` 和已有 routed tile pipeline，输入为一个请求 MTP3（当前 token + 三个候选，即 `--mtp --samples 4`）。保持 staged KDA 和 `[S+1]` 状态快照链。目标是解释融合的性能损失，并验证针对热点的修改；单个 kernel 或 overlap 本身不算性能收益。

## 打点实现与测量口径

`routed_pipeline.py` 使用 GLM 的 `mem_realtime()`，对应 `s_memrealtime`。这是设备 realtime counter，按 100 ticks/μs 换算，**不是 shader clock cycles**。额外的 `soft_durations[grid, 8, 16]` 为 int64 GPU 输出，benchmark 和 replay 正确性检查后复制到 CPU，并按 rank 写 JSON。模型输出不用于存储诊断信息。

`--routed-soft-profile`：

- `0`：关闭；TP8 编译后的 ISA 中无 `s_memrealtime`，不分配 duration buffer。
- `1`：所有 CTA 的 wave 0，记录 UG、down、pack、TP push、TP wait/reduce、相对时序和 epoch。
- `2`：每隔 31 个 CTA 采样一次，记录该 CTA 的全部 8 个 wave，并累计 tagged intermediate polling 时间。

S4 / D16 下，只有 wave 0 执行最终 tile packing 和 TP reduction；分析这些阶段时只使用 wave 0。down 的时间包含 polling、权重读取、解包、LDS、MFMA 和同步。`down - polling` 也不是纯计算时间。TP push 记录提交和原有 barrier，不独立证明 peer store 已完成。插桩没有增加 GPU barrier。

每个 rank 的 JSON 是最后一次 graph replay、最后一层的快照。CTA/wave 分布不是多次运行分布；各 rank counter 不做时钟对齐。重叠的阶段、wave 时间不能相加当成 kernel 延迟。相邻 counter 读取的中位差为 0，但 ISA 保留了两次独立读取；计数器分辨率和异步读取使该差值不能衡量真实打点开销，开销以单独的 GPU-event 对照衡量。

## 资源与正确性

MI355X / gfx950，Torch 2.9.1 / ROCm 7.2。node46 出现另一组 EP16 作业后停止本轮未完成采集，后续在 node47 使用相同源码和编译包完成。node47 的数据单独报告，不与此前 node46 性能样本合并。

| TP8 模式 | VGPR | SGPR | LDS / CTA | scratch / spill | HIP 驻留 CTA / CU |
|---|---:|---:|---:|---|---:|
| 0 | 82 | 85 | 15,360 B | 0 | 2 |
| 1 | 124 | 95 | 15,360 B | 0 | 2 |
| 2 | 124 | 99 | 15,360 B | 0 | 2 |

TP1/TP8 × 三种模式均先编译和检查 occupancy，再启动轮询 grid。单卡独立 expert 参考检查通过。TP8 两个 scope × 三种模式均在 8/8 rank 通过 routing、incoming snapshot、routed TP、tail reduction/residual 精确检查，graph replay 后重查；attention/state/output 及独立 expert 参考满足既定 relative-L2 门槛。三组 ATT 应用也完成同样的检查。

node46 的 soft0 整层首次 forward 曾出现一次异步非法访问。同步执行重试和正常重试通过；不能据此宣称已找到或修复原因。原始错误、受竞争影响的运行和正常 node47 数据均分别保留。

## Soft 结果

固定 `grid=512, producers=256, U32, D16, prefetch=4, interleaved`；256 个 producer、224 个 consumer、32 个空闲 CTA。以下为各 rank 内 CTA/wave 中位数，再列出八个 rank 的范围，单位 μs。

| 阶段 | MoE 段上下文 | 完整 KDA/MTP 层上下文 | 模式 |
|---|---:|---:|---|
| UG | 19.24–19.68 | 20.76–23.10 | coarse |
| down（含 polling） | 25.42–25.80 | 28.20–30.32 | coarse |
| pack | 0.20 | 0.20 | coarse |
| TP push | 0.40 | 0.40 | coarse |
| TP wait/reduce | 1.04–7.52 | 1.16–2.56 | coarse |
| down 内 polling | 9.42–9.80 | 10.00–11.08 | detailed |
| down 扣除 polling | 15.90–16.52 | 18.04–18.78 | detailed |

全部采样 consumer 都在最后一个 UG 提交之前进入 down。详细样本中 polling 占 down 约 35–38%。说明 producer/consumer 已 overlap，但下游仍有明显的输入等待；同时不能把剩余约 16–19 μs 全部归为 MFMA。整层上下文中的 UG/down 比 MoE-only 更长，TP wait 反而更短；这两种上下文不能通过相减拆出 attention 开销。

打点的观察者效应采用独立 GPU-event 测量：graph 内 16 层，50 次重复，seed 1234，每次取最慢 rank。下表仅评估打点扰动，不替代此前优化 A/B。

| scope | 关闭 | coarse | detailed |
|---|---:|---:|---:|
| MoE 段 | 71.678 | 71.268 | 72.151 |
| 整层 | 146.694 | 147.361 | 149.756 |

detailed 在这组整层对照中增加 3.063 μs（2.09%）；coarse 变化约 0.46%。MoE coarse 的负差值不能解读成打点收益。

## ATT 结果

使用仓库 `capture-kernel-trace` / `kernel-trace-analysis` 工作流和其 `hotspot_analyzer.py`。decoder 0.1.6 安装在实验目录；三组 trace 均成功解码，没有报告截断或 `INVALID_SHADER_DATA`。使用独立 debug-info cache，soft 关闭，选第 64 次目标 dispatch，GPU0 / CU1 / 4 个 SE / 全 SIMD，`att_serialize_all=false`。

| 目标 | 已解码 ISA 行 | source mapping | VMEM-wait / 全部 stall | VMEM-load / 全部 stall | LDS/SMEM-wait / 全部 stall |
|---|---:|---:|---:|---:|---:|
| routed tile pipeline | 1,738 | 1,738 / 1,739 code 行 | 53.3% | 15.2% | 12.9% |
| staged GMM1 | 3,107 | 3,107 / 3,108 | 39.7% | 5.2% | 16.7% |
| staged GMM2 | 454 | 454 / 455 | 56.4% | 11.6% | 6.7% |

这里是采样 wave 的累计 stall 分布，**不是 kernel 延迟占比或可直接兑现的收益上限**。ATT 使被采样 rank 的启动延后，其他 rank 会等待它，因此 ATT 的 TP 等待与总时延不用作性能结果。分析脚本因 BF16 ISA 自动识别为 gfx942，且 profiler CSV 的 VGPR 数偏低；资源结论使用上表的 gfx950 编译元数据和真实 HIP occupancy。

融合 kernel 的主要 source 热点：

1. `monokernel/ops.py:21`（`uniform` / `readfirstlane` 前 VMEM wait）：24.57% 的累计 stall。ISA 上多数位于 GMM2 的 expert/weight 读取和 descriptor 构造；每个 K128 chunk 都重载元数据，引入等待并打断预取。
2. `routed_pipeline.py:212`（tagged intermediate polling）：20.88%。与 soft 记录的等待一致，但两者分母和采样不同，不应要求比例相等。
3. `routed_pipeline.py:267`：7.83%，主要是 UG 的 LDS 读取等待；source 标在 MFMA 行，实际热点指令是 `s_waitcnt lgkmcnt(0)`，不能据此判为 MFMA 吞吐瓶颈。
4. `routed_pipeline.py:179/176` 的 scale/weight load，以及 `:183` 的 scale 消费处等待，说明 load 调度仍值得优化。

staged GMM2 的主要热点为 `gemm2.py:373` barrier 前的 `vmcnt(0)`；因此 baseline 本身也存在等待。融合必须用实际 overlap 抵消更细的任务、元数据和交接成本，减少 launch 数并不自动保证总收益。

## Route 元数据预取实验

在独立 `candidate_route` 副本中，把同一路由的三个 K128 chunk 一起构造：expert ID 和 coefficient 每 route 只加载一次，保持原有数学顺序、tagged payload、输出 ownership 与 TP。此阶段限制 S4/D16；后续已接入显式实验开关，见 [768 CTA 实验](kimi_mtp_grid768_zh.md)，默认实现没有切换。

node47 上使用 seed 1234、2025、42，部分反转 A/B 顺序，graph 每次 16 层、重复 50 次；下表是三个 seed 的中位数，单位 μs。所有 rank 的既有检查与 replay 重查均通过。

| scope | staged | 原 tile | route-prefetch |
|---|---:|---:|---:|
| MoE 段 | 67.063 | 69.576 | 69.362 |
| 整层 | 133.707 | 145.827 | 144.365 |

整层逐 seed 相对原 tile 降低 1.405、1.424、1.805 μs，但仍比 staged 慢约 8%。MoE 的 seed 2025 局部偏快到 66.14 μs，其余约 69.36/69.87 μs，不能据单个 seed 宣称超过 baseline。

独立 soft 快照中，各 rank 的 down 中位数再取中位数，原版本 29.09 μs，route-prefetch 28.04 μs；polling 从 10.54 增到 16.52 μs，down 扣除 polling 从 18.37 降到 11.34 μs。对应的未插桩 ATT 中，metadata scalarization 的累计 stall 占比从 24.57% 降到 11.83%，而 polling 从 20.88% 升到 33.02%。这支持“consumer 更早追上 producer、部分元数据等待转为输入等待”的解释，不能把 stall 占比差直接兑现为延迟收益。

TP8/soft0 的寄存器降为 VGPR 78 / SGPR 73，LDS 不变、scratch 为零，真实 HIP occupancy 提升至 **3 CTA/CU**。随后完成了 `grid=768, producers=544` 的三 seed 实验，整层较 512/P256 改善约 2 μs，仍慢于 staged；数据和显式开关见 [768 CTA 实验](kimi_mtp_grid768_zh.md)。soft1/2 仍为 2 CTA/CU，不能直接把原有插桩版本放进 768-CTA 轮询 grid。

GLM 的实际调度与 ATT/source mapping 对照见 [GLM 与 Kimi 的流水比较](glm_kimi_pipeline_att_zh.md)。GLM 的 GMM2 会先预取权重，再等待全部 mid；本次 S4 路径未观察到 GMM1/GMM2 MFMA 重叠，因此“是否 overlap”必须区分内存预取、计算和 TP。

## 重现与证据

```bash
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --staged --mtp --samples 4 --check --bench --layers 16 --repeats 50 \
  --routed-pipeline --routed-grid 512 --routed-producers 256 \
  --routed-up-order interleaved --routed-soft-profile 2 \
  --soft-output-dir /tmp/kimi-soft --output /tmp/kimi-soft-check.json

python kernels/kimi_k3_monokernel/tools/analyze_soft_profile.py \
  /tmp/kimi-soft --output /tmp/kimi-soft-summary.json
```

MoE-only 在第一条命令增加 `--moe-only-bench`。关闭打点使用 `--routed-soft-profile 0`；当前打点限制为 S4 / D16 / 默认 sample grouping / wave handoff / complete TP。

证据根目录：`/home/zihuang/work/mega_transformer/results/kimi_mtp_soft_att_20260928`。`node47` 保存原版本六组 soft 数据、route-prefetch 的额外 soft 数据、30 次 TP8 运行的逐 rank 正确性和 GPU-event 数组、四组 ATT 的原始 `.att`、对应 code object、decoded UI/source、CSV、分析报告、命令、exit code 和 source snapshot。未参与 trace 的大体积库 code object 保留在远端，没有全部下载。`node46` 保存首次异常与中断记录。最终 archive SHA256 为 `f1bf223b55fe42f06d133330eefda5bd96bf2da14672c6b3feefffb7b0cd5237`；本地核对了 1,402 个源文件及完整结果。原正确性 commit `cb147e89` 未修改，没有推送。
