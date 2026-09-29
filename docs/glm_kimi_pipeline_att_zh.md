# GLM / Kimi K3：S4 流水、ATT 和源码映射

本轮直接采集 PR #1204 (`3563ef29`) 的 GLM-5 indexed MonoKernel：MI355X/gfx950、TP8、S4、位置 3000、topk 2048、native MXFP4 experts。目标是解释可迁移到 Kimi K3 true-MTP3 的调度方式。Kimi 的目标仍为真实 GPU-event 性能收益，不能用融合成功或出现 overlap 替代。

## 结论

1. **GLM 的下一 sample 权重预取与当前 sample 的 GMM1 MFMA 确实交错执行。** 未插桩 ATT 的 ISA 和动态 wave 记录均能看到这一点。
2. **S4 的 GMM2 权重预取提前，但 GMM2 MFMA 等待全部 intermediate 就绪。** 40 个 TP8 soft 快照里，没有一个 down task 的计算起点早于最后的 UG emit marker。不能把这条 GLM 路径的收益解释为 GMM1/GMM2 MFMA 重叠。
3. **GMM2 与 TP 按输出 tile 流水。** 早完成的 tile 提交 TP 时，其他 tile 仍在计算。
4. GLM 每个 CTA 完成 UG 后继续承担 down；routing 元数据留在 LDS；小 UG tile 合并 gate/up，下一 sample 权重提前加载。当前 Kimi S4 候选则有 256 个专用 producer、224 个专用 consumer 和 32 个无任务 CTA，还承担逐 K128 chunk 的 poll、LDS staging 和元数据构造。两者的调度成本不同。
5. 本次没有做 GLM fusion-on/off 的受控消融，不能给出“某项机制贡献多少加速”的因果比例。已有 Kimi A/B 仍显示融合候选未超过 staged baseline。

## 调试信息与采集完整性

在进程启动、导入 FlyDSL 和编译之前设置 `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`。base 和 timeline 使用全新的独立 `FLYDSL_RUNTIME_CACHE_DIR`，避免复用不带调试信息的旧二进制。

- ATT kernel：`glm5_monokernel_0`，dispatch 9330；目标 iteration index 64，CSV 中为第 65 次匹配 dispatch，位于 16-layer graph 的预热 replay 中。
- 原始 ISA 共 13,330 条，**13,330 / 13,330 均有 source mapping**；只有额外的 kernel-header 行没有源码位置。
- HSACO 包含 `.debug_info` 和 `.debug_line`；ATT 的 code object 38 与预编译的 TP8/16-layer 产物逐字节相同。
- `source_2_kernel.py` 与实际编译的 base `kernel.py` 逐字节相同；辅助 primitives/ops 的源码快照一并保留。
- GPU0/CU1、4 个 SE、全 SIMD，96 MiB ATT buffer，`att_serialize_all=false`。32 个 wave 的 `num_insts == num_stitched`，日志没有解码错误或截断报告。
- `hotspot_analyzer.py` 来自仓库的 kernel-trace-analysis skill。scoped 分析直接调用它的分类和汇总函数，仅用已核对的静态 PC 区间过滤 GLM 的 UG/down。

当前分析器因 BF16 ISA 自动判断为 gfx942，profiler CSV 的 VGPR 也偏低；这些自动资源推算不作为结论。实际硬件是 gfx950，资源以编译元数据和 HIP occupancy API 为准。

## 正确性、资源与观察者效应

base/timeline 分别先 compile-only，再验证驻留条件，然后运行 TP1 和 TP8。两版都通过现有独立 golden、中间结果、精确 routing、KV/index cache 检查及 graph replay；所有 rank 的 output/routing 精确相同。本次没有触发参考实现的 near-tie 路由导致端到端检查跳过。ATT 应用本身也通过相同检查。

| TP8 / 16-layer 编译产物 | VGPR | SGPR | 编译器 SGPR spill 计数 | LDS / CTA | private scratch | HIP CTA/CU |
|---|---:|---:|---:|---:|---:|---:|
| base | 194 | 106 | 155 | 95,824 B | 0 | 1 |
| timeline | 187 | 106 | 36 | 95,824 B | 0 | 1 |

SGPR spill 计数非零，但没有 private scratch 分配；不能写成“完全没有 spill”。256 CTA × 512 threads 的 grid 满足全部驻留条件。打点还改变了编译器调度及寄存器分配，所以不能根据 VGPR 下降认为打点没有干扰。

相同确定性输入、graph 内 16 层、预热后 20 次 event 测量，每次取最慢 rank，单位 μs/layer：

| 模式 | TP1 | TP8 |
|---|---:|---:|
| base，无时间戳 | 72.854 | 78.889 |
| timeline | 79.782 | 85.476 |

本次 TP8 插桩增加 6.586 μs，约 **8.35%**。这组值用于估计观察者效应；没有与 Kimi 的 MoE-only 数值做横向速度比较。ATT 测得的延迟也不作为优化结果。

## GLM 的实际 S4 路径

以下指向 base `kernels/glm5_monokernel/kernel.py` 的源码位置。

| 位置 | 行为 | 对性能分析的含义 |
|---|---|---|
| 2006–2111 | 活跃的 `S > 1` UG 分支；每 CTA 8 个 intermediate，8 gate + 8 up 合为一个 16-row group，8 waves 分 K | 每 sample 24 次 BF16 MFMA/wave；CTA 顺序计算 4 个 sample |
| 2092、1814 | `dn_route(load_bias())`，expert ID/coefficient 保存至 LDS | UG/down 复用元数据，不在每个 down chunk 从 global 重新取 ID |
| 2094–2100 | 先预取 shared/current 权重并 stage 所有输入；部分 CTA 额外执行 shared expert | 32 个 shared CTA 存在额外工作，不能只用普通 UG 的 start/end 概括全部准备成本 |
| 2103–2107 | 预取下一 sample，再消费当前 sample | 编译器可以将 load 移到当前 MFMA 中间；需看实际 ISA |
| 2330–2356 | GMM2 先发出首批 9 个 K128 unit/wave 的权重，再 poll 全部 mid 并写 LDS，最后 barrier | 预取能与尾部 UG overlap，GMM2 MFMA 依赖全部 mid |
| 601–614 | `run_units` 提前构造下一批 unit | GMM2 后续权重也有软件预取 |
| 2374、782 | 每个 24-row 输出 tile 直接 push 到 peers，再按 rank 顺序求和并加 residual | TP 随 tile 完成推进 |

源码中后面的 `elif ug_split(S) is not None` 是旧调度，S4 已由前面的 `S > 1` 分支处理，不能拿旧分支解释本次性能。`hint_wait` 在 422 行只是 marker + barrier；真正的 readiness wait 是 `poll/get2_many`。原有 report 的 “hint seen” 列名不代表看到 producer 就绪。

## Timeline 结果

timeline 诊断副本仅添加 timestamp stores，没有增加 barrier 或改动数学运算。采集预热 graph 最后一层，8 ranks × 5 replays，共 40 个快照；按 rank 内部的 realtime counter 分析，不跨 GPU 对齐。`s_memrealtime` 为 100 ticks/μs，不能与 ATT 的 shader cycle 数直接换算。

| 指标 | 40 个快照范围 | 中位数 |
|---|---:|---:|
| 最早 GMM2 prefetch marker 比最后 UG emit 提前 | 2.80–3.68 μs | 3.25 μs |
| prefetch marker 提前的 down CTA / 256 | 224–231 | 224 |
| GMM2 compute marker 早于最后 UG emit 的 CTA | **0** | **0** |
| 最早 GMM2 compute marker 比最后 UG emit 晚 | 0.59–0.94 μs | 0.725 μs |
| TP push marker 早于最后 down compute 的 CTA / 256 | 243–255 | 252 |
| 最早 TP push marker 比最后 down compute 提前 | 1.74–2.30 μs | 1.96 μs |

这里的 emit/push marker 表示对应源码提交阶段的边界，不能独立证明所有 store 已在 peer 可见。GMM2 的 LDS 后 barrier 才是各 wave 的全 mid 依赖边界。

按 CTA 绘制的图：`node47/timeline_analysis_v2/glm_tp8_rank0_snapshot2.png`（同名 SVG 可缩放）。行按该 CTA 最后 UG emit 排序；上方 32 个 CTA 包含 shared-expert 工作，出现更晚的 UG 尾部。图中 GMM2 预取在最后 UG marker 左侧，GMM2 计算在其右侧，TP 起点分散在 down 尾部。

特别保留了相邻 UG entry markers 之间的灰色区间。源码上两次 stamp 相邻，但编译器移动 load、控制流汇合及插桩会让该区间明显非零。不能把所有 marker 差值都命名为“纯预取时间”或“纯 MFMA 时间”。粗粒度依赖结论还有未插桩 ATT 和源码支持。

## 未插桩 ATT 的具体证据

`code.json` 中的 PC index 是解码器行索引，不是字节地址。以下为同一份已校验二进制：

| 当前 sample | 当前 sample 的 24 次 MFMA，首/末 PC index | 下一 sample 的权重 + scale load PC index |
|---|---|---|
| 0 | 10022 / 10284 | 10103–10114，插在当前计算中间 |
| 1 | 10415 / 10723 | 10538–10549，插在当前计算中间 |
| 2 | 10910 / 11146 | 10885–10896，提前于当前计算 |
| 3 | 11301 / 11539 | 无下一 sample |

这些 load 映射到 `kernel.py:537/539`，MFMA 映射到 `:574`。因此不仅是 Python 源码表达了预取，编译结果也实际保留了跨 sample 的安排。动态 wave 图在 `node47/att_scoped/glm_att_ug_prefetch.png`，下半图展示一个 wave 中 sample 1 load 出现在 sample 0 的前后两段 MFMA 之间。

GMM2 权重 load 在 PC 11767 开始；全部 mid staging 后的 barrier 为 PC 12134，映射到 `kernel.py:2356`；首个 down MFMA 为 PC 12389。32 个采样 wave 中，最早 down MFMA 也晚于最后 sampled UG MFMA，差 11,224 ATT cycles。这个差值只描述选中 wave 的这次 trace，不推算全卡微秒延迟。

全 kernel 的最大两个 source hotspot 是 `pre_poll` 后的 barrier (`:435`) 和 payload tag 检查 (`:407`)，各约占累计 stall 的 25.5%。前者用于 attention 的 UV 依赖，不能当作 MoE 瓶颈。因此另外按 ISA 区间分析：

| 累计 sampled-wave stall 分布 | UG（routing 后，PC 9255–11642） | down + TP（PC 11643–13330） |
|---|---:|---:|
| barrier | 51.06% | 53.93% |
| VMEM-load | 25.27% | 13.86% |
| VMEM-wait | 10.07% | 13.70% |
| LDS/SMEM-wait | 3.49% | 5.35% |

down 的 `:2356` staging barrier 占该区域 stall 的 16.02%；最终 `:2375` barrier 占 26.83%，其中其他 waves 会等待承担 TP/reduction 的少量 threads。不能直接删除它们或把这些百分比作为延迟收益上限。`MFMA/FMA stall` 很低也不等于 MFMA 没有执行成本。

## 对 Kimi 的可迁移方案

两者算子结构相似，但实际 workload 不相等。GLM routed hidden/inter/topk 为 6144/256/8，另有 1 个 shared expert；Kimi routed 为 3584/384/16。若每个 route 的 packed 权重读取一次，暂不计算 scale、cache reuse，S4 下 GLM（含一次 shared）约 77.86 MB，Kimi routed 约 132.12 MB。这是名义字节量，不是 PMC 测出的 HBM 流量。GLM UG 还使用 FP8-rounded activation，而 Kimi 为 BF16；不能为模仿调度而擅自改变 Kimi 精度语义。

优先级如下，每项都必须用关闭 soft/ATT 的同输入 A/B 判断：

1. **减少 metadata/descriptor 开销并解决 producer 供给。** 已有 Kimi route-prefetch 将 metadata stall 占比约减半，整层只改善约 1.4 μs，更多等待转成 polling。其 soft0 HIP occupancy 已为 3 CTA/CU；后续已完成 768 CTA / 544 producer 实验，整层再改善约 2 μs，仍未超过 staged。详见 [768 CTA 实验](kimi_mtp_grid768_zh.md)。
2. **借鉴 GLM 的跨任务预取和紧凑 gate/up tile。** Kimi 当前 producer 每个 U32 任务重新 stage activation；可尝试以 sample/route 批次复用 LDS activation，并提前发下一任务的权重。U8 gate+up 合并需要重新设计 K 分割：Kimi 有 28 个 K128 unit，不能照抄 GLM 的 48/8 均分。可比较 4 waves × 7 chunks 与 8 waves 的 masked 分割，记录重复读取、MFMA 数和实际资源。
3. **控制细粒度 handoff 成本。** 比较 route 粒度的三个 K128 连续消费、提前加载下一 route，以及一次 stage 更大的 mid 小组；保留 tagged readiness 和原有 math/TP 顺序。GLM 的全 mid staging 成本更低，但会放弃两级 MFMA 重叠，必须把总延迟作为判据。
4. **检查 CTA 在不同阶段的利用率。** GLM 的 256 个 CTA 都有 down 工作；Kimi 当前固定 224 个 down task，producer 完工后不能再帮忙。后续可以比较阶段间复用 CTA 或调整 down 工作分割。实际 CU 分布应由 trace/硬件 CU 标识验证，不能根据 grid 大小假设均匀分布。此前简单扩大 D tile 已回退，不重复把它当作新方案。
5. **保留按 tile 的 TP。** GLM 的 243–255/256 个 tile 能在最后 down 计算结束前提交 TP，说明这一方向有实际流水空间。Kimi 已有 down/TP overlap；当前数据不支持先把主要精力放在只减少 TP 字节数上。

这些结论支持按 S、路由数、tile 和实际 residency 选择配置。它们尚不构成自动 tuning 表，也不支持直接把 GLM 参数搬到 Kimi。

## 证据位置与重现

本地根目录：`/home/zihuang/work/mega_transformer/results/glm_kimi_att_20260928`。远端 node47：`/tmp/flydsl-glm-kimi-att-20260928`。

最终 archive 为 `glm-kimi-att-final-node47.tgz`，SHA256 `4c25bfaf4bf483d765cc7db87117d8ab8b6514611185efe3214f237242d9647c`。本地核对了 manifest 的 1,563 个文件、全部 26 份 rank 结果和 45 个 timeline 快照（其中 TP8 为 40 个），汇总见 `verification_summary.json`。

- `node47/controller.py`、逐次 `*.command.json` 和 `*.rc`：debug env、compile-only、资源 gate、TP1/TP8、ATT 的完整命令与退出状态。
- `node47/base` / `node47/timeline`：实际编译源码；生产 worktree 的 GLM kernel 没有修改。
- `node47/att_base/ui_output_agent_32706_dispatch_9330`：ISA/source mapping、32 wave 动态指令、源码快照。原始 `.att`、目标 code object、8-rank CSV 同目录保留；未参与目标 trace 的库 code object 留在远端。
- `node47/timeline_tp8`：40 个原始 timeline 快照与正确性结果；`timeline_analysis_v2` / `att_scoped`：图表、汇总与分析脚本输出。
- `node47/source_mapping_validation.json`、`debug_sections.txt`、`resources_base` / `resources_timeline`、`manifest.json`：源码、DWARF、二进制和资源证据。

分析只需要结果和 Python；绘图需 matplotlib：

```bash
python analyze_glm_timeline.py node47/timeline_tp8 --output timeline_analysis
python analyze_glm_att.py node47/att_base/ui_output_agent_32706_dispatch_9330 \
  --analyzer node47/hotspot_analyzer.py --output att_scoped
```

本轮没有新 commit，也没有 push；原 correctness commit `cb147e89d02a38dcc35a7929cf6593aa84784d18` 保持独立。
