# Kimi K3 MTP3：768 CTA 之后的优化方向

2026-09-28，接续 session `01a0e652-c9e1-7693-9511-b050789773e8` 最后未回答的问题“当前还有什么方向”。工作区 HEAD 为 `3563ef29`；本次是现有结果与源码复核，没有产生新的 GPU 性能结果。

后续执行结果见 [无 soft 插桩 ATT 与八个候选实验](kimi_mtp_att_followup_zh.md)：已确认并删除末轮三次冗余 load，但多 seed 整层收益不稳定；activation 复用、UG 提前预取、polling 退避均未显示足够收益，未接入新 kernel 改动。以下内容保留为实验前的方向与判据。

最新的 [HBM/compact 数据流审计](kimi_mtp_dataflow_zh.md) 已用三 seed PMC 核对：当前 pipeline 比 staged routed GEMM1+2 多约 23% 的 HBM 读请求。后续优先研究两级共同按 expert 分组，复用同专家的多 sample 列权重；该结构性候选尚未实现。

## 已验证的起点

输入保持一个请求的 true-MTP3（S4）、TP8、staged KDA、native MXFP4 routed experts。当前融合范围是 routed UG/SiTU/down/combine/TP；router/latent-down/shared-up 与最终 tail 仍是独立 kernel。

三 seed、无 soft/ATT 的整层 GPU-event 中位数：staged **133.815 μs**，512/P256 **144.121 μs**，768/P544 **142.086 μs**。768 比 staged 仍慢 **8.271 μs / 6.18%**。MoE-only 与整层不可相减拆出 attention 时间。

已保存的验证汇总为 30 次 TP8 全通过、3,770 个文件校验通过；本次只读取汇总，没有重新执行这些 GPU 验证。实验与资源见 [768 CTA 报告](kimi_mtp_grid768_zh.md)。

## 1. 先单独验证 consumer 循环的尾部预取

入口：`routed_pipeline.py` 的 `down_prefetch` 和 `consume_output`，当前循环位于 366–399 行。

S4/D16 时，每个 wave 消费 24 个 K128 unit，即 8 个 route，每组预取 3 个 unit。源码先构造一次 `cur`，随后 8 次循环均构造 `nxt`。最后一轮的 `next_begin` 被 clamp 到 21，再次构造最后一个 route；循环结束后这一组 `nxt` 不再消费。

这是源码层的预取构造计数 **27 对 24**，不是测得的 HBM 流量或可兑现的 12.5% 加速。需要先检查最终 ISA 是否仍保留该组读取，编译器可能消除未使用操作。

实验仅拆分“首组预取、7 轮主体、最后一组消费”，保持 route/chunk 的原数学顺序与 tagged payload 协议。先保持运行时 route 循环，不同时展开全部 8 个 route。此前失败变体把“全展开”和“删除末轮预取”一起修改，VGPR150、1 CTA/CU 的结果不能单独否定尾部处理。

判据：保留 768 grid 所需的实际 3 CTA/CU；若 ISA 没有冗余读取或新版本增加寄存器/分支代价，停止该变体。已有 consumer 本来就预取下一组，新增实验不应重复声称“首次加入预取”。

## 2. producer 按 sample 分组，复用 activation

入口：`routed_pipeline.py` 的 producer task 分配和 activation staging，当前在 234–280 行。

每个 U32 任务都重新加载一个 sample 的 3,584 个 BF16 activation 到 LDS，并进行 CTA 同步。总 UG 任务为 `4 × 16 × (384/32) = 768`。

对当前 interleaved 映射做穷举，结果如下：

| 配置 | 每个 producer 的任务数 | 相邻任务转换数 | 相邻任务保持同 sample |
|---|---|---:|---:|
| P256 | 256 个 CTA 各 3 个 | 512 | 0 |
| P544 | 224 个 CTA 各 2 个，320 个各 1 个 | 224 | 0 |

因此不能只在当前循环外提升 activation load：下一任务的 sample 会变，必须先调整任务映射。

一个可验证的静态映射是每个 sample 分配 `P/4` 个 producer，sample 内按交错 route 次序 `0,8,1,9,...,7,15` 分配 U32 tile。每个 producer 始终处理同一 sample，activation 只需初始化一次；独立校验每个 `(sample, route, U32 tile)` 恰好执行一次，并保留各 consumer wave 第一 K128 的供给优先级。

本次用 Python 穷举 `sample=pid%4`、`local_pid=pid//4`、`local_task=local_pid+n*(P/4)`：P256/P544 均覆盖全部 768 个任务且无重复，每个 producer 的任务数分布与原配置一致。这只验证任务映射，不验证 GPU 调度、kernel 数值或性能。

在全部 producer 都有任务的前提下，源码层 activation staging 次数可从 768 降到 P：P544 少 224 次（29.2%），P256 少 512 次（66.7%）。这不是 HBM 流量降低比例，activation 可能来自缓存；还需关注 LDS 和同步成本。必须保留保护 `red` 跨任务复用的同步，不能直接删去所有 task 边界 barrier。

这与之前“同专家多 token 复用权重”的实验不同。后者已经做过且未获益，本方向复用的是同一个 sample 的 activation，并要求新的任务分配。

若该调整有效，再单独尝试提前加载下一 UG 任务的一小组权重/scale；限定预取深度，检查 ISA 是否真正穿插 load/MFMA。一次只改变一个因素。

## 3. 再考虑 producer/consumer 角色复用

当前 S4/768 配置由 544 个 producer 和 224 个 consumer 构成。UG 完成后，producer 的 CTA 不承担 down；源码的通用“producer 后转 consumer”只在输出 ownership 与 producer 集合相交的其他配置中成立。

这是潜在的后半程利用率问题，但不能直接把更多 CTA 加到当前 224 个输出 tile 上。需要重新划分 down 工作、保持确定性的跨 rank 输出顺序，并证明任何 consumer 等待都不会阻止其依赖的 producer 执行。

优先研究有限的静态角色切换；不先引入任意任务窃取或无约束跨 rank 队列。原 256-CTA 混合调度已经测过且较慢，简单恢复它不算新方案。扩大 D tile 也已测过退化。

## 4. 若仍有差距，检查 routed 与 tail 的边界

`staged.py::_moe` 当前先等 routed pipeline 整个 kernel 结束，再启动 fused tail。tail 内部已让 shared-down 与 routed 归约/归一化并行，但 shared-down 尚未与前一个 routed kernel 重叠。

可研究让部分完成 UG 的 CTA 承担独立 shared-down，或有界地衔接 tail。latent-up 必须等待 routed TP 与完整 RMSNorm 依赖就绪；不能仅凭某个 routed tile 已完成就提前归一化。

这是比前两项更大的改动，会改变资源占用与通信依赖。只有在逐 kernel 的整层时间线证明该边界位于关键路径后再实施，不能承诺直接节省全部 8.271 μs 差距。

## 实验顺序与停止条件

1. consumer 尾部预取单因素变体；核对 ISA、资源，再决定是否运行。
2. sample 固定的 producer 映射与 activation 复用；正确性通过后再加跨 UG 任务预取。
3. 前两项仍无法超过 staged 时，再投入角色复用及 routed/tail 边界重构。

每个 kernel 修改先静态检查、compile-only、实际 HIP occupancy，再做 TP1 独立参考、TP8 精确 routing/TP/tail 与 MTP 状态检查及改变 payload 的 replay。768 CTA 要求 3 CTA/CU；降到 2 时不能继续启动原 grid。

性能先用一个 seed 筛选明显退化，再对有希望的变体执行三个 seed、轮换次序的 staged/原768/新候选对照。整层无插桩 GPU events 是主判据；soft 与 ATT 仅解释结果。debug info 在编译时开启。

若再次采集 768 轻量 timeline，应仍关注 last UG、first down chunk、down end 和 TP end。此前 UG 尾部提前 4.29 μs，但首个 down chunk 晚 1.09 μs，说明输入供给与资源竞争需要一起评估。继续增加 CTA、三个 K128 一次 staging、全 route 展开均不作为未经区分的新方向重复尝试。

最终按 S/TP/MTP shape 选择经过验证的配置；当前仅验证 S4 的候选不能作为通用动态调优结论。默认继续使用 staged。
