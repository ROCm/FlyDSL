# Kimi MTP：以 kernel 边界替代 intermediate polling

本轮实现并验证了 compact UG 与 compact down/combine/TP 分离的两阶段版本。完整层单 seed 筛选为 **150.139 μs**，未超过既有 public pipeline 的约 143.577 μs，也未达到约 129.18 μs 的目标。此版本保留在独立实验目录，没有接入默认 kernel。

## 改变的依赖与数据流

UG 继续按 expert/sample compact，使用 U32 小 tile；最多 768 个 task，每个 task 恰好由一个 CTA 计算。UG 写出 48 KiB 的 raw BF16 intermediate，另保留 diagnostic tagged mid 供独立 reference 检查。随后同一 stream 启动 224 个 down 输出 owner，直接读取 raw intermediate；kernel 顺序保证数据已就绪，不再轮询每组 intermediate 的 tag。

down 仍按 expert 复用权重，以原来的 BF16 MFMA 和逐 K128 的 FP32 权重归约产生输出，再按静态输出 tile 完成 TP。没有大的 down partial buffer，也没有 atomic combine。两次 launch 均包含在完整层 graph/event 时间内；同一实例、同一 stream 的要求保持不变。

这项改变同时失去 UG/down 的跨 CTA overlap，换取独立的计算资源与更少的中间结果 load/poll。结果表明这种交换目前没有带来完整层收益。

## 实际 geometry 与资源

外层命令行沿用 `--routed-grid 512 --routed-producers 256` 作为实验 host 接口参数；编译产物和真实 launch 的 geometry 如下，不能把命令行别名误认为实际 grid。

| 阶段 | grid | threads/CTA | TP1 VGPR / CTA每CU | TP8 VGPR / CTA每CU |
|---|---:|---:|---:|---:|
| compact UG | 768 | 512 | 94 / 2 | 94 / 2 |
| compact down + TP | 224 | 512 | 78 / 3 | 72 / 3 |

四份产物均完成 compile-only 和真实 HIP occupancy 查询，无 SGPR/VGPR spill 或 private scratch。UG 不包含任何 intermediate/peer polling，任务间无 GPU 全局完成等待，因此不要求整个 768 grid 同时驻留；有 TP 等待的 down 阶段只有 224 CTA，满足设备至少 256 CU 的原有约束。没有在仅 2 CTA/CU 的情况下启动一个 768-CTA polling grid。

## 验证与计时

TP1 原输入与 varied-scale reference 检查全部通过，包括 expert 数 16/17/47/48/63/64、改变 routing/payload、零系数和多次 16-layer graph 重放。TP8 通过 exact routing、rank equality、独立 mid/output、TP/tail 及 true-MTP state 检查，并在 graph replay 后复查。

node50、TP8、S4、seed1234、16-layer graph、50 repeats、无插桩 GPU events，取最慢 rank 的中位数：150.138594 μs。rank0 的 pipeline mid/output relative L2 分别约 7.16e-6 / 0.001653，TP 精确相等；所有 rank 经原验证器检查。原 mid < 2e-3、output < 1e-2 阈值保持不变。

该版本没有采集 ATT/PMC，也没有多 seed 性能确认，不能把额外耗时定量拆给 launch、UG 或 down。U192 原子/partial 路线的独立 ATT 与流量结果见 [U192 报告](kimi_mtp_atomicdown_zh.md)。

源码、patch、编译资源、ISA、命令与验证记录在 [splitphase 实验目录](../../results/kimi_mtp_splitphase_20260929/)。下一步若继续研究这条边界，应先与 tuned production UG 做受控替换，确定自定义 UG 本身的差距，再判断是否继续投入 down/TP 或 tail 边界；不能从取消 polling 直接推断性能必然改善。
