# Kimi K3 MTP：CTA 内 UG → SiTU → down 实验

本轮没有性能收益。把中间 BF16 留在 CTA 内，立即算 down，确实移除了消费者对 BF16 中间量的读取；但输出需要跨 CTA 汇总带 epoch 的 FP32 部分和，增加了写事务和 L2 读取。最快的安全变体仍慢于现有 pipeline，因此未修改公开内核实现。

实验目录：`/home/zihuang/work/mega_transformer/results/kimi_mtp_localdown_20260928`。节点为 MI355 node50，S4、TP8、true MTP。性能数字均来自未插桩的完整层 GPU events，取最慢 rank；本轮仅作单 seed 筛选，不能视为多 seed 最终结论。

## 计算和数据流

每个生产 CTA 处理一个 `(expert, intermediate tile)`，为四个 sample 共享权重，执行 UG、BF16 SiTU，并立即覆盖全部 3584 个 down 输出行。消费者按固定顺序读取、检查 epoch 并归约 FP32 部分和，随后执行原来的 BF16/TP 尾部。

U64 方案的部分和有效载荷加 tag，逻辑上写入 10.5 MiB、读取 10.5 MiB。实际写事务以 sector 计费，非连续有效 sample lane 使事务占用高于有效字节；旧 tagged BF16 中间缓冲仍写入，用于独立正确性检查。

## 资源与性能筛选

| 变体 | 结果 | 完整层耗时，μs |
|---|---|---:|
| U64 初版 | VGPR149，1 CTA/CU；不满足 grid512 的驻留要求，未启动轮询内核 | — |
| U32 初版 | 通过 TP1、改变 routing 的图重放和 TP8 | 200.930 |
| U64 限制循环展开、预取 4 个输出行组 | VGPR88，2 CTA/CU；通过正确性 | 157.720 |
| U32 相同预取方式 | 通过正确性 | 175.537 |
| 旧 compact，仅限制 UG 展开 | 通过正确性 | 145.821 |

采集 profile 前的同期完整层对照：公开 pipeline grid768/P544 为 **144.081 μs**，U64 预取方案 grid512/P288 为 **157.221 μs**，慢约 **9.1%**。此前公开 pipeline 三 seed 中位数为 143.528 μs，至少降低 10% 耗时的目标仍约为 129.175 μs。

## 实际读写流量

同一 seed、每 rank 一次匹配 dispatch，表中为八个 rank 的均值。HBM 使用 `TCC_EA0_RDREQ_DRAM_32B_sum` 和 `TCC_EA0_WRREQ_WRITE_DRAM_32B_sum`；L2 使用 `TCC_READ_SECTORS_sum` 和 `TCC_WRITE_SECTORS_sum`，均乘 32 转为字节。计数器验证可在同一 pass 收集。

| 流量，MiB/rank/dispatch | 公开 pipeline | U64 CTA 内 down |
|---|---:|---:|
| HBM 读 | 125.817 | 123.671 |
| HBM 写 | 0.203 | 21.203 |
| L2 读 sectors | 206.011 | 259.904 |
| L2 写 sectors | 0.641 | 21.641 |

HBM 读取减少 1.7%，但总读写增加约 15.0%；L2 读取增加约 26.2%。该数据流没有减少总体内存代价。

## ATT 解释

64 条采样 wave 全部拼接成功，动态计数与 code table 一致，源文件快照、编译代码对象 `.text`、debug info 均已核对。48 条生产 wave 每条执行 112 条 UG MFMA 和 56 条 local down MFMA；16 条消费 wave 不执行 MFMA，只读取并归约部分和。

分析按角色选择截止位置：生产者包含最后一条 local down MFMA；消费者包含最终归约的 LDS store。后续打包和 TP 等待不计入范围。消费者约 90.3% 的 scoped stall 归于 VMEM wait，其中约 89.1% 的全部 stall 定位在部分和初次检查/重试附近。生产者的 VMEM load 与 wait 合计约 55.9%。这些是 ATT cycles 的归因，不能换算成微秒或当作无插桩性能。

## 校验与留档

6 组编译资源记录已审核；所有实际启动的 polling 几何均满足 HIP 实际 block size 的驻留要求。4 次 TP1 routing replay 各覆盖 16、17、47、48、63、64 个唯一 expert，同时改变 payload 并使用独立 FP32 reference。TP8 结果和 profile 的源代码匹配检查通过。

证据包 `kimi-localdown-node50-evidence.tgz` 已下载并逐文件校验 5,057 个 manifest 项，SHA256：

```
70d58bf4a8135cc17972942469d7687f1778c4b8c67284e3f94e80ef0c30934f
```

后续转向减少 UG 的 FP4 解码与矩阵指令：将 BF16 输入表示成两个带 scale 的 FP8 分量，逐块验证精确重构，无法重构时回退 BF16。GPU 打包费用必须计入完整层，仍需完整正确性、资源和性能验证后才决定是否保留。
