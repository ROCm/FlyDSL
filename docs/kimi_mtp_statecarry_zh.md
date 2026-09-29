# Kimi MTP：递归任务调度与状态驻留

本轮在独立 overlay 中实现了七个 attention 变体。最有效的是将 S4 MTP 的 96 个递归任务从 CTA 0–95 移到 128–223，使其避开卷积 CTA 0–47。三组 seed 交替测量确认：public routed pipeline 完整层中位数 **142.693 → 137.422 μs，降低 3.69%**；三组都改善，节省 4.94–5.63 μs。

同一 attention 改动用于生产 staged 路径时，中位数 **133.295 → 128.218 μs，降低 3.81%**。两个执行路径都稳定节省约 5 μs。生产路径组合值低于历史 public 参考所对应的 129.18 μs 目标，但该组合还包含原有 MoE 执行路径的差异；不能把全部差异计为新 attention 或原 routed pipeline 的 10% 优化。原 routed pipeline 自身仍未达 10%，目标保持进行中。

## 完整层筛选

node50、TP8、S4 true-MTP、seed1234、16-layer CUDA graph、50 repeats；使用无插桩 GPU events，每次取最慢 rank，再取中位数。各候选均通过正确性后计时。

| 版本 | 改动 | 完整层 μs |
|---|---|---:|
| base | 原 public pipeline | 143.279 |
| relocate | 递归任务移到 CTA 128–223 | **138.108** |
| carry200 | CTA 200–223 连续处理 4 token，保留 FP32 状态 | 140.795 |
| carry0 | 相同状态复用，owner 为 CTA 0–23 | 144.116 |
| headcarry | 12 CTA 各负责整 head 的卷积、递归、RMS | 141.310 |
| carrydefer | 保留状态，将跨 split RMS 等待移到状态链之后 | 139.455 |
| wide512 | attention grid512；400 个输入投影任务，递归 CTA 416–511 | 139.016 |
| map200 | 卷积 CTA 200–247；递归 CTA 112–207 | 138.594 |

carry 系列保留全部输出快照，仅省去连续有效 token 的状态重读。按源码计算，S4 可减少 2.25 MiB 逻辑状态读取；本轮没有采集对应 PMC，不能将它表述为实测 HBM 降幅。直接缓存状态也使同一 CTA 连续承担输出归一化等待。将 RMS 延后确实比 carry200 更快，但尚未超过只改变任务分工的版本。整 head 合并与更细的投影任务也未超过它。

## 多 seed 确认

使用 seed1234/5678/9012，反转中间一组的运行顺序，交替运行原版和 relocate；生产 staged 控制同样交替测量。每次 50 repeats。

| seed | public 原版 | public relocate | 降幅 | staged 原版 | staged relocate | 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| 1234 | 143.270 | 138.196 | 3.54% | 134.400 | 129.356 | 3.75% |
| 5678 | 142.693 | 137.066 | 3.94% | 133.295 | 128.218 | 3.81% |
| 9012 | 142.358 | 137.422 | 3.47% | 132.583 | 127.465 | 3.86% |

单位为 μs/layer。图在实验目录的 `confirmation.png` / `confirmation.svg`。

## 状态正确性与资源

所有八个版本均通过 TP8 独立 FP32 golden 与同一个 graph 的动态输入重放。每个版本覆盖四条 slot 链，每条更换八组 prefix、卷积和递归初始状态，graph 内包含 16 次 layer 调用：

- `[0,1,2,3,4]`：常规快照链。
- `[5,2,6,1,4]`：非连续 slot。
- `[0,1,-1,3,4]`：无效 token 后从已有 slot 重新开始。
- `[-1,-1,-1,-1,-1]`：全部无效。

检查 output、每个 state slot 和未写 slot 的精确保持。原阈值 attention <2e-3、conv/recurrent state 各 <5e-4 未放宽。重放最大 attention relative L2 约 7.82e-4，最大单 slot recurrent relative L2 约 4.37e-4。完整层还验证 routing、mid/output、rank equality、TP/tail 和 incoming state；多 seed 运行也全部通过。

每个候选均先 compile-only，再对真实 512-thread block 查询 HIP occupancy。base/relocate/map200 为 114 VGPR，carry 系列 118 VGPR，wide512 为 113 VGPR，均为 2 CTA/CU；headcarry 为 175 VGPR、1 CTA/CU。所有版本 private scratch 和 VGPR spill 为零。headcarry 有 8 个 SGPR spill，首次零 spill 筛选因而拒绝；保留初次失败日志后，按实测驻留条件和明确的 spill 上限复核，通过后才运行。没有将该版本描述为“零 spill”。

attention grid256 需要至少 1 CTA/CU，wide512 需要至少 2；所有 public routed grid768 仍使用原来的 512 threads，并验证至少 3 CTA/CU。

## ATT 与下一步

base 和 relocate 已采集新的 attention ATT，启用 debug info，并核对 code object `.text`、源码快照和完整动态指令统计。首次 CU1 采样各 32 waves，静态源码映射分别 3656/3656 与 3664/3664。base 采到 16 个含递归/输出等待的 wave；relocate 采到卷积/输出等待等不同角色。补充 CU8 采样有 16 个 wave，仍主要是卷积/输出等待角色。两者的累计 stall 不能直接作为同类 CTA 的因果对比；本轮没有用这些采样给 5 μs 收益做定量归因。ATT shader cycles 不换算为 GPU event 微秒，也不跨 GPU 对齐。

另一次 CU20 请求在 profiler 配置阶段被 SDK 以 invalid-argument error19 拒绝，未创建 benchmark 子进程；失败返回码和日志保留，没有使用该次性能数据。三次成功 capture 共 80 waves 全部完成 stitching 和动态总数校验，代码段及全部源码快照匹配。源码快照中的 Python 标准库 `functools.py` 也已单独保存并逐字节验证。

下一步的计算原理候选是短块 gated delta recurrence：将初始状态对四个 token 的投影并行化，再求四个小 value 向量的三角递推，独立生成 attention 输出与各 token 状态快照。160 个 CPU NumPy FP32 用例已验证公式，包括非连续、无效重启、别名 slot、零 beta/value 和较极端 decay；最大 raw-output relative L2 约 2.11e-7。

这仍只是公式与数值可行性验证，没有 GPU 性能结论。独立构造四个快照会增加 rank-one 运算；别名 slot 也需要保证初始状态不可被并行写者提前覆盖。具体公式、实现约束及代价在 [block_delta_design.md](../../results/kimi_mtp_statecarry_20260929/block_delta_design.md)。

全部候选、patch、检查脚本与数据在 [statecarry 实验目录](../../results/kimi_mtp_statecarry_20260929/)。工作区原有 dirty kernel 文件及正确性 commit 保持不变；候选没有接入默认实现。

归档已下载并校验全部 6,837 项 manifest；SHA256 为 `b9aa7dbbc89c5e29ad265d1b2f67b9b79b4a85ad9204950be0bba9e32c29c672`。可直接查看 [三 seed 对照图](../../results/kimi_mtp_statecarry_20260929/node50/confirmation.png) 与 [完整审计](../../results/kimi_mtp_statecarry_20260929/node50/experiment_audit.json)。
