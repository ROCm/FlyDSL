# Kimi K3 MTP3：无 soft 插桩 ATT 对照与八个候选实验

2026-09-28，接续 [768 CTA 后续方向](kimi_mtp_next_steps_zh.md) 与 session `01a0e652-c9e1-7693-9511-b050789773e8`。节点为 node47 / MI355X gfx950，容器为 `ziming_huang_work`。工作区 HEAD 保持 `3563ef29b7c9f9030e5f9645940948299b03ca46`。

**ATT 确认末轮存在未使用的权重预取，但删除后没有获得稳定的整层加速。按 sample 复用 activation、提前 UG 预取、consumer polling 退避也未显示足够收益，本轮八个候选均未接入工作区 kernel。** 实验源码、补丁、资源、正确性结果和四份 ATT 已独立归档。默认仍采用 staged；此前已验证的 512/768 实验接口保持原状。

## 测量口径

输入为一个请求的 true-MTP3（S4）、TP8、staged KDA、native MXFP4 routed experts。融合范围为 routed UG/SiTU/down/combine/TP；router/projections、shared/latent tail 仍独立。对照为 U32/D16、prefetch4、interleaved、route-prefetch，512/P256 与 768/P544；两者都有 224 个 consumer CTA。

性能数字仅来自 **关闭 soft 和 ATT 的完整层 GPU events**：graph 内 16 层，每次取最慢 rank 的每层时间，再取 replay 中位数。筛选使用 seed1234、50 replay；尾部预取候选复测使用 seed1234/2025/42、各 100 replay，并轮换 staged/base/候选的顺序。筛选与多 seed 确认分别列出，不把 MoE-only 与整层相减推算 attention 时间。

ATT 使用相同的第 65 次匹配 dispatch（iteration `[64]`，dispatch10896），GPU0、CU1、四个 SE、全部 SIMD、96 MiB buffer，`att_serialize_all=False`，没有 soft 计时写入。ATT 本身仍会扰动执行，其 event 时间不参与性能表。

所有变体在编译前启用 debug info，使用独立缓存。每次首次启动 polling kernel 前先 compile-only，再检查实际 HIP occupancy；768 CTA 至少要求 3 CTA/CU，512 CTA 至少要求 2 CTA/CU。先做 TP1 独立参考与改变 payload 的 replay，再运行 TP8 正确性和性能。

## 512 与 768 的 ATT 对照

本轮重新编译的两个 baseline `.text` 与前一轮测量产物逐字节一致：512 为 9,920 bytes，768 为 9,984 bytes。四份 capture 的代码对象均与相应编译资源匹配，源码快照逐字节一致；全部 352 个采样 wave 完整 stitched，动态指令计数、stall 和 interval 累加与 decoder code table 一致。

| 采样指标 | baseline 512 | baseline 768 |
|---|---:|---:|
| producer / consumer waves | 32 / 32 | 72 / 24 |
| 每 consumer wave 的 down MFMA | 96 | 96 |
| 每 consumer wave 的 raw weight fragment load | **27** | **27** |
| consumer entry → first MFMA，中位数，ATT cycles | 19,586 | 22,812 |
| consumer entry → 第一 K128 完成，中位数，ATT cycles | 19,752 | 22,972 |
| consumer entry → last MFMA 结束，中位数，ATT cycles | 60,666 | 55,002 |
| `stage_mid_chunk` 的 while 行占 scoped stall | 64.91% | 66.51% |

每 consumer wave 数学上消费 24 个 K128 fragment、执行 96 个 MFMA。ATT 实际记录了 27 次 raw weight fragment load，因此末轮预取没有被编译器消除。这是动态 wave 指令计数，不能等同于 27 次 HBM transaction。

768 的采样 consumer 更晚开始第一批计算，但更早结束 down 的 MFMA 区间，与之前轻量 timeline 的供给/竞争趋势一致。这里的区间包含等待；不是纯 MFMA 时间，也不是整层时间。ATT 单位保持 shader cycles，不换算成 μs，不比较不同 GPU 的绝对时间戳。

producer 的工作量也不同：512 的 32 个采样 wave 各执行 3 个 UG task（168 MFMA / 42 raw loads）；768 中 40 个 wave 做 1 个 task、32 个 wave 做 2 个 task（56/112 MFMA，14/28 raw loads）。768 producer 的权重和 scale 读取源码行分别占其 scoped stall 的 25.61% 与 21.46%。由于采样角色和 task 数不同，不用两个 capture 的全体 stall 总量判定加速。

图：[512/768 wave timeline PNG](../../results/kimi_mtp_att_followup_20260928/node47/att_analysis/att_wave_comparison.png) / [SVG](../../results/kimi_mtp_att_followup_20260928/node47/att_analysis/att_wave_comparison.svg)。上图按角色分组，纵轴不是逻辑 CTA ID；下图展示一个 consumer wave 的指令区间。各 capture 独立以最早采样 wave entry 为原点，统计 scope 从各 wave 自身 entry 到最后一个 MFMA，排除后续 pack/TP 的 profiler 干扰。

## 尾部预取：确实少了三次 load，但收益不稳定

`epilogue` 保留前七组 route 的运行时循环，把最后一组消费拆出，不完整展开八组 route，不改变数学顺序或 tagged payload 协议。TP8 仍为 VGPR78、SGPR73、LDS15,360 B、无 spill/scratch、3 CTA/CU。

初筛 base 的两次完整层为 143.040 / 143.036 μs，epilogue 为 142.364 μs，因而进入三 seed 复测：

| seed，整层 μs | staged | baseline 768 | epilogue 768 | epilogue − baseline |
|---|---:|---:|---:|---:|
| 1234 | 134.059 | 142.343 | 142.730 | +0.387 |
| 2025 | 133.824 | 142.931 | 141.851 | −1.080 |
| 42 | 133.505 | 142.426 | 142.206 | −0.220 |
| 三 seed 中位数 | 133.824 | 142.426 | 142.206 | — |

中位数差约 −0.220 μs（0.15%），但逐 seed 方向不一致，没有证明稳定收益。staged 仍明显更快。各 seed 下的逐 replay 样本保留在 JSON 的 `critical_times_us` 中；三个运行的中位数不是置信区间。

修改后的 ATT 采样到 32 个 consumer wave，每个均为 **24 raw weight loads / 96 MFMA**，确认改动达到了预期。consumer entry→last MFMA 中位数为 55,084 cycles（baseline 为 55,002）；while 行仍占 scoped stall 的 67.58%。减少冗余读取未解除主要的 intermediate 依赖等待，因此不推广这次改动。

图：[尾部预取对照 PNG](../../results/kimi_mtp_att_followup_20260928/node47/att_epilogue_analysis/att_wave_comparison.png) / [SVG](../../results/kimi_mtp_att_followup_20260928/node47/att_epilogue_analysis/att_wave_comparison.svg)。

## 其余七个候选的筛选结果

下表均为 seed1234、50 replay 的完整层筛选，**不是多 seed 性能结论**。各阶段单独重复 base，以观察运行间变化。

| 候选 | 变更 | 完整层 μs | 同阶段 base μs |
|---|---|---:|---:|
| sample_schedule | sample 固定的 producer 任务分配 | 144.327 | 143.040 / 143.036 |
| sample_reuse | 上述映射，activation 每 producer 只 staging 一次 | 144.006 | 143.040 / 143.036 |
| prefix_schedule | 保持原始第一任务，后续任务固定 sample | 144.422 | 143.036 / 142.726 |
| prefix_reuse | 上述映射，再复用 activation | 143.510 | 143.036 / 142.726 |
| ug_early_prefetch | 首组 UG 权重/scale 预取移到 activation staging 前 | 142.935 | 142.726 / 142.545 |
| poll_sleep1 | 仅 intermediate 重试加入 `s_sleep 1` | 142.914 | 142.545 / 142.762 |
| poll_sleep4 | 仅 intermediate 重试加入 `s_sleep 4` | 142.545 | 142.545 / 142.762 |

sample 分组映射与 prefix 映射均覆盖全部 768 个 UG task 且无重复。prefix 保留各 producer 的原始第一任务；在 P544 时仍是 224 CTA 做两个 task、320 CTA 做一个 task。activation 复用仍保留保护 `red` 跨任务复用的 barrier。减少源码层 staging 次数未转化为完整层收益；没有单独采集这些映射的 ATT，不能把退化原因确定归结于某一个硬件因素。

UG 提前预取使 TP8 VGPR 从 78 降至 74，仍有 3 CTA/CU，但完整层没有显示加速。资源改善本身不足以作为接入依据。

MoE-only 数字保存在原始结果中。尤其 prefix 两个候选的单次 MoE-only 测量差异很大，与完整层排序不一致，没有将这些单次局部数字作为获益结论或用于推导 attention 延迟。

### Polling 退避的 ATT 核验

原 `spin_pause()` 仅执行 `s_nop 0`。两个候选只替换 `stage_mid_chunk` 内的暂停指令，全局 spin helper、TP 等待和 tagged payload 读取协议保持原状。

`sleep4` 的 ATT 显示暂停确实执行；每个 consumer wave 的 sleep 次数与 retry load 次数相等。其采样数据如下：

| 指标 | baseline 768 | sleep4 768 |
|---|---:|---:|
| producer / consumer waves | 72 / 24 | 64 / 32 |
| 每 consumer wave retry load 数，中位数 | 20.5 | 19.5 |
| 每 consumer wave sleep 数，中位数 | 0 | 19.5 |
| 每 consumer wave raw weight loads / down MFMA | 27 / 96 | 27 / 96 |
| consumer entry → last MFMA，中位数，ATT cycles | 55,002 | 59,330 |
| while 行占 scoped stall | 66.51% | 69.72% |

retry 计数仅统计 `stage_mid_chunk` 内失败后重读的 load，不包含预取的首次 tagged payload 读取。每份 capture 采样的角色和 task 组合不同，这个小幅计数变化不能证明全 GPU 访存流量下降。完整层筛选也没有足够收益，所以不做推广；不能把 ATT scoped stall 百分比解释为可直接消除的整层耗时比例。

图：[polling 对照 PNG](../../results/kimi_mtp_att_followup_20260928/node47/att_poll_analysis/att_wave_comparison.png) / [SVG](../../results/kimi_mtp_att_followup_20260928/node47/att_poll_analysis/att_wave_comparison.svg)。

## 验证与归档

最终完成 **39 次 TP8 检查运行**：2 次初始检查、24 次筛选、9 次多 seed 确认、4 次 ATT 采集。8 个候选均通过 TP1 独立 FP32 参考、staged 对照，以及三次改变 activation payload 后的 graph replay。9 组源码（baseline + 8 候选）共 27 个编译资源记录通过 occupancy gate；均无 spill/scratch。

TP8 检查包括 attention 与 conv/recurrent state 误差门槛、MTP incoming snapshot 不变、独立 expert mid/routed 参考、精确 routing、routed TP、tail reduce/residual 和 AttnRes delta，以及 graph replay 重查。39 次运行的所有 rank 都通过。验证日志中的 ATT event 时间仅作为采集过程产物保存，不纳入性能汇总。

证据目录：[node47](../../results/kimi_mtp_att_followup_20260928/node47/)。主要入口：

- [TP8 与 ATT 核验汇总](../../results/kimi_mtp_att_followup_20260928/node47/verification_summary.json)、[TP1/资源/源码审计](../../results/kimi_mtp_att_followup_20260928/node47/experiment_audit.json)。
- [512/768 ISA 一致性](../../results/kimi_mtp_att_followup_20260928/node47/base_isa_comparison.json)、[各变体补丁](../../results/kimi_mtp_att_followup_20260928/node47/variant_patches/)。
- 四个 `att_*_g*` 目录：原始 `.att`、decoded `ui*`、CSV、源码快照和该 kernel 引用的 code objects；各 `resources_*` 目录另保留 HSACO 与实际 HIP occupancy。
- `*.command.json`、`*.log`、`*.rc`、`*.json` 保留每次执行的命令、结果与退出状态；`base/` 和各候选目录保留源码。
- [文件 SHA256 清单](../../results/kimi_mtp_att_followup_20260928/node47/manifest.json)、[归档选择清单](../../results/kimi_mtp_att_followup_20260928/node47/archive_inventory.json)。仅省略与采样 kernel 无关的大型库 code objects、缓存和 `.git` 等；引用的 kernel code objects 均保留。

在下载目录可用 `ATT_RUN="$PWD" python3 verify_results.py` 复核 TP8、源码快照与代码对象；脚本使用归档内 `previous_validation.py`，支持路径迁移。`analyze_att.py` 使用随包保存的 `hotspot_analyzer.py` 分类器，可重新生成对应图表。GPU 重跑仍需技能规定的 ROCm/FlyDSL/容器环境，不能只靠归档在任意机器执行。

本轮未修改工作区 kernel，也未提交或推送。四个关键入口文件与实验前 SHA256 一致，原有 dirty worktree 和独立正确性提交保持不变。

## 后续判断

用户随后要求从计算数据流和 HBM 重复读取重新审视方向，已补充 [三 seed PMC 与 compact 数据流审计](kimi_mtp_dataflow_zh.md)。硬件计数显示当前 pipeline 的 HBM 读请求比 staged routed GEMM1+2 多约 23%，因此优先级调整为两级 expert 分组与多 sample 列权重复用；下面保留局部实验结束时对角色复用/tail 边界的判断。

这轮实验排除了几种便宜改动能直接消除差距的假设。当前更值得研究的是 **UG 结束后的 CTA 角色复用，以及 routed 与 shared/tail 的边界**：先通过完整层逐 kernel timeline 确定该边界是否处于关键路径，再设计有限、可证明无死锁的任务接续。增加 producer 数量、删除少量预取或加入固定 sleep，均未证明能解决当前的依赖等待与资源竞争。

该结构性方向尚未实现或验证，不计入本轮成果。它必须保持确定的输出 ownership、完整 RMSNorm/TP 依赖、tagged payload 协议与驻留条件；也不能简单恢复此前已退化的 256-CTA 混合调度，或重复增大 D tile。当前实测选型仍为 staged。
