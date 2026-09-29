# K3 one-kernel 优化归档

2026-09-29，按用户要求暂停优化和测试，整理到 `codex/kimi-k3_one_kernel`。
分支基点为 `3563ef29b7c9f9030e5f9645940948299b03ca46`。原 `codex/kimi-k3-mtp-tuning` 工作区及其未提交文件保持不变。

## 当前启用版本

当前 [kernel.py](../../kernels/kimi_k3_monokernel/kernel.py) 精确恢复 `tail_rotate192_downsplit_events`。
全部140个kernel Python文件与该实验快照逐字节一致；kernel SHA256：
`cd72bbdffb19c4bad0c21257aa9e8bb0a678701f680176b88030af4f7033a0da`。

本分支入口仍限定完整融合实验为 TP8、B1/S4 true MTP。16组合的batch/seq草稿作为补丁保存，未应用到运行源码。

## 性能与证据范围

节点46，TP8，S4 true MTP，layer1，16层graph、50 repeats，无插桩GPU-event计时；每次取最慢rank，随后取中位数。
当前轮只有seed1234筛选，没有完成新的三seed配对。

| 同轮来源 | 完整层耗时 μs |
|---|---:|
| 前一版 front_wave_pf7_recur_events | 127.111405 |
| 当前 tail_rotate192_downsplit_events | 119.808812 |
| staged / 7-kernel | 128.750190 |
| staged × 0.90 目标 | 115.875171 |

当前筛选结果相对staged降低6.94475%，10%目标尚未达到，仍差3.93364 μs。
最后完成三seed配对的版本是 `front_wave_pf7_recur_events`，126.16–127.12 μs；这些数据来自不同轮次，不能跨轮直接当作配对比较。

当前快照的144条阶段回放指标与前一版一致，严格routing和slot重放检查保持通过。
既有端到端诊断偏差仍存在：attention relL2最大0.002040724、recurrent snapshot最大0.000779645；阶段回放通过不能说明这两个偏差已修复，阈值没有放宽。
资源：121 VGPR，LDS67456 bytes，private scratch0，实际2 CTA/CU。精确结果见
[选用记录](../../experiments/kimi_one_kernel/evidence/tail/selected_candidate_node46.json)和
[原始计时/检查数据](../../experiments/kimi_one_kernel/evidence/tail/raw/)。

## 按方向的提交

下列14个提交保存实现、验证或独立实验方向；随后还有一条归档索引与完整性校验提交。

| Commit | 方向 |
|---|---|
| `ea2f6809` | staged / routed-pipeline基础快照 |
| `6f4ee6e1` | 计时与完整重放验证 |
| `7098c085` | recurrence CTA重排、batched tail |
| `06edcf13` | projection split-K4、UG32及服务CTA |
| `93ab0a81` | 严格selector原生wave归约 |
| `a7b56f42` | 前端预取、wave协作读取统计 |
| `7f8dbc3a` | recurrence无状态计算提前 |
| `db1a414f` | tail先全部发送再收集 |
| `49b148b5` | rank相对tail旋转 |
| `5f8af0f6` | down先全部发送再收集（当前选用） |
| `e88c80eb` | latent预取实验归档 |
| `2fa685c2` | UG预取 / 联合发布实验归档 |
| `da966abd` | 向量搬运 / 编译器调度实验归档 |
| `cfc6142c` | batch/seq未编译草稿归档 |

实际运行源码按上述实现提交逐步演进，最终停在已测tail/down版本。实验归档提交仅加入补丁和证据，不改变选用kernel。

## 历史实验与未完成工作

[experiments/kimi_one_kernel](../../experiments/kimi_one_kernel/) 保存90个独立历史源码补丁及2个batch/seq草稿补丁。
历史补丁统一以当前选用源码为基底，可能回退后续优化以精确恢复当时版本，不能依次叠加。每个方向的manifest记录来源、文件前后SHA256、整树指纹及已有验证。

- latent首批预取 / norm gain：pf7的单轮差异只有0.0025 μs，未作为收益启用。
- UG pf1/pf2、联合发布：实现候选已有回放，但性能筛选未完成；相关新增profile只有编译证据。
- raw BF16向量搬运、编译器调度屏障：仅compile-only；未补做HIP occupancy、回放或性能测试。
- batch∈{1,2,4,8}、seq∈{1,2,3,4}：解释为独立状态链数×每链连续token数。草稿未编译、未运行、无性能矩阵；已发现MXFP8 projection缺少16行之后的row tile，其他布局/覆盖/依赖审计未完成。详见[草稿状态](../../experiments/kimi_one_kernel/batch_seq_draft/status.json)。

原始大体积GPU二进制、缓存、完整逐CTA profile和压缩包继续留在外部results目录。提交的profile摘要保留聚合数据及原始文件SHA256/路径，避免把数十MB逐CTA明细放入Git。
旧报告与historical脚本保留原实验路径语义，用于追溯；不把它们当作无需配置即可重跑的便携入口。

## 本次归档校验

这次没有新增GPU运行或编译。仅进行CPU静态与归档完整性检查：
140个运行源码文件一致、92个独立补丁可应用并精确还原目标文件/整树指纹、所有归档Python可解析，原工作区155个文件及Git状态保持不变。
节点46没有本任务遗留的GPU工作或等待控制器；SSH已退出。

复核归档无需GPU或FlyDSL：

```bash
python3 experiments/kimi_one_kernel/verify_archive.py
```

[机器可读校验](../../experiments/kimi_one_kernel/archive_verification.json)记录这次检查范围。
此分支为本地提交，尚未push；后续恢复优化前应重新确认节点空闲及既定验证门槛。
