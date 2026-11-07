# EP MegaMoE 全融合重构：给 Claude Opus 5.5 的任务 prompt

以下内容可以直接作为新会话任务。知识基线更新至 2026-11-06；执行时必须重新核对源码和环境。

---

你负责 AITER main 上 EP MegaMoEV2 的路径归一和冗余清理。#6881 已合入，当前同时存在 staged fixedslot/compact、small-M 和 large-M 实现，配置、计算 helper、通信资源和 host 入口重复严重。请从源码出发实施重构，不是再加一层包装。

## 先读知识库，再核对当前代码

私用知识库在 FlyDSL PR https://github.com/ROCm/FlyDSL/pull/1111 ，分支 `ghu/megamoe-engineering-skill`，**DO NOT MERGE**。可以 fetch 此分支到独立 worktree 阅读，不要把这个 FlyDSL 分支作为 AITER 的代码基线。

先完整阅读该分支的 `.claude/skills/megamoe-engineering/SKILL.md`，再按 router 阅读本任务必需的参考：

- architecture-and-protocol.md
- ep-fusion-and-unification.md
- tp-megamoe-entry.md
- validation.md
- correctness-and-debugging.md
- review-and-cleanup.md
- serving-and-workspace.md
- performance-and-tracing.md
- repository-map.md、environment-and-safety.md，以及相关模型章节

再完整阅读实际采用的 FlyDSL 版本对应的 `.claude/skills/kernel-code-cleanup/SKILL.md`。API 必须在 AITER 支持的 FlyDSL build 上可用；不要为套用最新 API 随意更换 serving 栈。

只处理这次 AITER EP 重构。不要恢复旧的 Byte/SmoothQuant、GLM、EP4 调优或自研 WS 实验，也不要修改 TP 算子。其他模型/精度是回归覆盖，不是顺手重构目标。

## 代码基线与工作方式

从官方 AITER 最新 main 新建干净 worktree/分支，记录完整 SHA、依赖版本、dirty 状态和真实 import 路径。知识库审计锚点为 `4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00`，不是要求回退到它。保留用户其他 worktree 和未提交修改。

该审计 main 已固定 FlyDSL 0.4.2，旧 11 月 4 日实验用的是 0.4.1。执行时按最新 main 的 pin 对齐真实安装，再重跑 baseline；不要混合旧 Python 包、不同版本 native binding 或外部 build 路径。

先画清楚现有调用链、入口选择、buffer/epoch 所有权和 AOT 覆盖，再给出文件级迁移清单。机械去重与协议/资源布局变更分阶段，分别保留可回退、已验证的 checkpoint。不要混改数值顺序、通信精度、模型分片和无关文件。设计性变更先说明，不要伪装成 cleanup。

## 目标架构

1. 唯一生产 `MegaMoEV2.forward()` 入口、同一份 canonical config/BundlePlan，以及统一的 prepare/bind/preload/lifecycle。尽量扩展已有结构，不新增 Manager/Factory 层级。
2. small-M/large-M 收敛为调度和资源特化；fixedslot/compact 保留确有必要的协议/布局差异，不再各带完整算子系统。允许不同 GPU kernel 模板，避免一个巨大运行时分支拖累所有 shape。
3. 去除 `mega_moe_fused_gemm.py` 中重复的配置类型/selector，以及真正等价的 load/MFMA/quant/scatter helper。复用现有配置和 GEMM 公共模块，不要挪到另一个文件继续重复。
4. SmallMRuntime/LargeMRuntime 不再各自持有完整独立通信 op 和大缓冲。由统一 owner 管理兼容布局的重型 arena，按计划提供 view；计数器、epoch、ready 和在途状态，只有完成生命周期证明后才能共享。
5. 保留 dispatch 去重/fanout、indexed payload、producer 退役/consumer 补位、ready-order 安全性和既有 AOT 能力。不要为了统一而回退已有性能优化。
6. 移除生产 SMALL_M/LARGE_M 环境变量选路依赖；默认通过确定性配置自动选。测试可以显式固定计划。不要把实验开关改名为更多公开参数，也不要顺手删除无关运行时开关。

## 参数、入口和框架契约

- MTPR 是容量/ABI；本地 TPR 是输入物理行数；runtime max_tokens 是本次整个 EP 组一致的最大物理行数或安全上界。Graph padding 算物理行，不能只看有效请求数。
- 所有 rank 一致选择协议和容量桶。不能按各自 `x.shape[0]`、mask、cache 命中等独立切 small/large。零 token rank 仍参加协议。
- 使用框架已有全组元数据，不每层新增 all-reduce。没有 runtime hint 的旧调用保持安全可运行，采用保守容量计划，不推断成 local TPR。
- 统一 `forward`、`forward_prequant` 的选路和输出契约，保留显式 stream、无效 slot mask、权重重绑及旧接口兼容。尤其检查 `slice_output=False` 的父容量/子 bank 输出形状差异。
- 旧专用入口可暂时作为薄兼容包装，不能留第二套 selector。capture 前准备所有可达资源/产物；forward 不能临时分配、编译或依赖 Graph replay 不会执行的 Python 状态更新。

## large-M 容量和全 shape 覆盖

去掉 `MTPR==8192` 人为白名单。当前融合目标首先是已实现的 gfx950 / EP8 / V4-Pro H7168 I3072 E384 topk6 / A8W4 / SiLU limit10 组合，覆盖其有效 TPR<=MTPR，不宣称所有网络天然都已融合。

之前的本地扩容 patch 基于 PR head `0a85ae56868bc4bd8ddb19271e5b7b9040e809c3`，未提交/未合入；1024～32768 的 88 组配对功能/Graph 通过，证明 8192 不是算法限制，但不是当前重构已验收。

必须处理两个已知问题：

- 32768 满载 uniform 曾从 19149.3 增至 21398.0 us，耗时回退 11.74%。不能默认强推后称全 shape 优化。
- 32768 仅 rx_em：staged indexed 约 1.75 GiB/卡，扩容 large-M 非 indexed 约 10.54 GiB/卡，而且当时两套同时分配。需要迁入合适的 indexed/dedup 能力或给出经验证的等效方案，不能只复用大尺寸非去重分配。

保留实际编码范围、内存容量和地址宽度检查。65536 曾触发既有约 2 GiB scatter/resource ABI 限制，不能把移除 whitelist 理解成可以删除安全检查。

全 shape 功能覆盖不等于每个 shape 强制同一融合实现。未适配或确认回退的场景可暂用公共组件组成的 staged 实现，明确标注；不要用四套长期复制实现掩盖覆盖缺口。

## 实施与验收

先保存同栈 baseline，再逐阶段实施，每阶段做对应验证：

1. CPU/config：支持矩阵、边界、跨 rank wire/schema 一致、地址编码、buffer 容量、编译 key 和所有 bundle variant。
2. 机械重构：各原路径前后数值一致；确定性路径要求逐位一致。不要假定旧 small-M 与 staged 本来就是零误差。
3. 8 卡 eager/FP32 reference/Graph/stress：TPR 1、2、4、8…32768，MTPR=TPR/MAX 和中间有效组合；加入非二次幂输入及 511/512/513、1023/1024/1025 边界。
4. 路由：uniform、hot destination/expert、rank-balanced hot、Zipf/mixed skew、fanout 边界、uneven/zero ranks、无效/填充 slots。路由倾斜不等于输入长度不均，分别构造。
5. 生命周期：small bank/compact 切换、权重与 layer slot 交替、生产层数 queued Graph、重复调用/epoch wrap、保存输出值；并发只按实际支持的契约验，不宣称顺序通过等于 TBO 安全。
6. AOT：生产枚举入口、新空缓存、expected/actual job 数、全部 EP rank、新进程 strict run-only GPU 执行和 Graph；单独构建 16384-only/32768-only profile，确认 small 子 bank 不漏编；缺失产物负对照必须失败。CPU lookup 不冒充 GPU 验收。
7. ISA/资源：比较公共 API 替换前后 load/store 宽度、地址、fence/atomic scope、wait/MFMA 调度、LDS/VGPR/scratch 和 residency。裸 MLIR/inline asm 没有等价高阶 API时记录原因，不藏到新 facade 后假装清掉。
8. 性能：相同输入、路由、权重、runtime 和计时方法，8-rank max，3 轮×10 次 Graph replay，报告中位数/范围。近噪声时增加配对；不要拿不同 rank 的 Stage1/Stage2 max 相加当 E2E。发生回退时用 ATT/PMC 锁定主要矛盾。
9. 框架相关的接口、协议、AOT 或默认选路改变后，验证 ATOM 整网；直接给出可独立复制的原生 server/bench 命令，不依赖私人脚本。

验证只能用获准的 46/47 自有 guoliang 容器；先核对 GPU holder 和归属，不继承旧会话杀别人进程/容器的权限。没有空卡可先完成 CPU、编译和代码审查，明确剩余 GPU gate；不要编造结果。SSH 配置和 key 从用户本地配置查询，私钥内容不得读取或发布。

## 交付要求

- 源码级设计和文件迁移清单、公开接口与框架兼容说明。
- 分阶段最小 diff、去除哪些重复模块/配置/资源及真实显存变化。
- 完整精度/Graph/AOT/压力结果，已跑与未跑分开。
- 性能表按 TPR/MTPR、实际 fixedslot/compact 和 fused/staged 列明 main / this change / Speedup，公式为 baseline_us/candidate_us。
- 可复现原生命令、版本、原始日志位置、当前任务状态和可恢复 checkpoint。
- 不自动 push 或提 PR；完成验证和整体 self-review 后先向我汇报确认。不能为了省代码牺牲正确性、既有优化或强行给所有 shape 同一个配置。
