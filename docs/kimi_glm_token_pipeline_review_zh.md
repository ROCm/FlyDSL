# GLM 与 K3 完整 MonoKernel：流水复核及下一步改进

日期：2026-09-29。基于工作区 HEAD `3563ef29`，对照本地实际源码与 2026-09-28 的 GLM ATT / timeline 归档。本次为源码、依赖与 CPU 任务枚举分析，没有新 GPU 编译、运行或性能结果，没有改动 kernel。

**结论：GLM 的主要可迁移经验是按依赖安排 CTA、在等待前预取、保留高效多 token 计算和按 tile 归约。GLM S4 自己也等待整组 attention 输出和 expert intermediate，不能把它描述为已实现的逐 token 全层流水。K3 的下一步适合选择性拆除跨 token 等待，并在同一个完整 kernel 中改变任务遍历顺序；不能只将 `staged_samples` 从 4 改成 1。**

## 1. 比较范围与证据

三条路径保持区分：

| 路径 | S4 每层计算 launch | 本次如何使用 |
|---|---:|---|
| K3 staged | 7 | 性能基线；使用相同 attention 优化时另有 128.218 μs 三 seed 结果 |
| K3 routed pipeline | 6 | 局部 UG/SiTU/down/combine/TP 实验，提供已知成功与失败机制的证据 |
| K3 / GLM 完整 MonoKernel | 各 1 | 本次源代码调度比较的主体；二者模型与精度语义不同 |

K3 完整 MTP MonoKernel 的既有同 seed 对照为 165.1855 μs，7-kernel 为 134.2789 μs；这不是此次产生的新结果。137.42 / 137.73 μs 等近期数值属于 6-kernel 路径，不能据此宣称完整 MonoKernel 已达到同样性能。

核对了当前 GLM `kernel.py`、`layout.py`、公共 `config.py` / `layout.py`，四份文件与实际 ATT 编译源逐字节一致。GLM 主 kernel SHA256 为 `a3bb6656c95c98a518ca2d4e6aa39cb581659536d7b225fa38b0f6739841943b`。

证据：

- [已有 GLM ATT/source/timeline 报告](glm_kimi_pipeline_att_zh.md)。
- [本次 CPU 审计脚本](../../results/kimi_glm_pipeline_review_20260929/audit_schedule.py)及[机器可读结果](../../results/kimi_glm_pipeline_review_20260929/schedule_audit.json)。
- GLM 原始 `timeline_analysis_v2/summary.json` 的 40 个 TP8 快照全部 marker 检查通过；原 ATT 共 32 个 wave，13,330 条 ISA 全部映射到源码。这次复用归档，没有重新采集。

## 2. GLM 实际采用什么流水

### 2.1 静态任务图与 CTA 偏移，而非全局动态 ready 队列

[GLM kernel.py](../kernels/glm5_monokernel/kernel.py) 的 `base`（225 行附近）与 `start()`（833 行附近）将任务映射到 `(stage_base + task) % 256`。每个 CTA 仍依次走各阶段，但只执行自己拥有的任务；依赖通过 mailbox 等待。

S4、TP8、indexed/topk2048 的静态枚举：

| 阶段 | 任务数量 | 逻辑 CTA base | 布置的意义 |
|---|---:|---:|---|
| qkv_a | 164 | 0 | CTA 0–163 做输入投影 |
| q_norm | 4 | 164 | 放到没有 qkv_a 工作的 CTA |
| cache | 1 | 168 | 同样避开 qkv_a 计算 |
| q_b | 128 | 169 | 占一半 CTA |
| index_q | 256 | 41 | 前半复用 q_b 的 LDS，后半由互补 CTA 计算；不是简单按统一 base 执行全部 256 项 |
| split | 128 | 41 | 配置使其使用更早释放的 CTA |
| uk | 32 | 169 | 与其依赖的 q_b CTA 复用 |
| down | 256 | 137 | 每个 CTA 承担一个 24-row 输出 tile |

这里是逻辑任务所有权，不是测得的物理 CU 分布。K3 当前完整 kernel 的 input、conv、recurrence、output、projection、UG/down 多从 `task=bid` 开始，post-AttnRes、selector、norm 等服务工作也集中在低编号 CTA。调整这种布置比笼统增加 grid 更有针对性；6-kernel attention 的 relocate 已给出约 5 μs 的正向证据，但尚不能直接外推到完整 MonoKernel。

### 2.2 多数 GEMM 在等待 activation 前先发权重读取

GLM 的 q_b / UK / output / router 都先构造 `pre=[unit(...)]`，再 stage 输入，最后交给 `run_units(..., pre)`。`run_units` 又提前发下一批权重，避免最后一批无用预取。实际 UG ATT 确认下一 sample 的权重读取与当前 sample MFMA 交错。

K3 的 `bf16_mfma` 已有 K-loop 双批次预取；缺少的是许多调用点的**第一次权重读取发生在 input wait 之前**：例如 output 先 `stage_norm()`，再进入 `bf16_mfma()`。因此新实验应是“完整 kernel 的等待前预取”，不能重新把已有 K-loop 预取称为新改进。共享/输出投影权重与 routing 无关，可预取；routed 权重必须先知道 expert ID。

源码表达不能证明编译器最终安排，后续需用带 debug info 的 ISA/ATT 确认。扩大预取会增加活跃寄存器，应该限制首批尺寸。

### 2.3 GLM S4 仍然有整组依赖

- W_o 在 1706–1711 行附近 stage 所有 `S * O_K` attention 输出后一起计算。
- S>1 UG 在 2092–2097 行附近计算所有 sample 的 routing，并 `stage_xq(list(range(S)))`。
- down 在 2340–2356 行附近把全部 sample / route mid 放入 LDS，然后才开始 MFMA。
- UG 的不同 sample 按次序计算并跨 sample 预取；shared 权重同时服务所有 sample 列。

已有 40 个快照显示：

| 观察 | 结果 |
|---|---:|
| 首个 down 预取 marker 比最后 UG emit 提前 | 中位 3.25 μs |
| down compute marker 早于最后 UG emit 的 CTA | 每个快照均为 0 |
| TP push marker 早于最后 down compute 的 CTA | 243–255 / 256，中位 252 |

这些是插桩 marker 关系，不是纯计算时长或可兑现的收益。GLM 的证据支持 UG 尾部与 down 预取、down 与 TP 的重叠，不支持 S4 UG/down MFMA 本身重叠。GLM MLA 的多 query 计算也不具有 KDA 跨 token 的 recurrent-state 链，不能直接搬其 token 分组结论。

## 3. K3 中新核实的四处跨 token 依赖

下面针对 [K3 完整 kernel.py](../kernels/kimi_k3_monokernel/kernel.py)，不是六 launch 的 `routed_pipeline.py`。

### A. 输出分组与 post-AttnRes 的 CTA 所有权需要一起改变

`stage_norm()` 等待组内所有 token 的 head/split；当前组大小为 4，token0 的输出投影等待 token3。

即使假设只把 output 分成单 token，112 个输出任务/token 变成 448 个，现有 `output_task += 256` 会让 CTA0 在完成 token0 的 task0 后，还做 token2 的 task256，然后才进入自己负责的 token0 post-AttnRes。CPU 枚举结果为 CTA0 的先行输出 token 集合 `{0,2}`。

所以 output 分组、output task 遍历顺序和 post-AttnRes owner 都要共同设计。仅拆一次 `stage_norm` 不足以让 token0 进入 MoE。

### B. top-k selector 的 CTA barrier 等齐四个 token

2488 行附近，selector 的四个 wave 各处理一个 token，但都先等待各自 router-ready，再经过同一个 `gpu.barrier()`。较早 token 的 route 发布被较晚 token 拦住。

可比较每 token 独立 selector CTA，或者重写成正确的 wave-local 协作。还必须让该 selector CTA 在执行完无关的后续 token projection 之前就获得运行机会。GLM 的每 CTA 重算 top-k 不宜原样复制：它是 256 experts/top8，K3 为 896/top16，且 packed-key 近似 tie 行为不能替换 K3 的严格 routing 契约。

### C. UG 的阶段循环在进入 down 前遍历四个 token

当前 UG16 每 token `16 routes × 24 tiles = 384` 个任务，S4 共 1536 个。256 个 CTA 各做六个 `bid+n*256` 任务；枚举证明 **256/256 CTA 的最后一个 UG 任务均属于 token3**，然后才进入 down。

因此，即使前端提前产生 token0，也无法在现有阶段循环中让 token0 down 与后续 token UG 充分重叠。需要改成有界 token-group 的 UG/down 推进、可证明进度的角色分配，或阶段间角色复用；不能保留整组 UG 循环仅增加 ready 标记。

这不是说旧实现完全没有任何 UG/down 重叠：某些 CTA 可比其他 CTA 更早完成自己的 token3 UG 并进入 down。结论仅是 token0 down 被本 CTA 的 token3 UG 控制流依赖挡住。

### D. shared-down 被 routed RMSNorm 等待挡住

2833 行附近，tail task 先 `load_f32(routed_inv, sample)`，再 stage shared mid 并开始 shared-down。数学上 shared-down 独立于 routed 分支，应在 shared mid ready 后推进；只让 latent-up 等待 routed TP / RMSNorm，最后合并后做 final TP。

重新分配任务时需明确 shared-down 结果的所有权和存储。为提早计算引入大 partial mailbox 或让长寿命寄存器损害驻留，可能抵消收益；不能默认相当于省去整段 shared-down 时间。

## 4. 只改 token 分组还会造成任务覆盖缺失

`staged_samples` 同时控制 input、output、router/latent/shared 的分组。S4 目前：router 56 tasks，latent 75，shared-up 32，每组共 163。

| 假设 group size | projection tasks | 当前 `projection_task=bid; if ...` 覆盖 | 缺失 |
|---|---:|---:|---:|
| 4（当前合法配置） | 163 | 163 | 0 |
| 2（假设修改） | 326 | 256 | 70 |
| 1（假设修改） | 652 | 256 | 396 |

这不是当前默认配置的 bug，而是朴素修改造成的覆盖错误。需要独立的 input/output/MoE 分组参数与完整的任务遍历；循环补齐后仍要处理上一节的跨 token 控制流依赖。上述反事实只做了 CPU 数学枚举，没有启动这些不完整配置。

## 5. 改进优先级与已排除的重复方向

| 优先级 | 建议实验 | 与既有实验的区别 | 成功证据 |
|---|---|---|---|
| P0 | 建立完整 kernel 的 token×stage/tile 任务图，枚举覆盖和等待环；重排 norm、selector、output 等 owner | 全层调度；此前 relocate 只验证 staged attention | 所有任务覆盖，生产者有执行机会，TP/状态别名语义保留 |
| P1 | input 保留4行，独立比较 output/MoE 的1/2/4行；同时拆除 A/B/C 的隐含整组依赖 | 不等于把全局 group 改为1，也不等于旧6-kernel局部融合 | token0后端与后续KDA确实重叠，且最后token完成时间下降 |
| P1 | shared-down 提前，latent-up 单独等 routed norm，按输出 tile 合流 | 完整 tail 任务图重构，尚未由 local-down 试验覆盖 | 原语义全通过，新增partial/活跃值成本小于隐藏的等待 |
| P2 | output/router/latent/shared 的首批权重在等待前预取，路由metadata跨任务复用 | 局部UG的提前预取已试过；这是完整kernel不同等待位置 | ISA确认顺序，资源合格，关闭插桩整层A/B改善 |
| P2 | 在token间流水基础上比较token-group较粗交接和tile细交接，允许CTA角色复用 | GLM没有证明越细越好；旧route三个chunk staging已回退 | 同计算、同数值下比较poll/LDS/读请求与整层延迟 |
| P3 | 清理静态LDS区与控制生命周期 | K3已有共享X区，不应把它描述为完全没有复用；GLM自身资源占用也不低 | 编译资源或访存有实证瓶颈时再推进 |

不要作为“新方向”重复推荐：

- 同时按 expert 分组 UG/down：compact 已减少约17–18% HBM读请求，但整层慢约2 μs。
- 中间 BF16 全留 CTA：U64/U192 local-down 已试，部分和读写和归约代价导致回退。
- 独立 ready flag：实现已通过检查，但完整层394.773 μs；不能靠删可见性操作宣称解决。
- 仅提高 grid、全展开 routes、调 polling sleep：已有资源门槛或负结果。
- 直接复制 GLM UG8：GLM K128 chunks=48，K3=28，8-wave 均分不适用；K3 U8 会增加任务数，需联合设计。
- 假设 GLM 没有全局 mid、没有 polling、使用完全不同的原生FP4指令：与已核实源码/ATT不符。

## 6. 下一轮验证口径

先固定 TP8、B1/true-MTP3、S4、相同权重/路由/精度。完整 MonoKernel 原版、候选以及应用相同 attention 优化的7-kernel都作为对照；既有6-kernel结果用于解释局部变化，不替代完整单launch结果。

轻量 timeline 应新增 `state_ready(token,head,split)`、`norm_ready`、output/attention-TP、post-AttnRes、router/top-k、latent/shared、UG/down、routed-norm、shared-down、final-TP等边界，并区分“开始轮询”“首个实际计算”和“结果发布”。跨 rank 只比较各自相对时间，ATT cycles 不换算成微秒。

先 compile-only 和真实 occupancy，再进行既有独立参考、严格 routing、全部状态快照、无效/重启/别名slot及变化payload的graph重放检查。最后以关闭 soft/ATT/PMC 的完整层事件计时、最慢rank、多个seed交替测量评价。token0更早完成不能替代最后token完成时间的改善；不承诺仅靠融合实现10%。

CPU 审计复现：

```bash
python3 /home/zihuang/work/mega_transformer/results/kimi_glm_pipeline_review_20260929/audit_schedule.py
```

`schedule_audit.json` 保存源文件哈希、GLM任务base、K3全部UG任务到token的映射、分组反事实覆盖和既有timeline关键统计。它不证明GPU调度时长、无死锁或数值正确性；这些仍是实际实现后需要完成的检查。
