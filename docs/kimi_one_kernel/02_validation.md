# 计时与重放验证

关闭用于筛选源码的内置 CTA0 时间戳；保留完整层 GPU-event 计时与全 rank 原始数据。验证脚本包含 compile-only、实际 HIP 驻留门槛，以及六类 slot 模式的三轮变化数据重放。保持已有精度阈值。端到端 attention/state 诊断偏差仍需单独记录，不能用阶段回放通过替代。

完整回放入口位于 `kernels.kimi_k3_monokernel.tools.full_replay`，不再依赖额外的 oracle PYTHONPATH。seq 1–8 的编译及 occupancy 工具位于 `experiments/kimi_one_kernel/seq8/`。MTP 回放覆盖六类 slot 模式，每类使用三轮变化数据；seq 8 包含完整九个状态快照，batch 间的状态池独立。
