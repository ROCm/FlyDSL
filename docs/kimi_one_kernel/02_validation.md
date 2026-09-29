# 计时与重放验证

关闭用于筛选源码的内置 CTA0 时间戳；保留完整层 GPU-event 计时与全 rank 原始数据。验证脚本包含 compile-only、实际 HIP 驻留门槛，以及六类 slot 模式的三轮变化数据重放。保持已有精度阈值。端到端 attention/state 诊断偏差仍需单独记录，不能用阶段回放通过替代。

完整回放入口依赖顶层 `full_replay` 模块，运行时需将 `experiments/kimi_one_kernel/validation` 加入 `PYTHONPATH`。编译及 occupancy 脚本仅在之后重新授权 GPU 工作时使用。
