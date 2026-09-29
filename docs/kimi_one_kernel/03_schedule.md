# CTA 调度与 batched tail

将 MTP recurrence 服务任务偏移128个CTA；tail 按4-token组复用权重，通过 wave 协作进行归约与TP交接。保留 BF16 边界和 peer 求和顺序。逐 token / 双 token 拆分试验以独立补丁归档。
