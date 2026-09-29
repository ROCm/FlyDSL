# 任务调度与 tail 多 token 计算

保留 token 分组、CTA 重排、shared-down 提前及 wave-tail 对照快照。

这些补丁是历史实验快照，相互独立，均以本分支最终选用的 `tail_rotate192_downsplit_events` 源码为基底。它们可能回退后续优化以精确恢复当时的测量源码，不能依次叠加，也不表示建议启用。manifest 记录改动文件的前后 SHA256、完整源码树指纹及已有验证。未记录检查/性能的项目仍是未验证或未测量状态。
