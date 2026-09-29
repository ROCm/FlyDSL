# Projection / UG 并行度

latent/shared projection 使用 split-K4，UG 将相邻16行任务配对为32行工作；post-AttnRes、selector、shared 与 norm 服务CTA错开。完整实验入口仍限制为 TP8、S4 true MTP。源码直接取自已经验证的 `full_tail4_proj4_ug32_events`，不引入新算术变更。
