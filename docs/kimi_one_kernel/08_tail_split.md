# Tail 全部发送后再收集

tail 拆为本CTA所有任务的compute/push阶段和collect阶段，沿用现有tagged mailbox。这是rank相关任务旋转的进度前提，不能单独恢复阻塞的逐任务push+collect再旋转。
