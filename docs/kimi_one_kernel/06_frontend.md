# 等待前预取和 AttnRes 统计读取

input/router 在等待 activation 之前预取首7个 K64 权重单元；AttnRes tagged 统计由 wave0 协作读取、readlane 取值，保持原串行相加顺序。资源和性能以既有 node46 编译/测量证据为准。
