# 严格 selector 归约

用原生 wave MAX score 和 MIN expert ID 归约减少 top-k 开销，保持严格score/id tie规则。偏置保留、local比较、双token MoE流水和down预取实验保存在独立补丁中。
