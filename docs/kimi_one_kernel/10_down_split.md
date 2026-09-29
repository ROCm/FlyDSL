# Down 全部发送后再收集

同样分离down的本CTA计算/发送与collect/norm统计发布，保留原任务及浮点归约顺序。这是本分支最终启用的快照 `tail_rotate192_downsplit_events`。

node46、TP8、B1/S4 true MTP、layer1、16层graph、50 repeats，seed1234：119.808812 μs；同轮staged128.750190 μs，降低6.94475%。10%目标未达到；该轮三个seed正式配对尚未完成。144条阶段回放指标与上一版一致，既有端到端attention/state诊断偏差仍保留。
