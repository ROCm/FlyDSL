# Recurrence 无状态计算提前

将不依赖前一token状态的输入计算放到 predecessor fence 之前；状态读取和写回仍在原fence之后。invalid/restart/alias slot 的顺序保持不变。

该快照完成三个seed同机配对：126.992501 / 126.162499 / 127.121255 μs；对应 staged 为128.621280 / 128.616281 / 128.708810 μs。仍未达到10%目标。
