# Batch / seq 未验证草稿

用户要求停止时，B∈{1,2,4,8}、L∈{1,2,3,4}矩阵尚未编译或运行，无性能结果。这里只保存可审阅补丁和当时的生成/回放脚本，主kernel仍保持已测B1/S4版本。

`current.patch`基于分支最终选用源码；`staged.patch`基于先恢复的 `../baselines/staged_relocate_events.patch`。两者不能叠加到同一源码树。修改guard不足以支持大shape：已发现MXFP8 projection缺少16行之后的row tile，其余布局、覆盖和等待依赖审计尚未完成。禁止把此草稿描述为已经支持16种组合。

同名旧结果目录中更早的node50矩阵使用不同版本，未纳入此次当前版本矩阵结果。historical脚本保留原路径假设，仅用于续接分析，未作为可直接运行的新benchmark入口发布。
