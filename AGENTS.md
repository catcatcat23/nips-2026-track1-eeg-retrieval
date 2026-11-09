# 接手规则

先读 README.md、docs/HANDOFF.md、docs/EXPERIMENT_MAP.md 和 STORAGE_POLICY.md。

- 所有训练产物、缓存、权重写数据盘，不能写系统盘或 /tmp。
- 原始结果在 evidence/runs；历史实际代码用 source_snapshots.json 查找。当前研究源码不一定等于旧run版本。
- 新实验先记录数据、checkpoint、特征层、候选空间、聚合方式和选模规则。不要仅凭实验ID判断配置。
- 默认聚合为逐试次预测后按同人同图求原始均值。不要静默改成先平均EEG。
- blur本地指标与clean提交指标分开。公开测试选模结果称开发结果。LOSO不得让留出人进入源训练。
- 负结果同样保留。不能通过挑图片、种子或测试最佳步数证明假设。
- 不自动运行历史launcher，不删除共享cache或选中checkpoint，不中断其他用户进程。
- 中文汇报直接、准确；方法变化与证据一起说明，不把性能改善写成机制证明。
