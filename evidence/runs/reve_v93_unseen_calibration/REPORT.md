# v93：新受试者校准迁移实验

主模型只在 S3—S10 上训练和选权重。校准模块在 S2 上训练后测试 S1，另一个模块在 S1 上训练后测试 S2。
无身份监督与身份监督模型共享样本、划分和种子。测试时只输入当前 EEG，不利用目标受试者标签或整体统计量。

| 模型 | 当前状态 | 运行日志 |
| --- | --- | --- |
| base | complete | [base](base/) |
| identity | complete | [identity](identity/) |

完整对照表见 [校准实验报告](calibration_report/REPORT.md)。
原始逐种子指标和完整运行状态见 [summary.json](summary.json)。
方法来源及实验边界见 [reference.md](reference.md)，固定配置见 [calibration_protocol.json](calibration_protocol.json)。

seen 对照使用八位源受试者的混合数据，novel 校准使用一位新人；两者的差异也可能来自受试者多样性。主要对照是同池、同种子下身份模型与无身份模型的绝对检索成绩。
这不是 MLDG 或 Episodic-DG 的完整复现，也不把身份聚类更清楚当作适配能力更强的证据。
