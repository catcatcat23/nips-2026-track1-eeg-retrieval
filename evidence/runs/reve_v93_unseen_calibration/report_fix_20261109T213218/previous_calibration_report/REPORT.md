# 未见受试者校准器迁移实验汇报

更新时间：2026-11-08T21:54:46.741387+00:00。实验目录：`/mnt/DATA-4/anteng/eeg-retrieval/experiments/reve_v93_unseen_calibration`。

本次预期 30 个校准器训练结果、40 条受试者评估记录；有效完成 0 条。每个 seen 校准器评估 S1、S2，novel 校准器只评估另一位受试者。

主模型仅使用 S3—S10 训练和验证。novel 在 S2 训练 C 后测试 S1，另一个 C 在 S1 训练后测试 S2；seen 对照在 S3—S10 的混合单试次上训练 C。所有测试指标仅供汇报，没有用于此脚本的选模或排序。

## 完成状态

| 主模型 | worker | pipeline | 完成评估／预期 |
| --- | --- | --- | --- |
| base | complete | complete | 0／20 |
| identity | complete | complete | 0／20 |

## 逐人结果

数值为已完成种子的均值，单位为 %，每格按 Top-1／Top-5 展示。缺失种子不填零；未满全部种子的结果标为 partial。两模型之间的配对结论以下一表的共同完成种子为准。

| 模型 | C 训练→测试 | 完成种子 | 状态 | clean 单次 | clean 聚合 | blur 单次 | blur 聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | S3-S10→S1 | 0／5 | missing | —／— | —／— | —／— | —／— |
| base | S3-S10→S2 | 0／5 | missing | —／— | —／— | —／— | —／— |
| base | S1→S2 | 0／5 | missing | —／— | —／— | —／— | —／— |
| base | S2→S1 | 0／5 | missing | —／— | —／— | —／— | —／— |
| identity | S3-S10→S1 | 0／5 | missing | —／— | —／— | —／— | —／— |
| identity | S3-S10→S2 | 0／5 | missing | —／— | —／— | —／— | —／— |
| identity | S1→S2 | 0／5 | missing | —／— | —／— | —／— | —／— |
| identity | S2→S1 | 0／5 | missing | —／— | —／— | —／— | —／— |

## 零校准与校准增量

下面的零步和增量取自与最终结果相同的完成种子。零步在同一主模型、同一测试受试者下重复出现，是配对参照，不是额外独立实验。增量单位为百分点。

| 模型 | C 训练→测试 | clean 聚合零步 | clean 聚合增量 | blur 聚合零步 | blur 聚合增量 |
| --- | --- | --- | --- | --- | --- |
| base | S3-S10→S1 | —／— | —／— | —／— | —／— |
| base | S3-S10→S2 | —／— | —／— | —／— | —／— |
| base | S1→S2 | —／— | —／— | —／— | —／— |
| base | S2→S1 | —／— | —／— | —／— | —／— |
| identity | S3-S10→S1 | —／— | —／— | —／— | —／— |
| identity | S3-S10→S2 | —／— | —／— | —／— | —／— |
| identity | S1→S2 | —／— | —／— | —／— | —／— |
| identity | S2→S1 | —／— | —／— | —／— | —／— |

## 配对差值

identity−base 为同一 C 数据池、同一种子下两种监督模型的最终成绩差值；novel−seen 为同一主模型、同一种子下两种 C 训练来源的差值。正值表示前者更高，单位为百分点。只纳入双方都完成的种子。

| 对比 | 条件 | 测试人 | 配对种子 | clean 单次差值 | clean 聚合差值 | blur 单次差值 | blur 聚合差值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| novel_minus_seen | base | S2 | 0／5 | —／— | —／— | —／— | —／— |
| novel_minus_seen | base | S1 | 0／5 | —／— | —／— | —／— | —／— |
| identity_minus_base | seen | S1 | 0／5 | —／— | —／— | —／— | —／— |
| identity_minus_base | seen | S2 | 0／5 | —／— | —／— | —／— | —／— |
| identity_minus_base | novel01 | S2 | 0／5 | —／— | —／— | —／— | —／— |
| novel_minus_seen | identity | S2 | 0／5 | —／— | —／— | —／— | —／— |
| identity_minus_base | novel02 | S1 | 0／5 | —／— | —／— | —／— | —／— |
| novel_minus_seen | identity | S1 | 0／5 | —／— | —／— | —／— | —／— |

## 缺失、失败与数据检查

| 模型 | 池 | 种子 | 测试人 | 状态 | 说明 |
| --- | --- | --- | --- | --- | --- |
| base | seen | 33 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 33 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 34 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 34 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 35 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 35 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 36 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 36 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 37 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | seen | 37 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel01 | 33 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel01 | 34 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel01 | 35 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel01 | 36 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel01 | 37 | S2 | invalid | Result checkpoint differs from selected_source.json |
| base | novel02 | 33 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | novel02 | 34 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | novel02 | 35 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | novel02 | 36 | S1 | invalid | Result checkpoint differs from selected_source.json |
| base | novel02 | 37 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 33 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 33 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 34 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 34 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 35 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 35 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 36 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 36 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 37 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | seen | 37 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel01 | 33 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel01 | 34 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel01 | 35 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel01 | 36 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel01 | 37 | S2 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel02 | 33 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel02 | 34 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel02 | 35 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel02 | 36 | S1 | invalid | Result checkpoint differs from selected_source.json |
| identity | novel02 | 37 | S1 | invalid | Result checkpoint differs from selected_source.json |
另有 40 条读取或一致性问题，见 `summary.json` 的 `issues` 字段。

## 解释范围与结果文件

本次统计单位是 S1、S2 两位受试者。五个种子是同一受试者上的配对重复，CSV 中的标准差表示种子间波动，不把两个人与五个种子合成十位独立受试者，也不计算这种伪独立样本下的显著性。

seen 与 novel 的样本数、图片划分及 C 容量配对，但 seen 包含八位源受试者，novel 每方向仅一位新人。两者差异同时可能涉及主干是否见过和 C 训练受试者的多样性。

每位受试者的聚合使用同人同图全部有效试次，先推理、再平均原始预测；各主模型保留自己的冻结 blur 图像 adapter。开发集选择策略逐项保存在 CSV 的 selection 列；本脚本不会重新挑测试表现最好的 epoch。

- `per_seed.csv`：所有预期评估记录，包括缺失／失败、零步、校准后及增量的全部八项指标。
- `means.csv`：逐模型、数据池、测试人的种子均值和种子标准差。
- `paired_per_seed.csv`：identity−base、novel−seen 的逐种子差值。
- `paired_means.csv`：共同完成种子的平均配对差值和种子标准差。
- `summary.json`：完整结构化结果、状态、来源及检查问题。
