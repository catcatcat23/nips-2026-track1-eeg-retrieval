# 固定模型的 SATTC／SCORE 部署对照

本实验固定 S1、S2 的无身份监督与身份监督模型，检查测试时校准能否改善原生 blur 检索，以及身份监督模型是否获得更多收益。入口是 `compare_deployment.py`；不训练或替换 REVE、投影头和图像 adapter。

| 项目 | 设置 |
|---|---|
| 模型 | 已固定的 v79／v91 LOSO checkpoint，保持此前选择 |
| 查询 | 每位受试者全部 16,224 条单试次预测，维度为 1,536 |
| 候选 | 相同 200 张测试图片；保留各模型冻结的原生 blur 图像 adapter |
| 校准标签 | 不使用；方法函数不接收标签或真实图片分组 |
| SATTC 图像白化参考 | 14,886 张源训练图片，每图一个原生 adapter 目标，排除源验证和测试图片 |
| SCORE 配对 | 通过目标 EEG 与候选图像的 CSLS 互近邻产生伪配对 |
| 聚合 | 校准后，同人同图全部有效试次的候选分数取均值 |
| 参数选择 | 固定作者默认／论文参数，公开材料未说明的数值约定写入说明；不按测试分数搜索 |
| 资源 | CPU，4 线程；全部输出在数据盘 |

原始点积时，分数均值与原预测向量均值再检索等价。归一化和候选相关校准会改变这种关系，因此保留 `raw_dot` 和 `raw_cosine` 两项，区分归一化本身的影响。不能把分数均值称作先平均 EEG 输入。

SATTC 使用固定作者源码的白化、Ada-CSLS、结构证据和 PoE 函数，校准失败时明确报错。SCORE 根据公开 v1 公式 3—9 实现矩匹配、伪配对、加权正交恢复和再次 CSLS 检索。具体来源、参数及未说明项分别见 `sattc_deployment.md` 与 `SCORE_DEPLOYMENT.md`。

这里使用整批无标签测试查询和候选库，是传导式检索校准；与此前独立支持集的少标签适配分开报告。论文的源模型训练、预处理和先平均 EEG 的测试协议均未替换，结果用于检验部署方法在本项目上的作用。方法需要读取候选库，因此不直接作为只接收 EEG、输出特征向量的比赛提交接口。

## 运行与核验

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /mnt/DATA-4/anteng/eeg-retrieval/env/bin/python \
  tools/v64_controls/compare_deployment.py \
  --sources /mnt/DATA-4/anteng/eeg-retrieval/experiments/sattc_score_deployment_20261108/sources/SATTC-CVPR2026-617cba60db52 \
  --out /mnt/DATA-4/anteng/eeg-retrieval/experiments/sattc_score_deployment_20261108/run
```

输出目录必须不存在，避免覆盖旧结果。每个模型保存各阶段分数矩阵、指标、数值诊断和来源哈希；SCORE 还保存伪配对与变换参数。伪配对正确率只在所有变换固定之后作事后诊断，不参与筛选。

`audit_deployment_inputs.py --run <目录>` 独立重算全部分数指标，核对 EEG 行序、候选图片、checkpoint 哈希和历史零步结果。分数聚合由独立实现用 float64 再算，避免仅重复调用主评估函数。
