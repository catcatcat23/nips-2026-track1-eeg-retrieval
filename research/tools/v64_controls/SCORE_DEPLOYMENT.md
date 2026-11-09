# SCORE 部署算子：实现与复现边界

实现文件：`score_deployment.py`。来源是 Cui 等的 [SCORE，arXiv:2608.19134v1](https://arxiv.org/html/2608.19134v1)，部署部分公式 3—9。这次固定现有 REVE checkpoint，只迁移部署算法；没有执行论文的 recovery-aware 源训练，不把结果命名为完整 SCORE 复现。

## 函数合同

```python
result = run_score_deployment(query, gallery, ScoreConfig())
```

`query` 为一个目标受试者的浮点张量 `[N,D]`，`gallery` 为相同特征空间中的 `[M,D]`。两者设备必须相同。函数不接收真值标签、图片 ID、受试者 ID，不修改模型和输入，不访问文件，不使用随机数。CPU 测试通过；本次实验使用 CPU，并在外部限制线程数。

返回值：

- `scores`：`raw_cosine`、`raw_csls`、`moment_cosine`、`moment_csls`、`recovered_cosine`、`recovered_csls`，全部为 `[N,M]`。
- `features`：矩匹配及恢复后的查询特征。
- `mapping`：矩匹配统计量，以及加权正交映射和中心。
- `landmarks`：全部互近邻与保留互近邻的行索引、候选索引、margin、权重。
- `diagnostics`：配置、范数、hub 计数、landmark 数量、正交误差、拟合目标、数值状态及协议说明。可直接序列化为 JSON。

`raw_csls` 是分离 hubness 校正收益的额外对照。`recovered_csls` 是完整部署路径的最终得分。真实标签只允许在外部 evaluator 中使用。

## 实际计算顺序

先按各维度将 query 的均值和标准差匹配到 gallery。随后计算余弦相似度和双向 CSLS 邻域修正，保留双方都选择对方为 Top-1 的互近邻。以每条互近邻的第一、第二名 CSLS 分数差构造正权重，并将权重均值归一到 1。

对伪配对两侧分别减去加权中心，形成加权交叉矩阵。在它上面加入强度为其谱范数乘 `rho` 的单位矩阵，用 SVD 求解正交映射。所有 query 按相同的中心和映射恢复，再重新计算 CSLS 的两个邻域项。只做一次恢复；没有反复自训练、真实配对筛选或全局一对一分配。

## 超参数与未说明细节

| 设置 | 当前取值 | 来源或处理方式 |
|---|---:|---|
| CSLS 邻居数 | 10 | 正文实验设置明确给出 |
| `rho` | 0.1 | 正文明确给出；乘加权交叉矩阵的谱范数 |
| landmark 数量范围 | 12—160 | 正文明确给出 |
| 恢复次数 | 1 | 正文明确给出 |
| margin 权重 | `m * max(margin, eta) / sum(max(margin, eta))` | 公式 5 |
| `epsilon` | `1e-6` | 本地固定数值保护；公开 v1 未给数值 |
| `eta`／`margin_floor` | `1e-6` | 本地固定数值保护；公开 v1 未给数值 |
| 标准差 | 总体标准差，`ddof=0` | 本地固定约定；公开 v1 未说明 |
| 超过 160 对 | 按 margin 降序保留 160；并列保留原顺序 | 本地固定约定；正文未说明截断方式 |
| 少于 12 对 | 跳过恢复，标记 `insufficient_landmarks` | 不构造额外伪配对；正文未说明回退方式 |
| 零交叉矩阵 | 跳过恢复，标记 `degenerate_cross_covariance` | 没有支持旋转的信息，避免任意 SVD 方向 |
| 初始行归一化 | 不额外执行 | 公式 3 使用传入特征；余弦打分内部归一化 |
| 计算精度 | float64 | 本地数值实现选择，未声称是作者配置 |
| 并列最近邻 | 最先出现的索引 | 固定的无标签约定 |
| 小规模检查中的 `k` | 分别裁剪到两侧样本数 | 正式 `16224×200` 不触发 |

以上本地约定预先固定，不根据测试指标修改。源码中的 `ScoreConfig` 将它们显式暴露给运行记录。

如果互近邻不足，返回的 `recovered_*` 与 `moment_*` 一致以便统一输出，但 `recovery_complete=False`，不能把它算作完整恢复成功。发生非有限值、SVD 失败或明显非正交时抛异常，由外部 runner 写失败状态。

## 本次输入与论文协议的差别

本次输入是全部单试次 query，例如 `16224×1536`，候选库是 `200×1536`。论文 THINGS 流程使用重复 EEG 平均后的输入。本实现不会利用真实图片 ID 把 query 先分为 200 组，因为该操作会把标签信息送进部署算法。

矩形 CSLS 在数学上成立：query 侧在 200 个候选中取近邻，gallery 侧在 16224 条 query 中取近邻。但是它改变了统计问题。gallery 的近邻可能来自同图重复试次；不同图片的有效重复数会改变总体均值、方差和邻域密度。互近邻还限制每张候选至多提供一条 landmark，所以最多 200 对；它不会把每张候选的所有高分试次都当作可靠配对。

因此，本次结果回答“该部署算子是否改善现有单试次 REVE 系统”，不能直接与论文的重复平均结果相减。所有参数对 baseline 与 identity 固定一致，labels 仅在部署结束后算单试次和分组指标。

当前 caller 传入未逐行归一化的原 blur 预测，以及原生冻结图像 adapter 的归一化 gallery。raw cosine 的单试次排名与原点积一致，但逐条余弦分数再平均会改变原始预测范数的聚合权重。因此，原始预测先平均后的历史聚合指标应由 runner 单独列出，不能与 `mean(raw_cosine scores)` 混称。

该方法读取整批目标 EEG 和候选特征，属于传导式部署。它也需要候选图像特征；不能直接装进仅接收 EEG、没有 gallery 的比赛输出接口。

## 来源核查与验证

已保存公开 v1 的 HTML、9 页 PDF 和 arXiv TeX 源码包。源码包中的正文是 `main.tex`，没有附录文件；正文两次提及 supplement，但本次未找到对应公开附件或作者实现。代码按公式移植，并保留上述未说明项，不声称对齐了无法取得的作者源码。

来源和检查记录目录：

`/mnt/DATA-4/anteng/eeg-retrieval/experiments/sattc_score_deployment_20261108/sources/`

模块内 `self_test()` 使用合成数据验证矩形 CSLS、旋转方向、加权中心、正交目标、恒等映射、互近邻唯一性、margin 权重、最少 landmark 回退、排序等变性，以及不改变输入和随机状态。运行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /mnt/DATA-4/anteng/eeg-retrieval/env/bin/python \
  tools/v64_controls/score_deployment.py
```
