# SATTC 部署算法迁移与实现核对

本模块把 SATTC 的测试时校准计算接到已经冻结的 REVE 检索模型上，比较无身份监督和有身份监督的 checkpoint。它不重新训练源模型，不复现作者的 EEG 编码器，也不把迁移后的结果当作 SATTC 原论文成绩。

## 来源与版本

论文：[SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](https://arxiv.org/html/2603.20738v2)。作者代码：[QunjieHuang/SATTC-CVPR2026](https://github.com/QunjieHuang/SATTC-CVPR2026)。固定版本为 `617cba60db52fa2e51e52f05b8f75bc13259f8ef`。

原始快照保存在数据盘：

```text
/mnt/DATA-4/anteng/eeg-retrieval/experiments/sattc_score_deployment_20261108/sources/SATTC-CVPR2026-617cba60db52
```

`source_manifest.json` 记录每个文件的下载地址、SHA-256、字节数和获取时间；目录内保留作者的 MIT LICENSE。主要依据是 `Retrieval/run_sattc_loso.py` 与 `Retrieval/utils/cli_args.py`。模块读取前会验证这两个文件的固定哈希，随后通过 Python AST 只加载纯计算函数和 `evaluate_model` 的证据构建区段，不导入或执行作者训练入口。

## 接口与数据流

```python
result = calibrate_sattc(
    query,                 # 一个受试者的全部单试次 EEG 投影，N×D，未经 L2 归一化
    gallery,               # 当前模型冻结 image adapter 的测试候选，M×D
    gallery_reference=ref, # 当前模型源训练集的唯一图片特征，R×D
    source_dir=source_dir,
)
scores = result["scores"]
diagnostics = result["diagnostics"]
```

接口不接受图片标签、受试者标签或正确匹配索引。一个调用只能包含一个受试者的完整 query 集合，因为 query 的均值、协方差以及行列近邻均在该集合上计算。分批各自校准会改变结果。输入转成 CPU float32，不修改调用者张量，不计算梯度。

| 输出矩阵 | 实际计算 | 比较目的 |
|---|---|---|
| `raw_dot` | 原始 `Q @ G.T` | 保留现有检索打分 |
| `raw_cosine` | Q、G 各自 L2 归一化后点积 | 单独检查归一化影响 |
| `whiten_cosine` | EEG 的 SAW 与图像白化后各自归一化，再点积 | 单独检查白化影响 |
| `whiten` | 对 `whiten_cosine` 整个矩阵做作者的全局中心化和标准差缩放 | 后续几何校准的输入 |
| `geometry` | 在 `whiten` 上执行作者 Ada-CSLS | 检查局部密度校正 |
| `full` | `geometry + beta × structural` | 检查加入结构证据后的完整 PoE |

这些输出都是 N×M 的分数矩阵。标签只在函数返回后进入评估器。当前实验的单试次指标直接对每行分数排序；聚合指标将同人同图全部有效试次的对应分数行取算术平均后排序。不会先把已校准分数转成 softmax 概率，也不会先平均 EEG。

## 白化统计量来自哪里

EEG 的 SAW 从当前受试者的全部无标签 query 特征估计均值和样本协方差。协方差采用收缩形式 `C = (1 − a) × S + a × trace(S) / D × I`，再计算 `C^(-1/2)`；特征中心化、白化后做 L2 归一化。这里使用测试 query 的统计量，因此属于传导式测试时校准，与仅使用独立校准集的非传导式方法需要分表报告。

图像侧需要特别区分：作者 `train_model` 第 1093 行先根据训练图像全库估计白化状态，之后第 1098 行才进行 `[::10]` 下采样。`evaluate_model` 第 1243—1250 行复用已有状态。训练图像库按 `[1654, 10, D]` 组织，每张图一条特征，不按 EEG 试次数或受试者人数重复加权。

当前主实验传入实际源训练集的唯一图片经冻结 adapter 生成的特征，排除源验证图片和原始测试图片。若不传 `gallery_reference`，模块会在候选库估计图像白化，并在诊断中标记 `gallery_only_transfer`；这只是可选迁移方式，不能冒充作者训练参考的主实验。函数无法凭特征张量判断来源，图片 ID 排除和哈希检查由外部 runner 完成。

作者 loader 直接读取 `ViT-H-14_features_train.pt["img_features"]`，没有在估计白化前再次归一化。公开仓库没有提供该缓存的生成脚本，无法仅凭当前源码确认缓存是否已经单位化。此次 REVE 迁移使用项目已有的单位化 image-adapter 目标作为 G 和源参考，不改变目标定义。作者训练损失也使用白化后的图像目标，而既有 REVE checkpoint 的训练目标没有这项白化；这同样属于模型接入差异，结果必须按部署迁移来解释。

## 固定参数及来源

默认值取 CLI 的实际运行默认，不取同名函数的备用值。模块会在加载时再次核对源码中的默认值。

| 参数 | 使用值 | 来源 |
|---|---:|---|
| EEG SAW shrink | 0.6 | `run_sattc_loso.py` 的 `--saw_shrink` |
| 图像白化 shrink | 0.5 | 同文件的 `--global_whiten_auto_shrink` |
| 协方差形式 | 全矩阵，非对角近似 | 两项 `diag=False` |
| 特征值下限 | 1e-6 | `_eigh_inv_sqrt` |
| 全局分数中心化／缩放 | 两项开启 | `logits_auto_center / scale` |
| Ada-CSLS 行侧 k／范围 | 12；5—20 | `cli_args.py` `_CANONICAL_DEFAULTS` |
| 行侧 density m／alpha | 10／1.0 | 同上 |
| 列侧 k／范围 | 12；8—24 | 同上 |
| 列侧 density m／alpha | 20／1.2 | 同上 |
| 双向安全邻域 | 行 35，列 35 | `_STRUCTURAL_DEFAULTS` |
| 图像流行度统计 top-L | 5 | 同上 |
| case2 阈值比例／列惩罚 | 0.5／0.5 | 同上 |
| 高／中 hub 分位数 | 0.85／0.70 | 同上 |
| 背景惩罚 | 0.3 | 同上 |
| case6 高／中惩罚 | 0.5／0.25 | 同上 |
| PoE beta | 1.9 | 同上 |
| lambda_pen／lambda_bonus | 0.5／0.25 × 几何分数标准差 | `build_structural_expert` |

完整参数也写入每次返回的 `diagnostics`。固定参数根据矩阵尺寸进行的 k 上限截断与官方 `evaluate_model` 保持一致。REVE 没有作者模型的 `logit_scale`，这里设为 1；默认的全矩阵 z-score 会消除统一的正比例缩放。

Ada-CSLS 的列侧密度使用候选图片在各 query top-m 中出现的频率；并非直接取列侧相似度均值。结构专家使用 CSLS 之前的矩阵构建互为最近邻、双向排名、保护区与 hub 惩罚，再与几何专家相加。实现直接执行作者对应函数，避免将这些细节简化成另一种算法。

## 与官方测试协议的差别

作者 `eegdatasets_leaveone.py` 第 291 行对每张测试图片的 EEG 重复先取平均，再送入编码器，每图得到一个 query。此次主实验保留项目原协议：每条 EEG 单独推理、校准其分数，之后平均同人同图的分数。作者常见的 200×200 矩阵因此变成本地约 16,224×200 的矩阵，列侧排名和结构证据也会随之变化。这是必须报告的迁移条件，不能把同名指标直接当作原论文复现。

SATTC 需要完整候选库，并联合使用一批目标受试者 query。它不会返回可在无候选库情况下独立使用的 1536 维提交特征；当前实现用于本地适配研究。是否可接入比赛提交需要另行核对当期接口。

## 验证与错误处理

`verify_author_parity()` 使用三种固定随机输入，独立执行作者 `evaluate_model` 中从 SAW 到 PoE 的完整后处理区段，再与本模块输出对拍。测试覆盖 48×23、200×200、160×17 的 query／候选矩阵形状。三组的 `whiten`、`geometry` 和 `full` 最大绝对误差均为 0。原始点积与余弦各自精确复现，输入张量不变，诊断可直接写入 JSON。测试不读取真实模型或 EEG，也不产生性能结论。

作者代码对 SAW、CSLS 和 PoE 的异常含有静默跳过或降级逻辑。此实现将异常处理改为抛出错误，避免组件失效后仍以完整 SATTC 名义汇报。正常计算路径与作者保持一致。另外，官方结构证据代码仅在 case6 非空时定义两个惩罚诊断变量，随后无条件引用；本模块提前按相同配置初始化，解决空 case6 时的变量未定义问题，不改变分数计算。

PoE 的真实计算使用 `S_geom.std()`，即 `unbiased=True`；官方日志另用 `unbiased=False` 估计 lambda 并打印，两者略有差异。本模块记录真实算子的有效 lambda。

作者源码遵循 MIT License，版权为 Copyright (c) 2024-2026 Qunjie Huang, Weina Zhu — Yunnan University；完整许可与免责声明随快照保存。分发包含这些算子的代码时须同时保留该 LICENSE。
