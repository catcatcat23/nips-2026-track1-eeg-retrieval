# REVE + Local OT + EMA：EEG→图像检索（v64，当前最优方案）

本仓库只保留当前最优策略 **v64**：单阶段全参数微调 REVE，训练时加 Local OT 区域对齐、混合试次平均和通道遮挡增强，并用 EMA 权重做推理。当前提交为 **`reve_v64_ema0999_ep9_track1.zip`**（EMA0.999，第 9 个 epoch）。

同一套配方不做任何调参，迁移到 Alljoined2025 上，记为 v67。旧的 v39 两阶段方案可以在 git 历史里找到（commit `74e8732`）。

## 实验结果

所有检索都是在 200 张图里选，随机水平为 Top1 0.5% / Top5 2.5%。下表格式为 **Top1 / Top5（%）**。

- **单试次**：每条 EEG 单独检索。
- **聚合**：先把同一被试、同一张图的全部试次的原始输出取平均，再归一化后检索。
- **Blur**：在模糊图像库中检索，是训练时的对齐目标。
- **Clean**：在原始清晰图像库中检索，是比赛的官方输出。

### THINGS-EEG2（10 名被试，训练集为 train+val，测试 200 张图）

| 方法 | Blur 单试次 | Blur 聚合 | Clean 单试次 | Clean 聚合 |
|---|---|---|---|---|
| 官方 NeuralBench REVE 基线（3 个种子平均） | — | — | — / 38.04 | 49.17 / 81.40 |
| v64 raw 权重（ep8） | 19.61 / 45.89 | 80.20 / 97.20 | 17.05 / 41.50 | 61.80 / 89.80 |
| v64 EMA0.999（ep8） | 22.13 / 49.46 | 86.45 / 98.75 | 19.52 / 45.20 | 68.35 / 93.75 |
| **v64 EMA0.999（ep9，当前提交）** | **22.07 / 49.36** | **87.80 / 98.65** | **19.40 / 45.08** | **70.80 / 94.45** |
| v64 EMA0.9998（ep10） | 22.35 / 49.72 | 86.65 / 98.60 | 19.76 / 45.58 | 69.80 / 93.95 |

当前提交和官方基线相比：Clean 单试次 Top5 高 7.0 个点，Clean 聚合 Top1 高 21.6 个点，Clean 聚合 Top5 高 13.1 个点。

### Alljoined2025（20 名被试，训练集为 stim_train：16540 张图、134 万条试次，测试 200 张图）

| 方法 | Blur 单试次 | Blur 聚合 | Clean 单试次 | Clean 聚合 |
|---|---|---|---|---|
| 旧配方基线（ep10） | 2.88 / 10.82 | 23.75 / 54.52 | 2.76 / 10.35 | 15.70 / 41.60 |
| v67 raw 权重（ep6） | 3.57 / 12.57 | 28.35 / 58.45 | 3.27 / 11.94 | 19.28 / 47.95 |
| v67 EMA0.999（ep5） | 3.96 / 13.70 | 37.30 / 68.40 | 3.70 / 13.07 | 23.62 / 53.75 |
| **v67 EMA0.9998（ep6）** | **4.05 / 13.85** | 38.15 / 68.95 | **3.79 / 13.22** | 24.43 / 54.30 |
| v67 EMA0.9998（ep7，聚合最好） | 4.02 / 13.82 | **40.10 / 70.65** | 3.77 / 13.21 | **26.35 / 56.25** |

旧配方基线的设置：编码器学习率 1e-5、ReduceLROnPlateau 调度、adapter dropout 0.85，不用增强、OT 和 EMA。

v67 EMA0.9998（ep6）和旧配方基线相比：Blur 聚合 Top5 高 14.4 个点，Clean 聚合 Top5 高 12.7 个点。

### 结论

1. **同一套配方在两个数据集上都有效。** 迁移到 Alljoined2025 时没有调任何超参数，聚合 Top5 提高了 12–14 个点。
2. **EMA 是稳定有效的组件。** 和 raw 权重相比，单试次 Top5 高约 1.3–4 个点，聚合 Top5 高约 5–10 个点。
3. **没有采用的方案：**
   - SWA（对第 7–11 个 epoch 的 EMA 权重取平均），和最好的单个 EMA checkpoint 相差不到 0.3 个点；
   - 在 stage1 的 EMA 权重上继续训 stage2，单试次低 1.5–2 个点。
4. **限制：**
   - checkpoint 都是在 public test 上选的，属于开发集结果，不是独立测试或平台分数；
   - 两个数据集都只跑了 seed33；
   - 训练在第 11–15 个 epoch 就早停了，而 OneCycle 是按 40 个 epoch 设计的，学习率还没退火。

每个 epoch 的完整指标见 `evidence/things_v64_history.jsonl` 和 `evidence/alljoined_v67_history.jsonl`。

## 模型架构

```
EEG [B,C,120] → 线性插值到 200 点 → 逐通道 z-score（clamp ±15）
  → REVE-base（brain-bzh/reve-base，全参数微调） → token 均值 → Linear(512→1536) = h        （blur 空间）
  → ProjectedHead：LN → Linear → 2×ResidualBlock → LN → Linear = z  （clean 空间，推理输出 [B,1536]）

只在训练时使用的分支：
  blur 图像特征 → image adapter：Linear(1536→768) → ELU → Dropout(.2) → Linear(768→1536) = a
  REVE tokens → LN → Linear(512→256) → 6 个可学习 query 做注意力 = q
  图像 3×3 区域 token → Linear(1536→256) → GELU → LN，加上位置编码 = v
```

- **损失**：`ClipLoss(z, clean) + ClipLoss(h, a) + 0.1 × LocalOT(q, v)`。
  - ClipLoss 是单向的，只对图像侧做归一化，温度固定不学习。
- **Local OT**：
  - 每条样本选 5 个图像区域、3 个 query，做 6 次迭代的 Sinkhorn，ε=0.1；
  - 正样本按同一张图判定（multi-positive），每条样本配 31 个困难负样本，双向对比；
  - 各区域的权重按被试分别计算：用训练集 EEG 做 5 折 Ridge 回归，看每个区域的可预测性（`src/local_ot_weights.py`）。
- **增强**：
  - 混合试次：同一被试、同一张图的试次，以 50% / 25% / 25% 的概率分别取 1 / 2 / 4 条做平均；
  - 通道遮挡：20% 的样本随机把 1 个通道置零。
- **EMA**：每次优化器更新后同步更新 0.999 和 0.9998 两份权重，不参与训练。
- **优化**：
  - AdamW，编码器和检索头的最大学习率都是 1e-4，weight decay 0.05；
  - OneCycle 调度，40 个 epoch，pct_start=0.1；
  - batch 256，梯度裁剪 1，FP32 计算，关闭 TF32，seed33。
- **选模型与早停**：每个变体（raw / EMA0.999 / EMA0.9998）按 Blur 单试次 Top5 各自选最佳 epoch；如果连续 5 个 epoch 三个变体都没有提升，就停止训练。
- **推理**：只保留 encoder + blur_head + recovery，输入 EEG，输出原始的 `[B,1536]`，不需要图像，也不跑 OT。

## 数据与资产

数据和权重不放进 Git。需要准备的资产及其路径写在 `configs/assets.local.json`：

| 资产 | 说明 |
|---|---|
| `things_data_root` | THINGS-EEG2 预处理缓存。`data/{train,val,test}_{eeg,ids,subjects}.npy` 是官方 120 Hz、63 通道数据。`data/clean.npy` 是 DINOv2-giant 提取的 clean 特征。`features/blur.npy` 是 blur 特征，schema 为 `upstream_per_scale_l2_dino_v1`，用 12 级高斯模糊、第 0.6667 层、逐级 L2 归一化后取平均。 |
| `things_test_blur_shards` | 测试集 200 张图的 blur 特征，评估时用。 |
| `things_local_features` | Local OT 用的图像区域特征 `local.npy`，形状 `[N,9,1536]`，由 `cache_reference/local_ot_extract.py` 生成。 |
| `alljoined_data_manifest` | Alljoined2025 的试次清单，使用官方 session 划分。 |
| `hf_home` | REVE 权重（snapshot `dc2a075c`）和 DINOv2-giant 权重的位置。 |
| `local_ot_upstream_script` | 上游 `train_intrasubject_local_ot.py`，只在 preflight 时用来校验 OT 实现一致。 |

## 运行

环境版本见 `evidence/environment.versions.json`。所有输出都写到 `--output` 指定的新目录，不会覆盖已有结果。

```bash
python run.py check                                           # 校验缓存形状和提交 checkpoint 的 SHA
python run.py prepare  --output runs/things_v64 [--reuse-local] # 生成 Local OT 区域特征和被试区域权重
python run.py preflight --output runs/things_v64              # 小样本过拟合、OT 一致性、EMA 公式等检查
python run.py train     --output runs/things_v64              # 训练 v64
python run.py evaluate  --output runs/things_v64 [--checkpoint X.pt]  # 评估 Blur/Clean 的四组指标
python run.py alljoined --output runs/alljoined_v67 [--preflight]     # Alljoined2025 迁移实验 v67
python tools/export_submission.py --checkpoint X.pt --data-root <things_data_root> \
       --competition-root <2026-competition checkout> --output <新目录>   # 导出提交包，并核对预测与训练模型一致
```

## 打包验证（2026-10-26）

详细记录见 `evidence/validation.json`。

- 用本仓库代码对提交的 checkpoint 运行 `run.py evaluate`，四组指标都和原训练日志完全一致：Clean 19.40/45.08、70.80/94.45；Blur 22.07/49.36、87.80/98.65。
- `prepare` 重新算出的被试区域权重和原始运行逐项相同。
- `export_submission.py` 导出的 `weights.pt` 和提交版本的张量完全相同，`submission.py` 也逐字节相同。
- 提交 ZIP 的 SHA 见 `evidence/checksums.json`。在本地已经通过官方的读取和打分流程，并在完整测试集上做过回放（`evidence/submission_*.json`）。这一步用的不是官方 Docker，所以不能保证平台上一定成功。
- 本次打包只做了上面的验证，没有重新完整训练。

## 目录

```
run.py                     入口
src/                       v64 训练、评估代码（原样拷贝，路径改为可配置，见 evidence/provenance.json）
alljoined/alljoined_v67.py Alljoined2025 迁移实验，复用 src/ 中的模型、损失、增强和 EMA
cache_reference/           图像 blur 特征和 Local OT 区域特征的提取脚本
configs/                   things_v64.json、alljoined_v67.json（*.original.json 是原始运行配置），assets.local.json
submission/                提交用的推理代码（Solver）、电极位置、许可证
tools/export_submission.py 把 checkpoint 导出成提交格式
evidence/                  每个 epoch 的指标、preflight 结果、提交审计、校验和、来源记录
```

部分 `src/` 模块（如 `stability_ablation.py`、`recovery_ablation.py`、`v33_recovery_stage2.py`）保留下来，只是因为 v64 的模型类和初始化顺序依赖它们，并不是额外的实验方案。
