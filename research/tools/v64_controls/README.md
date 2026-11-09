# v64 matched controls and downstream LOSO

| Run | Purpose | Only experimental change | Protocol |
|---|---|---|---|
| v77_v64_mean4 | Test 100% within-subject four-trial averaging | Replace 50/25/25 repeat mix with four distinct same-subject/same-image trials for every input | Original v64 train+val and public-test selection |
| v78_v64_identity | Test whether subject auxiliary supervision helps retrieval | Add pooled512 -> LN -> Linear256 -> GELU -> Linear128 and supervised subject contrastive loss, weight0.1 temperature0.1 | Original v64 augmentation and public-test selection |
| v79_v64_loso | Measure generalization to unseen downstream subjects | Ten independent THINGS leave-one-subject-out folds | Heldout subject excluded from training/selection; fixed image-disjoint validation |

All use v64 pretrained initialization, full-parameter REVE tuning, clean+blur+0.1 Local OT, image adapter dropout0.2, 20% one-channel masking, batch256, AdamW1e-4 weight_decay0.05, OneCycle40 epochs pct_start0.1, early stopping5, FP32 TF32off, raw plus EMA0.999/0.9998. v77/v78 retain every epoch and select blur single-trial Top5 exactly as v64. These are new stage1 training runs, not v64-weight continuation. Checkpoints/logs/cache links live exclusively on DATA-4.

Identity positives are other rows belonging to the same subject, negatives other subjects, self excluded. The loss averages log probability over all same-subject positives. It reaches the shared REVE through a separate 165248-parameter projection; it does not require subject identity during inference. The new head initialization preserves the base RNG state and all v64 initial parameters. Improving identity separation is a hypothesis, not a guaranteed retrieval improvement.

LOSO original local train/val overlap in image identity (16500 images), so they are combined and redivided by image: seed33 selects1654 of16540 images for validation;14886 remain for training. Each fold uses only nine seen subjects for both subsets. The heldout subject's original200 test images are evaluated once after selecting variant and epoch on seen-subject validation blur single Top5. Regional predictability weights are reestimated independently per subject on the14886 training images only; heldout-subject weights are never accessed by training or test scoring. The shared weight-preparation job computes independent per-subject entries; there is no EEG pooling across subjects. This is downstream LOSO, not a statement that REVE foundation pretraining excluded these people. Report each fold and subject-macro mean, not a single fold as final LOSO performance.

Queues: GPU3 v75 -> v77 -> v79 folds01..10; GPU2 v76 -> v78. Existing jobs are not interrupted. Each new training run first executes real-data forward/backward and tiny-set overfit preflight; failure stops that queue. Waiting does not mean training has begun.

Build: run prepare.py then fix_loso_split.py BEFORE launching (these intentionally fail rather than overwrite existing run roots). verify_snapshots.py verifies hashes and ten fold splits. run_control.py wraps the frozen v64 training loop; extensions.py holds new sampling, loss and evaluation logic. prepare.py extends training loss logging with the identity term, including zero for other arms. summarize_loso.py creates the ten-fold result only after all heldout tests exist.

Validation performed: syntax, real-cache minimum repeats and train/val/test identity audit, four-trial exact pairing, subject loss finite gradients/permutation/no-positive behavior, base-weight and RNG initialization parity. GPU preflight remains pending until assigned GPU is idle. Scores are not yet available.

## v92：用图片监督替换身份监督

`prepare_image_contrastive.py` 从 v82 的已验证源码创建独立只读快照，输出到数据盘 `reve_v92_image_contrastive`。`run_image_contrastive.py` 是本次入口；`image_contrastive.py` 实现辅助损失和 CPU／GPU 预检。配置、源码差异、来源和文件 SHA256 均保存在新实验目录，旧实验不变。

原检索 batch 仍为随机 256 条，保留 clean＋blur＋0.1 Local OT、同人 1／2／4 次混合、通道遮挡和 EMA。辅助 batch 沿用 v82 的 32 张图×4 人×每人 2 条不同单试次，共 256 条；不做辅助 EEG 增强。新损失直接作用于 REVE 的 512 维池化特征，温度 0.1、权重 1.0，无新增网络参数。每条 EEG 的 7 个同图样本均为正例（1 个同人、6 个跨人），其他图片的 248 条为负例，自身排除。先分别计算检索与辅助梯度，再统一裁剪并更新一次。

这是从原预训练 REVE 重新训练，未加载 v82 的检索权重。沿用 v82 的 40 轮上限、patience＝5，以及公开测试 blur 单试次 Top-5 的选模口径，同时保留每轮 Raw／EMA 的 clean 和 blur 单试次、预测后聚合指标。公开测试承担开发选模作用，本次不是 LOSO。v82 与 v92 的辅助采样和预算一致，但正负例组成及比较范围改变；与没有辅助 batch 的 v64 比较时，还包含额外 EEG 暴露和计算量的差异。

预检包括真实数据与测试图片隔离、采样顺序与 v82 一致、损失与稠密目标交叉熵等价、仅 REVE 获得辅助梯度、4 张真实图片小样本过拟合、保存重载及初始化哈希一致。训练记录实际梯度范数、辅助／检索梯度比和余弦。每个锚点有 7 个正例，因此该多正例损失的下界为 ln(7)，不能要求其降到零。运行状态以数据盘中的 `launch_status.json`、预检报告和 `train.log` 为准。

## 固定 LOSO 模型的同图配对适配

`paired_subject_alignment.py` 检验身份监督后的特征是否更容易通过少量同图配对恢复到原检索头可用的坐标。它复用 `identity_reference_loso_pilot` 的缓存，冻结 S1／S2 的无身份监督、身份监督 checkpoint 和各自的原生 blur 图像 adapter。50／200 张目标校准图片各取一条固定 EEG，种子 33—37，两模型共享图片及试次。源模板为其余九位受试者同图四次编码输出的平均，再对人平均。

比较原模型、仅中心平移、正交配对对齐、打乱配对的正交对齐。求解器参考 [SCORE 式 6—8](https://arxiv.org/html/2608.19134v1)，固定 rho＝0.1，向单位矩阵正则化；打乱组沿用同一正确配对组的 lambda，仅破坏图片对应。测试数据不参与拟合或选参，仍按全部有效试次的原始预测均值聚合。它是有标签 EEG→EEG 校准诊断，不是完整的无标签 SCORE 复现。

运行：`python tools/v64_controls/paired_subject_alignment.py --out /mnt/DATA-4/anteng/eeg-retrieval/experiments/<new-name>`。输出目录必须全新。无需 GPU，默认四个 CPU 线程；每次运行保存只读源码、哈希、支持集索引、变换参数、五种子逐项结果及 `REPORT.md`。本次结果位于数据盘 `paired_subject_alignment_20261108/run`。
