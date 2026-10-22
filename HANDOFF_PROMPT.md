请复现本仓库唯一保留的实验：`reve_v39b_epoch8_track1.zip` 对应的两阶段模型。先读 README.md、configs 和 evidence，不能将它替换为最新实验或另一个 B 组。

目标：v33/dropout 第7轮 REVE与投影 → 冻结提取h → v39/B 第8轮固定Ridge＋可训练残差恢复头。历史本地200候选结果：单试次17.72929/42.49753%，subject-aggregated 66.35/93.05%。这些是公共测试集参与选模型后的结果，不是独立测试。

严格固定：THINGS-EEG2 train+val训练，原test评估；seed33，batch256，40epoch，patience5，AdamW最大LR1e-4（第一阶段backbone也是1e-4）、wd.05，OneCycle pct_start .1，FP32、TF32关闭。第一阶段图像adapter dropout=.2，无增强、无OT；原单向ClipLoss，仅图像归一化、无learned temperature，clean+blur等权。第二阶段冻结EEG和投影，Ridge alpha=.01，仅训练残差网络，clean ClipLoss，无额外cosine。两阶段都按clean单试次Top5选最佳，不能换成blur聚合Top5保留规则。

执行：
1. 核查本机资源，修改configs/assets.local.json的路径；检查数据授权及缓存可用性，不重复下载已有资产。
2. 运行run.py check，校验原权重SHA、通道、120Hz缓存和图片ID。查看environment.versions.json。
3. 先运行stage1 --preflight；通过后训练stage1。初始化来自原始REVE预训练，不加载历史检索checkpoint冒充从头训练。
4. 用本次stage1最佳checkpoint运行extract。h缓存必须绑定该checkpoint SHA。
5. 运行stage2，仅B臂；保持Ridge冻结，验收残差初始化与梯度。保存配置、源码版本、日志和checkpoint。
6. 报告单试次及聚合Top1/Top5。聚合是同人同图全部有效试次的原始输出均值，之后归一化。同时做blur诊断；不要把blur指标当作官方clean结果。
7. 核对最终推理只能接收EEG、输出[B,1536]，再使用官方评分环境验证。仓库中的ZIP历史回放通过不等于当前平台提交必然成功。

当前仓库是从已经准备好的EEG/图像缓存起复现；cache_reference是图像提取参考，不是完整原始数据下载管线。数据或预训练资产缺失时明确报告，不自行替换预处理或图像编码配置。新跑的最佳epoch可能不是7/8，必须如实记录，不能修改记录迎合历史数字。
