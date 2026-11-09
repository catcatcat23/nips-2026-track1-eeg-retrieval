# 适配方法复现范围

## LoRA

原文：https://arxiv.org/abs/2106.09685 。官方实现：https://github.com/microsoft/LoRA/blob/main/loralib/layers.py 。实现低秩增量BA、A随机初始化/B零初始化、alpha/r缩放；本实验alpha=r=4，因此缩放为1。合并QKV分别使用rank4，与三个独立低秩分支一致。测试覆盖零初始化、梯度、冻结原权重、显式合并权重等价性。

迁移设置：REVE最后两个block，32,768参数，EEG→blur检索；不复现论文语言模型基准。设备错误修复使用base.weight.new_empty/new_zeros，复用已有经过核查的token缓存。

## CORAL

原文：https://arxiv.org/html/1511.05547v2 ，3.2节Algorithm1：C_source=cov(source)+I，C_target=cov(target)+I，source'=source C_source^(-1/2) C_target^(1/2)，随后训练源标签读出器。

`coral_forward_reproduction.py`复现源→目标的协方差变换。采用源标签Ridge视觉读出器，另有同样读出器但无CORAL的对照。Ridge正则10固定，不使用目标标签选参。该下游任务与原论文分类基准不同。

`closed_form_adaptation.py`中的coral_identity_regularizer为反方向部署：新受试者→源分布，复用原投影。shrink_coral再改为10%协方差收缩，属于本项目改造，不称为原论文完整复现。均值/对角方差是诊断基线。

所有统计均来自校准EEG或源训练EEG，测试EEG不参与估计。数值测试检查A^T C_source A=C_target及同域恒等变换。

## Ridge

求解目标为||XW+b-Y||²/n +lambda||W||²_F，偏置不惩罚。验证对偶/原始求解等价与目标整体平移不改变W。标签预算含内部开发数据；lambda仅在1/10/100中依据开发集平均排名选择。

残差Ridge的基线先逐试次L2归一化，故必须同时呈现unit_prediction_zero，不能将归一化带来的聚合变化归给校准。
