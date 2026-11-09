"""Aggregate completed runs without selecting any checkpoint by test scores."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments/staged_adaptation_20261108')

def read(p):
    return json.loads(p.read_text())

def vector(m):
    return [100*m[g][k] for g in ['single_trial','subject_aggregated'] for k in ['top1','top5']]

def main():
    rows = read(ROOT/'closed_form/summary.json')
    for row in rows:
        row['labels'] = row['mode'] == 'ridge'
    for row in read(ROOT/'stage2_report/summary.json'):
        rows.append(dict(method='residual_lr3e-5', fold=row['fold'], budget=row['budget'], arm=row['arm'], metrics=row['final'], labels=True))
    for row in read(ROOT/'stage1_report/summary.json'):
        rows.append(dict(method='residual_lr1e-4', fold=row['fold'], budget=row['budget'], arm=row['arm'], metrics=row['final'], labels=True))
    for fold in [1,2]:
        for arm in ['baseline','identity']:
            first=read(ROOT/'closed_form/stats'/f's{fold}_{arm}_n50_seed33.json')
            for key in ['zero','unit_prediction_zero']:
                rows.append(dict(method=key,fold=fold,budget=0,arm=arm,metrics=vector(first[key]),labels=False))
            vals=[read(ROOT/'lora_fixed'/f's{fold}_{arm}_n200_seed{s}.json') for s in range(33,38)]
            rows.append(dict(method='lora',fold=fold,budget=200,arm=arm,metrics=np.mean([vector(v['scores']) for v in vals],0).tolist(),labels=True))
            for n in [50,200]:
                vals=[read(ROOT/'coral_forward'/f's{fold}_{arm}_n{n}_seed{s}.json') for s in range(33,38)]
                for method in vals[0]['methods']:
                    rows.append(dict(method=method,fold=fold,budget=n,arm=arm,metrics=np.mean([vector(v['methods'][method]['metrics']) for v in vals],0).tolist(),labels=False))
    names={'zero':'无校准','unit_prediction_zero':'仅归一化预测','residual_lr3e-5':'512维残差适配，LR 3e-5','lora':'最后两层注意力 LoRA','ridge':'从头 Ridge','residual_ridge':'残差 Ridge','mean':'均值校正','diagonal':'均值＋对角方差校正','shrink_coral':'目标→源，收缩 CORAL','coral_identity_regularizer':'目标→源，协方差加 I','source_ridge_no_coral':'源标签 Ridge，无 CORAL','forward_coral_source_ridge':'源→目标 CORAL＋源标签 Ridge'}
    names['residual_lr1e-4']='已有512维残差适配，LR 1e-4'
    text=['# LOSO 身份监督适配实验报告','',
          '## 实验回答的问题','',
          '在相同的新受试者标注预算下，身份监督能否提高最终 blur 检索成绩？本轮固定源模型，对比无身份监督模型与身份监督模型，没有重新筛选源 checkpoint。结果表明，身份监督的收益依赖受试者；目前没有方法达到预设的跨 S1、S2 一致优势标准。', '',
          '## 方法与评测','',
          '开发受试者为 S1、S2。两模型共享支持集种子 33—37、图片和真实 EEG 试次。50、200 张校准图片均来自既定 1000 图片池，每图一条 EEG，与原始 200 张测试图片分离。聚合是在单试次推理后，对同人同图全部有效试次的原始预测求平均，并非先平均 EEG。原生 blur 候选库由各模型冻结的图像 adapter 生成。因此，这里比较的是两套原生检索系统，不能将全部差异单独归因于 REVE 特征。', '',
          '无标签方法使用校准 EEG 估计统计量，不读取目标图片标签，也不使用测试 EEG 拟合变换。源参考为九位训练受试者每人 1000 条单试次特征，共 9000 条；选择源图片不依赖目标支持集图片身份。CORAL 原方向实验使用源图片标签训练 Ridge 读出器，但不使用目标标签。', '',
          '有标签 Ridge 在预算内按 80%／20% 划分拟合与开发图片；开发候选库为 1000 图片缓存中除拟合图片外的图片。正则化从 1、10、100 中按开发集正确图片平均排名选择，再用全部预算拟合。LoRA 固定 200 步、学习率 1e-4、batch 50，在最后两个注意力模块的 Q、K、V 和输出投影上加入 rank 4、alpha 4 的增量，共 32768 个参数。残差适配保留原投影，最后一层零初始化，只将学习率降至 3e-5。', '',
          '## 全部结果','',
          '以下为五种子均值，单位为百分比。每格依次为单试次 Top-1／Top-5；聚合 Top-1／Top-5。零校准结果没有支持集随机性。差值为身份监督减去无身份监督的聚合 Top-5，单位为百分点。', '',
          '| 方法 | 使用目标标签 | 校准图片数 | 受试者 | 无身份监督 | 身份监督 | 聚合 Top-5 差值 |',
          '|---|---|---:|---|---|---|---:|']
    for method in names:
        for n in sorted({r['budget'] for r in rows if r['method']==method}):
            for fold in [1,2]:
                pair={r['arm']:r for r in rows if r['method']==method and r['budget']==n and r['fold']==fold}
                a,b=pair['baseline'],pair['identity']
                def fmt(r):
                    v=r['metrics'];return f'{v[0]:.2f}／{v[1]:.2f}；{v[2]:.2f}／{v[3]:.2f}'
                text.append(f"| {names[method]} | {'是' if a['labels'] else '否'} | {n} | S{fold} | {fmt(a)} | {fmt(b)} | {b['metrics'][3]-a['metrics'][3]:+.2f} |")
    text += ['', '## 结果分析与下一步决定','',
             '身份监督没有通过本轮筛选。低学习率残差适配的身份优势平均为 0.52 个百分点，S1 为 −3.90，S2 为 +4.95；残差 Ridge 平均为 +1.18，S1 为 −4.00，S2 为 +6.35。两种方法均只在 S2 获得优势，不能将其解释为普遍提高新受试者适配能力。LoRA 的 200 图片对照中，身份模型平均低 3.60 个百分点，S1、S2 分别低 5.30、1.90 个百分点，因此未触发 50 图片分支。', '',
             '对角方差校正是值得继续研究的无标签基线。它保留原投影，只校正特征中心和逐维尺度；完整协方差变换并未稳定优于这一简单方法。当前结果支持“部分域偏移可以通过简单统计校正缓解”，但没有证明这些偏移全部来自身份，也没有证明身份监督使统计校正更有效。', '',
             '从头 Ridge 和残差 Ridge 的绝对成绩较低。它们使用平方误差拟合视觉向量，与原检索对比目标不同；残差 Ridge 还包含逐试次归一化，因此报告单列了仅归一化预测的基线。不能只看身份模型相对另一模型的差值，而忽略两者是否都低于原有检索系统。', '',
             '本轮不启动 S3、S4 扩展验证，不再增加适配网络容量。结论限定为：在当前 checkpoint、原生 blur 目标、S1／S2 和少标签预算下，身份监督未表现出稳定的最终检索优势。若继续开展研究，应将“简单无标签统计校正能否跨更多受试者有效”作为独立问题，而不是继续用更复杂的适配器寻找身份监督的正向结果。', '',
             '## 论文复现与实现检查','',
             'LoRA 复现低秩增量、零输出初始化与 alpha/r 缩放，迁移到 REVE 的注意力投影。CORAL 复现原文 Algorithm 1 的源→目标协方差变换，并设置相同源标签 Ridge、无 CORAL 的对照；原文的分类基准没有在此重复。目标→源的冻结读出版本及协方差收缩版本是项目改造，不能称为原论文完整复现。', '',
             '参考：LoRA，https://arxiv.org/abs/2106.09685 ；CORAL，https://arxiv.org/html/1511.05547v2 ，第 3.2 节 Algorithm 1。', '',
             'LoRA 原 CPU／GPU 参数错误已经修复，旧失败记录保留；新运行完成。零初始化、梯度有限、冻结原权重、权重合并等价性经过检查。Ridge 检查了原始／对偶解一致性与不正则化偏置；CORAL 检查了变换后的协方差等式。保存参数重载抽查覆盖 S1 基线、50 图片、seed 33 的两种 Ridge 和四种统计变换，全部四项检索指标准确复现。', '',
             '所有已触发实验完成。未触发项为 LoRA 50 图片、校准集内部选步数和 S3／S4 扩展验证。五个种子是同一受试者内的配对重复，不能当作五位独立受试者，也不据此宣称统计显著性。逐种子结果、诊断、失败记录和源码快照保存在本报告所在目录的对应子目录中。']
    (ROOT/'OVERNIGHT_REPORT.md').write_text('\n'.join(text)+'\n')
    (ROOT/'all_adaptation_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (ROOT/'overnight_completion.json').write_text(json.dumps(dict(state='complete',extension_triggered=False,reason='No supervised method passes preset gate',rows=len(rows)),indent=2))
    print('Saved report and',len(rows),'result rows')

if __name__=='__main__':main()
