import json
from pathlib import Path
import numpy as np
import argparse
from common_target_readout import write

def main(root):
    rows=[];gates={}
    for mode in ['ridge','stats']:
        folder=root/mode
        if not folder.exists():continue
        files=list(folder.glob('s*_seed*.json'))
        if not files:continue
        methods=list(json.loads(files[0].read_text())['methods'])
        for method in methods:
            diffs={};methodrows=[]
            for f in [1,2]:
                for n in [50,200]:
                    arms={}
                    for arm in ['baseline','identity']:
                        vals=[json.loads((folder/f's{f}_{arm}_n{n}_seed{s}.json').read_text()) for s in range(33,38)]
                        metrics=[v['methods'][method]['metrics'] for v in vals]
                        arr=np.array([[m[g][k]*100 for g in ['single_trial','subject_aggregated'] for k in ['top1','top5']] for m in metrics]);arms[arm]=arr
                        methodrows.append(dict(mode=mode,method=method,fold=f,budget=n,arm=arm,metrics=arr.mean(0).tolist(),raw_zero=vals[0]['zero'],unit_prediction_zero=vals[0]['unit_prediction_zero']))
                    for k,s in enumerate(range(33,38)):diffs.setdefault(f'{f}_{s}',[]).append(float(arms['identity'][k,-1]-arms['baseline'][k,-1]))
            paired={k:float(np.mean(v)) for k,v in diffs.items()};means={str(f):float(np.mean([v for k,v in paired.items() if k.startswith(str(f)+'_')])) for f in [1,2]}
            gates[method]=dict(mode=mode,paired=paired,mean=float(np.mean(list(paired.values()))),subject_means=means,positive=sum(v>0 for v in paired.values()),passes=bool(np.mean(list(paired.values()))>=2 and min(means.values())>0 and sum(v>0 for v in paired.values())>=7),interpretation='unsupervised descriptive only' if mode=='stats' else 'supervised extension gate')
            rows.extend(methodrows)
    write(root/'summary.json',rows);write(root/'decisions.json',gates)
    text=['# LOSO 校准对照结果','指标为五种子平均百分比；单试次/聚合均按Top1/Top5列出。身份与基线使用各自原生视觉空间。','|模式|方法|受试者|图片数|模型|单试次Top1/5|聚合Top1/5|','|---|---|---|---|---|---|---|']
    for r in rows:
        m=r['metrics'];text.append(f"|{r['mode']}|{r['method']}|S{r['fold']}|{r['budget']}|{r['arm']}|{m[0]:.2f}/{m[1]:.2f}|{m[2]:.2f}/{m[3]:.2f}|")
    text+=['','## 预设门槛判断']
    for k,v in gates.items():text.append(f"- {k}：身份差值{v['mean']:.2f}个百分点，S1/S2为{v['subject_means']}，正向{v['positive']}/10，门槛{'通过' if v['passes'] else '未通过'}。{v['interpretation']}")
    text+=['','## 解释边界','CORAL为论文协方差变换迁移：原文变换源特征再训练分类器，本实验反向变换新受试者特征以复用冻结投影；不是原论文完整基准复现。','残差Ridge与原始输出相比还包含逐试次L2归一化。各结果JSON均记录unit_prediction_zero，必须与此基线一起解释。','source只使用固定源训练图；目标测试EEG不用于统计拟合。支持集标签只用于Ridge，不用于stats变换。']
    (root/'REPORT.md').write_text('\n'.join(text))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
