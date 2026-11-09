# LOSO 校准对照结果
指标为五种子平均百分比；单试次/聚合均按Top1/Top5列出。身份与基线使用各自原生视觉空间。
|模式|方法|受试者|图片数|模型|单试次Top1/5|聚合Top1/5|
|---|---|---|---|---|---|---|
|ridge|ridge|S1|50|baseline|1.40/6.06|1.60/7.80|
|ridge|ridge|S1|50|identity|0.99/4.76|1.70/6.50|
|ridge|ridge|S1|200|baseline|2.69/10.79|4.00/17.80|
|ridge|ridge|S1|200|identity|2.15/9.15|3.60/13.00|
|ridge|ridge|S2|50|baseline|1.52/6.59|2.60/8.70|
|ridge|ridge|S2|50|identity|1.25/5.42|1.60/7.80|
|ridge|ridge|S2|200|baseline|2.22/9.45|4.60/13.70|
|ridge|ridge|S2|200|identity|2.09/8.78|3.50/11.40|
|ridge|residual_ridge|S1|50|baseline|5.13/17.40|12.90/37.60|
|ridge|residual_ridge|S1|50|identity|4.31/15.40|12.40/35.20|
|ridge|residual_ridge|S1|200|baseline|5.27/17.76|11.90/37.30|
|ridge|residual_ridge|S1|200|identity|4.34/15.74|9.70/31.70|
|ridge|residual_ridge|S2|50|baseline|3.84/13.76|7.80/24.20|
|ridge|residual_ridge|S2|50|identity|4.16/14.69|9.30/31.30|
|ridge|residual_ridge|S2|200|baseline|3.98/14.61|9.10/27.80|
|ridge|residual_ridge|S2|200|identity|4.30/15.75|10.20/33.40|
|stats|mean|S1|50|baseline|6.43/20.45|36.40/71.20|
|stats|mean|S1|50|identity|5.50/17.73|30.40/61.40|
|stats|mean|S1|200|baseline|6.59/20.76|37.80/73.90|
|stats|mean|S1|200|identity|5.64/18.01|32.60/64.50|
|stats|mean|S2|50|baseline|4.94/16.48|19.20/45.40|
|stats|mean|S2|50|identity|4.77/16.23|20.10/45.00|
|stats|mean|S2|200|baseline|5.04/16.83|20.70/48.60|
|stats|mean|S2|200|identity|4.92/16.30|22.00/47.30|
|stats|diagonal|S1|50|baseline|6.57/20.72|39.90/75.20|
|stats|diagonal|S1|50|identity|5.62/18.46|40.80/72.10|
|stats|diagonal|S1|200|baseline|6.74/21.04|44.70/78.40|
|stats|diagonal|S1|200|identity|5.77/18.70|43.10/75.60|
|stats|diagonal|S2|50|baseline|5.46/17.80|29.10/60.10|
|stats|diagonal|S2|50|identity|5.17/17.50|31.20/62.10|
|stats|diagonal|S2|200|baseline|5.60/18.04|34.80/66.10|
|stats|diagonal|S2|200|identity|5.24/17.82|36.60/68.20|
|stats|shrink_coral|S1|50|baseline|4.32/15.32|26.40/59.30|
|stats|shrink_coral|S1|50|identity|4.01/14.00|21.20/48.10|
|stats|shrink_coral|S1|200|baseline|4.85/16.51|40.90/78.30|
|stats|shrink_coral|S1|200|identity|4.39/14.98|33.90/66.40|
|stats|shrink_coral|S2|50|baseline|4.17/14.65|22.40/53.40|
|stats|shrink_coral|S2|50|identity|3.92/14.15|20.60/49.10|
|stats|shrink_coral|S2|200|baseline|4.07/13.92|36.30/66.70|
|stats|shrink_coral|S2|200|identity|4.15/14.43|36.60/66.20|
|stats|coral_identity_regularizer|S1|50|baseline|4.62/16.14|29.70/61.60|
|stats|coral_identity_regularizer|S1|50|identity|4.10/14.31|22.50/50.70|
|stats|coral_identity_regularizer|S1|200|baseline|5.23/17.54|42.90/79.60|
|stats|coral_identity_regularizer|S1|200|identity|4.54/15.45|34.90/67.50|
|stats|coral_identity_regularizer|S2|50|baseline|4.51/15.46|25.70/57.10|
|stats|coral_identity_regularizer|S2|50|identity|4.10/14.64|22.40/51.60|
|stats|coral_identity_regularizer|S2|200|baseline|4.68/15.48|38.60/68.80|
|stats|coral_identity_regularizer|S2|200|identity|4.44/15.33|38.60/67.00|

## 预设门槛判断
- ridge：身份差值-2.33个百分点，S1/S2为{'1': -3.0500001460313797, '2': -1.5999999456107616}，正向1/10，门槛未通过。supervised extension gate
- residual_ridge：身份差值1.18个百分点，S1/S2为{'1': -3.999999463558197, '2': 6.350000202655792}，正向5/10，门槛未通过。supervised extension gate
- mean：身份差值-5.23个百分点，S1/S2为{'1': -9.600000977516174, '2': -0.8500000834465027}，正向1/10，门槛未通过。unsupervised descriptive only
- diagonal：身份差值-0.45个百分点，S1/S2为{'1': -2.94999897480011, '2': 2.049999237060547}，正向6/10，门槛未通过。unsupervised descriptive only
- shrink_coral：身份差值-6.97个百分点，S1/S2为{'1': -11.550000607967377, '2': -2.399998903274536}，正向2/10，门槛未通过。unsupervised descriptive only
- coral_identity_regularizer：身份差值-7.58个百分点，S1/S2为{'1': -11.500000357627869, '2': -3.650001287460327}，正向1/10，门槛未通过。unsupervised descriptive only

## 解释边界
CORAL为论文协方差变换迁移：原文变换源特征再训练分类器，本实验反向变换新受试者特征以复用冻结投影；不是原论文完整基准复现。
残差Ridge与原始输出相比还包含逐试次L2归一化。各结果JSON均记录unit_prediction_zero，必须与此基线一起解释。
source只使用固定源训练图；目标测试EEG不用于统计拟合。支持集标签只用于Ridge，不用于stats变换。