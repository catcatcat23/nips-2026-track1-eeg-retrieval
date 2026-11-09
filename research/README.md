# 研究源码

此处按原目录结构保存后续实验的当前源码。`tools/eeg_multiblur`提供模型及损失依赖，`joint_stage1`提供联合训练，`v64_controls`提供增强、身份、LOSO和适配控制，`diagnostics`提供聚类和集成诊断，`scripts/reve_submission`保存提交工具。

这是研究代码档案，不是第二套已经迁移验证的CLI。脚本中有本机路径、缓存假设和历史启动逻辑。迁移时先定位相应 `evidence/runs/<run>/config.json`，配置新机器资产；运行前检查 import 和参数。可直接使用的基线入口仍为根目录 `run.py`。

每次实验实际源码与当前版本可能不同。需要严格复现时，使用 `evidence/source_snapshots.json` 将对应 blob 恢复为原文件名，核对 SHA256。v93 汇总修复后的当前脚本位于 `tools/v64_controls/summarize_unseen_calibration.py`，其测试与旧快照均保留。

从仓库根目录运行研究代码时，将 `research` 加入 Python 路径。例如已验证的汇总回归测试：

```bash
PYTHONPATH=research python -B -m unittest discover \
  -s research/tools/v64_controls -p test_summarize_unseen_calibration.py
```

直接从仓库根目录省略 `PYTHONPATH` 会找不到 `tools.v64_controls`。这与根目录的基线工具目录不是同一个包。
