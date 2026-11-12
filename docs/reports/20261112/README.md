# EEG 图像检索实验汇报（2026-11-12）

[LaTeX 正文](report.tex) 按实验目的、设置、结果和结论整理 THINGS-EEG2 与 Alljoined 2025 的工作，保留 16 张对比表。章节覆盖试次增强、EMA、身份监督、联合训练、集成、LOSO、少标签适配及伪试次扩充。

## 编译

使用 XeLaTeX，安装 `ctex`、`booktabs`、`longtable`、`xurl`、`hyperref`、`fancyhdr`，以及 Noto CJK 字体。Overleaf 可选择 XeLaTeX；若没有 Noto 字体，可将文档类改为 `fontset=fandol` 并删除三个 `\setCJK...font` 命令。

在本目录执行以下命令。将 `REPORT_BUILD` 指向数据盘已有目录，编译两遍生成目录及交叉引用：

```bash
REPORT_BUILD=/mnt/DATA-4/anteng/eeg-retrieval/experiments/reports_20261112/latex_build
mkdir -p "$REPORT_BUILD"
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$REPORT_BUILD" report.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$REPORT_BUILD" report.tex
```

也可使用 Tectonic：`tectonic --outdir "$REPORT_BUILD" report.tex`。编译缓存应设置到数据盘。

## 本次编译检查

已用 Tectonic 0.15.0（XeTeX 引擎）实际编译，输出 11 页 A4。已检查表格页面，编译日志无 Overfull box 或缺失字符报告。PDF 与编译缓存留在数据盘 `experiments/reports_20261112/latex_build/`，仓库保留可编辑源码。

## 结果来源

[sources.json](sources.json) 记录动态指标表所用文件、仓库位置和 SHA256。已归档文件直接引用 `evidence/runs/`；新文件在本目录 `evidence/` 保留快照。报告是固定日期记录，不随训练自动变化。v95 在该快照中尚无完整轮次结果。

历史增强、恢复头及适配结果还参见报告附录、[项目交接](../../HANDOFF.md)及[实验路线表](../../EXPERIMENT_MAP.md)。本报告不替代逐运行配置，正文也明确说明了不同训练预算与公开测试选模的比较限制。
