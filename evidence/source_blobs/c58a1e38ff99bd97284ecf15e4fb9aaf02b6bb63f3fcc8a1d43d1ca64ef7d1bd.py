"""Summarize paired v93 calibration runs without selecting on test scores.

Usage: python -B summarize_unseen_calibration.py --root DATA_DISK_RUN_ROOT
Only ROOT/calibration_report is written; training files and queue reports stay
untouched. Percentages and percentage-point differences are explicit in names.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


DATA_ROOT = Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
ARMS = ('base', 'identity')
POOLS = {'seen': ('test01', 'test02'), 'novel01': ('test02',), 'novel02': ('test01',)}
METRICS = tuple(f'{space}_{mode}_{top}' for space in ('clean', 'blur')
                for mode in ('single_trial', 'subject_aggregated') for top in ('top1', 'top5'))


def read_json(path, issues):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        issues.append({'file': str(path), 'error': str(exc)})
        return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def flatten(metrics):
    values = {}
    for space in ('clean', 'blur'):
        for mode in ('single_trial', 'subject_aggregated'):
            for top in ('top1', 'top5'):
                value = metrics[space][mode][top]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f'{space}/{mode}/{top}: expected a numeric fraction')
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{space}/{mode}/{top}: outside [0, 1]')
                values[f'{space}_{mode}_{top}'] = 100 * value
    return values


def direction(pool, test):
    target = 'S1' if test == 'test01' else 'S2'
    origin = {'seen': 'S3-S10', 'novel01': 'S1', 'novel02': 'S2'}[pool]
    return f'{origin}→{target}'


def pending_state(run_status, worker, pipeline):
    state = (run_status or {}).get('state', (run_status or {}).get('status'))
    if state in ('failed', 'error'):
        return 'failed'
    if state in ('training', 'running', 'calibrating', 'resuming'):
        return 'running'
    if any((s or {}).get('state') == 'failed' for s in (worker, pipeline)):
        return 'missing_after_arm_failure'
    if any((s or {}).get('state') == 'complete' for s in (worker, pipeline)):
        return 'missing_after_arm_complete'
    return 'missing'


def collect(root, seeds):
    rows, issues, arm_states = [], [], {}
    for arm in ARMS:
        folder = root / arm
        worker = read_json(folder / 'worker_status.json', issues)
        pipeline = read_json(folder / 'pipeline_status.json', issues)
        zero_file = read_json(folder / 'calibration/zero.json', issues)
        arm_states[arm] = {'worker': worker, 'pipeline': pipeline,
                           'source': read_json(folder / 'selected_source.json', issues)}
        for pool, tests in POOLS.items():
            for seed in seeds:
                run = folder / 'calibration' / f'{pool}_seed{seed}'
                result_path = run / 'result.json'
                result = read_json(result_path, issues)
                run_status = read_json(run / 'status.json', issues)
                for test in tests:
                    row = dict(arm=arm, pool=pool, seed=seed, test=test,
                               subject='S1' if test == 'test01' else 'S2',
                               direction=direction(pool, test), result_path=str(result_path),
                               status=pending_state(run_status, worker, pipeline), error='')
                    for metric in METRICS:
                        row[f'zero_{metric}_pct'] = None
                        row[f'calibrated_{metric}_pct'] = None
                        row[f'gain_{metric}_pp'] = None
                    if zero_file is not None:
                        try:
                            for metric, value in flatten(zero_file['metrics'][test]).items():
                                row[f'zero_{metric}_pct'] = value
                        except (KeyError, TypeError, ValueError) as exc:
                            issues.append({'file': str(folder / 'calibration/zero.json'), 'error': str(exc)})
                    if result_path.exists() and result is None:
                        row.update(status='invalid', error='Cannot read result.json')
                    elif result is not None:
                        try:
                            if result['pool'] != pool or result['seed'] != seed:
                                raise ValueError('Result pool/seed does not match directory')
                            if (result.get('state', result.get('status')) in ('failed', 'error')
                                    or row['status'] == 'failed'):
                                raise RuntimeError('Result or run status explicitly reports failure')
                            record = result['tests'][test]
                            baseline, calibrated = flatten(record['zero']), flatten(record['calibrated'])
                            for metric in METRICS:
                                previous = row[f'zero_{metric}_pct']
                                if previous is not None and abs(previous - baseline[metric]) > 1e-7:
                                    raise ValueError('Result zero metrics differ from calibration/zero.json')
                            checkpoint = result['checkpoint']
                            selected = arm_states[arm]['source']
                            if selected and selected.get('sha256') != checkpoint.get('sha256'):
                                raise ValueError('Result checkpoint differs from selected_source.json')
                            for mode in ('zero', 'calibrated'):
                                if record[mode].get('candidate_count') != 200:
                                    raise ValueError(f'{mode}: candidate_count must be 200')
                            for metric in METRICS:
                                row[f'zero_{metric}_pct'] = baseline[metric]
                                row[f'calibrated_{metric}_pct'] = calibrated[metric]
                                row[f'gain_{metric}_pp'] = calibrated[metric] - baseline[metric]
                            row.update(status='complete', best_epoch=result['best_epoch'],
                                       trained_epochs=result['trained_epochs'],
                                       fit_images=result['fit_images'], fit_trials=result['fit_trials'],
                                       dev_images=result['dev_images'],
                                       trainable_parameters=result['trainable_parameters'],
                                       train_subjects=';'.join(result['train_subjects']),
                                       selection=result['development_selection'],
                                       checkpoint_sha256=checkpoint.get('sha256'),
                                       checkpoint_epoch=checkpoint.get('epoch'),
                                       checkpoint_variant=checkpoint.get('variant'),
                                       candidate_count=record['calibrated']['candidate_count'],
                                       repeats_min=record['calibrated'].get('repeats_min'),
                                       repeats_max=record['calibrated'].get('repeats_max'),
                                       correction_ratio=record['calibrated'].get('correction_ratio'))
                        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                            row.update(status='failed' if isinstance(exc, RuntimeError) else 'invalid', error=str(exc))
                            issues.append({'file': str(result_path), 'test': test, 'error': str(exc)})
                    if row['status'] != 'complete' and not row['error']:
                        failures = [s for s in (run_status, worker, pipeline) if isinstance(s, dict) and s.get('error')]
                        row['error'] = '; '.join(str(s['error']) for s in failures)
                    rows.append(row)
    return rows, arm_states, issues


def descriptive(values):
    if not values:
        return None, None
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None


def aggregate(rows, keys, fields, expected):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, group in groups.items():
        complete = [r for r in group if r['status'] == 'complete']
        item = dict(zip(keys, key))
        item.update(n_complete=len(complete), n_expected=expected,
                    complete_seeds=','.join(str(r['seed']) for r in complete),
                    status='complete' if len(complete) == expected else 'partial' if complete else 'missing',
                    seed_statuses=json.dumps({str(r['seed']): r['status'] for r in group}, ensure_ascii=False))
        for field in fields:
            mean, std = descriptive([r[field] for r in complete])
            item[f'{field}_mean'] = mean
            item[f'{field}_seed_sd'] = std
        summaries.append(item)
    return summaries


def paired_rows(rows):
    lookup = {(r['arm'], r['pool'], r['seed'], r['test']): r for r in rows}
    pairs = []
    for row in rows:
        if row['arm'] == 'identity':
            left = lookup['base', row['pool'], row['seed'], row['test']]
            pairs.append(make_pair('identity_minus_base', row['pool'], row['subject'], row, left))
        if row['pool'].startswith('novel'):
            left = lookup[row['arm'], 'seen', row['seed'], row['test']]
            pairs.append(make_pair('novel_minus_seen', row['arm'], row['subject'], row, left))
    return pairs


def make_pair(contrast, condition, subject, right, left):
    pair = dict(contrast=contrast, condition=condition, subject=subject, seed=right['seed'],
                direction=right['direction'], right=f"{right['arm']}/{right['pool']}",
                left=f"{left['arm']}/{left['pool']}",
                right_status=right['status'], left_status=left['status'],
                status='complete' if right['status'] == left['status'] == 'complete' else 'unpaired')
    for metric in METRICS:
        for field, suffix in [('zero', 'pct'), ('calibrated', 'pct'), ('gain', 'pp')]:
            key = f'{field}_{metric}_{suffix}'
            pair[f'{field}_{metric}_difference_pp'] = (
                right[key] - left[key] if pair['status'] == 'complete' else None)
    return pair


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, signed=False):
    return '—' if value is None else f'{value:+.2f}' if signed else f'{value:.2f}'


def metric_cell(row, prefix, space, mode, suffix, signed=False):
    return '／'.join(number(row.get(f'{prefix}_{space}_{mode}_{top}_{suffix}'), signed)
                    for top in ('top1', 'top5'))


def make_report(root, rows, means, pairs, pair_means, states, issues, seeds):
    counts = Counter(row['status'] for row in rows)
    lines = ['# 未见受试者校准器迁移实验汇报', '',
             f'更新时间：{datetime.now(timezone.utc).isoformat()}。实验目录：`{root}`。', '',
             f'本次预期 {len(seeds) * 3 * 2} 个校准器训练结果、{len(rows)} 条受试者评估记录；'
             f'有效完成 {counts["complete"]} 条。每个 seen 校准器评估 S1、S2，novel 校准器只评估另一位受试者。', '',
             '主模型仅使用 S3—S10 训练和验证。novel 在 S2 训练 C 后测试 S1，另一个 C 在 S1 训练后测试 S2；'
             'seen 对照在 S3—S10 的混合单试次上训练 C。所有测试指标仅供汇报，没有用于此脚本的选模或排序。', '',
             '## 完成状态', '', '| 主模型 | worker | pipeline | 完成评估／预期 |', '| --- | --- | --- | --- |']
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        worker = (states[arm]['worker'] or {}).get('state', 'missing')
        pipeline = (states[arm]['pipeline'] or {}).get('state', 'missing')
        lines.append(f'| {arm} | {worker} | {pipeline} | {sum(r["status"] == "complete" for r in group)}／{len(group)} |')
    lines += ['', '## 逐人结果', '',
              '数值为已完成种子的均值，单位为 %，每格按 Top-1／Top-5 展示。缺失种子不填零；'
              '未满全部种子的结果标为 partial。两模型之间的配对结论以下一表的共同完成种子为准。', '',
              '| 模型 | C 训练→测试 | 完成种子 | 状态 | clean 单次 | clean 聚合 | blur 单次 | blur 聚合 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in means:
        cells = [metric_cell(row, 'calibrated', s, m, 'pct_mean') for s in ('clean', 'blur')
                 for m in ('single_trial', 'subject_aggregated')]
        lines.append(f'| {row["arm"]} | {direction(row["pool"], row["test"])} | '
                     f'{row["n_complete"]}／{row["n_expected"]} | {row["status"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 零校准与校准增量', '',
              '下面的零步和增量取自与最终结果相同的完成种子。零步在同一主模型、同一测试受试者下重复出现，'
              '是配对参照，不是额外独立实验。增量单位为百分点。', '',
              '| 模型 | C 训练→测试 | clean 聚合零步 | clean 聚合增量 | blur 聚合零步 | blur 聚合增量 |',
              '| --- | --- | --- | --- | --- | --- |']
    for row in means:
        cells = []
        for space in ('clean', 'blur'):
            cells.extend([metric_cell(row, 'zero', space, 'subject_aggregated', 'pct_mean'),
                          metric_cell(row, 'gain', space, 'subject_aggregated', 'pp_mean', signed=True)])
        lines.append(f'| {row["arm"]} | {direction(row["pool"], row["test"])} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 配对差值', '',
              'identity−base 为同一 C 数据池、同一种子下两种监督模型的最终成绩差值；novel−seen 为同一主模型、'
              '同一种子下两种 C 训练来源的差值。正值表示前者更高，单位为百分点。只纳入双方都完成的种子。', '',
              '| 对比 | 条件 | 测试人 | 配对种子 | clean 单次差值 | clean 聚合差值 | blur 单次差值 | blur 聚合差值 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in pair_means:
        cells = [metric_cell(row, 'calibrated', s, m, 'difference_pp_mean', signed=True)
                 for s in ('clean', 'blur') for m in ('single_trial', 'subject_aggregated')]
        lines.append(f'| {row["contrast"]} | {row["condition"]} | {row["subject"]} | '
                     f'{row["n_complete"]}／{row["n_expected"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 缺失、失败与数据检查', '']
    incomplete = [r for r in rows if r['status'] != 'complete']
    if incomplete:
        lines += ['| 模型 | 池 | 种子 | 测试人 | 状态 | 说明 |', '| --- | --- | --- | --- | --- | --- |']
        for row in incomplete:
            error = row['error'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f'| {row["arm"]} | {row["pool"]} | {row["seed"]} | {row["subject"]} | {row["status"]} | {error} |')
    else:
        lines.append('预期的逐种子评估记录均已完成并通过指标格式核查。')
    if issues:
        lines.append(f'另有 {len(issues)} 条读取或一致性问题，见 `summary.json` 的 `issues` 字段。')
    lines += ['', '## 解释范围与结果文件', '',
              '本次统计单位是 S1、S2 两位受试者。五个种子是同一受试者上的配对重复，CSV 中的标准差表示种子间波动，'
              '不把两个人与五个种子合成十位独立受试者，也不计算这种伪独立样本下的显著性。', '',
              'seen 与 novel 的样本数、图片划分及 C 容量配对，但 seen 包含八位源受试者，novel 每方向仅一位新人。'
              '两者差异同时可能涉及主干是否见过和 C 训练受试者的多样性。', '',
              '每位受试者的聚合使用同人同图全部有效试次，先推理、再平均原始预测；各主模型保留自己的冻结 blur 图像 adapter。'
              '开发集选择策略逐项保存在 CSV 的 selection 列；本脚本不会重新挑测试表现最好的 epoch。', '',
              '- `per_seed.csv`：所有预期评估记录，包括缺失／失败、零步、校准后及增量的全部八项指标。',
              '- `means.csv`：逐模型、数据池、测试人的种子均值和种子标准差。',
              '- `paired_per_seed.csv`：identity−base、novel−seen 的逐种子差值。',
              '- `paired_means.csv`：共同完成种子的平均配对差值和种子标准差。',
              '- `summary.json`：完整结构化结果、状态、来源及检查问题。', '']
    return '\n'.join(lines)


def summarize(root):
    root = root.resolve()
    if not root.is_relative_to(DATA_ROOT.resolve()) or root == DATA_ROOT.resolve():
        raise ValueError(f'--root must be a run directory below {DATA_ROOT}')
    if not root.is_dir():
        raise FileNotFoundError(root)
    protocol = json.loads((root / 'calibration_protocol.json').read_text())
    seeds = protocol.get('seeds', list(range(33, 38)))
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds)):
        raise ValueError('Protocol seeds must be a nonempty list of distinct integers')
    rows, states, issues = collect(root, seeds)
    # Explicit order keeps all eight clean/blur metrics aligned in every export.
    fields = [f'{kind}_{metric}_{unit}'
              for kind, unit in [('zero', 'pct'), ('calibrated', 'pct'), ('gain', 'pp')]
              for metric in METRICS]
    means = aggregate(rows, ('arm', 'pool', 'test', 'subject'), fields, len(seeds))
    pairs = paired_rows(rows)
    pair_fields = [f'{kind}_{metric}_difference_pp' for kind in ('zero', 'calibrated', 'gain') for metric in METRICS]
    pair_means = aggregate(pairs, ('contrast', 'condition', 'subject'), pair_fields, len(seeds))
    out = root / 'calibration_report'
    out.mkdir(exist_ok=True)
    for name, records in [('per_seed', rows), ('means', means), ('paired_per_seed', pairs), ('paired_means', pair_means)]:
        write_csv(out / f'{name}.csv', records)
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(), root=str(root), protocol=protocol,
                   metric_units='accuracy: percent; difference/gain: percentage points',
                   statistical_unit='subject; seeds are paired repeats, not independent subjects',
                   states=states, issues=issues, rows=rows, means=means, pairs=pairs, paired_means=pair_means)
    write_json(out / 'summary.json', summary)
    (out / 'REPORT.md').write_text(make_report(root, rows, means, pairs, pair_means, states, issues, seeds))
    return summary, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    summary, out = summarize(args.root)
    print(json.dumps({'output': str(out), 'evaluation_statuses': dict(Counter(r['status'] for r in summary['rows'])),
                      'issues': len(summary['issues'])}, ensure_ascii=False))
