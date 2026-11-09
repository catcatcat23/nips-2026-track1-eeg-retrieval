"""Package and replay the actual ZIP through official ingestion and scoring.

Run with the project Python environment. Uses synthetic matching-shape data;
this checks the submission workflow, not accuracy or the worker Docker image.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import yaml


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=project / 'submissions/reve_seed33')
    parser.add_argument('--output', type=Path, default=project / 'submissions/reve_seed33_track1_v2.zip')
    parser.add_argument('--n-chans', type=int, default=63)
    args = parser.parse_args()
    files = ['submission.py', 'weights.pt', 'positions.json', 'LICENSE_BRAINDECODE.txt']
    for name in files:
        if not (args.source / name).is_file():
            raise FileNotFoundError(args.source / name)
    tree = ast.parse((args.source / 'submission.py').read_text())
    solver = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solver')
    assert any(isinstance(n, ast.FunctionDef) and n.name == 'load_model' for n in solver.body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the candidate before publishing the final ZIP path.
    candidate = args.output.with_suffix('.candidate.zip')
    with zipfile.ZipFile(candidate, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (args.source / name).read_bytes())
    root = Path(tempfile.mkdtemp(prefix='reve_ingestion_audit_'))
    extracted = root / 'submission'
    with zipfile.ZipFile(candidate) as archive:
        assert archive.namelist() == files
        assert archive.testzip() is None
        archive.extractall(extracted)
    for file in extracted.iterdir():
        file.chmod(0o444)
    extracted.chmod(0o555)
    official = project / 'competition-2026'
    phase = root / 'phase'
    phase.mkdir()
    config = yaml.safe_load((official / 'codabench/phases/warmup/image_decoding/config.yaml').read_text())
    config['dataset'] = [f'Simulated[n_chans={args.n_chans},n_times=120,n_outputs=1536,n_images=8,n_train=16,n_test=16,sfreq=120]']
    (phase / 'config.yaml').write_text(yaml.safe_dump(config))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH'] = str(official) + os.pathsep + env.get('PYTHONPATH', '')
    commands = [
        [sys.executable, str(official / 'codabench/ingestion_program/ingestion.py'),
         '--submission-dir', str(extracted), '--output-dir', str(root / 'predictions'),
         '--input-data', str(phase), '--benchmark-dir', str(official / 'tracks/image_decoding')],
        [sys.executable, str(official / 'codabench/scoring_program/scoring.py'),
         '--prediction-dir', str(root / 'predictions'), '--output-dir', str(root / 'scores')],
    ]
    for label, command in zip(['ingestion', 'scoring'], commands):
        with (root / f'{label}.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(label, 'exit', result.returncode, 'log', root / f'{label}.log', flush=True)
        result.check_returncode()
    scores = json.loads((root / 'scores/scores.json').read_text())
    assert set(config['scoring']['columns']).issubset(scores)
    candidate.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(f'{digest}  {args.output.name}\n')
    report = {'zip': str(args.output), 'sha256': digest, 'files': files,
              'audit_dir': str(root), 'official_ingestion': 'passed', 'official_scoring': 'passed',
              'data': f'synthetic {args.n_chans}-channel 120Hz input; not an accuracy evaluation',
              'environment': 'local project environment, not official Docker',
              'platform_failure_cause': 'unknown; no server logs available'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
