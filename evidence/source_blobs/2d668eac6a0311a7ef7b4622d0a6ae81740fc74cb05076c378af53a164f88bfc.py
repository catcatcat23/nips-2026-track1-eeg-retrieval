import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=[json.loads((f/'heldout_test.json').read_text()) for f in sorted(a.root.glob('fold_*'))]
assert len(rows)==10
out={'folds':rows,'macro_mean':{}}
for space in ['clean','blur']:
 out['macro_mean'][space]={}
 for group in ['single_trial','subject_aggregated']:
  out['macro_mean'][space][group]={k:float(np.mean([r['metrics'][space][group][k] for r in rows])) for k in ['top1','top5']}
(a.root/'result.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['macro_mean'],indent=2))
