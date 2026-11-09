"""Apply this handoff's exact cleanup manifest. Does not discover new candidates."""
import json, os, time
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'evidence/cleanup_plan.json').read_text())
base=Path('/mnt/DATA-4/anteng/eeg-retrieval/experiments')
for item in m['delete']:
 f=Path(item['path'])
 assert f.is_relative_to(base) and not f.is_symlink() and f.resolve()==f
 assert f.suffix=='.pt' and 'epoch_' in f.name
 s=f.stat();assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns'],f
print('Validated',len(m['delete']),'files;',m['bytes'],'bytes')
if a.apply:
 audit=base/'handoff_cleanup_20261109';audit.mkdir(exist_ok=True)
 (audit/'manifest.json').write_text(json.dumps(m,indent=2))
 deleted=[]
 with (audit/'deletion_journal.jsonl').open('a') as j:
  for item in m['delete']:
   Path(item['path']).unlink()
   deleted.append(item)
   j.write(json.dumps(item)+'\n');j.flush();os.fsync(j.fileno())
 result={'date':time.strftime('%Y-%m-%d %H:%M:%S %z'),'deleted_files':len(deleted),'reclaimed_bytes':sum(x['bytes'] for x in deleted),'preserved_epoch_files':len(m['keep']),'audit':str(audit),'all_kept_exist':all(Path(x['path']).is_file() for x in m['keep'])}
 (audit/'result.json').write_text(json.dumps(result,indent=2))
 (root/'evidence/cleanup_result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
