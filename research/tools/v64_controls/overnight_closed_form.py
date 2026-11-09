"""Sequential source extraction then closed-form controls, with terminal report."""
import subprocess,sys,json,datetime
from pathlib import Path
r=Path(__file__).parent
try:
 for mode in ['source','stats','ridge']:
  with (r/f'{mode}.log').open('w') as f:
   subprocess.run([sys.executable,'-u',str(r/'closed_form_adaptation.py'),'--root',str(r),'--mode',mode],stdout=f,stderr=subprocess.STDOUT,check=True)
  if mode!='source':subprocess.run([sys.executable,str(r/'summarize_closed_form.py'),'--root',str(r)],check=True)
 (r/'status.json').write_text(json.dumps(dict(state='complete',time=datetime.datetime.now(datetime.timezone.utc).isoformat())))
except Exception as exc:
 (r/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc))));raise
