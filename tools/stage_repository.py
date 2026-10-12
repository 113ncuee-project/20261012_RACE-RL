"""Stage the current repository and active vendor files; no commit or network."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
subprocess.run(['git','add','--','.'],cwd=ROOT,check=True)
vendor=ROOT/'third_party/rapidchiplet-main'
paths=[str(p.relative_to(ROOT)) for p in vendor.rglob('*') if p.is_file()
       and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.name!='.DS_Store']
ignored=subprocess.run(['git','check-ignore','--stdin'],cwd=ROOT,input='\n'.join(paths)+'\n',text=True,capture_output=True)
if ignored.stdout.splitlines():subprocess.run(['git','add','-f','--',*ignored.stdout.splitlines()],cwd=ROOT,check=True)
print('Staged current implementation, current evidence and active native dependency.')
