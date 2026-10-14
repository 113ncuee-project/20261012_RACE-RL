"""UTF-8 test log on every runner; preserve the exact test exit status."""
import os
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    folder=Path('experiments/runs/ci_logs');folder.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([sys.executable,'-m','unittest','discover','-s','tests','-v'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',
        env=dict(os.environ,PYTHONUTF8='1'))
    with (folder/'tests.log').open('w',encoding='utf-8') as log:
        for line in process.stdout:
            log.write(line);log.flush()
            sys.stdout.write(line)
    process.stdout.close()
    raise SystemExit(process.wait())
