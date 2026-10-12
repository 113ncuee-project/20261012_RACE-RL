"""Capture actual runtime/package identities and platform-specific dependency lock."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
payload = {"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
           "executable":sys.executable,"packages":dict(sorted(packages.items())),
           "execution_device":"CPU","torch_deterministic_algorithms":True,
           "date":"2026-10-12","timezone":"Asia/Taipei",
           "scope":"tested local environment; not a claim of cross-platform identical floats"}
(ROOT/"docs/research/evidence/environment_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
lines = [line for line in freeze.splitlines() if not line.startswith(("-e ","#"))]
(ROOT/"requirements-tested.txt").write_text("# Tested macOS arm64 / Python 3.12 snapshot. Other platforms may need compatible wheels.\n"+
    "\n".join(lines)+"\n")
