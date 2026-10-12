"""Verify active native RapidChiplet dependency against its pinned source lock."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    lock=json.loads((ROOT/'rapidchiplet.lock.json').read_text())
    failures=[]
    for name,expected in lock['reference_sha256'].items():
        path=ROOT/'third_party/rapidchiplet-main'/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failures.append(name)
    print(json.dumps({'dependency':'native RapidChiplet','files_checked':len(lock['reference_sha256']),'mismatches':failures},indent=2))
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
