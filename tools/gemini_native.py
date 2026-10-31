#!/usr/bin/env python3
"""Build and run the pinned HPCA 2024 Gemini artifact as a separate engine.

Native Gemini results are NOT directly comparable with the RACE CNN model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'third_party/gemini-hpca2024/GEMINI_AE.zip'
EXPECTED = '9726a34f501a4e8ea71569795b955121493b2ce9e0e3f251776b38bdb661be60'
DEFAULT_WORK = ROOT/'.cache/gemini-hpca2024-v6'
VERSION = 6
EXPORT_SOURCE = ROOT/'tools/gemini_mapping_export.inc'


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(source_dir):
    paths = [source_dir/'makefile']
    for folder in ('src', 'include'):
        paths.extend(path for path in (source_dir/folder).rglob('*') if path.is_file())
    return {str(path.relative_to(source_dir)):digest_file(path) for path in sorted(paths)}


def replace_once(path, old, new):
    source = path.read_text(encoding='utf-8')
    if source.count(old) != 1:
        raise ValueError(f'unexpected upstream source in {path}: cannot apply portability patch')
    path.write_text(source.replace(old, new), encoding='utf-8')


def parse_results(text):
    parsed = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 36:
            raise ValueError(f'unexpected Gemini result row with {len(fields)} fields')
        parsed.append({'mapping_index':int(fields[21]), 'raw_inputs':list(map(int, fields[:16])),
            'monetary_cost':float(fields[16]), 'energy_raw':float(fields[17]),
            'cycles':int(fields[18]), 'energy_delay_product_raw':float(fields[19]),
            'objective_raw':float(fields[20]), 'raw_columns':fields,
            'note':'native Gemini quantities, not RACE PPA'})
    return parsed


def prepare(work, jobs):
    if digest_file(ARCHIVE) != EXPECTED:
        raise ValueError('Gemini artifact SHA-256 mismatch')
    marker = work/'race_adapter.json'
    identity = {'archive_sha256':EXPECTED, 'portability_patch_version':VERSION,
                'export_source_sha256':digest_file(EXPORT_SOURCE)}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (any(saved.get(k) != v for k, v in identity.items()) or
                saved.get('source_hashes') != source_hashes(work/'GEMINI')):
            raise ValueError('installed artifact identity mismatch')
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError('choose an empty work directory for Gemini')
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE) as archive:
            for member in archive.infolist():
                target = (work/member.filename).resolve()
                if work.resolve() not in target.parents:
                    raise ValueError('archive contains an unsafe path')
            archive.extractall(work)
        source = work/'GEMINI/src'
        bitset = source/'bitset.cpp'
        replace_once(bitset, 'return static_cast<Bitset::bitlen_t>(_Find_first());',
            'for (bitlen_t i=0; i<std_bs::size(); ++i) if (test(i)) return i;\n'
            '\treturn static_cast<bitlen_t>(std_bs::size());')
        replace_once(bitset, 'return static_cast<Bitset::bitlen_t>(_Find_next(bit));',
            'for (bitlen_t i=bit+1; i<std_bs::size(); ++i) if (test(i)) return i;\n'
            '\treturn static_cast<bitlen_t>(std_bs::size());')
        util = source/'util.cpp'
        replace_once(util, 'std::__gcd(a, b)', 'std::gcd(a, b)')
        util.write_text('#include <numeric>\n'+util.read_text(encoding='utf-8'), encoding='utf-8')
        shutil.copyfile(EXPORT_SOURCE, source/'race_mapping_export.inc')
        replace_once(source/'main.cpp', 'output(3);',
                     'output(3);\n#include "race_mapping_export.inc"')
        identity['source_hashes'] = source_hashes(work/'GEMINI')
        marker.write_text(json.dumps(identity, indent=2)+'\n')
    source_dir = work/'GEMINI'
    command = ['make', '-j', str(jobs), 'release',
               'CXXFLAGS=--std=c++17 -O3 -Wno-parentheses', 'LDFLAGS=-lm -lpthread']
    with (work/'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=source_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    executable = source_dir/'build/stschedule'
    if not executable.is_file(): raise RuntimeError('Gemini executable missing after build')
    print(f'Gemini compiled: {executable}\nBuild log: {work / "build.log"}', flush=True)


def run(work, values, out, timeout):
    if not (work/'race_adapter.json').exists() or not (work/'GEMINI/build/stschedule').exists():
        raise ValueError('prepare the pinned Gemini artifact before running it')
    identity = json.loads((work/'race_adapter.json').read_text())
    if (identity.get('archive_sha256') != EXPECTED or identity.get('portability_patch_version') != VERSION or
            identity.get('export_source_sha256') != digest_file(EXPORT_SOURCE) or
            identity.get('source_hashes') != source_hashes(work/'GEMINI')):
        raise ValueError('Gemini source identity changed; prepare a fresh copy')
    if len(values) != 16: raise ValueError('Gemini requires exactly 16 integer inputs')
    if values[6] < 30: raise ValueError('Gemini rounds must be >=30: upstream uses rounds/30 as a divisor')
    if out.exists() and any(out.iterdir()): raise ValueError('choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    executable = (work/'GEMINI/build/stschedule').resolve()
    result_path = (out/'gemini_result.txt').resolve()
    mapping_path = (out/'gemini_mapping.json').resolve()
    started = time.perf_counter()
    record = {'engine':'GEMINI_HPCA2024_NATIVE', 'archive_sha256':EXPECTED,
              'executable_sha256':digest_file(executable), 'inputs':values,
              'status':'RUNNING', 'not_common_cnn_ppa_evaluator':True,
              'seed':'upstream chooses wall-clock time; read stdout',
              'warning':'wafer_util contains an upstream chained comparison; unchanged'}
    (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    try:
        with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
            result = subprocess.run([str(executable), str(result_path), str(mapping_path)], cwd=work/'GEMINI',
                input=' '.join(map(str, values))+'\n', text=True, stdout=stdout, stderr=stderr,
                timeout=timeout, check=False)
        rows = parse_results(result_path.read_text()) if result_path.exists() else []
        (out/'parsed_results.json').write_text(json.dumps(rows, indent=2)+'\n')
        gemini_rows = [row for row in rows if row['mapping_index'] == 3]
        mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else None
        record.update(status='COMPLETED' if result.returncode == 0 and gemini_rows and mapping else 'FAILED',
                      returncode=result.returncode, gemini_mapping_rows=len(gemini_rows))
    except subprocess.TimeoutExpired:
        record['status'] = 'TIMEOUT'
    except KeyboardInterrupt:
        record['status'] = 'INTERRUPTED'
        raise
    except (ValueError, OSError) as error:
        record.update(status='FAILED', error=str(error))
        raise
    finally:
        record['wall_time_s'] = time.perf_counter()-started
        (out/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    if record['status'] != 'COMPLETED':
        raise RuntimeError(f"Gemini {record['status']}; see {out/'stderr.txt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('prepare')
    build.add_argument('--jobs', type=int, default=4)
    execute = sub.add_parser('run')
    execute.add_argument('--inputs', type=int, nargs=16, required=True)
    execute.add_argument('--out', type=Path, required=True)
    execute.add_argument('--timeout', type=float, default=None)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.jobs < 1: parser.error('jobs must be positive')
        prepare(args.work, args.jobs)
    else:
        if args.timeout is not None and args.timeout <= 0: parser.error('timeout must be positive')
        run(args.work, args.inputs, args.out, args.timeout)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'Gemini: {error}', file=sys.stderr)
        raise SystemExit(1)
