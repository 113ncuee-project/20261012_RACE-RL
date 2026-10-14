"""Reproduce Git for Windows checkout conversion without a Windows host."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('git'),'checkout regression requires Git')
class CheckoutLineEndingTests(unittest.TestCase):
    def test_windows_autocrlf_preserves_pinned_sources_with_attributes(self):
        lock=json.loads((ROOT/'rapidchiplet.lock.json').read_text(encoding='utf-8'))
        expected=lock['reference_sha256']
        sources={name:(ROOT/'third_party/rapidchiplet-main'/name).read_bytes() for name in expected}
        gemmini=json.loads((ROOT/'gemmini.lock.json').read_text(encoding='utf-8'))
        expected={**expected,**{'gemmini/'+name:record['sha256'] for name,record in gemmini['reference_files'].items()}}
        sources.update({'gemmini/'+name:(ROOT/'third_party/gemmini-reference'/name).read_bytes()
                        for name in gemmini['reference_files']})
        # Isolate both configuration and attributes from the developer's Git.
        env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_ATTR_NOSYSTEM='1')
        with tempfile.TemporaryDirectory(prefix='race checkout ') as folder:
            repo=Path(folder)
            def git(*args,data=None):
                return subprocess.run(['git','-C',str(repo),*args],input=data,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,env=env,check=True,timeout=20).stdout
            git('init','--quiet')
            git('config','core.autocrlf','true')
            git('config','core.attributesFile',os.devnull)
            def index(name,data):
                sha=git('hash-object','--no-filters','-w','--stdin',data=data).decode('ascii').strip()
                git('update-index','--add','--cacheinfo',f'100644,{sha},{name}')
            for name,data in sources.items():index(name,data)
            git('checkout-index','--all','--force')
            before={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in expected}
            # This is the original Windows failure: Git inserts CR bytes.
            self.assertTrue(all(before[name]!=sha for name,sha in expected.items()))
            self.assertTrue(all(b'\r\n' in (repo/name).read_bytes() for name in expected))
            index('.gitattributes',(ROOT/'.gitattributes').read_bytes())
            git('checkout-index','--all','--force')
            after={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in expected}
            self.assertEqual(after,expected)
            for name,data in sources.items():self.assertEqual((repo/name).read_bytes(),data)
