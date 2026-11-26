#!/usr/bin/env python3
"""Lexical negative-cfg coverage for GGUF FFI; never a compiler/link proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
MODULE='#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]\nmod ffi;'

def mask(source):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                  lambda m:re.sub(r'[^\n]',' ',m.group()),source)

def guarded_ranges(source):
    code=mask(source)
    result=[]
    for g in re.finditer(re.escape(GUARD),source):
        start=g.end()
        while True:
            attr=re.match(r'\s*#\[[^\]]*\]',code[start:])
            if not attr:break
            start+=attr.end()
        depth=0
        for i in range(start,len(code)):
            ch=code[i]
            if ch in '([':depth+=1
            elif ch in ')]':depth-=1
            elif depth==0 and ch==';':
                result.append((start,i+1));break
            elif depth==0 and ch=='{':
                nesting=1
                for j in range(i+1,len(code)):
                    if code[j]=='{':nesting+=1
                    elif code[j]=='}':nesting-=1
                    if not nesting:
                        result.append((start,j+1));break
                break
    return result

def audit(root):
    directory=root/'mistralrs-quant/src/gguf'
    if MODULE not in (directory/'mod.rs').read_text():raise ValueError('FFI module not excluded')
    files={}
    for path in sorted(directory.glob('*.rs')):
        if path.name=='ffi.rs':continue
        source=path.read_text()
        spans=guarded_ranges(source)
        refs=list(re.finditer(r'(?<!std::)(?<!core::)\bffi::(?!c_void)',mask(source)))
        bad=[source.count('\n',0,m.start())+1 for m in refs if not any(a<=m.start()<b for a,b in spans)]
        if bad:raise ValueError(f'{path.name}: unguarded FFI reference lines {bad}')
        files[path.name]=len(refs)
    # Both packages must remain within the same hard per-shard limits.
    plans={}
    identities=set()
    for family,expected,images in [('gguf-residual',126,3),('gguf-grouped-row',22,1)]:
        path=root/'kernel-packages/cuda'/family/'build.py'
        spec=importlib.util.spec_from_file_location(family,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        plan=module.plan(61)
        rows=[p for rows in plan.values() for p in rows]
        assert len(plan)==images and len(rows)==expected
        for p in rows:
            identity=p['implementation_id']
            if identity in identities:raise ValueError('duplicate implementation')
            identities.add(identity)
        assert all(len(rows)<=64 for rows in plan.values())
        plans[family]={'images':images,'profiles':expected}
    return dict(schema='dkr.r126.gguf.residual.source.v1',files=files,
                guarded_ffi_refs=sum(files.values()),unguarded_ffi_refs=0,plans=plans,
                projected_physical_upper=523,projected_logical=3761,
                compilation=False,link_qualified=False,numeric_qualified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path.cwd())
    p.add_argument('--json-out',type=Path)
    a=p.parse_args()
    report=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(f"DKR_R126_S9C2B5_GGUF_SOURCE=PASS guarded_ffi_refs={report['guarded_ffi_refs']} unguarded=0 profiles=148 images=4 compilation=false link_qualified=false")
if __name__=='__main__':main()
