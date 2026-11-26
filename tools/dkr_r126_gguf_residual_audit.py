#!/usr/bin/env python3
"""Lexical negative-cfg coverage for GGUF FFI; never a compiler/link proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
MODULE='#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]\nmod ffi;'
PACKED_MODULE='#[cfg(all(feature = "cuda", has_marlin_kernels, not(feature = "dkr-static-closure-probe")))]\nmod packed_affine;'
FFI_REF=r'(?<!std::)(?<!core::)\bffi::(?!c_void)'

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

def excluded_modules(source):
    # Accept only the exact negative cfg at the sole direct module
    # declaration; a comment or a second unguarded declaration is not proof.
    modules={}
    for name,guard in [('ffi',MODULE),('packed_affine',PACKED_MODULE)]:
        declaration=re.compile(r'^\s*(?:pub(?:\([^\n]*\))?\s+)?mod\s+'+name+r'\s*;',re.M)
        matches=list(declaration.finditer(mask(source)))
        if len(matches)!=1 or source.count(guard)!=1:
            raise ValueError(f'{name}: module not uniquely excluded')
        start=source.index(guard)
        # The actual mod declaration must be the one following the guard.
        if not (start<=matches[0].start()<=start+len(guard) and matches[0].end()==start+len(guard)):
            raise ValueError(f'{name}: exclusion does not guard declaration')
        modules[name+'.rs']=guard.split('\n')[0]
    return modules

def count_refs(source, name, excluded):
    refs=list(re.finditer(FFI_REF,mask(source)))
    spans=guarded_ranges(source)
    bad=[source.count('\n',0,m.start())+1 for m in refs
         if name not in excluded and not any(a<=m.start()<b for a,b in spans)]
    if bad:raise ValueError(f'{name}: unguarded FFI reference lines {bad}')
    return len(refs)

def self_test():
    source=MODULE+'\n'+PACKED_MODULE+'\n'
    excluded=excluded_modules(source)
    assert count_refs('fn launch(){ ffi::launch(); }','packed_affine.rs',excluded)==1
    assert count_refs(GUARD+'\nfn launch(){ ffi::launch(); }','cuda.rs',excluded)==1
    for bad in [source.replace(PACKED_MODULE,'#[cfg(has_marlin_kernels)]\nmod packed_affine;'),
                source+'mod packed_affine;\n',
                source.replace(PACKED_MODULE,'/*'+PACKED_MODULE+'*/\nmod packed_affine;')]:
        try:excluded_modules(bad)
        except ValueError:pass
        else:raise AssertionError('unguarded or duplicate module admitted')
    try:count_refs('fn launch(){ ffi::launch(); }','cuda.rs',excluded)
    except ValueError:pass
    else:raise AssertionError('unguarded call admitted')
    print('DKR_R126_S9C2B5_GGUF_AUDIT_SELF_TEST=PASS module_exclusion=true negative_cases=4')

def audit(root):
    directory=root/'mistralrs-quant/src/gguf'
    excluded=excluded_modules((directory/'mod.rs').read_text())
    files={}
    for path in sorted(directory.glob('*.rs')):
        files[path.name]=count_refs(path.read_text(),path.name,excluded)
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
                excluded_modules=excluded,
                module_guarded_ffi_refs=sum(files.get(name,0) for name in excluded),
                local_guarded_ffi_refs=sum(count for name,count in files.items() if name not in excluded),
                projected_physical_upper=523,projected_logical=3761,
                compilation=False,link_qualified=False,numeric_qualified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path.cwd())
    p.add_argument('--json-out',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test();return
    report=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(f"DKR_R126_S9C2B5_GGUF_SOURCE=PASS guarded_ffi_refs={report['guarded_ffi_refs']} unguarded=0 profiles=148 images=4 compilation=false link_qualified=false")
if __name__=='__main__':main()
