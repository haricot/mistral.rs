#!/usr/bin/env python3
"""S9C.3A: conservative workspace source inventory, not Rust expansion/link proof.

Exit 0 means the inventory and cumulative gates completed, NOT that all
candidates are closed. --require-closed fails if review candidates remain.
No Cargo, compiler, cache inspection or GPU calls.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

from dkr_r126_static_ffi_inventory import git, PIN
from dkr_r126_s9c2e6_audit import audit as previous

COMPONENTS = ("mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn")
SEEDS = {
    "mistralrs-core": {"dkr-sm61-cubin-qualified", "dkr-static-closure-probe"},
}
EXCLUDED_FEATURES = {"metal", "flash-attn", "flash-attn-v3", "cutile", "nccl",
                     "cuda-legacy", "cuda-legacy-bf16", "cuda-legacy-fp8", "cuda-legacy-fp4"}
NATIVE = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".ptx", ".cubin", ".a", ".so", ".lib", ".o", ".fatbin"}
TOKEN = re.compile(r'(?:br|cr|r)#*\"|(?:b|c)?\"|//|/\*|(?:b)?\x27(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|[^\n])|[^\x27\\\n])\x27|[A-Za-z_][A-Za-z_0-9]*|::|=>|[^\s]')
IDENT = re.compile(r"[A-Za-z_]\w*\Z")
PRODUCER = re.compile(r"\b(?:KernelBuilder|CudaBuilder|build_lib|source_glob|source_files|nvcc|ptxas|cc::Build|compile_ptx|compile_cuda|rustc-link-lib|rustc-link-search|link_name)\b")
LIB_LINK = re.compile(r"cargo::?rustc-link-lib=([^\"\s]+)")
# The explicit archive names are evidence candidates, never a link allowlist.
ARCHIVES = {"mistralrscuda", "mistralrsquant", "mistralrspagedattention",
            "mistralrsflashinfergdn", "mistralrsdeepgemm", "mistralrsnvfp4",
            "mistralrsfa3paged", "flashattention"}

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def tokens(source):
    """Keep offsets; skip comments (including nested comments), strings as single tokens."""
    out = []
    pos = 0
    while (m := TOKEN.search(source, pos)) is not None:
        start, end = m.span()
        t = m.group()
        if t == "//":
            end = source.find("\n", end)
            if end < 0: end = len(source)
        elif t == "/*":
            depth = 1
            while depth:
                mark = re.search(r"/\*|\*/", source[end:])
                require(mark is not None, "unterminated block comment")
                depth += 1 if mark.group() == "/*" else -1
                end += mark.end()
        elif re.fullmatch(r"(?:b|c)?r#*\"", t):
            close = '"' + "#" * t.count("#")
            finish = source.find(close, end)
            require(finish >= 0, "unterminated raw string")
            end = finish + len(close)
            out.append((source[start:end], start, end))
        elif t.endswith('"') and t in ('"', 'b"', 'c"'):
            while end < len(source):
                ch = source[end]
                end += 1
                if ch == "\\":
                    end += 1
                elif ch == '"':
                    break
            else:
                raise ValueError("unterminated string")
            out.append((source[start:end], start, end))
        else:
            out.append((t, start, end))
        pos = end
    return out

def pairs(ts):
    stack = []
    result = {}
    for i, t in enumerate(ts):
        if t in ("(", "[", "{"):
            stack.append((t, i))
        elif t in (")", "]", "}"):
            if not stack or stack[-1][0] != {")":"(", "]":"[", "}":"{"}[t]:
                # Macros can contain token patterns, but mismatches are never
                # used as positive exclusion evidence.
                continue
            _, j = stack.pop()
            result[j] = i
    return result

def tri_and(values):
    return False if False in values else None if None in values else True

def tri_or(values):
    return True if True in values else None if None in values else False

def evaluate(expr, features, declared, provider_flags):
    """Only prove simple cfg expressions in the intended Linux CUDA scenario.
    Unknown predicates and cfg_attr are not assumed false.
    """
    if not expr: return None
    if len(expr) == 1:
        if expr[0] in ("test", "windows"): return False
        if expr[0] == "unix": return True
        if expr[0] in provider_flags: return False
        return None
    if len(expr) == 3 and expr[1] == "=":
        k, _, v = expr
        if not (v.startswith('"') and v.endswith('"')): return None
        v = v[1:-1]
        if k == "feature":
            if v in features: return True
            if v in declared or v in EXCLUDED_FEATURES: return False
        if k == "target_family": return v == "unix"
        if k == "target_os": return v == "linux"
        return None
    if len(expr) < 3 or expr[1] != "(" or expr[-1] != ")": return None
    op = expr[0]
    groups, current, depth = [], [], 0
    for t in expr[2:-1]:
        if t == "," and depth == 0:
            if current: groups.append(current)
            current = []
            continue
        current.append(t)
        if t == "(": depth += 1
        elif t == ")": depth -= 1
    if current: groups.append(current)
    vals = [evaluate(x, features, declared, provider_flags) for x in groups]
    if op == "all": return tri_and(vals)
    if op == "any": return tri_or(vals)
    if op == "not" and len(vals) == 1: return None if vals[0] is None else not vals[0]
    return None

class Rust:
    def __init__(self, path, source, features=(), declared=(), provider_flags=()):
        self.path, self.source = path, source
        self.tok = tokens(source)
        self.ts = [x[0] for x in self.tok]
        self.pairs = pairs(self.ts)
        self.attrs, self.guards, self.modules = [], [], []
        ts = self.ts
        i = 0
        while i < len(ts)-1:
            if ts[i:i+2] != ["#", "["] or i+1 not in self.pairs:
                i += 1
                continue
            # One chain of attributes belonging to a single item.
            begin = i
            cfgs, override, attr_text = [], None, []
            while ts[i:i+2] == ["#", "["] and i+1 in self.pairs:
                end = self.pairs[i+1]
                a = ts[i+2:end]
                attr_text.append(" ".join(a))
                if a[:2] == ["cfg", "("] and a[-1:] == [")"]:
                    cfgs.append(evaluate(a[2:-1], features, declared, provider_flags))
                if len(a)==3 and a[:2]==["path","="] and a[2].startswith('"'):
                    override = a[2][1:-1]
                i = end+1
            start = i
            end = self.item_end(start)
            self.attrs.append((start,end,attr_text,override))
            if False in cfgs and end > start:
                self.guards.append((start,end,attr_text))
            if i == begin: i += 1
        # External module declarations, including alternate #[path] modules.
        for i in range(len(ts)-2):
            if ts[i]=="mod" and IDENT.fullmatch(ts[i+1]) and ts[i+2]==";":
                attr = [a for a in self.attrs if a[0]<=i<a[1] and i-a[0]<=5]
                override = next((a[3] for a in reversed(attr) if a[3] is not None),None)
                self.modules.append((i,ts[i+1],override))

    def item_end(self, i):
        ts = self.ts
        j = i
        while j < len(ts):
            t = ts[j]
            if t == ";": return j+1
            if t == "{": return self.pairs.get(j,j)+1
            if t in ("(", "[") and j in self.pairs:
                j = self.pairs[j]+1
                continue
            if t in ("}", "]", ")"): return j
            j += 1
        return j

    def guard(self,i):
        return next((a for start,end,a in self.guards if start<=i<end),None)

    def loc(self,i):
        return f"{self.path}:{self.source.count(chr(10),0,self.tok[i][1])+1}"

    def externs(self):
        ts=self.ts
        rows=[]
        for i in range(len(ts)-2):
            if ts[i]!="extern": continue
            # extern fn definitions are not imports. Omitted ABI defaults to C.
            abi=ts[i+1] if ts[i+1].startswith('"') else '"C"'
            opening=i+2 if ts[i+1].startswith('"') else i+1
            if ts[opening]!="{" or opening not in self.pairs: continue
            end=self.pairs[opening]
            block=ts[opening+1:end]
            macros=("$" in block or "!" in block)
            for j in range(opening+1,end-1):
                if ts[j]=="fn" and IDENT.fullmatch(ts[j+1]) and ts[j+2:j+3]==["("]:
                    attrs=[a for a in self.attrs if a[0]<=j<a[1] and j-a[0]<=8]
                    alias=next((x.split('"')[1] for a in attrs for x in a[2]
                                if x.startswith("link_name = \"")),None)
                    rows.append(dict(kind="symbol",name=ts[j+1],link_name=alias or ts[j+1],
                                     abi=abi,site=self.loc(j),guard=self.guard(j)))
                elif ts[j] in ("static","type"):
                    name=ts[j+2] if ts[j]=="static" and ts[j+1]=="mut" and j+2<end else ts[j+1]
                    if IDENT.fullmatch(name):
                        rows.append(dict(kind=ts[j],name=name,link_name=name,abi=abi,
                                         site=self.loc(j),guard=self.guard(j)))
            if macros:
                rows.append(dict(kind="macro_block",name=None,link_name=None,abi=abi,
                                 site=self.loc(i),guard=self.guard(i),
                                 macro_invocations=sorted(set(block[k-1] for k,t in enumerate(block)
                                                              if t=="!" and k and IDENT.fullmatch(block[k-1])))))
        return rows

def feature_intent(root, members):
    tables={c:tomllib.loads((root/c/"Cargo.toml").read_text()).get("features",{}) for c in members}
    active=defaultdict(set,{c:set(v) for c,v in SEEDS.items()})
    changed=True
    external=defaultdict(set)
    while changed:
        before=sum(map(len,active.values()))
        for c in list(active):
            for name in list(active[c]):
                for dep in tables.get(c,{}).get(name,[]):
                    if dep.startswith("dep:"): continue
                    if "/" in dep:
                        pkg, feat=dep.split("/",1)
                        pkg=pkg.rstrip("?")
                        if pkg in tables: active[pkg].add(feat)
                        else: external[pkg].add(feat)
                    else:
                        active[c].add(dep)
        changed=sum(map(len,active.values()))!=before
    return tables,active,external

def module_exclusions(parsed):
    """Only mark a module excluded if all resolved declarations exclude it.
    Unknown or missing module paths remain review candidates.
    """
    edges=defaultdict(list)
    for path,r in parsed.items():
        p=Path(path)
        # Standard Rust file layout; #[path] relative to containing file.
        directory=p.parent if p.name in ("lib.rs","main.rs","mod.rs") else p.parent/p.stem
        for i,name,override in r.modules:
            if override is not None:
                candidates=[p.parent/override, directory/override]
            else:
                candidates=[directory/(name+".rs"),directory/name/"mod.rs"]
            for candidate in set(candidates):
                target=candidate.as_posix()
                if target in parsed:
                    edges[target].append((path,r.guard(i),r.loc(i)))
    excluded={}
    changed=True
    while changed:
        changed=False
        for target,parents in edges.items():
            if target not in excluded and parents and all(g is not None or p in excluded for p,g,_ in parents):
                excluded[target]=[site for _,_,site in parents]
                changed=True
    return excluded,edges

def build_inventory(path,source,features,declared):
    r=Rust(path,source,features,declared)
    rows=[]
    # Enumerate call/directive candidates independently of legacy archive list.
    for i,(t,start,_) in enumerate(r.tok):
        if not PRODUCER.search(t) and not (t=="Build" and r.ts[max(0,i-2):i]==["cc","::"]): continue
        if t.startswith('"') and not ("rustc-link-" in t or t.strip('"') in ("nvcc","ptxas")): continue
        if t not in ("KernelBuilder","CudaBuilder","build_lib","source_glob","source_files","nvcc","ptxas",
                     "compile_ptx","compile_cuda","Build","link_name") and not t.startswith('"'): continue
        rows.append(dict(site=r.loc(i),token=t,guard=r.guard(i),line=source.splitlines()[source.count("\n",0,start)].strip()))
    return rows

def paged_early_return(source):
    r=Rust("paged/build.rs",source)
    ts=r.ts
    pattern=["if","cfg","!","(","feature","=",'"dkr-paged-catalog-only"',")","{"]
    hits=[i for i in range(len(ts)) if ts[i:i+len(pattern)]==pattern]
    require(len(hits)==1,"paged producer early-return guard")
    start=hits[0]+len(pattern)-1
    end=r.pairs[start]
    require(ts[end-6:end]==["return","Ok","(","(",")",")"] or
            ts[end-7:end]==["return","Ok","(","(",")",")",";"],"paged early return must be unconditional tail")
    # Prove each archive build/link is lexically later in the same main body.
    mains=[i for i in range(len(ts)-2) if ts[i:i+3]==["fn","main","("]]
    owners=[i for i in mains if i<hits[0]<r.item_end(i)]
    require(len(owners)==1,"paged guard must belong to main")
    main_end=r.item_end(owners[0])
    sites=[i for i,t in enumerate(ts) if t=="build_lib" or LIB_LINK.search(t)]
    require(sites and all(end<i<main_end for i in sites),"paged producer/link outside early-return fence")
    return [r.loc(i) for i in sites]

def self_test():
    sample=r'''
// extern "C" { fn ignored(); }
/* outer /* nested */ extern "C" { fn ignored2(); } */
const TEXT: &str = r#"extern "C" { fn fake(); }"#;
#[cfg(not(feature="strict"))]
extern "C" { pub fn closed(); }
extern "C" {
    #[cfg(not(feature="strict"))]
    pub(crate) fn closed2();
    #[link_name="native_alias"] pub fn alias();
    fn ordinary();
    static FLAG: i32;
    declare!(generated);
}
extern { fn implicit_abi(); }
#[cfg(unknown_flag)]
extern "C" { pub fn unknown(); }
#[cfg(any(not(feature="strict"),unknown_flag))]
extern "C" { pub fn maybe(); }
#[cfg(not(feature="strict"))] mod old;
mod open;
'''
    r=Rust("crate/src/lib.rs",sample,{"strict"},{"strict"})
    rows=r.externs()
    symbols={x["name"]:x for x in rows if x["kind"]=="symbol"}
    require(set(symbols)=={"closed","closed2","alias","ordinary","unknown","maybe","implicit_abi"},"literal extern inventory")
    require(any(x["kind"]=="static" and x["name"]=="FLAG" for x in rows),"extern static inventory")
    require(symbols["closed"]["guard"] and symbols["closed2"]["guard"],"negative guard")
    require(all(symbols[x]["guard"] is None for x in ("alias","ordinary","unknown","maybe")),"unproven cfg is not exclusion")
    require(symbols["alias"]["link_name"]=="native_alias" and any(x["kind"]=="macro_block" for x in rows),"alias/macro")
    parsed={r.path:r,"crate/src/old.rs":Rust("crate/src/old.rs",'extern "C" { fn a(); }'),
            "crate/src/open.rs":Rust("crate/src/open.rs",'extern "C" { fn b(); }')}
    excluded,_=module_exclusions(parsed)
    require(set(excluded)=={"crate/src/old.rs"},"module fence")
    duplicate=Rust(r.path,sample+"\nmod old;",{"strict"},{"strict"})
    parsed[r.path]=duplicate
    require("crate/src/old.rs" not in module_exclusions(parsed)[0],"duplicate module bypass")
    paged='fn main(){ if cfg!(feature="dkr-paged-catalog-only"){return Ok(());} b.build_lib(x); println!("cargo:rustc-link-lib=mistralrspagedattention"); }'
    require(len(paged_early_return(paged))==2,"paged producer fence")
    failures=0
    for bad in (paged.replace("return Ok(());","let _ = Ok(());"),
                paged.replace("b.build_lib(x);","")+ "fn leak(){ b.build_lib(x); }",
                paged.replace('if cfg!', 'if other!')):
        try: paged_early_return(bad)
        except ValueError: failures+=1
        else: raise AssertionError("invalid paged producer fence accepted")
    print(f"DKR_R126_S9C3A_SELF_TEST=PASS negative_cases={failures+3} macro_inventory=true aliases=true cfg_unknown_conservative=true")

def audit(root):
    head=git(root,"rev-parse","HEAD").decode().strip()
    tracked=sorted(x.decode() for x in git(root,"ls-files","-z").split(b"\0") if x)
    ws=tomllib.loads((root/"Cargo.toml").read_text())
    members=ws["workspace"]["members"]
    require(all((root/c/"Cargo.toml").is_file() for c in members),"workspace member inventory")
    def in_scope(p):
        return (p=="Cargo.toml" or p.startswith(".cargo/") and p.endswith(".toml")
                or p.startswith("tools/") and p.endswith((".py",".rs"))
                or any(p.startswith(c+"/") for c in members) and
                   (p.endswith((".rs",".toml")) or Path(p).suffix in NATIVE)
                or p.startswith("kernel-packages/") and (p.endswith((".py",".rs",".toml")) or Path(p).suffix in NATIVE))
    selected=[p for p in tracked if in_scope(p)]
    untracked=[p.decode() for p in git(root,"ls-files","--others","--exclude-standard","-z").split(b"\0") if p]
    require(not any(in_scope(p) for p in untracked),"untracked source in audit scope")
    require(not git(root,"status","--porcelain","--untracked-files=no","--",*selected).strip(),"audited tracked source dirty")
    prior=previous(root)
    require(prior["closed_inventory"]==60 and prior["residual_ffi"]==0,"S9C.2E cumulative gate")
    tables,active,external=feature_intent(root,members)
    require(all(active[c] for c in COMPONENTS),"missing strict source feature propagation")
    require(all(not EXCLUDED_FEATURES.intersection(active[c]) for c in COMPONENTS),"forbidden source feature")
    require("dkr-paged-catalog-only" in active["mistralrs-paged-attn"],"paged source intent")
    # Build-generated cfg predicates: false only for known disabled producer
    # flags in this source scenario, not arbitrary has_* names.
    provider_flags=defaultdict(set)
    builds={}
    for c in COMPONENTS:
        path=c+"/build.rs"
        source=(root/path).read_text()
        provider_flags[c]=set(re.findall(r'cargo::?rustc-cfg=(\w+)"',source))
        rows=build_inventory(path,source,active[c],tables[c])
        require("DKR_STATIC_PRODUCER_DISABLED component="+c in source,"producer marker "+c)
        if c=="mistralrs-paged-attn":
            fenced=paged_early_return(source)
        else:
            r=Rust(path,source,active[c],tables[c])
            sites=[i for i,t in enumerate(r.ts) if t=="build_lib" or LIB_LINK.search(t)]
            require(sites and all(r.guard(i) is not None for i in sites),"unfenced static build/link "+c)
            fenced=[r.loc(i) for i in sites]
        builds[c]=dict(static_build_link_sites=fenced,candidates=rows,producer_cfgs_absent_in_scenario=sorted(provider_flags[c]))
    parsed={}
    supplemental={}
    for p in selected:
        c=p.split("/",1)[0]
        if not p.endswith(".rs"): continue
        try:
            declared=tables.get(c,{}) if c in active else {}
            r=Rust(p,(root/p).read_text(),active.get(c,set()),declared,provider_flags[c])
            if c in members and p.startswith(c+"/src/"):
                parsed[p]=r
            else:
                supplemental[p]=r
        except ValueError as exc:
            raise ValueError(f"{p}: {exc}") from exc
    excluded,edges=module_exclusions(parsed)
    rows=[]
    for p,r in parsed.items():
        c=p.split("/",1)[0]
        for x in r.externs():
            x.update(component=c,path=p)
            if c=="mistralrs-flash-attn":
                x["classification"]="separate_producer_excluded_by_intended_features"
            elif p in excluded:
                x.update(classification="module_cfg_excluded",module_evidence=excluded[p])
            elif x["guard"] is not None:
                x["classification"]="local_cfg_excluded"
            elif c in COMPONENTS:
                x["classification"]="review_legacy_candidate"
            else:
                x["classification"]="review_external_abi"
            rows.append(x)
    # Index literal identifier references once. A same-name reference is NOT
    # a resolved call; record qualified occurrences and module/local guards.
    names=defaultdict(set)
    for x in rows:
        if x["name"]: names[x["component"]].add(x["name"])
    refs=defaultdict(list)
    for p,r in parsed.items():
        c=p.split("/",1)[0]
        for i,t in enumerate(r.ts):
            if t not in names[c]: continue
            refs[c,t].append(dict(site=r.loc(i),qualified=(i>0 and r.ts[i-1]=="::"),
                                  guard=r.guard(i),module_excluded=p in excluded))
    for x in rows:
        if x["name"]:
            x["references"]=[v for v in refs[x["component"],x["name"]] if v["site"]!=x["site"]]
            x["unexcluded_qualified_candidates"]=sum(v["qualified"] and v["guard"] is None and not v["module_excluded"] for v in x["references"])
    all_builds={}
    for p in selected:
        if not p.endswith("/build.rs"): continue
        c=p.split("/",1)[0]
        source=(root/p).read_text()
        all_builds[p]=build_inventory(p,source,active[c],tables.get(c,{}))
    # Also inventory other Rust native link attributes, build helper calls,
    # nonstandard producer source files, and separate host tooling.
    additional=[]
    qualified=[]
    generated=[]
    supplemental_externs=[]
    for p,r in supplemental.items():
        supplemental_externs.extend(dict(x,path=p,classification="review_non_library_target") for x in r.externs())
    for p,r in {**parsed,**supplemental}.items():
        for i,t in enumerate(r.ts):
            if t in ("link","link_name") and i>0 and r.ts[i-1]=="[":
                additional.append(dict(site=r.loc(i),kind="link_attribute",guard=r.guard(i),module_excluded=p in excluded))
            if t in ("build_lib","compile_ptx","compile_cuda","KernelBuilder") or (t=="Build" and r.ts[max(0,i-2):i]==["cc","::"]):
                additional.append(dict(site=r.loc(i),kind="producer_candidate",guard=r.guard(i),module_excluded=p in excluded))
            if t in ("include","include_bytes","include_str") and r.ts[i+1:i+2]==["!"]:
                generated.append(dict(site=r.loc(i),kind=t,guard=r.guard(i),module_excluded=p in excluded))
            if (t=="ffi" or t.endswith("_ffi")) and r.ts[i+1:i+2]==["::"] and i+2<len(r.ts):
                if i>=2 and r.ts[i-2] in ("std","core") and r.ts[i-1]=="::":
                    continue
                name=r.ts[i+2]
                if not IDENT.fullmatch(name): continue
                c=p.split("/",1)[0]
                qualified.append(dict(site=r.loc(i),module=t,name=name,literal_declared_in_component=name in names[c],
                                      guard=r.guard(i),module_excluded=p in excluded,
                                      scope="library_source" if p in parsed else "supplemental_target"))
    native=[p for p in selected if Path(p).suffix in NATIVE]
    counts=Counter(x["classification"] for x in rows)
    per_component={}
    for c in members:
        rs=[x for x in rows if x["component"]==c]
        if rs: per_component[c]=dict(Counter(x["classification"] for x in rs))
    review=[x for x in rows if x["classification"].startswith("review_")]
    families=Counter("/".join(x["path"].split("/")[:4]) for x in review)
    lock=(root/"Cargo.lock").read_bytes()
    lock_dirty=bool(git(root,"status","--porcelain","--","Cargo.lock").strip())
    require(head==git(root,"rev-parse","HEAD").decode().strip() and
            not git(root,"status","--porcelain","--untracked-files=no","--",*selected).strip(),"source changed during audit")
    return dict(schema="dkr.r126.s9c3a.source.v1",status="INVENTORY_COMPLETE_REVIEW_REQUIRED" if review else "INVENTORY_COMPLETE",
                source_head=head,workspace_members=members,scanned_rust_files=len(parsed)+len(supplemental),
                library_rust_files=len(parsed),supplemental_rust_files=len(supplemental),
                supplemental_extern_inventory=supplemental_externs,
                qualified_ffi_inventory=qualified,generated_input_sites=generated,
                tracked_source_paths=len(selected),native_source_inventory=native,
                source_feature_intent={c:sorted(active[c]) for c in sorted(active)},
                external_feature_intent={c:sorted(v) for c,v in sorted(external.items())},
                candle_pin_expected=PIN,candle_pin_verified_in_metadata=False,
                cargo_lock_sha256=sha(lock),cargo_lock_dirty=lock_dirty,
                producer_guards=builds,all_build_scripts=all_builds,additional_native_sites=additional,
                module_exclusions=excluded,extern_counts=dict(counts),extern_by_component=per_component,
                extern_inventory=rows,review_families=dict(sorted(families.items())),review_candidates=len(review),
                previous_inventory_closed=60,previous_inventory_residual=0,
                projected_master_logical=4268,projected_master_physical_upper=552,
                source_only=True,compilation=False,link_qualified=False,full_closure=False,
                limitations=["source feature intent is not resolved Cargo metadata",
                    "lexical token and module analysis is not rustc cfg/macro expansion or name resolution",
                    "declarations and identifier references do not establish live native calls",
                    "unknown cfgs, generated bindings, include! and external dependencies require compiler/link evidence",
                    "test=false production scenario; GPU tests and model coverage remain separate",
                    "Candle embedded catalogue, CUDA driver and vendor libraries require dependency evidence in S9C.3B/S10"])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    p.add_argument("--require-closed",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f: json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    for c,counts in r["extern_by_component"].items():
        print(f"DKR_R126_S9C3A_COMPONENT component={c} extern_records={sum(counts.values())} review={sum(v for k,v in counts.items() if k.startswith('review_'))}")
    for family,count in r["review_families"].items():
        print(f"DKR_R126_S9C3A_REVIEW family={family} records={count}")
    print(f"DKR_R126_S9C3A_NATIVE_SITES supplemental_extern_records={len(r['supplemental_extern_inventory'])} qualified_ffi_sites={len(r['qualified_ffi_inventory'])} generated_input_sites={len(r['generated_input_sites'])}")
    print(f"DKR_R126_S9C3A_SOURCE=INVENTORY_COMPLETE producers_guarded=3 previous_residual=0 review_candidates={r['review_candidates']} rust_files={r['scanned_rust_files']} cargo_lock_dirty={str(r['cargo_lock_dirty']).lower()} compilation=false link_qualified=false full_closure=false")
    if a.require_closed and (r["review_candidates"] or r["supplemental_extern_inventory"]):
        raise SystemExit("DKR_R126_S9C3A_CLOSURE_GATE=BLOCKED unresolved source review candidates")

if __name__=="__main__":
    main()
