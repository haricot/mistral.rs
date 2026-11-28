#!/usr/bin/env python3
"""Argsort/F64 source closure and independent index-network oracle."""
import argparse
import copy
import importlib.util
import itertools
import json
import random
import re
import tempfile
from pathlib import Path
from dkr_r126_s9c2e4_audit import audit as previous
from dkr_r126_s9c2e0_audit import ARG_SORT, GRAPH, self_test as partition_self_test
from dkr_r126_s9c2e1_audit import check_declarations, checked_refs
from dkr_r126_gguf_residual_audit import mask, guarded_ranges
from dkr_r126_s9c2d3c_audit import strip_comments
import dkr_r126_model_catalog_scope as scope

CORE=Path("mistralrs-core/src")
RUNTIME=Path("mistralrs-kernel-runtime")
PKG=Path("kernel-packages/cuda/argsort")
TYPES={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","f64":"double","u8":"uint8_t","u32":"uint32_t","i64":"int64_t"}

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def load(root):
    spec=importlib.util.spec_from_file_location("argsort_build",root/PKG/"build.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def normalize(s):
    return re.sub(r"\s*\*\s*","* "," ".join(s.split()))

def check(source,rows,cc):
    found_list=re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{(.*?)\}',source,re.S)
    found={name:(params,body) for name,params,body in found_list}
    require(len(found_list)==len(found)==len(rows)==23,"entry inventory")
    require(set(found)=={r["entry"] for r in rows},"entry set")
    require(len({r["implementation_id"] for r in rows})==23,"identity inventory")
    for row in rows:
        entry=row["entry"]
        params,body=found[entry]
        expected=[]
        for a in row["arguments"]:
            ty=TYPES[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
                require(a["rank"]==1,"tensor rank")
            expected.append(ty+" "+a["name"])
        require([normalize(p) for p in params.split(",")]==expected,"CUDA ABI "+entry)
        require(row["implementation_id"]=="cuda."+entry+".v1","identity "+entry)
        require(row["target_cc"]==row["min_cc"]==row["max_cc"]==cc,"target "+entry)
        require(row["block"]==[256,1,1] and row["shared_mem_bytes"]==0,"geometry "+entry)
        require(row["artifact"]=="argsort.ptx","image")
        names=[a["name"] for a in row["arguments"]]
        suffix=entry.removeprefix("dkr_argsort_")
        if suffix in ("init","finish"):
            op=suffix
            call=f"argsort_{suffix}("+", ".join(names)+");"
        elif suffix.startswith("gather_"):
            dt=suffix.removeprefix("gather_")
            op="gather"
            require(row["arguments"][0]["dtype"]==row["arguments"][2]["dtype"]==dt,"gather dtype")
            call=f"argsort_gather<{TYPES[dt]}>("+", ".join(names)+");"
        else:
            direction,dt=suffix.split("_")
            op="compare."+direction
            require(direction in ("asc","desc") and row["arguments"][0]["dtype"]==dt,"compare dtype")
            require(row["arguments"][1]==dict(kind="tensor",name="work",dtype="u32",rank=1,access="read_write"),"mutable workspace")
            call=f"argsort_compare<{TYPES[dt]}, {str(direction=='asc').lower()}>("+", ".join(names)+");"
        require(row["operation"]=="tensor.argsort."+op+".v1","operation "+entry)
        require(" ".join(body.split())==call,"template/argument order "+entry)
    require(max(len(r["arguments"]) for r in rows)==6,"argument limit")

def oracle(values,asc):
    """Model the pair network, using indices and an explicit padding class."""
    n=len(values)
    require(n>0,"empty oracle row")
    padded=1<<(n-1).bit_length()
    work=list(range(padded))
    k=2
    while k<=padded:
        j=k//2
        while j:
            for i in range(padded):
                other=i^j
                if other<=i:
                    continue
                a,b=work[i],work[other]
                a_pad,b_pad=a>=n,b>=n
                up=((i&k)==0) if asc else ((i&k)!=0)
                if a_pad!=b_pad:
                    exchange=a_pad if up==asc else b_pad
                elif not a_pad:
                    exchange=values[a]>values[b] if up else values[a]<values[b]
                else:
                    exchange=False
                if exchange:
                    work[i],work[other]=b,a
            j//=2
        k*=2
    return work[:n]

def legacy_power2(values,asc):
    """Independent value-swapping reference for unpadded historical rows."""
    data=list(values)
    ids=list(range(len(values)))
    k=2
    while k<=len(data):
        j=k//2
        while j:
            for i in range(len(data)):
                other=i^j
                if other>i:
                    up=((i&k)==0) if asc else ((i&k)!=0)
                    if (data[i]>data[other]) if up else (data[i]<data[other]):
                        data[i],data[other]=data[other],data[i]
                        ids[i],ids[other]=ids[other],ids[i]
            j//=2
        k*=2
    return ids

def oracle_self_test():
    count=0
    def test(values,asc):
        nonlocal count
        ids=oracle(values,asc)
        require(sorted(ids)==list(range(len(values))),"padding leaked or invalid permutation")
        require([values[i] for i in ids]==sorted(values,reverse=not asc),"numeric order")
        if len(values)&(len(values)-1)==0:
            require(ids==legacy_power2(values,asc),"legacy unpadded tie order")
        count+=1
    for n in range(1,8):
        for values in itertools.product((-2,0,2),repeat=n):
            for asc in (False,True):
                test(values,asc)
    rng=random.Random(126)
    for n in (31,32,33,255,256,257,1023,1024,1025):
        values=[rng.randrange(-1000,1000) for _ in range(n)]
        for asc in (False,True):
            test(values,asc)
    for values in ((-3.,-1.,-2.),(1.+2.**-52,0.,1.),
                   (-(1<<63),(1<<63)-1,(1<<63)-2),(0,2**32-1,2**32-2),
                   (float("inf"),-float("inf"),0.,-0.,5e-324),
                   (0,255,255,0,0)):
        for asc in (False,True):
            test(values,asc)
    for values in ((float("nan"),2.,-3.),(float("nan"),)*5):
        for asc in (False,True):
            require(sorted(oracle(values,asc))==list(range(len(values))),"NaN padding leak")
            count+=1
    return count

def scope_example(m):
    chosen={"dkr_argsort_init","dkr_argsort_finish","dkr_argsort_asc_f64",
            "dkr_argsort_desc_f64","dkr_argsort_asc_u32","dkr_argsort_gather_f64"}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/"master"
        rows=m.plan(61,1)
        ptx=".version 8.0\n.target sm_61\n.address_size 64\n"
        for row in rows:
            params=",".join(f".param .b{64 if a['kind']=='tensor' else 32} a{j}"
                            for j,a in enumerate(row["arguments"]))
            ptx+=f".visible .entry {row['entry']}({params}) {{ret;}}\n"
        m.stage(root/"image-01",61,rows,ptx.encode())
        root,records,manifests=scope.inventory(root)
        keys=[key for key in records if key[1].removeprefix("cuda.").removesuffix(".v1") in chosen]
        require(len(keys)==len(chosen)==6,"argsort scope inventory")
        spec=dict(schema=scope.SELECTION,model="synthetic-f64-u32-argsort",device_cc=61,
                  profiles=[dict(zip(("operation","implementation_id","target_cc"),key)) for key in keys])
        group=scope.selection(spec,records,manifests)
        report,images=scope.report(root,records,manifests,spec,group)
        dest=Path(tmp)/"active"
        scope.materialize(root,dest,group,manifests,images,61)
        _,selected,selected_manifests=scope.inventory(dest)
        require(set(selected)==set(keys) and len(selected_manifests)==1,"argsort scope materialization")
        require(report["scope_receipt"]==scope.decode((dest/scope.SCOPE_FILENAME).read_bytes()),"scope receipt")
        return dict(master_profiles=23,master_manifests=1,union_profiles=6,union_manifests=1,synthetic_only=True)

def producer(root,negative=False):
    m=load(root)
    source=(root/PKG/"argsort.cu").read_text()
    for cc in (61,80):
        rows=m.plan(cc,1)
        check(source,rows,cc)
        if negative:
            m.self_test(cc,rows)
    header=(root/CORE/"cuda/argsort_device.cuh").read_text()
    for marker in ("auto argsort_value(T x) { return x; }","__half2float(x)","__bfloat162float(x)",
                   "const bool a_pad = a >= cols, b_pad = b >= cols;",
                   "exchange = (up == ASC) ? a_pad : b_pad;",
                   "exchange = up ? av > bv : av < bv;",
                   "if (other <= i || other >= padded) return;",
                   "output[i] = input[base + (id < cols ? id : 0)]"):
        require(marker in header,"device contract "+marker)
    require("numeric_limits" not in header and "const T *input" in header,"no sentinel and immutable input")
    runtime=(root/RUNTIME/"src/universal.rs").read_text()
    require("pub enum ElementType {\n    F64," in runtime and "Self::F64 | Self::I64 | Self::U64 => 8" in runtime,
            "native typed F64")
    require("fn dkr_argsort_f64_typed_abi()" in runtime,"F64 runtime test")
    shard=(root/"kernel-packages/cuda/dkr_s9c2d_shard.py").read_text()
    require('("i64","u64","f64")' in shard,"F64 scalar width in producer")
    builder=(root/RUNTIME/"src/bin/dkr_package_builder.rs").read_text()
    require('name: "argsort", source: "argsort.cu", fast_math: false' in builder and
            'pkg.name == "argsort"' in builder and "argsort_device.cuh" in builder,"Cudaforge registration")
    failures=0
    if negative:
        rows=m.plan(61,1)
        f64=next(i for i,row in enumerate(rows) if row["entry"]=="dkr_argsort_asc_f64")
        wrong=copy.deepcopy(rows);wrong[f64]["arguments"][0]["dtype"]="f32"
        block=copy.deepcopy(rows);block[f64]["block"]=[128,1,1]
        access=copy.deepcopy(rows);access[f64]["arguments"][1]["access"]="read"
        for code,manifest in ((source,wrong),(source,block),(source,access),
                              (source.replace("argsort_compare<double, true>","argsort_compare<float, true>",1),rows),
                              (source.replace("work, cols, padded, j, k","work, cols, padded, k, j",1),rows)):
            try:
                check(code,manifest,61)
            except ValueError:
                failures+=1
            else:
                raise AssertionError("invalid argsort ABI accepted")
    return dict(profiles=23,images=1,max_args=6,negative_cases=failures,scope_example=scope_example(m))

def audit(root):
    prior=previous(root)
    require(prior["residual_ffi"] in (2,0) and set(prior["closed_argsort_declarations"])==ARG_SORT,"cumulative argsort closure")
    result=producer(root)
    ffi=(root/CORE/"cuda/ffi.rs").read_text()
    check_declarations(ffi,ARG_SORT)
    clean=strip_comments(ffi)
    spans=guarded_ranges(clean)
    residual={m.group(1) for m in re.finditer(r'\bpub\(crate\)\s+fn\s+(\w+)\s*\(',mask(clean))
              if not any(a<=m.start()<b for a,b in spans)}
    closed_graph=set(prior["closed_graph_declarations"])
    require(closed_graph in (set(),GRAPH) and residual==GRAPH-closed_graph and len(residual)==prior["residual_ffi"],"remaining FFI must be exactly unclosed graph copies")
    pattern=re.compile(r"\bffi::("+"|".join(sorted(ARG_SORT))+r")\b")
    found=[]
    for path in (root/CORE).rglob("*.rs"):
        if path==root/CORE/"cuda/ffi.rs":
            continue
        code=strip_comments(path.read_text())
        spans=guarded_ranges(code)
        for match in pattern.finditer(mask(code)):
            require(path==root/CORE/"ops.rs","unexpected Argsort call site")
            require(any(a<=match.start()<b for a,b in spans),"unguarded Argsort call "+match.group(1))
            found.append(match.group(1))
    require(len(found)==14 and set(found)==ARG_SORT,"legacy call inventory")
    ops=(root/CORE/"ops.rs").read_text()
    adapter=(root/CORE/"cuda/argsort_dynamic.rs").read_text()
    for marker in ("argsort_dynamic::indices(storage, layout, self.last_dim, self.asc)",
                   "self.apply_op2_no_bwd(&asort, &crate::cuda::argsort_dynamic::ArgsortGather",
                   "indices.arg_sort(true)?"):
        require(marker in ops,"dispatch "+marker)
    for marker in ("LaunchPlanCache<SortKey,SortPlan>","LaunchPlanCache<GatherKey,PrevalidatedLaunchPlan>",
                   "prevalidate_launch_with_default_shared(&request,device,0)","p.manifest.arguments!=params",
                   "offset.checked_add(count)","layout.start_offset()","checked_next_power_of_two()",
                   "work=unsafe{dev.alloc::<u32>(g.padded as usize)}","||sort_plan(&device,key)",
                   "DType::F64=>Ok(ElementType::F64)","gather!(F64,f64,s)"):
        require(marker in adapter,"adapter contract "+marker)
    require("pub use cuda::argsort_dynamic::dkr_argsort_requirements;" in (root/CORE/"lib.rs").read_text(),"capability export")
    require(prior["projected_master_logical"]+23==4268 and prior["projected_master_physical_upper"]+1==552,"master budget")
    return dict(result,status="SOURCE_COMPLETE",guarded_ffi=14,residual_ffi=len(residual),closed_graph_declarations=sorted(closed_graph),
                projected_master_logical=4268,projected_master_physical_upper=552,
                padding_policy="explicit_valid_index",f64_native=True,nan_order_unspecified=True,
                active_limits_unchanged=True,model_scope_fit_proven=False,
                compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    root=a.source_root.resolve()
    if a.self_test:
        result=producer(root,True)
        count=oracle_self_test()
        partition_self_test()
        print(f"DKR_R126_S9C2E5_AUDIT_SELF_TEST=PASS negative_cases={result['negative_cases']} oracle_cases={count} scoped_fixture=true synthetic_ptx=true")
        return
    r=audit(root)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2E5_SOURCE=COMPLETE guarded_ffi=14 residual={r['residual_ffi']} profiles=23 images=1 max_args=6 f64_native=true padding=explicit_valid_index projected_master_logical=4268 projected_master_physical=552 model_scope_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
