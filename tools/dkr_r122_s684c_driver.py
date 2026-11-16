#!/usr/bin/env python3
"""S6.8.4-C: real, non-primary CUDA Driver contexts; no CUDA build."""
import argparse
import ctypes as C
import json
import re
from pathlib import Path
from dkr_r122_s6_evidence import native_audit
from dkr_r122_s683_audit import need, sha

P=C.c_void_p
I=C.c_int
U=C.c_uint
S=C.c_size_t
D=C.c_uint64
N=257

class Driver:
    def __init__(self):
        self.lib=C.CDLL("libcuda.so.1")
        signatures={
            "cuInit":[U],"cuDeviceGet":[C.POINTER(I),I],
            "cuDeviceGetAttribute":[C.POINTER(I),I,I],
            "cuCtxCreate_v2":[C.POINTER(P),U,I],
            "cuCtxDestroy_v2":[P],"cuCtxGetCurrent":[C.POINTER(P)],"cuCtxSetCurrent":[P],
            "cuCtxSynchronize":[],
            "cuModuleLoadData":[C.POINTER(P),P],"cuModuleUnload":[P],
            "cuModuleGetFunction":[C.POINTER(P),P,C.c_char_p],
            "cuMemAlloc_v2":[C.POINTER(D),S],"cuMemFree_v2":[D],
            "cuPointerGetAttribute":[P,I,D],
            "cuMemcpyDtoH_v2":[P,D,S],
            "cuLaunchKernel":[P,U,U,U,U,U,U,U,P,C.POINTER(P),C.POINTER(P)],
        }
        for name,args in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=I
        self.call("cuInit",0)
        dev=I();self.call("cuDeviceGet",C.byref(dev),0);self.dev=dev
        cc=[]
        for attr in (75,76):
            v=I();self.call("cuDeviceGetAttribute",C.byref(v),attr,dev);cc.append(v.value)
        need(cc==[6,1],"expected SM61")
    def call(self,name,*args):
        code=getattr(self.lib,name)(*args)
        if code: raise RuntimeError(f"{name}: CUDA Driver error {code}")
    def current(self):
        ctx=P();self.call("cuCtxGetCurrent",C.byref(ctx));return ctx.value
    def bind(self,ctx):
        self.call("cuCtxSetCurrent",ctx)
        need(self.current()==ctx.value,"context binding differs")

class Instance:
    def __init__(self,driver,ptx):
        self.d=driver;self.ctx=P();self.mod=P();self.func=P();self.ptr=D()
        try:
            self.d.call("cuCtxCreate_v2",C.byref(self.ctx),0,self.d.dev)
            self.d.bind(self.ctx)
            self.d.call("cuModuleLoadData",C.byref(self.mod),C.cast(ptx,P))
            self.d.call("cuModuleGetFunction",C.byref(self.func),self.mod,b"fill_f32")
            self.d.call("cuMemAlloc_v2",C.byref(self.ptr),N*4)
            owner=P()
            # CU_POINTER_ATTRIBUTE_CONTEXT = 1.
            self.d.call("cuPointerGetAttribute",C.byref(owner),1,self.ptr)
            need(owner.value==self.ctx.value,"allocation context differs")
        except BaseException:
            self.close()
            raise
    def launch(self,value):
        self.d.bind(self.ctx)
        f=C.c_float(value);n=S(N)
        params=(P*3)(C.cast(C.byref(self.ptr),P),C.cast(C.byref(f),P),C.cast(C.byref(n),P))
        self.d.call("cuLaunchKernel",self.func,3,1,1,128,1,1,0,None,params,None)
        self.d.call("cuCtxSynchronize")
        self.check(value)
    def check(self,value):
        self.d.bind(self.ctx)
        out=(C.c_float*N)()
        self.d.call("cuMemcpyDtoH_v2",out,self.ptr,N*4)
        need(all(x==value for x in out),"foreign or stale allocation contents")
    def close(self):
        if not self.ctx.value:return
        try:
            self.d.bind(self.ctx)
            self.d.call("cuCtxSynchronize")
            if self.ptr.value:self.d.call("cuMemFree_v2",self.ptr)
            if self.mod.value:self.d.call("cuModuleUnload",self.mod)
        finally:
            self.d.call("cuCtxDestroy_v2",self.ctx)
            self.ctx=P()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    need(not a.output.exists(),"refusing overwrite")
    native_audit(a.native_root)
    path=a.native_root/"fill/fill.ptx"
    payload=path.read_bytes()
    # fill_f32(float*, float, size_t) from the pinned Candle fill.cu.
    match=re.search(r"\.entry\s+fill_f32\s*\((.*?)\)",payload.decode(),re.S)
    need(match is not None,"fill_f32 missing")
    # NVCC may use .b32 or .f32 for the scalar float parameter.
    params=re.findall(r"\.param\s+\.(?:[bfsu])(\d+)\b",match.group(1))
    need(params==["64","32","64"],"fill_f32 ABI differs: "+str(params))
    ptx=C.create_string_buffer(payload)
    d=Driver();initial=d.current();a0=b=c=None
    try:
        a0=Instance(d,ptx);b=Instance(d,ptx)
        need(a0.ctx.value!=b.ctx.value,"contexts alias")
        need(a0.mod.value!=b.mod.value and a0.func.value!=b.func.value,"module/function handles alias")
        for i in range(12):
            x=i+0.25;y=-i-0.5
            a0.launch(x);b.launch(y)
            a0.check(x);b.check(y)
        a0.close();a0=None
        b.check(-11.5);b.launch(42.25)
        c=Instance(d,ptx)
        need(c.ctx.value!=b.ctx.value,"recreated context aliases live context")
        c.launch(-55.5);b.check(42.25)
        result={"status":"DRIVER_CONTEXT_ISOLATION_PASS",
                "contexts_simultaneously_live":2,"contexts_created":3,
                "alternating_launches":24,"total_launches":26,"elements_per_check":N,
                "surviving_context_after_peer_destruction":True,"recreated_context_verified":True,
                "native_ptx_sha256":sha(path),"entry":"fill_f32",
                "actual_dkr_registry_nonprimary_contexts_verified":False,
                "scope":"raw CUDA Driver module/function/allocation isolation; not Candle registry",
                "cuda_compilation":False}
    finally:
        for inst in (c,b,a0):
            if inst is not None:inst.close()
        d.call("cuCtxSetCurrent",P(initial))
    with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2));print("DKR_S684C_DRIVER=PASS")

if __name__=="__main__":main()
