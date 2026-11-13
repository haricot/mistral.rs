#!/usr/bin/env python3
"""DKR-R1.21/S8 independent CPU oracle for direct routed FP8/FP4 GEMV/GEMM.

Checks expert-indirection, token-vs-route activation addressing, E4M3,
E8M0 OCP, E2M1 nibble ordering and block/group scales. It is *not* a
CUDA/PTX compilation or a GPU performance/precision qualification.
"""
import itertools
import math
import random
import sys

EXPERTS = 4
N = 5


def fp8(bits):
    negative = bool(bits & 0x80)
    exp = (bits >> 3) & 15
    mantissa = bits & 7
    if exp == 15 and mantissa == 7:
        return float("nan")
    magnitude = (mantissa * 2**-9 if exp == 0
                 else math.ldexp(1 + mantissa / 8, exp - 7))
    return -magnitude if negative else magnitude


def fp4(code):
    exp = (code >> 1) & 3
    mantissa = code & 1
    magnitude = (mantissa / 2 if exp == 0
                 else math.ldexp(1 + mantissa / 2, exp - 1))
    return -magnitude if code & 8 else magnitude


def e8m0(exp):
    if exp == 255:
        return float("nan")
    return math.ldexp(1.0, exp - 127)


def setup(codec, k, rng):
    weights = []
    if codec in ("fp8", "mxfp8"):
        weights = [rng.choice((0x38, 0x30, 0xb0, 0x40, 0x28))
                   for _ in range(EXPERTS * N * k)]
    else:
        weights = [rng.choice((0x24, 0x26, 0x49, 0x82))
                   for _ in range(EXPERTS * N * k // 2)]
    by, bx = 2, 16
    sr = (N + by - 1)//by if codec == "fp8" else N
    sc = (k + bx - 1)//bx if codec == "fp8" else (
        k//16 if codec == "nvfp4" else k//32
    )
    if codec == "fp8":
        scales = [rng.choice((0.25, 0.5, 1.0))
                  for _ in range(EXPERTS * sr * sc)]
    elif codec == "nvfp4":
        scales = [rng.choice((0x30, 0x38, 0x40))
                  for _ in range(EXPERTS * sr * sc)]
    else:
        scales = [rng.choice((125, 127, 129))
                  for _ in range(EXPERTS * sr * sc)]
    globals_ = [0.25 + e*0.25 + n*0.125
                for e in range(EXPERTS) for n in range(N)]
    return weights, scales, globals_, sr, sc, by, bx


def decode(codec, w, scales, globals_, expert, col, idx, k, sr, sc, by, bx):
    row = expert*N + col
    if codec == "fp8":
        scalar = scales[(expert*sr + col//by)*sc + idx//bx]
        return fp8(w[row*k + idx]) * scalar
    if codec == "mxfp8":
        return fp8(w[row*k + idx]) * e8m0(scales[row*sc + idx//32])
    pair = w[row*(k//2) + idx//2]
    code = pair >> 4 if idx & 1 else pair & 15
    if codec == "mxfp4":
        return fp4(code) * e8m0(scales[row*sc + idx//32])
    return fp4(code) * fp8(scales[row*sc + idx//16]) * globals_[row]


def reference(codec, w, scales, globals_, ids, x, tokens, topk, input_routes,
              k, sr, sc, by, bx):
    dense = [
        [decode(codec,w,scales,globals_,e,col,j,k,sr,sc,by,bx)
         for j in range(k)]
        for e in range(EXPERTS) for col in range(N)
    ]
    result = []
    for t in range(tokens):
        for r in range(topk):
            expert = ids[t*topk+r]
            src = (t*input_routes+(r if input_routes>1 else 0))*k
            for col in range(N):
                result.append(math.fsum(
                    x[src+j]*dense[expert*N+col][j] for j in range(k)
                ))
    return result


def direct(codec,w,scales,globals_,ids,x,tokens,topk,input_routes,k,
           sr,sc,by,bx):
    result=[]
    for route in range(tokens*topk):
        token, top = divmod(route,topk)
        expert=ids[route]
        src=(token*input_routes+(top if input_routes>1 else 0))*k
        for col in range(N):
            # The CUDA implementation accumulates these terms directly.
            terms=(x[src+j]*decode(
                codec,w,scales,globals_,expert,col,j,k,sr,sc,by,bx
            ) for j in range(k))
            result.append(math.fsum(terms))
    return result


def main():
    rng=random.Random(6108121)
    ncases=0
    for codec,tokens,topk,k in itertools.product(
        ("fp8","mxfp8","mxfp4","nvfp4"),
        (1,2,17,33), (1,2,3), (32,64),
    ):
        w,scales,globals_,sr,sc,by,bx=setup(codec,k,rng)
        ids=[(t*3+r*7)%EXPERTS for t in range(tokens) for r in range(topk)]
        for input_routes in set((1,topk)):
            x=[rng.randrange(-9,10)/16
               for _ in range(tokens*input_routes*k)]
            expected=reference(codec,w,scales,globals_,ids,x,tokens,topk,
                               input_routes,k,sr,sc,by,bx)
            got=direct(codec,w,scales,globals_,ids,x,tokens,topk,
                       input_routes,k,sr,sc,by,bx)
            assert len(expected)==len(got)==tokens*topk*N
            for i,(a,b) in enumerate(zip(expected,got)):
                if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):
                    raise AssertionError(
                        f"{codec} tokens={tokens} topk={topk} R={input_routes} "
                        f"K={k}, index={i}: {a} vs {b}"
                    )
            ncases+=1
    assert math.isclose(fp4(0x4),2.0)
    assert math.isclose(fp4(0x2),1.0)
    assert math.isclose(e8m0(0),2**-127)
    assert math.isnan(e8m0(255))
    assert math.isnan(fp8(0x7f))
    assert math.isclose(fp8(0x38),1.0)
    print(
        f"DKR_R121_S8_CPU_PREFLIGHT=PASS cases={ncases} codecs=4 "
        "gemv_and_gemm=checked expert_order=checked "
        "fp8_fp4_group_scales=checked no_cuda_compilation"
    )


if __name__=="__main__":
    try:
        main()
    except (AssertionError,ValueError,OverflowError) as error:
        print(f"DKR_R121_S8_CPU_PREFLIGHT=FAIL {error}",file=sys.stderr)
        sys.exit(1)
