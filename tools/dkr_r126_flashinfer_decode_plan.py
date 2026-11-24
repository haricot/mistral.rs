#!/usr/bin/env python3
"""R1.26/S9B4d1: source-level FlashInfer decode geometry and catalogue budget.

No CUDA compiler, GPU, or package admission. Counts the complete Cartesian
product exposed by the historical dispatch, not a qualified support matrix.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "mistralrs-paged-attn/src/cuda"
DTYPES = (("f16", 2), ("bf16", 2), ("f32", 4))

def source_contract():
    util = (CUDA / "flashinfer/utils.cuh").read_text()
    groups = util.split("#define DISPATCH_GQA_GROUP_SIZE", 1)[1].split("#define", 1)[0]
    groups = tuple(int(v) for v in re.findall(r"constexpr size_t GROUP_SIZE = (\d+);", groups))
    host = (CUDA / "flashinfer_decode.cu").read_text()
    heads = tuple(int(v) for v in re.findall(r"if \(head_size == (\d+)\)", host.split("} // namespace mistralrs_flashinfer", 1)[0]))
    decode = (CUDA / "flashinfer/attention/decode.cuh").read_text()
    dispatch = decode.split("cudaError_t BatchDecodeWithPagedKVCacheDispatched(", 1)[1].split("template <", 1)[0]
    for marker in ("std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL)",
                   "std::max(128U, bdx * bdy)", "num_threads / (bdx * bdy)",
                   "GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U",
                   "2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV)",
                   "tile_size_per_bdx * num_threads * sizeof(DTypeKV*)",
                   "params.partition_kv = true", "VariableLengthMergeStates("):
        if marker not in dispatch:
            raise ValueError("historical dispatch changed: " + marker)
    if groups != (1,2,3,4,5,6,7,8,16) or heads != (64,128,256,512):
        raise ValueError(f"historical dispatch matrix changed: groups={groups} heads={heads}")
    if "compute_capacity.first >= 8" not in util:
        raise ValueError("historical stage policy changed")
    runtime = (ROOT / "mistralrs-kernel-runtime/src/universal.rs").read_text()
    cap = re.search(r"pub const MAX_DISCOVERED_PACKAGES: usize = (\d+);", runtime)
    if cap is None:
        raise ValueError("cannot establish runtime catalogue bound")
    logical = re.search(r"pub const MAX_CATALOG_PROFILES: usize = (\d+);", runtime)
    shard = re.search(r"pub const MAX_PROFILES_PER_SHARD: usize = (\d+);", runtime)
    if logical is None or shard is None:
        raise ValueError("cannot establish logical catalogue/shard bounds")
    return groups, heads, int(cap.group(1)), int(logical.group(1)), int(shard.group(1))

def geometry(cc, cache_width, head, group):
    stages = 2 if cc >= 80 else 1
    vec = max(16 // cache_width, head // 32)
    x = head // vec
    y = group
    intended_threads = max(128, x*y)
    z = intended_threads // (x*y)
    tile = (2 if cache_width == 1 else 4) if group == 1 else 1
    shared = (2*stages*tile*y*z*head*cache_width
              + max(tile*intended_threads*8, 2*y*z*4))
    return dict(block=[x,y,z], actual_threads=x*y*z, intended_threads=intended_threads,
                vector_elements=vec, tile=tile, stages=stages,
                dynamic_shared_bytes=shared)

def report(cc, base_profiles, shared_limit):
    groups, heads, limit, logical_limit, shard_limit = source_contract()
    rows = []
    for (dt, width), cache, head, group in itertools.product(DTYPES, ("plain","fp8"), heads, groups):
        g = geometry(cc, 1 if cache == "fp8" else width, head, group)
        rows.append(dict(dtype=dt,cache=cache,head=head,group=group,**g,
                         exceeds_requested_shared_limit=g["dynamic_shared_bytes"]>shared_limit,
                         partial_warp=g["actual_threads"]%32!=0))
    # Two optional features, each on/off, and two output contracts:
    # direct output versus typed partial vectors plus base-2 LSE.
    per_geometry = 4*2
    decode_profiles = len(rows)*per_geometry
    merge_profiles = len(DTYPES)*len(heads)
    over_shared = sum(r["exceeds_requested_shared_limit"] for r in rows)
    partial_warp = sum(r["partial_warp"] for r in rows)
    return dict(schema="dkr.r126.flashinfer.decode.source-plan.v1",target_cc=cc,
                base_profiles=base_profiles,catalogue_limit=limit,
                logical_profile_limit=logical_limit, profiles_per_shard_limit=shard_limit,
                prospective_logical_profiles_fit=base_profiles+decode_profiles+merge_profiles<=logical_limit,
                geometry_count=len(rows),profiles_per_geometry=per_geometry,
                decode_profiles_full_cartesian=decode_profiles,merge_profiles=merge_profiles,
                prospective_profiles=base_profiles+decode_profiles+merge_profiles,
                merge_only_profiles=base_profiles+merge_profiles,
                shared_limit_bytes=shared_limit,geometries_over_shared_limit=over_shared,
                geometries_partial_warp=partial_warp,
                catalogue_full_cartesian_fits=base_profiles+decode_profiles+merge_profiles<=limit,
                geometries=rows,compilation=False,gpu_qualified=False,closure_qualified=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,default=61)
    p.add_argument("--base-profiles",type=int,default=411)
    p.add_argument("--shared-limit-kib",type=int,default=48)
    p.add_argument("--json-out",type=Path,required=True)
    a=p.parse_args()
    if not 50<=a.cc<=120 or a.base_profiles<0 or a.shared_limit_kib<=0:
        p.error("invalid target, count, or shared-memory budget")
    result=report(a.cc,a.base_profiles,a.shared_limit_kib*1024)
    if a.cc==61 and a.shared_limit_kib==48:
        assert result["geometry_count"]==216
        assert result["geometries_over_shared_limit"]==2
        assert result["geometries_partial_warp"]==39
    assert result["decode_profiles_full_cartesian"]==1728
    assert result["merge_profiles"]==12
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    a.json_out.write_text(json.dumps(result,indent=2)+"\n")
    print(f"DKR_R126_S9B4D_GEOMETRY=COMPLETE cc={a.cc} geometries={result['geometry_count']} "
          f"over_shared={result['geometries_over_shared_limit']} partial_warp={result['geometries_partial_warp']}")
    print(f"DKR_R126_S9B4D_BUDGET base={a.base_profiles} merge={result['merge_profiles']} "
          f"decode_cartesian={result['decode_profiles_full_cartesian']} "
          f"total={result['prospective_profiles']} limit={result['catalogue_limit']} "
          f"fits={str(result['catalogue_full_cartesian_fits']).lower()}")
    print(f"DKR_R126_S9B4D2_LOGICAL_LIMIT profiles={result['prospective_profiles']} "
          f"limit={result['logical_profile_limit']} fits={str(result['prospective_logical_profiles_fit']).lower()} "
          f"shard_limit={result['profiles_per_shard_limit']} physical_fit=unverified")
    print("DKR_R126_S9B4D_SOURCE_PLAN=COMPLETE compilation=false closure_qualified=false "
          f"report={a.json_out}")

if __name__=="__main__":
    main()
