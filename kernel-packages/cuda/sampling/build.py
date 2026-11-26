#!/usr/bin/env python3
"""Typed sampling/DFlash/rejection producer. No compiler invocation here."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import profile, tensor, scalar, stage, self_test

IMAGES=(1,2,3)
def t(name,dtype,access):
    return tensor(name,dtype,1,access)
def s(name,dtype):
    return scalar(name,dtype)

PROFILES={
    1: (
        ("dkr_top1_stage1_single_f32", "sampling.top1.stage1.single.v1", 256, 0, (
            t("input", "f32", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage1_batch_f32", "sampling.top1.stage1.batch.v1", 256, 0, (
            t("input", "f32", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage1_single_f16", "sampling.top1.stage1.single.v1", 256, 0, (
            t("input", "f16", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage1_batch_f16", "sampling.top1.stage1.batch.v1", 256, 0, (
            t("input", "f16", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage1_single_bf16", "sampling.top1.stage1.single.v1", 256, 0, (
            t("input", "bf16", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage1_batch_bf16", "sampling.top1.stage1.batch.v1", 256, 0, (
            t("input", "bf16", "read"), t("block_values", "f32", "write"), t("block_indices", "u32", "write"), s("ncols", "i32"), s("chunk_size", "i32"),
        )),
        ("dkr_top1_stage2_single_tokens", "sampling.top1.stage2.single.tokens.v1", 256, 0, (
            t("block_values", "f32", "read"), t("block_indices", "u32", "read"), t("token_ids_out", "u32", "write"), s("nblocks", "i32"),
        )),
        ("dkr_top1_stage2_single_both", "sampling.top1.stage2.single.both.v1", 256, 0, (
            t("block_values", "f32", "read"), t("block_indices", "u32", "read"), t("packed_out", "f32", "write"), t("token_ids_out", "u32", "write"), s("nblocks", "i32"),
        )),
        ("dkr_top1_stage2_batch_tokens", "sampling.top1.stage2.batch.tokens.v1", 256, 0, (
            t("block_values", "f32", "read"), t("block_indices", "u32", "read"), t("token_ids_out", "u32", "write"), s("nblocks", "i32"),
        )),
        ("dkr_top1_stage2_batch_both", "sampling.top1.stage2.batch.both.v1", 256, 0, (
            t("block_values", "f32", "read"), t("block_indices", "u32", "read"), t("packed_out", "f32", "write"), t("token_ids_out", "u32", "write"), s("nblocks", "i32"),
        )),
        ("dkr_top1_stage2_batch_packed", "sampling.top1.stage2.batch.packed.v1", 256, 0, (
            t("block_values", "f32", "read"), t("block_indices", "u32", "read"), t("packed_out", "f32", "write"), s("nblocks", "i32"),
        )),
        ("dkr_categorical_stage1", "sampling.categorical.stage1.v1", 256, 0, (
            t("input", "f32", "read"), t("block_values", "f32", "write"), t("block_sums", "f32", "write"), s("ncols", "i32"), s("chunk_size", "i32"), t("inv_temperatures", "f32", "read"),
        )),
        ("dkr_categorical_stage2", "sampling.categorical.stage2.v1", 256, 16384, (
            t("input", "f32", "read"), t("inv_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("block_values", "f32", "read"), t("block_sums", "f32", "read"), t("packed_out", "f32", "write"), s("ncols", "i32"), s("chunk_size", "i32"), s("nblocks", "i32"),
        )),
        ("dkr_sample_ranked_topk", "sampling.ranked.sample.v1", 128, 0, (
            t("packed", "f32", "read"), t("params", "f32", "read"), t("tokens", "u32", "write"), s("nrows", "i32"), s("packed_k", "i32"),
        )),
    ),
    2: (
        ("dkr_dflash_greedy_f32_f32_f32", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_f32_f32_bf16", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_f32_bf16_f32", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_f32_bf16_bf16", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_bf16_f32_f32", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_bf16_f32_bf16", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_bf16_bf16_f32", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_greedy_bf16_bf16_bf16", "sampling.dflash.greedy.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("selected", "u32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_f32_f32_f32", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_f32_f32_bf16", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_f32_bf16_f32", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_f32_bf16_bf16", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "f32", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_bf16_f32_f32", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_bf16_f32_bf16", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "f32", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_bf16_bf16_f32", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "f32", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
        ("dkr_dflash_sample_bf16_bf16_bf16", "sampling.dflash.sample.v1", 256, 0, (
            t("packed_topk", "f32", "read"), t("projected_hidden", "bf16", "read"), t("predecessor_codebook", "bf16", "read"), t("successor_codebook", "bf16", "read"), t("anchors", "u32", "read"), t("inverse_temperatures", "f32", "read"), t("uniforms", "f32", "read"), t("selected", "u32", "write"), t("candidate_ids", "u32", "write"), t("candidate_probs", "f32", "write"), s("batch", "i32"), s("positions", "i32"), s("rank", "i32"), s("vocab", "i32"), s("k", "i32"), s("packed_width", "i32"),
        )),
    ),
    3: (
        ("dkr_sparse_rejection_categorical_f32", "sampling.rejection.categorical.v1", 256, 0, (
            t("target_logits", "f32", "read"), t("draft_tokens", "u32", "read"), t("q_token_ids", "u32", "read"), t("q_probs", "f32", "read"), t("inverse_temperatures", "f32", "read"), t("target_top_k", "u32", "read"), t("top_p", "f32", "read"), t("min_p", "f32", "read"), t("accept_uniforms", "f32", "read"), t("sample_uniforms", "f32", "read"), t("outcomes", "u32", "write"), s("batch", "i32"), s("drafts", "i32"), s("vocab", "i32"), s("q_width", "i32"),
        )),
        ("dkr_sparse_rejection_topk_f32", "sampling.rejection.topk.v1", 1, 0, (
            t("packed_target", "f32", "read"), t("draft_tokens", "u32", "read"), t("q_token_ids", "u32", "read"), t("q_probs", "f32", "read"), t("inverse_temperatures", "f32", "read"), t("target_top_k", "u32", "read"), t("top_p", "f32", "read"), t("min_p", "f32", "read"), t("accept_uniforms", "f32", "read"), t("sample_uniforms", "f32", "read"), t("outcomes", "u32", "write"), s("batch", "i32"), s("drafts", "i32"), s("vocab", "i32"), s("q_width", "i32"), s("packed_k", "i32"),
        )),
    ),
}

def plan(cc,image):
    if not 50<=cc<=255 or image not in IMAGES:
        raise ValueError("invalid sampling target/image")
    rows=[]
    for entry,op,block,shared,args in PROFILES[image]:
        row=profile(cc,op,f"cuda.{entry}.v1",entry,[block,1,1],list(args),f"sampling_{image}.ptx")
        row["shared_mem_bytes"]=shared
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--image",type=int,choices=IMAGES)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--json-out",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--ptx-file",type=Path)
    a=p.parse_args()
    if not 50<=a.cc<=255:
        p.error("invalid compute capability")
    ids=IMAGES if a.image is None else (a.image,)
    groups=[plan(a.cc,i) for i in ids]
    count=sum(map(len,groups))
    max_args=max(len(r["arguments"]) for g in groups for r in g)
    if a.self_test:
        for g in groups:
            self_test(a.cc,g)
        print(f"DKR_R126_S9C2E4_PRODUCER_SELF_TEST=PASS cc={a.cc} images={len(ids)} profiles={count} max_args={max_args} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=[r for g in groups for r in g],
            physical_manifests=len(ids),compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2E4_PRODUCER_PLAN=COMPLETE cc={a.cc} images={len(ids)} profiles={count} compilation=false")
    elif a.image is not None and a.output and a.ptx_file:
        stage(a.output/f"sm{a.cc}"/"sampling"/f"image-{a.image:02}",a.cc,groups[0],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out or --image --output --ptx-file")

if __name__=="__main__":
    main()
