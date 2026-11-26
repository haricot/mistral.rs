//! Large vocabulary selection through existing workspaces and prevalidated
//! typed launch plans. The cooperative algorithm can be rejected only before
//! a launch; an actual CUDA error propagates without an algorithm retry.
use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{
    cuda_backend::cudarc::driver::{CudaStream, PushKernelArg},
    DType, Device, Result,
};
use mistralrs_kernel_runtime::{
    model_scope::OperationRequirement,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{cooperative_device_limits, PrevalidatedCooperativeLaunchPlan, PrevalidatedLaunchPlan},
};
use std::cell::RefCell;

const CHUNK: i32 = 2048;
const STAGE2_CEILING: u32 = 47*1024;
const COOP_CEILING: u32 = 96*1024;
const COOP_FIXED_BYTES: u32 = 1184;
const COOP_HEADROOM: u32 = 2048;
const COOP_THREADS: u32 = 1024;
const COOP_MAX_GROUP_CTAS: u32 = 256;
/// Must match the C++ sizeof/offsetof static assertions in radix_topk.cuh.
pub(crate) const RADIX_STATE_WORDS_PER_ROW: usize = 1284;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DkrLargeTopKMode { SingleSplit, SinglePacked, BatchedPacked, Ranked }
impl DkrLargeTopKMode {
    fn batched(self)->bool { matches!(self,Self::BatchedPacked|Self::Ranked) }
    fn softmax(self)->bool { self!=Self::Ranked }
    fn stage1(self)->&'static str {
        match self {Self::SingleSplit|Self::SinglePacked=>"single_softmax",
            Self::BatchedPacked=>"batch_softmax",Self::Ranked=>"batch_ranked"}
    }
    fn stage2(self)->&'static str {
        match self {Self::SingleSplit=>"single_split",Self::SinglePacked=>"single_packed",
            Self::BatchedPacked=>"batch_packed",Self::Ranked=>"batch_ranked"}
    }
}
fn dtype(dt:DType)->Result<ElementType>{
    match dt {DType::F32=>Ok(ElementType::F32),DType::F16=>Ok(ElementType::F16),
        DType::BF16=>Ok(ElementType::Bf16),_=>candle_core::bail!("large top-k dtype")}
}
fn tag(dt:ElementType)->&'static str {
    match dt {ElementType::F32=>"f32",ElementType::F16=>"f16",ElementType::Bf16=>"bf16",
        _=>unreachable!("checked top-k dtype")}
}
fn req(op:String,entry:String)->OperationRequirement{
    OperationRequirement{operation:op,implementations:Some(vec![format!("cuda.dkr_{entry}.v1")])}
}
fn stage1(mode:DkrLargeTopKMode,dt:ElementType)->OperationRequirement{
    req(format!("sampling.large_topk.stage1.{}.v1",mode.stage1()),
        format!("large_topk_stage1_{}_{}",mode.stage1(),tag(dt)))
}
fn stage2(mode:DkrLargeTopKMode)->OperationRequirement{
    req(format!("sampling.large_topk.stage2.{}.v1",mode.stage2()),format!("large_topk_stage2_{}",mode.stage2()))
}
fn cub1(dt:ElementType)->OperationRequirement{
    req("sampling.ranked_radix.stage1.v1".into(),format!("ranked_radix_stage1_{}",tag(dt)))
}
fn cub2()->OperationRequirement{
    req("sampling.ranked_radix.stage2.v1".into(),"ranked_radix_stage2".into())
}
fn coop(dt:ElementType,vec:u32)->OperationRequirement{
    req(format!("cuda.cooperative.ranked_radix.vec{vec}.v1"),format!("ranked_cooperative_{}_v{vec}",tag(dt)))
}
fn sort_pack()->OperationRequirement{
    req("sampling.ranked_radix.sort_pack.v1".into(),"ranked_cooperative_sort".into())
}

/// Conservative capability producer. Include all ranked algorithms and
/// possible vector widths, since actual alignment/occupancy is not a model
/// configuration fact. Union with target, draft and other request paths.
pub fn dkr_large_topk_requirements(dt:DType,mode:DkrLargeTopKMode)->Result<Vec<OperationRequirement>>{
    let dt=dtype(dt)?;
    if !mode.batched() && dt!=ElementType::F32 {candle_core::bail!("single large top-k requires F32");}
    let mut out=vec![stage1(mode,dt),stage2(mode)];
    if mode==DkrLargeTopKMode::Ranked{
        out.extend([cub1(dt),cub2(),sort_pack()]);
        for v in [1,2,4,8]{
            if v as usize<=16/dt.size_bytes(){out.push(coop(dt,v));}
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry { rows:i32, cols:i32, k:i32, blocks:i32 }
impl Geometry{
    fn new(rows:i32,cols:i32,k:i32,chunk:i32,blocks:i32)->Result<Self>{
        if rows<=0 || rows>65535 || cols<=0 || k<=0 || k>cols || k>128 || chunk!=CHUNK ||
            blocks<=0 || i64::from(blocks)!=(i64::from(cols)+i64::from(CHUNK)-1)/i64::from(CHUNK) ||
            i64::from(blocks)*i64::from(k)>i64::from(STAGE2_CEILING) {
            candle_core::bail!("invalid large top-k launch geometry");
        }
        let g=Self{rows,cols,k,blocks};
        for width in [g.cols(),g.candidates(),g.blocks(),g.width(true)]{
            if g.rows().checked_mul(width).is_none(){candle_core::bail!("large top-k span overflow");}
        }
        Ok(g)
    }
    fn rows(self)->usize{self.rows as usize}
    fn cols(self)->usize{self.cols as usize}
    fn k(self)->usize{self.k as usize}
    fn blocks(self)->usize{self.blocks as usize}
    fn candidates(self)->usize{self.blocks()*self.k()}
    fn width(self,soft:bool)->usize{2*self.k()+2*usize::from(soft)}
    fn grid1(self)->[u32;3]{[self.blocks as u32,self.rows as u32,1]}
    fn grid2(self)->[u32;3]{[self.rows as u32,1,1]}
}
#[derive(Clone,Copy,Eq,PartialEq)]
struct Key{g:Geometry,mode:DkrLargeTopKMode,dt:ElementType,vec:u32}
struct Plan{first:First,second:PrevalidatedLaunchPlan}
enum First{
    Iterative(PrevalidatedLaunchPlan),
    Cub(PrevalidatedLaunchPlan),
    Cooperative{plan:PrevalidatedCooperativeLaunchPlan,chunk:u32,ctas:u32},
}
thread_local!{
    static PLANS:RefCell<LaunchPlanCache<Key,Plan>>=RefCell::new(LaunchPlanCache::new());
}

fn tensor(dt:ElementType,n:usize)->Argument{
    Argument::Tensor{dtype:dt,shape:vec![n],contiguous:true,start_offset:0,representation:None}
}
fn scalar(dt:ElementType)->Argument{Argument::Scalar{dtype:dt}}
fn request(r:&OperationRequirement,args:Vec<Argument>,grid:[u32;3],ceiling:u32)->LaunchRequest{
    LaunchRequest{operation:r.operation.clone(),cc:0,args,grid,max_dynamic_shared_mem_bytes:ceiling}
}
fn err(e:impl std::fmt::Display)->candle_core::Error{candle_core::Error::msg(e.to_string())}
fn ordinary(device:&Device,r:OperationRequirement,args:Vec<Argument>,grid:[u32;3],
            block:[u32;3],ceiling:u32,shared:u32)->Result<PrevalidatedLaunchPlan>{
    let mut request=request(&r,args,grid,ceiling);
    let p=kernel_cache()?.prepare_for_device(&mut request,device).map_err(err)?;
    if p.manifest.block!=block || p.manifest.shared_mem_bytes!=ceiling ||
        r.implementations.as_ref().is_none_or(|v|v.len()!=1 || v[0]!=p.manifest.implementation_id){
        candle_core::bail!("large top-k selected profile differs from exact launch contract");
    }
    // Zero-dynamic CUB stages have no shape-derived shared allocation.
    if ceiling==0 {p.prevalidate_launch(&request,device).map_err(err)}
    else {unsafe{p.prevalidate_launch_with_default_shared(&request,device,shared)}.map_err(err)}
}
fn ceil_div(a:u32,b:u32)->u32{a.div_ceil(b)}
fn gcd(mut a:u32,mut b:u32)->u32{while b!=0{let r=a%b;a=b;b=r;}a}
fn vector_width(cols:u32,ptr:u64,element_bytes:u32)->u32{
    let max=16/element_bytes;
    // At most 16-byte vector alignment is required. Narrowed contiguous
    // inputs can have a smaller alignment than their underlying allocation.
    let align=u32::try_from((ptr%16)/u64::from(element_bytes)).unwrap_or(0);
    gcd(gcd(max,cols),align)
}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
struct CoopGeometry{chunk:u32,ctas:u32,shared:u32}
fn coop_geometry(g:Geometry,dt:ElementType,vec:u32,max_shared:u32)->Option<CoopGeometry>{
    if !(8..=32).contains(&g.k)||g.cols<=g.k{return None;}
    let element=u32::try_from(dt.size_bytes()).ok()?;
    if vec==0 || !vec.is_power_of_two() || vec>16/element || g.cols as u32%vec!=0{return None;}
    let available=max_shared.min(COOP_CEILING).checked_sub(COOP_FIXED_BYTES+COOP_HEADROOM)?;
    let max_chunk=(available/element)/vec*vec;
    if max_chunk<COOP_THREADS*vec{return None;}
    let ctas=ceil_div(g.cols as u32,max_chunk);
    if ctas==0||ctas>COOP_MAX_GROUP_CTAS{return None;}
    let chunk=ceil_div(ceil_div(g.cols as u32,ctas),vec)*vec;
    if chunk>max_chunk{return None;}
    Some(CoopGeometry{chunk,ctas,shared:COOP_FIXED_BYTES+chunk*element})
}
fn iterative_plan(key:Key,device:&Device)->Result<Plan>{
    let g=key.g;let mode=key.mode;let soft=mode.softmax();
    let data=||tensor(ElementType::F32,g.rows()*g.candidates());
    let ids=||tensor(ElementType::U32,g.rows()*g.candidates());
    let blocks=||tensor(ElementType::F32,g.rows()*g.blocks());
    let mut args=vec![tensor(key.dt,g.rows()*g.cols()),data(),ids()];
    if soft {args.extend([blocks(),blocks()]);}
    args.extend(std::iter::repeat_n(scalar(ElementType::I32),3));
    if soft {args.push(if mode.batched(){tensor(ElementType::F32,g.rows())}else{scalar(ElementType::F32)});}
    let first=ordinary(device,stage1(mode,key.dt),args,g.grid1(),[256,1,1],CHUNK as u32,CHUNK as u32)?;
    let mut args=vec![data(),ids()];
    if soft{args.extend([blocks(),blocks()]);}
    if mode==DkrLargeTopKMode::SingleSplit{
        args.extend([tensor(ElementType::F32,g.k()),tensor(ElementType::U32,g.k()),tensor(ElementType::F32,2)]);
    }else{args.push(tensor(ElementType::F32,g.rows()*g.width(soft)));}
    args.extend(std::iter::repeat_n(scalar(ElementType::I32),2));
    let second=ordinary(device,stage2(mode),args,g.grid2(),[256,1,1],STAGE2_CEILING,
        u32::try_from(g.candidates()).map_err(err)?)?;
    Ok(Plan{first:First::Iterative(first),second})
}
fn cub_plan(key:Key,device:&Device)->Result<Plan>{
    let g=key.g;
    let data=||tensor(ElementType::F32,g.rows()*g.candidates());
    let ids=||tensor(ElementType::U32,g.rows()*g.candidates());
    let mut args=vec![tensor(key.dt,g.rows()*g.cols()),data(),ids()];
    args.extend(std::iter::repeat_n(scalar(ElementType::I32),2));
    let first=ordinary(device,cub1(key.dt),args,g.grid1(),[256,1,1],0,0)?;
    let mut args=vec![data(),ids(),tensor(ElementType::F32,g.rows()*g.width(false))];
    args.extend(std::iter::repeat_n(scalar(ElementType::I32),2));
    let second=ordinary(device,cub2(),args,g.grid2(),[256,1,1],0,0)?;
    Ok(Plan{first:First::Cub(first),second})
}
fn cooperative_plan(key:Key,device:&Device)->Result<Option<Plan>>{
    let g=key.g;
    if !(8..=32).contains(&g.k)||g.cols<=g.k{return Ok(None);}
    let Some(limits)=cooperative_device_limits(device).map_err(err)? else{return Ok(None);};
    let Some(geo)=coop_geometry(g,key.dt,key.vec,limits.max_shared_bytes) else{return Ok(None);};
    let r=coop(key.dt,key.vec);
    let mut args=vec![tensor(key.dt,g.rows()*g.cols()),tensor(ElementType::F32,g.rows()*g.width(false))];
    args.extend(std::iter::repeat_n(scalar(ElementType::I32),3));
    args.push(tensor(ElementType::U32,g.rows()*RADIX_STATE_WORDS_PER_ROW));
    args.extend(std::iter::repeat_n(scalar(ElementType::U32),2));
    let mut request=request(&r,args,[1,1,1],COOP_CEILING);
    let p=kernel_cache()?.prepare_for_device(&mut request,device).map_err(err)?;
    if p.manifest.block!=[COOP_THREADS,1,1]||p.manifest.shared_mem_bytes!=COOP_CEILING ||
        r.implementations.as_ref().is_none_or(|v|v.len()!=1 || v[0]!=p.manifest.implementation_id){
        candle_core::bail!("cooperative radix selected profile differs from exact contract");
    }
    let Some(plan)= (unsafe{p.prevalidate_cooperative_grouped(&request,device,geo.shared,g.rows as u32,geo.ctas)}.map_err(err)?)
        else{return Ok(None);};
    let second=ordinary(device,sort_pack(),vec![tensor(ElementType::F32,g.rows()*g.width(false)),scalar(ElementType::I32)],
        g.grid2(),[32,1,1],0,0)?;
    Ok(Some(Plan{first:First::Cooperative{plan,chunk:geo.chunk,ctas:geo.ctas},second}))
}
fn prepare(key:Key,device:&Device)->Result<Plan>{
    if key.mode==DkrLargeTopKMode::Ranked{
        if let Some(plan)=cooperative_plan(key,device)?{return Ok(plan);}
        if (8..=16).contains(&key.g.k)&&key.g.candidates()<=CHUNK as usize{
            return cub_plan(key,device);
        }
    }
    iterative_plan(key,device)
}

/// Raw CUDA addresses are bound only while the existing caller holds all
/// storage/stream guards. Output and workspace buffers are already allocated
/// by the public ops paths. No new tensor allocation or copy is introduced.
#[derive(Clone,Copy)]
pub(crate) struct WorkspacePtrs{
    pub input:u64,pub block_values:u64,pub block_indices:u64,
}
#[derive(Clone,Copy)]
pub(crate) enum SoftmaxPtrs{
    None, Single{maxes:u64,sums:u64,temperature:f32},
    Batched{maxes:u64,sums:u64,temperatures:u64},
}
#[derive(Clone,Copy)]
pub(crate) enum OutputPtrs{
    Split{values:u64,indices:u64,info:u64},Packed(u64),
    Ranked{packed:u64,state:u64},
}
impl OutputPtrs{
    fn packed(self)->u64{match self{Self::Packed(p)|Self::Ranked{packed:p,..}=>p,_=>unreachable!("packed output")}}
}
impl SoftmaxPtrs{
    fn stats(self)->(u64,u64){match self{Self::Single{maxes,sums,..}|Self::Batched{maxes,sums,..}=>(maxes,sums),Self::None=>unreachable!("softmax stats")}}
}
fn validate_ptr(ptr:u64,align:u64)->Result<()>{
    if ptr==0||ptr%align!=0{candle_core::bail!("large top-k null/misaligned device pointer");}
    Ok(())
}
/// Safety: all addresses denote the validated contiguous spans of geometry,
/// distinct non-aliasing writable workspaces, retained on the same stream.
/// The caller retains the original CUDA buffer guards until both stages have
/// been submitted. Reused radix state must have been zero-initialized, and
/// may not be shared by concurrent launches.
pub(crate) unsafe fn launch(
    device:&Device,stream:&CudaStream,dt:DType,rows:i32,cols:i32,k:i32,chunk:i32,blocks:i32,
    mode:DkrLargeTopKMode,work:WorkspacePtrs,soft:SoftmaxPtrs,out:OutputPtrs,
)->Result<()>{
    let dt=dtype(dt)?;let g=Geometry::new(rows,cols,k,chunk,blocks)?;
    if !mode.batched()&&(rows!=1||dt!=ElementType::F32){candle_core::bail!("single top-k geometry/dtype");}
    if mode!=DkrLargeTopKMode::SingleSplit&&g.cols()>((1usize<<24)+1){candle_core::bail!("packed index precision");}
    validate_ptr(work.input,dt.size_bytes() as u64)?;
    validate_ptr(work.block_values,4)?;validate_ptr(work.block_indices,4)?;
    match (mode,soft,out){
        (DkrLargeTopKMode::SingleSplit,SoftmaxPtrs::Single{maxes,sums,..},OutputPtrs::Split{values,indices,info})=>{
            for p in [maxes,sums,values,indices,info]{validate_ptr(p,4)?;}
        }
        (DkrLargeTopKMode::SinglePacked,SoftmaxPtrs::Single{maxes,sums,..},OutputPtrs::Packed(p))=>{
            for p in [maxes,sums,p]{validate_ptr(p,4)?;}
        }
        (DkrLargeTopKMode::BatchedPacked,SoftmaxPtrs::Batched{maxes,sums,temperatures},OutputPtrs::Packed(p))=>{
            for p in [maxes,sums,temperatures,p]{validate_ptr(p,4)?;}
        }
        (DkrLargeTopKMode::Ranked,SoftmaxPtrs::None,OutputPtrs::Ranked{packed,state})=>{
            validate_ptr(packed,4)?;validate_ptr(state,16)?;
        }
        _=>candle_core::bail!("large top-k binding mode mismatch"),
    }
    let vec=if mode==DkrLargeTopKMode::Ranked{vector_width(cols as u32,work.input,dt.size_bytes() as u32)}else{0};
    let key=Key{g,mode,dt,vec};
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,device,||prepare(key,device),|p|{
        // Every stage has been admitted before any CUDA work is submitted.
        match &p.first{
            First::Iterative(first)=>{
                let (maxes,sums)=if mode.softmax(){soft.stats()}else{(0,0)};
                unsafe{first.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&work.input).arg(&work.block_values).arg(&work.block_indices);
                    if mode.softmax(){b.arg(&maxes).arg(&sums);}
                    b.arg(&cols).arg(&k).arg(&chunk);
                    match &soft{SoftmaxPtrs::Single{temperature,..}=>{b.arg(temperature);},
                        SoftmaxPtrs::Batched{temperatures,..}=>{b.arg(temperatures);},SoftmaxPtrs::None=>{}}
                    b.launch(cfg)?;Ok(())
                })}.map_err(err)?;
                unsafe{p.second.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&work.block_values).arg(&work.block_indices);
                    if mode.softmax(){b.arg(&maxes).arg(&sums);}
                    match &out{OutputPtrs::Split{values,indices,info}=>{b.arg(values).arg(indices).arg(info);},
                        OutputPtrs::Packed(p)|OutputPtrs::Ranked{packed:p,..}=>{b.arg(p);}}
                    b.arg(&blocks).arg(&k);b.launch(cfg)?;Ok(())
                })}.map_err(err)
            }
            First::Cub(first)=>{
                let packed=out.packed();
                unsafe{first.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&work.input).arg(&work.block_values).arg(&work.block_indices).arg(&cols).arg(&k);
                    b.launch(cfg)?;Ok(())
                })}.map_err(err)?;
                unsafe{p.second.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&work.block_values).arg(&work.block_indices).arg(&packed).arg(&blocks).arg(&k);
                    b.launch(cfg)?;Ok(())
                })}.map_err(err)
            }
            First::Cooperative{plan,chunk,ctas}=>{
                let OutputPtrs::Ranked{packed,state}=out else{unreachable!("cooperative bindings");};
                unsafe{plan.launch_cooperative(stream,|b|{
                    b.arg(&work.input).arg(&packed).arg(&rows).arg(&cols).arg(&k).arg(&state).arg(chunk).arg(ctas);
                })}.map_err(err)?;
                unsafe{p.second.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&packed).arg(&k);b.launch(cfg)?;Ok(())
                })}.map_err(err)
            }
        }
    }))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_large_topk_geometry_and_requirements()->Result<()>{
        let g=Geometry::new(5,131072,16,2048,64)?;
        assert_eq!(vector_width(131072,0x1000,4),4);
        assert_eq!(vector_width(131072,0x1004,4),1);
        assert_eq!(vector_width(131072,0x1008,2),4);
        let c=coop_geometry(g,ElementType::F32,4,49152).unwrap();
        assert_eq!(c.ctas,12);
        assert!(c.chunk%4==0&&c.shared+COOP_HEADROOM<=49152);
        assert!(coop_geometry(g,ElementType::F32,4,4096).is_none());
        assert!(Geometry::new(5,131072,16,2048,63).is_err());
        assert_eq!(dkr_large_topk_requirements(DType::F32,DkrLargeTopKMode::Ranked)?.len(),8);
        assert_eq!(dkr_large_topk_requirements(DType::BF16,DkrLargeTopKMode::Ranked)?.len(),9);
        Ok(())
    }
}
