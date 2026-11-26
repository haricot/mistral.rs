//! Indirect bitonic Argsort. No legacy host FFI, no mutation through a read
//! pointer, and no numeric sentinel or F64/I64 narrowing.
use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda_backend::{cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice, PushKernelArg}, CudaStorageSlice},
    CpuStorage, CudaStorage, DType, Device, Layout, Result, Shape,
};
use mistralrs_kernel_runtime::{
    model_scope::OperationRequirement,
    universal::{Access, Argument, ElementType, LaunchRequest, Parameter, MAX_TENSOR_BYTES},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const BLOCK:u32=256;
fn err(e:impl std::fmt::Display)->candle_core::Error{candle_core::Error::msg(e.to_string())}
fn dtype(dt:DType)->Result<ElementType>{
    match dt{
        DType::F32=>Ok(ElementType::F32),DType::F16=>Ok(ElementType::F16),
        DType::BF16=>Ok(ElementType::Bf16),DType::F64=>Ok(ElementType::F64),
        DType::U8=>Ok(ElementType::U8),DType::U32=>Ok(ElementType::U32),
        DType::I64=>Ok(ElementType::I64),_=>candle_core::bail!("unsupported DKR argsort dtype"),
    }
}
fn tag(dt:ElementType)->&'static str{
    match dt{
        ElementType::F32=>"f32",ElementType::F16=>"f16",ElementType::Bf16=>"bf16",
        ElementType::F64=>"f64",ElementType::U8=>"u8",ElementType::U32=>"u32",
        ElementType::I64=>"i64",_=>unreachable!("checked argsort dtype"),
    }
}
fn requirement(operation:&str,entry:&str)->OperationRequirement{
    OperationRequirement{operation:format!("tensor.argsort.{operation}.v1"),
        implementations:Some(vec![format!("cuda.dkr_argsort_{entry}.v1")])}
}
fn compare_requirement(dt:ElementType,asc:bool)->OperationRequirement{
    let direction=if asc{"asc"}else{"desc"};
    requirement(&format!("compare.{direction}"),&format!("{direction}_{}",tag(dt)))
}
fn gather_requirement(dt:ElementType)->OperationRequirement{
    requirement("gather",&format!("gather_{}",tag(dt)))
}
/// Exact required profiles for one dtype, direction, and API route.
/// Union across all reachable callers, including U32 top-k index sorting.
pub fn dkr_argsort_requirements(dt:DType,asc:bool,sorted_values:bool)->Result<Vec<OperationRequirement>>{
    let dt=dtype(dt)?;
    let mut out=vec![requirement("init","init"),compare_requirement(dt,asc),requirement("finish","finish")];
    if sorted_values{out.push(gather_requirement(dt));}
    Ok(out)
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
struct Geometry { cols:u32,padded:u32,rows:usize,elements:usize }
impl Geometry{
    fn new(elements:usize,cols:usize,dt:ElementType)->Result<Self>{
        if cols==0||elements==0||elements%cols!=0||elements>u32::MAX as usize{
            candle_core::bail!("invalid argsort geometry");
        }
        let padded=cols.checked_next_power_of_two().ok_or_else(||err("argsort padding overflow"))?;
        if elements.checked_mul(dt.size_bytes()).is_none_or(|n|n>MAX_TENSOR_BYTES)||
            elements.checked_mul(4).is_none_or(|n|n>MAX_TENSOR_BYTES)||
            padded.checked_mul(4).is_none_or(|n|n>MAX_TENSOR_BYTES){
            candle_core::bail!("argsort tensor/workspace exceeds typed ABI budget");
        }
        Ok(Self{cols:u32::try_from(cols).map_err(err)?,padded:u32::try_from(padded).map_err(err)?,
            rows:elements/cols,elements})
    }
}
fn geometry(layout:&Layout,cols:usize,dt:ElementType)->Result<Geometry>{
    if !layout.is_contiguous()||layout.shape().dims().last().copied()!=Some(cols){
        candle_core::bail!("argsort layout must be contiguous along the last dimension");
    }
    Geometry::new(layout.shape().elem_count(),cols,dt)
}
fn span(base:u64,offset:usize,count:usize,len:usize,dt:ElementType)->Result<u64>{
    let bytes=dt.size_bytes();
    if base==0||base%(bytes as u64)!=0||count==0||offset.checked_add(count).is_none_or(|end|end>len){
        candle_core::bail!("argsort invalid CUDA storage span");
    }
    let start=offset.checked_mul(bytes).and_then(|n|u64::try_from(n).ok()).and_then(|n|base.checked_add(n))
        .ok_or_else(||err("argsort start address overflow"))?;
    let size=count.checked_mul(bytes).and_then(|n|u64::try_from(n).ok()).ok_or_else(||err("argsort byte span overflow"))?;
    start.checked_add(size).ok_or_else(||err("argsort end address overflow"))?;
    Ok(start)
}

#[derive(Clone,Copy,Eq,PartialEq)]
struct SortKey { dt:ElementType,asc:bool,cols:u32,padded:u32 }
#[derive(Clone,Copy,Eq,PartialEq)]
struct GatherKey { dt:ElementType,cols:u32,elements:u32 }
struct SortPlan { init:PrevalidatedLaunchPlan,compare:PrevalidatedLaunchPlan,finish:PrevalidatedLaunchPlan }
thread_local!{
    static SORT:RefCell<LaunchPlanCache<SortKey,SortPlan>>=RefCell::new(LaunchPlanCache::new());
    static GATHER:RefCell<LaunchPlanCache<GatherKey,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());
}
fn tensor(name:&str,dt:ElementType,count:usize,access:Access)->(Parameter,Argument){
    (Parameter::Tensor{name:name.into(),dtype:dt,rank:1,access,representation:None},
        Argument::Tensor{dtype:dt,shape:vec![count],contiguous:true,start_offset:0,representation:None})
}
fn scalar(name:&str)->(Parameter,Argument){
    (Parameter::Scalar{name:name.into(),dtype:ElementType::U32},Argument::Scalar{dtype:ElementType::U32})
}
fn prepare(device:&Device,r:OperationRequirement,elements:u32,bindings:Vec<(Parameter,Argument)>)->Result<PrevalidatedLaunchPlan>{
    let (params,args):(Vec<_>,Vec<_>)=bindings.into_iter().unzip();
    let mut request=LaunchRequest{operation:r.operation,cc:0,args,grid:[elements.div_ceil(BLOCK),1,1],max_dynamic_shared_mem_bytes:0};
    let p=kernel_cache()?.prepare_for_device(&mut request,device).map_err(err)?;
    let ids=r.implementations.ok_or_else(||err("argsort exact implementation required"))?;
    if ids.len()!=1||p.manifest.implementation_id!=ids[0]||
        ids[0]!=format!("cuda.{}.v1",p.manifest.entry)||p.manifest.block!=[BLOCK,1,1]||
        p.manifest.shared_mem_bytes!=0||p.manifest.arguments!=params{
        candle_core::bail!("argsort exact profile/ABI mismatch");
    }
    unsafe{p.prevalidate_launch_with_default_shared(&request,device,0)}.map_err(err)
}
fn sort_plan(device:&Device,key:SortKey)->Result<SortPlan>{
    let p=key.padded as usize;let c=key.cols as usize;
    Ok(SortPlan{
        init:prepare(device,requirement("init","init"),key.padded,vec![tensor("work",ElementType::U32,p,Access::Write),scalar("padded")])?,
        compare:prepare(device,compare_requirement(key.dt,key.asc),key.padded,vec![tensor("input",key.dt,c,Access::Read),
            tensor("work",ElementType::U32,p,Access::ReadWrite),scalar("cols"),scalar("padded"),scalar("j"),scalar("k")])?,
        finish:prepare(device,requirement("finish","finish"),key.cols,vec![tensor("work",ElementType::U32,p,Access::Read),
            tensor("indices",ElementType::U32,c,Access::Write),scalar("cols")])?,
    })
}
fn gather_plan(device:&Device,key:GatherKey)->Result<PrevalidatedLaunchPlan>{
    let n=key.elements as usize;
    prepare(device,gather_requirement(key.dt),key.elements,vec![tensor("input",key.dt,n,Access::Read),
        tensor("indices",ElementType::U32,n,Access::Read),tensor("output",key.dt,n,Access::Write),scalar("cols"),scalar("elements")])
}

/// All stages are prepared before the first launch. Reuse a single index
/// workspace for all rows, with stream ordering between passes.
pub(crate) fn indices(storage:&CudaStorage,layout:&Layout,cols:usize,asc:bool)->Result<(CudaStorage,Shape)>{
    let dt=dtype(storage.dtype())?;
    let g=geometry(layout,cols,dt)?;
    let dev=storage.device();
    let stream=dev.cuda_stream();
    let device=Device::Cuda(dev.clone());
    let mut dst=unsafe{dev.alloc::<u32>(g.elements)}?;
    let mut work=unsafe{dev.alloc::<u32>(g.padded as usize)}?;
    macro_rules! read_ptr{
        ($slice:expr)=>{{
            let (ptr,guard)=$slice.device_ptr(&stream);
            (span(ptr,layout.start_offset(),g.elements,$slice.len(),dt)?,guard)
        }};
    }
    let (input,src_guard)=match &storage.slice{
        CudaStorageSlice::F32(s)=>read_ptr!(s),CudaStorageSlice::F16(s)=>read_ptr!(s),
        CudaStorageSlice::BF16(s)=>read_ptr!(s),CudaStorageSlice::F64(s)=>read_ptr!(s),
        CudaStorageSlice::U8(s)=>read_ptr!(s),CudaStorageSlice::U32(s)=>read_ptr!(s),
        CudaStorageSlice::I64(s)=>read_ptr!(s),_=>candle_core::bail!("argsort storage dtype mismatch"),
    };
    let (dst_ptr,dst_guard)=dst.device_ptr_mut(&stream);
    let dst_ptr=span(dst_ptr,0,g.elements,g.elements,ElementType::U32)?;
    let (work_ptr,work_guard)=work.device_ptr_mut(&stream);
    let work_ptr=span(work_ptr,0,g.padded as usize,g.padded as usize,ElementType::U32)?;
    let key=SortKey{dt,asc,cols:g.cols,padded:g.padded};
    SORT.with(|cache|cache.borrow_mut().with_plan(key,&device,||sort_plan(&device,key),|plan|{
        for row in 0..g.rows{
            // The whole input span and element count were checked above.
            let row_input=input+(row*cols*dt.size_bytes()) as u64;
            let row_output=dst_ptr+(row*cols*4) as u64;
            unsafe{plan.init.with_prevalidated_launch(|func,cfg|{
                let mut b=stream.launch_builder(func);
                b.arg(&work_ptr).arg(&g.padded);b.launch(cfg)?;Ok(())
            })}.map_err(err)?;
            let mut k=2u32;
            while k<=g.padded{
                let mut j=k/2;
                while j!=0{
                    unsafe{plan.compare.with_prevalidated_launch(|func,cfg|{
                        let mut b=stream.launch_builder(func);
                        b.arg(&row_input).arg(&work_ptr).arg(&g.cols).arg(&g.padded).arg(&j).arg(&k);
                        b.launch(cfg)?;Ok(())
                    })}.map_err(err)?;
                    j/=2;
                }
                if k==g.padded{break;}
                k*=2;
            }
            unsafe{plan.finish.with_prevalidated_launch(|func,cfg|{
                let mut b=stream.launch_builder(func);
                b.arg(&work_ptr).arg(&row_output).arg(&g.cols);b.launch(cfg)?;Ok(())
            })}.map_err(err)?;
        }
        Ok(())
    }))?;
    drop(work_guard);drop(dst_guard);drop(src_guard);
    Ok((CudaStorage{slice:CudaStorageSlice::U32(dst),device:dev.clone()},layout.shape().clone()))
}

/// Internal consumer of the permutation produced by indices(). Do not use
/// to gather untrusted arbitrary indices. Avoid mutating a read-only input.
pub(crate) struct ArgsortGather { pub(crate) cols:usize }
impl candle_core::CustomOp2 for ArgsortGather{
    fn name(&self)->&'static str{"dkr_argsort_gather"}
    fn cpu_fwd(&self,_:&CpuStorage,_:&Layout,_:&CpuStorage,_:&Layout)->Result<(CpuStorage,Shape)>{
        candle_core::bail!("DKR argsort gather requires CUDA")
    }
    fn cuda_fwd(&self,storage:&CudaStorage,layout:&Layout,indices:&CudaStorage,il:&Layout)->Result<(CudaStorage,Shape)>{
        let dt=dtype(storage.dtype())?;let g=geometry(layout,self.cols,dt)?;
        let dev=storage.device();let device=Device::Cuda(dev.clone());
        if !device.same_device(&Device::Cuda(indices.device().clone()))||il.shape()!=layout.shape()||!il.is_contiguous(){
            candle_core::bail!("argsort permutation layout/device mismatch");
        }
        let CudaStorageSlice::U32(ids)=&indices.slice else{candle_core::bail!("argsort indices must be U32");};
        let stream=dev.cuda_stream();
        let (id_ptr,id_guard)=ids.device_ptr(&stream);
        let id_ptr=span(id_ptr,il.start_offset(),g.elements,ids.len(),ElementType::U32)?;
        let key=GatherKey{dt,cols:g.cols,elements:u32::try_from(g.elements).map_err(err)?};
        macro_rules! gather{
            ($variant:ident,$ty:ty,$src:expr)=>{{
                let (p,guard)=$src.device_ptr(&stream);
                let p=span(p,layout.start_offset(),g.elements,$src.len(),dt)?;
                let mut out=unsafe{dev.alloc::<$ty>(g.elements)}?;
                let (out_ptr,out_guard)=out.device_ptr_mut(&stream);
                let out_ptr=span(out_ptr,0,g.elements,g.elements,dt)?;
                GATHER.with(|cache|cache.borrow_mut().with_plan(key,&device,||gather_plan(&device,key),|plan|{
                    unsafe{plan.with_prevalidated_launch(|func,cfg|{
                        let mut b=stream.launch_builder(func);
                        b.arg(&p).arg(&id_ptr).arg(&out_ptr).arg(&key.cols).arg(&key.elements);
                        b.launch(cfg)?;Ok(())
                    })}.map_err(err)
                }))?;
                drop(out_guard);drop(guard);
                CudaStorageSlice::$variant(out)
            }};
        }
        let out=match &storage.slice{
            CudaStorageSlice::F32(s)=>gather!(F32,f32,s),CudaStorageSlice::F16(s)=>gather!(F16,half::f16,s),
            CudaStorageSlice::BF16(s)=>gather!(BF16,half::bf16,s),CudaStorageSlice::F64(s)=>gather!(F64,f64,s),
            CudaStorageSlice::U8(s)=>gather!(U8,u8,s),CudaStorageSlice::U32(s)=>gather!(U32,u32,s),
            CudaStorageSlice::I64(s)=>gather!(I64,i64,s),_=>candle_core::bail!("argsort gather dtype mismatch"),
        };
        drop(id_guard);
        Ok((CudaStorage{slice:out,device:dev.clone()},layout.shape().clone()))
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::ops::ArgSortOp;
    use candle_core::Tensor;
    #[test]
    fn dkr_argsort_capabilities_and_geometry(){
        let mut ids=std::collections::BTreeSet::new();
        for dt in [DType::F32,DType::F16,DType::BF16,DType::F64,DType::U8,DType::U32,DType::I64]{
            for asc in [true,false]{
                assert_eq!(dkr_argsort_requirements(dt,asc,false).unwrap().len(),3);
                let requirements=dkr_argsort_requirements(dt,asc,true).unwrap();
                assert_eq!(requirements.len(),4);
                for r in requirements{ids.extend(r.implementations.unwrap());}
            }
        }
        assert_eq!(ids.len(),23);
        assert!(ids.contains("cuda.dkr_argsort_desc_f64.v1"));
        assert_eq!(Geometry::new(6,3,ElementType::F64).unwrap().padded,4);
        assert!(Geometry::new(5,3,ElementType::F64).is_err());
        assert!(Geometry::new(0,0,ElementType::F64).is_err());
        assert!(Geometry::new(MAX_TENSOR_BYTES,1,ElementType::F64).is_err());
        assert_eq!(span(8,1,3,4,ElementType::F64).unwrap(),16);
        assert!(span(8,2,3,4,ElementType::F64).is_err());
    }
    #[test]
    #[ignore="requires CUDA and qualified Argsort catalogue"]
    fn dkr_argsort_gpu_all_dtypes()->Result<()>{
        let dev=Device::new_cuda(0)?;
        macro_rules! exercise{
            ($ty:ty,$a:expr,$b:expr,$c:expr)=>{{
                let input=Tensor::from_vec(vec![$c,$a,$b,$a,$c,$b],(2,3),&dev)?;
                for asc in [true,false]{
                    let (values,ids)=ArgSortOp::sort(&input,asc)?;
                    let (expected,permutation)=if asc{
                        (vec![vec![$a,$b,$c];2],vec![vec![1,2,0],vec![0,2,1]])
                    }else{
                        (vec![vec![$c,$b,$a];2],vec![vec![0,2,1],vec![1,2,0]])
                    };
                    assert_eq!(values.to_vec2::<$ty>()?,expected);
                    assert_eq!(ids.to_vec2::<u32>()?,permutation);
                    assert_eq!(input.arg_sort(asc)?.to_vec2::<u32>()?,permutation);
                }
                assert_eq!(input.to_vec2::<$ty>()?,vec![vec![$c,$a,$b],vec![$a,$c,$b]]);
            }};
        }
        exercise!(f32,-3f32,0f32,2f32);
        exercise!(f64,-3f64,0f64,2f64);
        exercise!(half::f16,half::f16::from_f32(-3.),half::f16::from_f32(0.),half::f16::from_f32(2.));
        exercise!(half::bf16,half::bf16::from_f32(-3.),half::bf16::from_f32(0.),half::bf16::from_f32(2.));
        exercise!(u8,0u8,2u8,u8::MAX);
        exercise!(u32,0u32,2u32,u32::MAX);
        exercise!(i64,i64::MIN,0i64,i64::MAX);
        Ok(())
    }
    #[test]
    #[ignore="requires CUDA and qualified Argsort catalogue"]
    fn dkr_argsort_gpu_f64_and_offsets()->Result<()>{
        let dev=Device::new_cuda(0)?;
        let a=Tensor::from_vec(vec![0f64,0.,0.,-3.,-1.,-2.,1.+f64::EPSILON,0.,1.],(3,3),&dev)?;
        let v=a.narrow(0,1,2)?;
        let (sorted,ids)=ArgSortOp::sort(&v,false)?;
        assert_eq!(ids.to_vec2::<u32>()?,vec![vec![1,2,0],vec![0,2,1]]);
        assert_eq!(sorted.to_vec2::<f64>()?,vec![vec![-1.,-2.,-3.],vec![1.+f64::EPSILON,1.,0.]]);
        assert_eq!(v.to_vec2::<f64>()?,vec![vec![-3.,-1.,-2.],vec![1.+f64::EPSILON,0.,1.]]);
        let x=Tensor::from_vec(vec![i64::MAX,i64::MIN,i64::MAX-1],(1,3),&dev)?;
        let (values,ids)=ArgSortOp::sort(&x,true)?;
        assert_eq!(ids.to_vec2::<u32>()?,vec![vec![1,2,0]]);
        assert_eq!(values.to_vec2::<i64>()?,vec![vec![i64::MIN,i64::MAX-1,i64::MAX]]);
        Ok(())
    }
}
