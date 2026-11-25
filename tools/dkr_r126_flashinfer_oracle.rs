//! Standalone F64 attention oracle: rustc --edition=2024 --test -O this_file.rs.
//! No Candle, CUDA, FlashInfer or DKR dependencies.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub page: usize, pub pages: usize, pub batch: usize, pub kv_heads: usize,
    pub heads: usize, pub hd: usize,
    pub indptr: Vec<i32>, pub indices: Vec<i32>, pub last: Vec<i32>,
    pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>,
}
#[derive(Debug, Clone)]
pub struct Schedule {
    pub requests: Vec<i32>, pub tiles: Vec<i32>, pub offsets: Vec<i32>,
    pub mask: Vec<u8>, pub chunk: usize,
}
impl Fixture {
    pub fn new(hd: usize, group: usize) -> Self {
        let (page,pages,batch,kv_heads)=(7,5,2,2);
        let heads=kv_heads*group;
        // Exact binary fractions, representable in FP8 E4M3, BF16 and F16.
        let values=|n:usize,seed:usize| (0..n).map(|i|
            (((i*seed+i/7+3)%15) as f32-7.)/16.).collect::<Vec<_>>();
        Self {page,pages,batch,kv_heads,heads,hd,
            indptr:vec![0,3,5], indices:vec![4,1,3,0,2], last:vec![3,2],
            q:values(batch*heads*hd,5), k:values(pages*kv_heads*page*hd,3),
            v:values(pages*kv_heads*page*hd,7)}
    }
    pub fn lengths(&self) -> Result<Vec<usize>,String> {
        if self.page==0 || self.batch==0 || self.kv_heads==0 || self.heads%self.kv_heads!=0
            || self.hd==0 || self.indptr.len()!=self.batch+1 || self.last.len()!=self.batch
            || self.indptr[0]!=0 {return Err("invalid CSR header".into());}
        let mut result=Vec::new();
        for b in 0..self.batch {
            let lo=usize::try_from(self.indptr[b]).map_err(|_|"negative indptr")?;
            let hi=usize::try_from(self.indptr[b+1]).map_err(|_|"negative indptr")?;
            let tail=usize::try_from(self.last[b]).map_err(|_|"negative last page")?;
            if hi<=lo || hi>self.indices.len() || tail==0 || tail>self.page ||
                self.indices[lo..hi].iter().any(|&i|i<0 || i as usize>=self.pages) {
                return Err("invalid active CSR span or page ID".into());
            }
            result.push((hi-lo-1)*self.page+tail);
        }
        Ok(result)
    }
    pub fn schedule(&self, split:bool) -> Schedule {
        let lengths=self.lengths().unwrap();
        let chunk=if split {self.page} else {*lengths.iter().max().unwrap()};
        let mut s=Schedule{requests:vec![],tiles:vec![],offsets:vec![0],mask:vec![],chunk};
        for (b,len) in lengths.into_iter().enumerate() {
            for tile in 0..len.div_ceil(chunk) {
                s.requests.push(b as i32);s.tiles.push(tile as i32);s.mask.push(1);
            }
            s.offsets.push(s.requests.len() as i32);
        }
        if split {s.requests.push(-1);s.tiles.push(-1);s.mask.push(0);}
        s
    }
    pub fn validate(&self,s:&Schedule) -> Result<(),String> {
        let lengths=self.lengths()?;
        let padded=s.requests.len();
        if s.chunk==0 || s.tiles.len()!=padded || s.mask.len()!=padded ||
            s.offsets.len()!=self.batch+1 || s.offsets[0]!=0 || padded<self.batch {
            return Err("invalid schedule header".into());
        }
        for (b,len) in lengths.into_iter().enumerate() {
            let lo=usize::try_from(s.offsets[b]).map_err(|_|"negative merge offset")?;
            let hi=usize::try_from(s.offsets[b+1]).map_err(|_|"negative merge offset")?;
            if hi<=lo || hi>padded || hi-lo!=len.div_ceil(s.chunk) {
                return Err("merge intervals do not cover context exactly".into());
            }
            for (tile,row) in (lo..hi).enumerate() {
                if s.mask[row]!=1 || s.requests[row]!=b as i32 || s.tiles[row]!=tile as i32 {
                    return Err("invalid active tile mapping".into());
                }
            }
        }
        let active=s.offsets[self.batch] as usize;
        if s.mask[active..].iter().any(|&m|m!=0) {return Err("active padded tile".into());}
        Ok(())
    }
    fn logits(&self,b:usize,h:usize,len:usize,scale:f64,ks:f64,
              window:Option<usize>,cap:Option<f64>) -> Vec<f64> {
        let kh=h/(self.heads/self.kv_heads);
        (0..len).map(|t| {
            if window.is_some_and(|w|len-1-t>w) {return f64::NEG_INFINITY;}
            let page=self.indices[self.indptr[b] as usize+t/self.page] as usize;
            let qi=(b*self.heads+h)*self.hd;
            let ki=((page*self.kv_heads+kh)*self.page+t%self.page)*self.hd;
            let z=(0..self.hd).map(|d|self.q[qi+d] as f64*self.k[ki+d] as f64).sum::<f64>()*scale*ks;
            if let Some(c)=cap {c*(z/c).tanh()}else{z}
        }).collect()
    }
    fn partial(&self,b:usize,h:usize,logits:&[f64],lo:usize,hi:usize,vs:f64) -> (Vec<f64>,f64) {
        let m=logits[lo..hi].iter().copied().fold(f64::NEG_INFINITY,f64::max);
        let mut out=vec![0.;self.hd];
        if m==f64::NEG_INFINITY {return (out,m);}
        let kh=h/(self.heads/self.kv_heads);
        let mut den=0.;
        for t in lo..hi {
            let w=(logits[t]-m).exp();
            let page=self.indices[self.indptr[b] as usize+t/self.page] as usize;
            let vi=((page*self.kv_heads+kh)*self.page+t%self.page)*self.hd;
            den+=w;
            for d in 0..self.hd {out[d]+=w*self.v[vi+d] as f64*vs;}
        }
        for x in &mut out {*x/=den;}
        (out,(m+den.ln())/std::f64::consts::LN_2)
    }
    pub fn reference(&self,s:&Schedule,scale:f64,ks:f64,vs:f64,window:Option<usize>,
                     cap:Option<f64>) -> Result<(Vec<f64>,Vec<f64>),String> {
        self.validate(s)?;
        let mut direct=Vec::new();let mut merged=Vec::new();
        for (b,len) in self.lengths()?.into_iter().enumerate() {
            for h in 0..self.heads {
                let logits=self.logits(b,h,len,scale,ks,window,cap);
                direct.extend(self.partial(b,h,&logits,0,len,vs).0);
                let parts=(0..len.div_ceil(s.chunk)).map(|tile|self.partial(b,h,&logits,
                    tile*s.chunk,((tile+1)*s.chunk).min(len),vs)).collect::<Vec<_>>();
                let m=parts.iter().map(|p|p.1).fold(f64::NEG_INFINITY,f64::max);
                let mut out=vec![0.;self.hd];let mut den=0.;
                if m.is_finite() {
                    for (v,lse) in parts {
                        let w=(lse-m).exp2();den+=w;
                        for d in 0..self.hd {out[d]+=w*v[d];}
                    }
                    for x in &mut out {*x/=den;}
                }
                merged.extend(out);
            }
        }
        Ok((direct,merged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f64_direct_split_matrix() {
        for hd in [64,128,256,512] {
            for group in [1,2,3,4,5,6,7,8,16] {
                let f=Fixture::new(hd,group);
                for (ks,vs) in [(1.,1.),(0.5,2.)] {
                    for window in [None,Some(2)] {
                        for cap in [None,Some(0.75)] {
                            let (a,b)=f.reference(&f.schedule(true),0.125,ks,vs,window,cap).unwrap();
                            assert!(a.iter().zip(b).all(|(a,b)|a.is_finite() && (a-b).abs()<1e-12));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn invalid_metadata_fails_closed_in_oracle() {
        let f=Fixture::new(64,1);
        let mut s=f.schedule(true);s.tiles[1]=0;assert!(f.validate(&s).is_err());
        let mut s=f.schedule(true);s.offsets[1]-=1;assert!(f.validate(&s).is_err());
        let mut s=f.schedule(true);*s.mask.last_mut().unwrap()=1;assert!(f.validate(&s).is_err());
        let mut s=f.schedule(true);s.chunk=0;assert!(f.validate(&s).is_err());
        let mut f=f;f.indices[0]=99;assert!(f.lengths().is_err());
    }
}
