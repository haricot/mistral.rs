//! Device-independent checked geometry for CUDA graph transport.
//! No CUDA code or catalogue requirement belongs in this module.
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CopyGeometry {
    pub width: usize,
    pub rows: usize,
    pub src_pitch: usize,
    pub dst_pitch: usize,
    src_offset: usize,
    dst_offset: usize,
    src_span: usize,
    dst_span: usize,
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("graph copy size overflow")
}
fn span(width: usize, rows: usize, pitch: usize) -> Result<usize> {
    if rows == 0 || width == 0 { return Ok(0); }
    ensure!(pitch >= width, "graph copy pitch smaller than row");
    mul(rows - 1, pitch)?.checked_add(width).context("graph copy span overflow")
}
fn contiguous(dims: &[usize], strides: &[usize]) -> Result<bool> {
    let mut expected = 1;
    for (&dim, &stride) in dims.iter().zip(strides).rev() {
        if dim > 1 && stride != expected { return Ok(false); }
        expected = mul(expected, dim)?;
    }
    Ok(true)
}
impl CopyGeometry {
    /// All strides, offsets and allocation lengths are in elements.
    /// Returned geometry and addresses are in bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize], src_strides: &[usize], dst_strides: &[usize],
        src_offset: usize, dst_offset: usize, src_len: usize, dst_len: usize,
        elem_size: usize,
    ) -> Result<Self> {
        ensure!(dims.len() == src_strides.len() && dims.len() == dst_strides.len(),
                "graph copy rank mismatch");
        ensure!(elem_size > 0, "graph copy invalid element size");
        ensure!(src_offset <= src_len && dst_offset <= dst_len, "graph copy offset exceeds allocation");
        let src_offset_bytes = mul(src_offset, elem_size)?;
        let dst_offset_bytes = mul(dst_offset, elem_size)?;
        // Validate even zero-length offsets, but do not submit a zero-size copy.
        if dims.contains(&0) {
            return Ok(Self {width:0, rows:0, src_pitch:0, dst_pitch:0,
                src_offset:src_offset_bytes, dst_offset:dst_offset_bytes, src_span:0, dst_span:0});
        }
        let elements = dims.iter().try_fold(1usize, |n, &d| mul(n,d))?;
        ensure!(contiguous(dims,dst_strides)?, "graph copy destination must be contiguous");
        let (width,rows,src_pitch) = if contiguous(dims,src_strides)? {
            (elements,1,elements)
        } else {
            let (&width, outer) = dims.split_last().context("graph copy rank")?;
            let (&last_stride, strides) = src_strides.split_last().context("graph copy stride")?;
            ensure!(!outer.is_empty() && (width == 1 || last_stride == 1),
                    "graph copy source is not dense by row");
            let pitch = outer.iter().zip(strides).rev().find(|pair| *pair.0 > 1)
                .map(|(_, &stride)| stride).context("graph copy source pitch")?;
            ensure!(pitch >= width, "graph copy pitch smaller than width");
            let mut expected = pitch;
            for (&dim,&stride) in outer.iter().zip(strides).rev() {
                if dim > 1 { ensure!(stride == expected, "graph copy source is not dense by row"); }
                expected = mul(expected,dim)?;
            }
            (width,elements/width,pitch)
        };
        let width = mul(width,elem_size)?;
        let src_pitch = mul(src_pitch,elem_size)?;
        let dst_pitch = width;
        let src_span = span(width,rows,src_pitch)?;
        let dst_span = span(width,rows,dst_pitch)?;
        ensure!(src_span <= mul(src_len-src_offset,elem_size)? &&
                dst_span <= mul(dst_len-dst_offset,elem_size)?,
                "graph copy span exceeds allocation");
        Ok(Self {width,rows,src_pitch,dst_pitch,src_offset:src_offset_bytes,
            dst_offset:dst_offset_bytes,src_span,dst_span})
    }

    /// Check complete address ranges before submitting any copy. Reject
    /// overlapping bounding ranges conservatively (even if holes are disjoint).
    /// Exact identity copies are no-ops only when the row geometry agrees.
    pub fn bind(self, src_base: u64, dst_base: u64) -> Result<Option<BoundCopy>> {
        if self.rows == 0 { return Ok(None); }
        ensure!(src_base != 0 && dst_base != 0, "graph copy null device address");
        let src = src_base.checked_add(u64::try_from(self.src_offset)?).context("graph copy source address overflow")?;
        let dst = dst_base.checked_add(u64::try_from(self.dst_offset)?).context("graph copy destination address overflow")?;
        let se = src.checked_add(u64::try_from(self.src_span)?).context("graph copy source end overflow")?;
        let de = dst.checked_add(u64::try_from(self.dst_span)?).context("graph copy destination end overflow")?;
        if src == dst && (self.rows == 1 || self.src_pitch == self.dst_pitch) { return Ok(None); }
        ensure!(se <= dst || de <= src, "graph copy overlapping device ranges");
        Ok(Some(BoundCopy {geometry:self,src,dst}))
    }
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct BoundCopy { geometry:CopyGeometry, src:u64, dst:u64 }
impl BoundCopy {
    /// One copy for contiguous memory; one per row for arbitrary dense views.
    /// No driver calls, allocations or fallible arithmetic in the iterator.
    pub fn rows(self) -> impl Iterator<Item=(u64,u64,usize)> {
        (0..self.geometry.rows).map(move |row| (
            self.src + (row*self.geometry.src_pitch) as u64,
            self.dst + (row*self.geometry.dst_pitch) as u64,
            self.geometry.width,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_graph_copy_geometry_contiguous_and_pitched() {
        let c=CopyGeometry::new(&[2,3],&[3,1],&[3,1],2,1,8,7,8).unwrap();
        assert_eq!(c.bind(100,300).unwrap().unwrap().rows().collect::<Vec<_>>(),vec![(116,308,48)]);
        let p=CopyGeometry::new(&[2,2,2],&[8,4,1],&[4,2,1],1,2,16,10,4).unwrap();
        assert_eq!(p.bind(100,300).unwrap().unwrap().rows().collect::<Vec<_>>(),
            vec![(104,308,8),(120,316,8),(136,324,8),(152,332,8)]);
        let singleton=CopyGeometry::new(&[2,1,2],&[4,99,1],&[2,99,1],0,0,6,4,1).unwrap();
        assert_eq!(singleton.bind(100,300).unwrap().unwrap().rows().collect::<Vec<_>>(),
            vec![(100,300,2),(104,302,2)]);
        let scalar=CopyGeometry::new(&[],&[],&[],0,0,1,1,4).unwrap();
        assert_eq!(scalar.bind(100,300).unwrap().unwrap().rows().collect::<Vec<_>>(),vec![(100,300,4)]);
    }
    #[test]
    fn dkr_graph_copy_geometry_rejects_invalid() {
        for (src,dst,so,doff,sl,dl) in [
            ([8,3,1],[4,2,1],0,0,32,8),
            ([8,4,2],[4,2,1],0,0,32,8),
            ([8,1,1],[4,2,1],0,0,32,8),
            ([8,4,1],[8,4,1],0,0,32,32),
            ([8,4,1],[4,2,1],3,0,16,8),
            ([8,4,1],[4,2,1],0,1,16,8),
            ([8,4,1],[4,2,1],17,0,16,8),
        ] {assert!(CopyGeometry::new(&[2,2,2],&src,&dst,so,doff,sl,dl,4).is_err());}
        assert!(CopyGeometry::new(&[usize::MAX,2],&[2,1],&[2,1],0,0,usize::MAX,usize::MAX,1).is_err());
        assert!(CopyGeometry::new(&[1],&[1],&[1],usize::MAX,0,usize::MAX,1,8).is_err());
        let c=CopyGeometry::new(&[4],&[1],&[1],0,0,4,4,4).unwrap();
        assert!(c.bind(100,108).is_err());
        assert!(c.bind(u64::MAX-4,300).is_err());
        assert!(c.bind(0,300).is_err());
        assert_eq!(c.bind(100,100).unwrap(),None);
        assert!(c.bind(100,116).unwrap().is_some());
        let zero=CopyGeometry::new(&[2,0],&[0,1],&[0,1],0,0,0,0,4).unwrap();
        assert_eq!(zero.bind(0,0).unwrap(),None);
        assert!(CopyGeometry::new(&[0],&[1],&[1],1,0,0,0,4).is_err());
    }
}
