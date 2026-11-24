//! Shared historical unsigned page divider, computed only on plan miss.
pub(super) fn fastdiv(d:u32)->(u32,u32,u32){
    debug_assert!(d>0);
    if d==1{return (0,0,0);}
    let nc=u32::MAX-(0u32.wrapping_sub(d)%d);
    let mut p=31u32;
    let mut q1=0x8000_0000u32/nc;
    let mut r1=0x8000_0000u32-q1*nc;
    let mut q2=0x7fff_ffffu32/d;
    let mut r2=0x7fff_ffffu32-q2*d;
    let mut a=0u32;
    loop{
        p+=1;
        if r1>=nc-r1{
            q1=q1.wrapping_mul(2).wrapping_add(1);
            r1=r1.wrapping_mul(2).wrapping_sub(nc);
        }else{q1=q1.wrapping_mul(2);r1=r1.wrapping_mul(2);}
        if r2+1>=d-r2{
            if q2>=0x7fff_ffff{a=1;}
            q2=q2.wrapping_mul(2).wrapping_add(1);
            r2=r2.wrapping_mul(2).wrapping_add(1).wrapping_sub(d);
        }else{
            if q2>=0x8000_0000{a=1;}
            q2=q2.wrapping_mul(2);
            r2=r2.wrapping_mul(2).wrapping_add(1);
        }
        let delta=d-1-r2;
        if p>=64 || !(q1<delta || (q1==delta && r1==0)){break;}
    }
    (q2.wrapping_add(1),p-32,a)
}
