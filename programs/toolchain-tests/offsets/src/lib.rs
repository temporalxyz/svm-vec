//! Constant displacements fold into the memory ops' `off` field, including
//! negative ones, around a base only known at run time. A displacement that
//! does not fit in i16 stays in the address.
//!
//! input: [0, 8) n (= 4096), [64, 128) a, [4088, 4092) x, [n + 28000, +64) b,
//! [n + 40000, +64) c. output: [128, 192) a + b at n - 3968,
//! [n + 32767, +64) c + splat(x).
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let p = input.add(input.cast::<u64>().read_unaligned() as usize);
        let a = _svm512_loadu_si512(p.sub(4032));
        let b = _svm512_loadu_si512(p.add(28000));
        _svm512_storeu_si512(p.sub(3968), _svm512_add_epi32(a, b));
        let c = _svm512_loadu_si512(p.add(40000));
        let x = _svm512_loadbc_epi32(p.sub(8).cast());
        _svm512_storeu_si512(p.add(32767), _svm512_add_epi32(c, x));
    }
    0
}
