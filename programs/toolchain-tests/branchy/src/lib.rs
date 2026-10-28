//! Data-dependent if/else inside a loop (phi merges) and a select.
//!
//! input: [0, 8) n ≤ 63 blocks, [8, 16) flags; [256, 256 + 64n) i32 data.
//! Block i goes to `hi` (lane max) if flag bit i is set, else to `lo`
//! (lane min). output: [64, 128) hi, [128, 192) lo, [192, 256) hi if flag
//! bit 63 is set else lo.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let n = input.cast::<u64>().read_unaligned() as usize;
        let flags = input.add(8).cast::<u64>().read_unaligned();
        let mut hi = _svm512_set1_epi32(i32::MIN);
        let mut lo = _svm512_set1_epi32(i32::MAX);
        for i in 0..n {
            let x = _svm512_loadu_si512(input.add(256 + i * 64));
            if flags >> i & 1 != 0 {
                hi = _svm512_max_epi32(hi, x);
            } else {
                lo = _svm512_min_epi32(lo, x);
            }
        }
        let pick = if flags >> 63 != 0 { hi } else { lo };
        _svm512_storeu_si512(input.add(64), hi);
        _svm512_storeu_si512(input.add(128), lo);
        _svm512_storeu_si512(input.add(192), pick);
    }
    0
}
