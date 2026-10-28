//! Straight line: `a + b`, `a - b` and the lane sum of `a`.
//!
//! input: [0, 64) a: 16×u32, [64, 128) b. output: [128, 192) a + b,
//! [192, 256) a - b, [256, 264) hsum(a) as u64.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        let b = _svm512_loadu_si512(input.add(64));
        _svm512_storeu_si512(input.add(128), _svm512_add_epi32(a, b));
        _svm512_storeu_si512(input.add(192), _svm512_sub_epi32(a, b));
        let sum = _svm512_reduce_add_epi32(a) as u32 as u64;
        input.add(256).cast::<u64>().write_unaligned(sum);
    }
    0
}
