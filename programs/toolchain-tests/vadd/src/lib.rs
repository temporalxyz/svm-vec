//! Straight line: `a + b`, `a - b` and the lane sum of `a`, through the stack.
//!
//! input: [0, 64) a: 8×u64, [64, 128) b. output: [128, 192) a + b,
//! [192, 256) a - b, [256, 264) the wrapping sum of the lanes of a.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        let b = _svm512_loadu_si512(input.add(64));
        _svm512_storeu_si512(input.add(128), _svm512_add_epi64(a, b));
        _svm512_storeu_si512(input.add(192), _svm512_sub_epi64(a, b));
        let mut lanes = core::mem::MaybeUninit::<[u64; 8]>::uninit();
        _svm512_storeu_si512(lanes.as_mut_ptr().cast(), a);
        let sum = lanes.assume_init().iter().fold(0u64, |s, &x| s.wrapping_add(x));
        input.add(256).cast::<u64>().write_unaligned(sum);
    }
    0
}
