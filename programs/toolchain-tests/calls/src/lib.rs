//! A vector live across a call to a function that uses vector registers.
//!
//! input: [0, 64) a, [64, 128) b. output: [64, 128) b + b (by `double`),
//! [128, 192) a, [192, 256) a ^ (b + b).
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[inline(never)]
unsafe fn double(p: *mut u8) {
    unsafe {
        let x = _svm512_loadu_si512(p);
        _svm512_storeu_si512(p, _svm512_add_epi32(x, x));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        double(input.add(64));
        _svm512_storeu_si512(input.add(128), a);
        let b2 = _svm512_loadu_si512(input.add(64));
        _svm512_storeu_si512(input.add(192), _svm512_xor_si512(a, b2));
    }
    0
}
