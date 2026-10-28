//! Loop with a loop-carried accumulator: u8·i8 dot product with `dpbusd`.
//!
//! input: [0, 8) n, the number of 64-byte blocks; [128, 128 + 64n) a: u8;
//! then b: i8. output: [8, 16) the dot product (i32, zero-extended);
//! [64, 128) the 16 lane accumulators.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let n = input.cast::<u64>().read_unaligned() as usize;
        let a = input.add(128);
        let b = a.add(n * 64);
        let mut acc = _svm512_setzero_si512();
        for i in 0..n {
            let x = _svm512_loadu_si512(a.add(i * 64));
            let y = _svm512_loadu_si512(b.add(i * 64));
            acc = _svm512_dpbusd_epi32(acc, x, y);
        }
        _svm512_storeu_si512(input.add(64), acc);
        let sum = _svm512_reduce_add_epi32(acc) as u32 as u64;
        input.add(8).cast::<u64>().write_unaligned(sum);
    }
    0
}
