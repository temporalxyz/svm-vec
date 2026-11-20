//! Nested loops: per-row sums (inner accumulator), their lane-wise minimum
//! and total (outer accumulators).
//!
//! input: [0, 8) rows, [8, 16) cols (64-byte blocks per row);
//! [256, 256 + 64·rows·cols) u64 data. output: [64, 128) min of row sums,
//! [128, 192) total.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let rows = input.cast::<u64>().read_unaligned() as usize;
        let cols = input.add(8).cast::<u64>().read_unaligned() as usize;
        let data = input.add(256);
        let mut best = _svm512_set1_epi32(-1);
        let mut total = _svm512_setzero_si512();
        for r in 0..rows {
            let mut row = _svm512_setzero_si512();
            for c in 0..cols {
                let x = _svm512_loadu_si512(data.add((r * cols + c) * 64));
                row = _svm512_add_epi64(row, x);
            }
            best = _svm512_min_epu64(best, row);
            total = _svm512_add_epi64(total, row);
        }
        _svm512_storeu_si512(input.add(64), best);
        _svm512_storeu_si512(input.add(128), total);
    }
    0
}
