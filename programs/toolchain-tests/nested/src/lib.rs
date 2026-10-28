//! Nested loops: per-row sums (inner accumulator), their lane-wise max and
//! total (outer accumulators).
//!
//! input: [0, 8) rows, [8, 16) cols (64-byte blocks per row);
//! [256, 256 + 64·rows·cols) i32 data. output: [64, 128) max of row sums,
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
        let mut best = _svm512_set1_epi32(i32::MIN);
        let mut total = _svm512_setzero_si512();
        for r in 0..rows {
            let mut row = _svm512_setzero_si512();
            for c in 0..cols {
                let x = _svm512_loadu_si512(data.add((r * cols + c) * 64));
                row = _svm512_add_epi32(row, x);
            }
            best = _svm512_max_epi32(best, row);
            total = _svm512_add_epi32(total, row);
        }
        _svm512_storeu_si512(input.add(64), best);
        _svm512_storeu_si512(input.add(128), total);
    }
    0
}
