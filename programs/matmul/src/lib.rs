//! int8 GEMM C = A x B (M = 32, K = 256, N = 128), no requantization.
//!
//! input region: [0..8192) A as u8, row-major; [8192..24576) C written as i32, row-major.
//! B is constant s8 in .rodata, packed by build.rs, which also emits the tile kernels.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

const K: usize = 256;
const N: usize = 128;
const C_OFF: usize = 32 * K;

include!(concat!(env!("OUT_DIR"), "/kernel.rs"));

/// row tiles (first row, rows)
const TILES: [(usize, usize); 6] = [(0, 6), (6, 6), (12, 5), (17, 5), (22, 5), (27, 5)];

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    for cb in 0..N / 16 / NB {
        for (r0, mb) in TILES {
            let a = input.add(r0 * K);
            let b = BP.0.as_ptr().add(64 * NB * cb);
            let c = input.add(C_OFF + 4 * (r0 * N + 16 * NB * cb));
            if mb == 6 {
                tile6(a, b, c)
            } else {
                tile5(a, b, c)
            }
        }
    }
    0
}
