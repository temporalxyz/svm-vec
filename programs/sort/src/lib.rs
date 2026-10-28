//! Sorts 256 u32 keys ascending with svm vector ops.
//!
//! input region: [0..1024) the keys (little-endian u32), [1024..2048) the sorted keys out.
//! build.rs generates the sorting network (net.rs) as straight-line intrinsics.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[repr(C, align(64))]
pub struct Tab<const N: usize>([[u8; 64]; N]);

include!(concat!(env!("OUT_DIR"), "/sort.rs"));

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    sort(input, input.add(1024));
    0
}
