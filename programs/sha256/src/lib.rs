//! SHA-256 compression of 16 independent 64-byte blocks with svm vector ops.
//!
//! input region: [0..1024) 16 blocks of 64 bytes, [1024..1536) the 16 outputs, 32 bytes each.
//! each output is the raw compression function from the standard IV over that one block, with no
//! padding or length block: the 8 state words after `compress256(IV, block)`, big-endian (what
//! `sha2::compress256` gives, serialized as SHA-256 serializes its state).
//!
//! one vector holds a word of all 16 blocks (lane = block). build.rs generates the whole thing as
//! straight-line code, see build.rs and gen.rs.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[repr(C, align(64))]
pub struct Tab32<const N: usize>([u32; N]);
#[repr(C, align(64))]
pub struct Tab8<const N: usize>([u8; N]);

include!(concat!(env!("OUT_DIR"), "/sha.rs"));

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    sha256x16(input, input.add(1024));
    0
}
