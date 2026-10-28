//! programs/mlp's vector path, fully unrolled (int8 mlp 256 -> 128 relu -> 16).
//!
//! input region: as programs/mlp (the mode word is ignored): [64..320) x as u8 (0..=127),
//! [320..384) logits out as i32. returns argmax.

#![no_std]
#![feature(asm_experimental_arch)]

use core::mem::MaybeUninit;
use svm_intrinsics::*;
use svm_rt as _;

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let out = input.add(320);
    let mut h = MaybeUninit::<[u8; 128]>::uninit();
    // hide the base from llvm, so the loads use one register plus `off` instead of an lddw each
    let mut p = BLOB.0.as_ptr().add(BASE);
    core::arch::asm!("/* {0} */", inout(reg) p, options(pure, nomem, nostack));
    let v = vector(input.add(64), out, p, h.as_mut_ptr() as *mut u8);
    argmax(v, p)
}

// index of the first maximal lane: max over all lanes by swapping dwords, qwords, 128 and 256
// bits, then the lowest lane equal to it from the byte sign mask (4 bits per lane).
#[inline(always)]
unsafe fn argmax(v: __svm512i, p: *const u8) -> u64 {
    let mut m = _svm512_max_epi32(v, _svm512_rol_epi64::<32>(v));
    for k in 0..3 {
        let idx = _svm512_loadu_si512(p.offset(PERM + 64 * k));
        m = _svm512_max_epi32(m, _svm512_permutexvar_epi64(idx, m));
    }
    let mask = _svm512_movepi8_mask(_svm512_cmpeq_epi32(v, m));
    // lowest set bit is 1 << 4i; the multiply moves i to the top nibble
    let low = mask & mask.wrapping_neg();
    low.wrapping_mul(MAGIC) >> 60
}

const MAGIC: u64 = {
    let mut c = 0u64;
    let mut k = 0;
    while k < 16 {
        c |= (k as u64) << (60 - 4 * k);
        k += 1;
    }
    c
};
