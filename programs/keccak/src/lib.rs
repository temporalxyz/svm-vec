//! Keccak-f[1600], scalar sBPF vs svm vector ops.
//!
//! input region: [0..8) mode (0 scalar, 1 vector), [8..16) k, [64..264) state (25 u64 lanes,
//! lane (x, y) at 64 + 8 (x + 5y)), [264..288) zero (the vector path stores 64 byte rows).
//! the state is permuted k times in place.
//!
//! vector path: the state is 5 vectors of 5 q-lanes. build.rs generates the 24 rounds as
//! straight-line code (theta with ternlog and lane rotations, rho with rolv, pi and the layout
//! changes with permq, chi with ternlog, iota folded into chi's copy), see build.rs.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[path = "../keccak.rs"]
mod keccak;

#[repr(C, align(64))]
pub struct Tab<const N: usize>([[u64; 8]; N]);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

static MASK: Tab<1> = Tab([[!0, !0, !0, !0, !0, 0, 0, 0]]);

const STATE: usize = 64;

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let k = (input.add(8) as *const u64).read_unaligned();
    if (input as *const u64).read_unaligned() == 0 {
        scalar(input.add(STATE), k)
    } else {
        vector(input.add(STATE), k)
    }
    0
}

#[inline(never)]
unsafe fn scalar(st: *mut u8, k: u64) {
    let p = st as *mut u64;
    let mut a = [0u64; 25];
    for i in 0..25 {
        a[i] = p.add(i).read_unaligned();
    }
    for _ in 0..k {
        keccak::keccak_f(&mut a);
    }
    for i in 0..25 {
        p.add(i).write_unaligned(a[i]);
    }
}

#[inline(never)]
unsafe fn vector(st: *mut u8, k: u64) {
    let rot = ROT.0.as_ptr() as *const u8;
    let r = [
        _svm512_loadu_si512(rot),
        _svm512_loadu_si512(rot.add(64)),
        _svm512_loadu_si512(rot.add(128)),
        _svm512_loadu_si512(rot.add(192)),
    ];
    // rows y = 0..5, lanes 5..8 cleared (they stay zero)
    let m = _svm512_loadu_si512(MASK.0.as_ptr() as _);
    let mut s = [
        _svm512_and_si512(_svm512_loadu_si512(st), m),
        _svm512_and_si512(_svm512_loadu_si512(st.add(40)), m),
        _svm512_and_si512(_svm512_loadu_si512(st.add(80)), m),
        _svm512_and_si512(_svm512_loadu_si512(st.add(120)), m),
        _svm512_and_si512(_svm512_loadu_si512(st.add(160)), m),
    ];
    for _ in 0..k {
        s = keccak_f(s, &r);
    }
    // in order: each store's zero lanes 5..8 are overwritten by the next row
    _svm512_storeu_si512(st, s[0]);
    _svm512_storeu_si512(st.add(40), s[1]);
    _svm512_storeu_si512(st.add(80), s[2]);
    _svm512_storeu_si512(st.add(120), s[3]);
    _svm512_storeu_si512(st.add(160), s[4]);
}
