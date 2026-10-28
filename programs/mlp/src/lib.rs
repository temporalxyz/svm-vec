//! int8 mlp (256 -> 128 relu -> 16), scalar sbpf vs svm vector ops.
//!
//! input region: [0..8) mode (0 scalar, 1 vector), [64..320) x as u8 (0..=127),
//! [320..384) logits out as i32. returns argmax.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

include!(concat!(env!("OUT_DIR"), "/weights.rs"));

const IN: usize = 256;
const HID: usize = 128;
const OUT: usize = 16;
const X: usize = 64;
const LOGITS: usize = 320;

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let x = input.add(X);
    let out = input.add(LOGITS) as *mut i32;
    // the `scalar-only` build has no vector ops, so it runs on the unmodified jit path (baseline)
    if cfg!(feature = "scalar-only") || (input as *const u64).read_unaligned() == 0 {
        scalar(x, out)
    } else {
        #[cfg(not(feature = "scalar-only"))]
        vector(x, out)
    }
    argmax(out)
}

#[inline(never)]
unsafe fn scalar(x: *const u8, out: *mut i32) {
    let mut h = [0u8; HID];
    for o in 0..HID {
        let mut acc = B1[o];
        for i in 0..IN {
            acc = acc.wrapping_add(*x.add(i) as i32 * *W1.get_unchecked(o * IN + i) as i8 as i32);
        }
        h[o] = (acc >> 10).clamp(0, 127) as u8;
    }
    for o in 0..OUT {
        let mut acc = B2[o];
        for i in 0..HID {
            acc = acc.wrapping_add(h[i] as i32 * *W2.get_unchecked(o * HID + i) as i8 as i32);
        }
        out.add(o).write_unaligned(acc);
    }
}

#[cfg(not(feature = "scalar-only"))]
#[inline(never)]
unsafe fn vector(x: *const u8, out: *mut i32) {
    // layer 1: 8 accumulators x 16 lanes = 128 outputs. per group of 4 inputs:
    // broadcast the 4 input bytes, dpbusd against 64 bytes of packed weights per accumulator.
    let b1 = B1.as_ptr() as *const u8;
    let (mut a0, mut a1, mut a2, mut a3) = (
        _svm512_loadu_si512(b1 as _),
        _svm512_loadu_si512(b1.add(64) as _),
        _svm512_loadu_si512(b1.add(128) as _),
        _svm512_loadu_si512(b1.add(192) as _),
    );
    let (mut a4, mut a5, mut a6, mut a7) = (
        _svm512_loadu_si512(b1.add(256) as _),
        _svm512_loadu_si512(b1.add(320) as _),
        _svm512_loadu_si512(b1.add(384) as _),
        _svm512_loadu_si512(b1.add(448) as _),
    );
    let mut g = 0;
    while g < IN / 4 {
        let xb = _svm512_loadbc_epi32(x.add(4 * g) as _);
        let w = W1P.as_ptr().add(g * HID * 4);
        a0 = _svm512_dpbusd_epi32(a0, xb, _svm512_loadu_si512(w as _));
        a1 = _svm512_dpbusd_epi32(a1, xb, _svm512_loadu_si512(w.add(64) as _));
        a2 = _svm512_dpbusd_epi32(a2, xb, _svm512_loadu_si512(w.add(128) as _));
        a3 = _svm512_dpbusd_epi32(a3, xb, _svm512_loadu_si512(w.add(192) as _));
        a4 = _svm512_dpbusd_epi32(a4, xb, _svm512_loadu_si512(w.add(256) as _));
        a5 = _svm512_dpbusd_epi32(a5, xb, _svm512_loadu_si512(w.add(320) as _));
        a6 = _svm512_dpbusd_epi32(a6, xb, _svm512_loadu_si512(w.add(384) as _));
        a7 = _svm512_dpbusd_epi32(a7, xb, _svm512_loadu_si512(w.add(448) as _));
        g += 1;
    }

    // relu + requantize to 0..=127, narrow 16 lanes to 16 bytes. each store writes 64 bytes,
    // so h has 48 bytes of slack and the stores go in order.
    let zero = _svm512_setzero_si512();
    let max = _svm512_set1_epi32(127);
    let mut h = [0u8; HID + 48];
    let hp = h.as_mut_ptr();
    macro_rules! relu {
        ($($a:ident $k:literal)*) => {$(
            let t = _svm512_min_epi32(_svm512_max_epi32(_svm512_srai_epi32::<10>($a), zero), max);
            _svm512_storeu_si512(hp.add(16 * $k) as _, _svm512_cvtusepi32_epi8(t));
        )*};
    }
    relu!(a0 0 a1 1 a2 2 a3 3 a4 4 a5 5 a6 6 a7 7);

    // layer 2: one accumulator, 16 outputs.
    let mut acc = _svm512_loadu_si512(B2.as_ptr() as _);
    let mut g = 0;
    while g < HID / 4 {
        let hb = _svm512_loadbc_epi32(hp.add(4 * g) as _);
        acc = _svm512_dpbusd_epi32(acc, hb, _svm512_loadu_si512(W2P.as_ptr().add(64 * g) as _));
        g += 1;
    }
    _svm512_storeu_si512(out as _, acc);
}

unsafe fn argmax(out: *const i32) -> u64 {
    let mut best = 0;
    for o in 1..OUT {
        if out.add(o).read_unaligned() > out.add(best).read_unaligned() {
            best = o;
        }
    }
    best as u64
}
