//! Batch perp risk math over 256 positions.
//!
//! input region, i64 arrays of 256 little-endian words: size, entry, mark, funding_delta,
//! collateral at [0..10240), then outputs pnl, funding, equity, healthy at [10240..18432).
//! Semantics are perps-host's `risk`.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

const N: usize = 256;

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let one = _svm512_set1_epi64(1);
    // field k of positions i..i + 8 is at p + 8 N k, with p = input + 8 i
    let f = |p: *mut u8, k: usize| p.add(8 * N * k);
    let mut p = input;
    for _ in 0..N / 8 {
        let s = _svm512_loadu_si512(f(p, 0));
        let e = _svm512_loadu_si512(f(p, 1));
        let m = _svm512_loadu_si512(f(p, 2));
        let fd = _svm512_loadu_si512(f(p, 3));
        let c = _svm512_loadu_si512(f(p, 4));

        let pnl = _svm512_mullo_epi64(s, _svm512_sub_epi64(m, e));
        let funding = _svm512_srai_epi64::<32>(_svm512_mullo_epi64(s, fd));
        let equity = _svm512_sub_epi64(_svm512_add_epi64(c, pnl), funding);
        // |size| = (s ^ sign) - sign
        let sign = _svm512_srai_epi64::<63>(s);
        let abs = _svm512_sub_epi64(_svm512_xor_si512(s, sign), sign);
        let req = _svm512_srai_epi64::<36>(_svm512_mullo_epi64(abs, m));
        // equity >= req  <=>  !(req > equity)
        let healthy = _svm512_andnot_si512(_svm512_cmpgt_epi64(req, equity), one);

        _svm512_storeu_si512(f(p, 5), pnl);
        _svm512_storeu_si512(f(p, 6), funding);
        _svm512_storeu_si512(f(p, 7), equity);
        _svm512_storeu_si512(f(p, 8), healthy);
        p = p.add(64);
    }
    0
}
