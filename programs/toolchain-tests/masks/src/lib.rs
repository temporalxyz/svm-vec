//! Masked ops, mask results and the memory ops of the mask foundation, with
//! a reference in `check.py`.
//!
//! input: [0, 192) vectors a, b, c (16×u32); [192, 200) the mask k. output:
//! the k-th vector at 256 + 64k, then the scalar results as u64.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let [a, b, c] = _svm512_loadu_n::<3>(input);
        let k = input.add(192).cast::<u64>().read_unaligned();
        let out = input.add(256);
        let o = |i: usize| out.add(64 * i);
        _svm512_storeu_si512(o(0), _svm512_mask_add_epi32(c, k as u16, a, b));
        _svm512_storeu_si512(o(1), _svm512_maskz_sub_epi32(k as u16, a, b));
        _svm512_storeu_si512(o(2), _svm512_mask_dpbusd_epi32(c, k as u16, a, b));
        _svm512_storeu_si512(o(3), _svm512_mask_blend_epi8(k, a, b));
        _svm512_storeu_si512(o(4), _svm512_maskz_loadu_epi16(k as u32, input.add(64).cast()));
        let idx = _svm512_and_si512(a, _svm512_set1_epi64(7));
        let g = _svm512_mask_i64gather_epi64::<8>(c, k as u8, idx, input.cast());
        _svm512_storeu_si512(o(5), g);
        _svm512_mask_storeu_epi32(o(6).cast(), k as u16, a);
        _svm512_i64scatter_epi64::<8>(o(7).cast(), _svm512_and_si512(b, _svm512_set1_epi64(7)), a);
        // a conditional update in a loop: acc += x where the lane's bit is set
        let mut acc = _svm512_setzero_si512();
        let mut m = k;
        for i in 0..8 {
            let x = _svm512_loadu_si512(input.add(64 * (i % 3)));
            acc = _svm512_mask_add_epi64(acc, m as u8, acc, x);
            m >>= 8;
        }
        _svm512_storeu_si512(o(8), acc);
        let sum = [
            _svm512_add_epi32(a, b),
            _svm512_add_epi32(b, c),
            _svm512_add_epi32(c, a),
        ];
        _svm512_storeu_n::<3>(o(9), sum);
        _svm512_storeu_si512(o(12), _svm512_maskz_srli_epi32::<3>(k as u16, c));
        let s = o(13).cast::<u64>();
        s.write_unaligned(_svm512_cmplt_epi32_mask(a, b) as u64);
        s.add(1).write_unaligned(_svm512_cmpgt_epu8_mask(a, b));
        s.add(2).write_unaligned(_svm512_popcnt_mask(k) as u64);
        s.add(3).write_unaligned(_svm512_tzcnt_mask(k) as u64);
        s.add(4).write_unaligned(_svm512_lzcnt_mask(k >> 7) as u64);
        s.add(5).write_unaligned(_svm512_bitshuffle_epi64_mask(a, b));
    }
    0
}
