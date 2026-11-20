//! Accumulators that stay live, imm8 ops and permutes.
//!
//! input: [0, 192) vectors a, b, c. output: ten vectors at 192 + 64k
//! (see `entrypoint`).
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        let b = _svm512_loadu_si512(input.add(64));
        let c = _svm512_loadu_si512(input.add(128));
        // `a` is the accumulator here but is still used below.
        let o0 = _svm512_dpbusd_epi32(a, b, c);
        let o1 = _svm512_ternarylogic_epi64::<0xca>(a, b, c);
        let o2 = _svm512_slli_epi64::<3>(a);
        let o3 = _svm512_srai_epi32::<5>(b);
        let o4 = _svm512_rol_epi64::<13>(c);
        let o5 = _svm512_srai_epi64::<17>(b);
        let o6 = _svm512_rolv_epi64(a, b);
        let o7 = _svm512_permutexvar_epi64(c, a);
        let o8 = _svm512_shuffle_epi8(a, b);
        let o9 = _svm512_permutex2var_epi64(c, a, b);
        let out = [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9];
        for (k, &o) in out.iter().enumerate() {
            _svm512_storeu_si512(input.add(192 + 64 * k), o);
        }
    }
    0
}
