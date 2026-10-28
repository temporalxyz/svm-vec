//! Register pressure: 30 vectors live at once.
//!
//! input: [0, 1920) v[0..30]. output: [1920 + 64k) v[k] + v[k + 1], and
//! v[29] + v[29] for k = 29. v[0] dies at the first add, so exactly 30
//! values are live at the peak.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;


#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let ld = |k: usize| _svm512_loadu_si512(input.add(64 * k));
        let v = [
            ld(0), ld(1), ld(2), ld(3), ld(4), ld(5), ld(6), ld(7), ld(8), ld(9),
            ld(10), ld(11), ld(12), ld(13), ld(14), ld(15), ld(16), ld(17), ld(18), ld(19),
            ld(20), ld(21), ld(22), ld(23), ld(24), ld(25), ld(26), ld(27), ld(28), ld(29),
        ];
        let st = |k: usize, a: __svm512i, b: __svm512i| {
            _svm512_storeu_si512(input.add(64 * (30 + k)), _svm512_add_epi64(a, b))
        };
        st(0, v[0], v[1]); st(1, v[1], v[2]); st(2, v[2], v[3]); st(3, v[3], v[4]);
        st(4, v[4], v[5]); st(5, v[5], v[6]); st(6, v[6], v[7]); st(7, v[7], v[8]);
        st(8, v[8], v[9]); st(9, v[9], v[10]); st(10, v[10], v[11]); st(11, v[11], v[12]);
        st(12, v[12], v[13]); st(13, v[13], v[14]); st(14, v[14], v[15]);
        st(15, v[15], v[16]); st(16, v[16], v[17]); st(17, v[17], v[18]);
        st(18, v[18], v[19]); st(19, v[19], v[20]); st(20, v[20], v[21]);
        st(21, v[21], v[22]); st(22, v[22], v[23]); st(23, v[23], v[24]);
        st(24, v[24], v[25]); st(25, v[25], v[26]); st(26, v[26], v[27]);
        st(27, v[27], v[28]); st(28, v[28], v[29]); st(29, v[29], v[29]);
    }
    0
}
