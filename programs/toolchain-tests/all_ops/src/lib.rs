//! Every intrinsic once, to check that each marker is known to the pass and
//! encodes to its opcode.
//!
//! input: [0, 192) vectors a, b, c; [192, 200) a u64 x. output: the k-th
//! vector result at 256 + 64k.
#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        let b = _svm512_loadu_si512(input.add(64));
        let c = _svm512_loadu_si512(input.add(128));
        let x = input.add(192).cast::<i64>().read_unaligned();
        let mut out = input.add(256);
        let mut put = |v: __svm512i| {
            _svm512_storeu_si512(out, v);
            out = out.add(64);
        };
        put(_svm512_loadbc_epi32(input.add(192).cast()));
        put(_svm512_loadbc_epi64(input.add(192).cast()));
        put(_svm512_set1_epi32(x as i32));
        put(_svm512_setzero_si512());
        put(_svm512_add_epi16(a, b));
        put(_svm512_add_epi64(a, b));
        put(_svm512_sub_epi16(a, b));
        put(_svm512_sub_epi64(a, b));
        put(_svm512_mullo_epi16(a, b));
        put(_svm512_mullo_epi64(a, b));
        put(_svm512_mulhi_epi16(a, b));
        put(_svm512_max_epi32(a, b));
        put(_svm512_min_epi32(a, b));
        put(_svm512_min_epu64(a, b));
        put(_svm512_and_si512(a, b));
        put(_svm512_xor_si512(a, b));
        put(_svm512_ternarylogic_epi64::<0x96>(a, b, c));
        put(_svm512_min_epu16(a, b));
        put(_svm512_slli_epi64::<33>(a));
        put(_svm512_srli_epi64::<33>(a));
        put(_svm512_srai_epi32::<7>(a));
        put(_svm512_srai_epi64::<33>(a));
        put(_svm512_rol_epi64::<33>(a));
        put(_svm512_rolv_epi64(a, c));
        put(_svm512_dpbusd_epi32(c, a, b));
        put(_svm512_dpwssd_epi32(c, a, b));
        put(_svm512_madd_epi16(a, b));
        put(_svm512_permutexvar_epi64(c, a));
        put(_svm512_shuffle_epi8(a, b));
        put(_svm512_permutex2var_epi16(a, c, b));
        put(_svm512_permutex2var_epi64(a, c, b));
        put(_svm512_cvtusepi32_epi8(a));
        put(_svm512_abs_epi64(a));
    }
    0
}
