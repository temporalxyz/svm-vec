//! Every intrinsic once, to check that each marker is known to the pass and
//! encodes to its opcode.
//!
//! input: [0, 192) vectors a, b, c; [192, 200) a u64 x. output: the k-th
//! vector result at 256 + 64k, then the scalar results as u64.
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
        put(_svm512_set1_epi8(x as i8));
        put(_svm512_set1_epi16(x as i16));
        put(_svm512_set1_epi32(x as i32));
        put(_svm512_set1_epi64(x));
        put(_svm512_setzero_si512());
        put(_svm512_add_epi8(a, b));
        put(_svm512_add_epi16(a, b));
        put(_svm512_add_epi32(a, b));
        put(_svm512_add_epi64(a, b));
        put(_svm512_sub_epi8(a, b));
        put(_svm512_sub_epi16(a, b));
        put(_svm512_sub_epi32(a, b));
        put(_svm512_sub_epi64(a, b));
        put(_svm512_mullo_epi16(a, b));
        put(_svm512_mullo_epi32(a, b));
        put(_svm512_mullo_epi64(a, b));
        put(_svm512_mul_epu32(a, b));
        put(_svm512_mul_epi32(a, b));
        put(_svm512_mulhi_epi16(a, b));
        put(_svm512_mulhi_epu16(a, b));
        put(_svm512_max_epi32(a, b));
        put(_svm512_min_epi32(a, b));
        put(_svm512_max_epu32(a, b));
        put(_svm512_min_epu32(a, b));
        put(_svm512_max_epi16(a, b));
        put(_svm512_min_epi16(a, b));
        put(_svm512_max_epu64(a, b));
        put(_svm512_min_epu64(a, b));
        put(_svm512_and_si512(a, b));
        put(_svm512_or_si512(a, b));
        put(_svm512_xor_si512(a, b));
        put(_svm512_andnot_si512(a, b));
        put(_svm512_max_epu16(a, b));
        put(_svm512_min_epu16(a, b));
        put(_svm512_ternarylogic_epi64::<0x96>(a, b, c));
        put(_svm512_slli_epi16::<3>(a));
        put(_svm512_slli_epi32::<7>(a));
        put(_svm512_slli_epi64::<33>(a));
        put(_svm512_srli_epi16::<3>(a));
        put(_svm512_srli_epi32::<7>(a));
        put(_svm512_srli_epi64::<33>(a));
        put(_svm512_srai_epi16::<3>(a));
        put(_svm512_srai_epi32::<7>(a));
        put(_svm512_srai_epi64::<33>(a));
        put(_svm512_rol_epi32::<7>(a));
        put(_svm512_rol_epi64::<33>(a));
        put(_svm512_sllv_epi32(a, c));
        put(_svm512_sllv_epi64(a, c));
        put(_svm512_srlv_epi32(a, c));
        put(_svm512_srlv_epi64(a, c));
        put(_svm512_rolv_epi32(a, c));
        put(_svm512_rolv_epi64(a, c));
        put(_svm512_cmpeq_epi8(a, b));
        put(_svm512_cmpeq_epi32(a, b));
        put(_svm512_cmpeq_epi64(a, b));
        put(_svm512_cmpgt_epi32(a, b));
        put(_svm512_cmpgt_epi64(a, b));
        put(_svm512_cmpgt_epu64(a, b));
        put(_svm512_dpbusd_epi32(c, a, b));
        put(_svm512_dpwssd_epi32(c, a, b));
        put(_svm512_maddubs_epi16(a, b));
        put(_svm512_madd_epi16(a, b));
        put(_svm512_madd52lo_epu64(c, a, b));
        put(_svm512_madd52hi_epu64(c, a, b));
        put(_svm512_packs_epi32(a, b));
        put(_svm512_packs_epi16(a, b));
        put(_svm512_packus_epi16(a, b));
        put(_svm512_permutexvar_epi32(c, a));
        put(_svm512_permutexvar_epi64(c, a));
        put(_svm512_shuffle_epi8(a, b));
        put(_svm512_permutex2var_epi16(a, c, b));
        put(_svm512_cvtusepi32_epi8(a));
        put(_svm512_cvtsepi32_epi8(a));
        put(_svm512_cvtsepi32_epi16(a));
        put(_svm512_cvtepi32_epi8(a));
        put(_svm512_cvtepu8_epi32(a));
        put(_svm512_cvtepi8_epi32(a));
        put(_svm512_cvtepu16_epi32(a));
        put(_svm512_cvtepi16_epi32(a));
        put(_svm512_cvtepu32_epi64(a));
        put(_svm512_cvtepi32_epi64(a));
        put(_svm512_popcnt_epi8(a));
        put(_svm512_popcnt_epi64(a));
        put(_svm512_insert_epi64::<6>(a, x));
        put(_svm512_clmulepi64_epi128::<0x10>(a, b));
        let s = out.cast::<u64>();
        s.write_unaligned(_svm512_reduce_add_epi32(a) as u32 as u64);
        s.add(1).write_unaligned(_svm512_reduce_add_epi64(a) as u64);
        s.add(2).write_unaligned(_svm512_extract_epi64::<5>(a) as u64);
        s.add(3).write_unaligned(_svm512_extract_epi32::<11>(a) as u32 as u64);
        s.add(4).write_unaligned(_svm512_movepi8_mask(a));
    }
    0
}
