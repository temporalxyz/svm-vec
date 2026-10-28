//! 512-bit vector intrinsics for sBPF, in the style of `core::arch`.
//!
//! Every intrinsic is an `#[inline(always)]` wrapper around an `extern "C"`
//! marker (`__svm_<op>`). The markers are never defined: the vector pass in
//! sbpf-linker allocates `v0`–`v29` over the whole program and replaces each
//! marker call with the encoded vector instruction. A program linked without
//! that pass fails with undefined `__svm_*` symbols.
//!
//! A [`__svm512i`] is an opaque SSA token naming a vector value, not a
//! register number and not memory. It must only flow between intrinsics (and
//! through `let`, `if`/`match` and loop variables). Storing one to memory,
//! passing it to or returning it from a non-inlined function is rejected by
//! the pass. See `SPEC.md` for the semantics of each op.
#![no_std]
#![allow(non_camel_case_types)]

/// A 512-bit vector value (16×u32, 8×u64, 32×u16 or 64×u8 lanes).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct __svm512i(u32);

/// Two-operand ops: `out ← op(a, b)` (kind `r3`).
macro_rules! r3 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, b: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __svm512i, b: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(a.0, b.0) })
        }
    )*};
}

/// Accumulating ops: `acc ← op(acc, a, b)` (kind `a3`).
macro_rules! a3 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident;)*) => {$(
        unsafe extern "C" { fn $marker(acc: u32, a: u32, b: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(acc: __svm512i, a: __svm512i, b: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(acc.0, a.0, b.0) })
        }
    )*};
}

/// One-operand ops: `out ← op(a)` (kind `r2`).
macro_rules! r2 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(a.0) })
        }
    )*};
}

/// One-operand ops with an immediate below `$lim`: `out ← op(a, IMM)`
/// (kind `r2i`).
macro_rules! r2i {
    ($($(#[$doc:meta])* $name:ident => $marker:ident < $lim:literal;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, imm: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const IMM: u32>(a: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            __svm512i(unsafe { $marker(a.0, IMM) })
        }
    )*};
}

// ---- memory and broadcast (kind `ls` / `st`) ----

unsafe extern "C" {
    fn __svm_ld(p: *const u8) -> u32;
    fn __svm_st(a: u32, p: *mut u8);
    fn __svm_ldbc_d(p: *const u8) -> u32;
    fn __svm_ldbc_q(p: *const u8) -> u32;
    fn __svm_splat_b(x: u64) -> u32;
    fn __svm_splat_w(x: u64) -> u32;
    fn __svm_splat_d(x: u64) -> u32;
    fn __svm_splat_q(x: u64) -> u32;
    fn __svm_zero() -> u32;
}

/// `ld` (0x00): loads 64 bytes from `p` (unaligned).
///
/// # Safety
/// `p..p+64` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadu_si512(p: *const u8) -> __svm512i {
    __svm512i(unsafe { __svm_ld(p) })
}

/// `st` (0x01): stores 64 bytes of `a` to `p` (unaligned).
///
/// # Safety
/// `p..p+64` must be writable.
#[inline(always)]
pub unsafe fn _svm512_storeu_si512(p: *mut u8, a: __svm512i) {
    unsafe { __svm_st(a.0, p) }
}

/// `ldbc.d` (0x02): broadcasts the u32 at `p` to all 16 lanes.
///
/// # Safety
/// `p..p+4` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_epi32(p: *const i32) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_d(p.cast()) })
}

/// `ldbc.q` (0x03): broadcasts the u64 at `p` to all 8 lanes.
///
/// # Safety
/// `p..p+8` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_epi64(p: *const i64) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_q(p.cast()) })
}

/// `splat.b` (0x04): broadcasts `x` to all 64 byte lanes.
#[inline(always)]
pub fn _svm512_set1_epi8(x: i8) -> __svm512i {
    __svm512i(unsafe { __svm_splat_b(x as u64) })
}

/// `splat.w` (0x05): broadcasts `x` to all 32 u16 lanes.
#[inline(always)]
pub fn _svm512_set1_epi16(x: i16) -> __svm512i {
    __svm512i(unsafe { __svm_splat_w(x as u64) })
}

/// `splat.d` (0x06): broadcasts `x` to all 16 u32 lanes.
#[inline(always)]
pub fn _svm512_set1_epi32(x: i32) -> __svm512i {
    __svm512i(unsafe { __svm_splat_d(x as u64) })
}

/// `splat.q` (0x07): broadcasts `x` to all 8 u64 lanes.
#[inline(always)]
pub fn _svm512_set1_epi64(x: i64) -> __svm512i {
    __svm512i(unsafe { __svm_splat_q(x as u64) })
}

/// `zero` (0x08): all bits zero.
#[inline(always)]
pub fn _svm512_setzero_si512() -> __svm512i {
    __svm512i(unsafe { __svm_zero() })
}

// ---- integer arithmetic, logic, compares (kind `r3`) ----

r3! {
    /// `add.b` (0x10): wrapping add, 64×u8.
    _svm512_add_epi8 => __svm_add_b;
    /// `add.w` (0x11): wrapping add, 32×u16.
    _svm512_add_epi16 => __svm_add_w;
    /// `add.d` (0x12): wrapping add, 16×u32.
    _svm512_add_epi32 => __svm_add_d;
    /// `add.q` (0x13): wrapping add, 8×u64.
    _svm512_add_epi64 => __svm_add_q;
    /// `sub.b` (0x14): wrapping `a - b`, 64×u8.
    _svm512_sub_epi8 => __svm_sub_b;
    /// `sub.w` (0x15): wrapping `a - b`, 32×u16.
    _svm512_sub_epi16 => __svm_sub_w;
    /// `sub.d` (0x16): wrapping `a - b`, 16×u32.
    _svm512_sub_epi32 => __svm_sub_d;
    /// `sub.q` (0x17): wrapping `a - b`, 8×u64.
    _svm512_sub_epi64 => __svm_sub_q;
    /// `mullo.w` (0x18): low 16 bits of the product, 32×u16.
    _svm512_mullo_epi16 => __svm_mullo_w;
    /// `mullo.d` (0x19): low 32 bits of the product, 16×u32.
    _svm512_mullo_epi32 => __svm_mullo_d;
    /// `mullo.q` (0x1a): low 64 bits of the product, 8×u64.
    _svm512_mullo_epi64 => __svm_mullo_q;
    /// `mulwu` (0x1b): `q[i] ← u32(a.d[2i]) · u32(b.d[2i])`.
    _svm512_mul_epu32 => __svm_mulwu;
    /// `mulws` (0x1c): `q[i] ← i32(a.d[2i]) · i32(b.d[2i])`.
    _svm512_mul_epi32 => __svm_mulws;
    /// `mulhi.w` (0x1d): high 16 bits of the signed product, 32×i16.
    _svm512_mulhi_epi16 => __svm_mulhi_w;
    /// `mulhiu.w` (0x1e): high 16 bits of the unsigned product, 32×u16.
    _svm512_mulhi_epu16 => __svm_mulhiu_w;
    /// `max.d` (0x20): signed max, 16×i32.
    _svm512_max_epi32 => __svm_max_d;
    /// `min.d` (0x21): signed min, 16×i32.
    _svm512_min_epi32 => __svm_min_d;
    /// `maxu.d` (0x22): unsigned max, 16×u32.
    _svm512_max_epu32 => __svm_maxu_d;
    /// `minu.d` (0x23): unsigned min, 16×u32.
    _svm512_min_epu32 => __svm_minu_d;
    /// `max.w` (0x24): signed max, 32×i16.
    _svm512_max_epi16 => __svm_max_w;
    /// `min.w` (0x25): signed min, 32×i16.
    _svm512_min_epi16 => __svm_min_w;
    /// `maxu.q` (0x26): unsigned max, 8×u64.
    _svm512_max_epu64 => __svm_maxu_q;
    /// `minu.q` (0x27): unsigned min, 8×u64.
    _svm512_min_epu64 => __svm_minu_q;
    /// `and` (0x28): `a & b`.
    _svm512_and_si512 => __svm_and;
    /// `or` (0x29): `a | b`.
    _svm512_or_si512 => __svm_or;
    /// `xor` (0x2a): `a ^ b`.
    _svm512_xor_si512 => __svm_xor;
    /// `andn` (0x2b): `!a & b`.
    _svm512_andnot_si512 => __svm_andn;
    /// `maxu.w` (0x2e): unsigned max, 32×u16.
    _svm512_max_epu16 => __svm_maxu_w;
    /// `minu.w` (0x2f): unsigned min, 32×u16.
    _svm512_min_epu16 => __svm_minu_w;
    /// `shlv.d` (0x3b): per-lane `a << b`, counts ≥ 32 give 0.
    _svm512_sllv_epi32 => __svm_shlv_d;
    /// `shlv.q` (0x3c): per-lane `a << b`, counts ≥ 64 give 0.
    _svm512_sllv_epi64 => __svm_shlv_q;
    /// `shrv.d` (0x3d): per-lane logical `a >> b`, counts ≥ 32 give 0.
    _svm512_srlv_epi32 => __svm_shrv_d;
    /// `shrv.q` (0x3e): per-lane logical `a >> b`, counts ≥ 64 give 0.
    _svm512_srlv_epi64 => __svm_shrv_q;
    /// `rolv.d` (0x3f): per-lane rotate left by `b mod 32`.
    _svm512_rolv_epi32 => __svm_rolv_d;
    /// `rolv.q` (0x40): per-lane rotate left by `b mod 64`.
    _svm512_rolv_epi64 => __svm_rolv_q;
    /// `cmpeq.b` (0x48): lane all ones if equal, else 0, 64×u8.
    _svm512_cmpeq_epi8 => __svm_cmpeq_b;
    /// `cmpeq.d` (0x49): lane all ones if equal, else 0, 16×u32.
    _svm512_cmpeq_epi32 => __svm_cmpeq_d;
    /// `cmpeq.q` (0x4a): lane all ones if equal, else 0, 8×u64.
    _svm512_cmpeq_epi64 => __svm_cmpeq_q;
    /// `cmpgt.d` (0x4b): lane all ones if signed `a > b`, 16×i32.
    _svm512_cmpgt_epi32 => __svm_cmpgt_d;
    /// `cmpgt.q` (0x4c): lane all ones if signed `a > b`, 8×i64.
    _svm512_cmpgt_epi64 => __svm_cmpgt_q;
    /// `cmpgtu.q` (0x4d): lane all ones if unsigned `a > b`, 8×u64.
    _svm512_cmpgt_epu64 => __svm_cmpgtu_q;
    /// `maddubs` (0x52): `w[i] ← sat16(Σ u8(a) · s8(b))` over byte pairs.
    _svm512_maddubs_epi16 => __svm_maddubs;
    /// `maddwd` (0x53): `d[i] ← Σ s16(a) · s16(b)` over word pairs, wrapping.
    _svm512_madd_epi16 => __svm_maddwd;
    /// `packs.d` (0x58): per 128-bit lane `[sat16(a.d×4), sat16(b.d×4)]`.
    _svm512_packs_epi32 => __svm_packs_d;
    /// `packs.w` (0x59): per 128-bit lane `[sat8(a.w×8), sat8(b.w×8)]`.
    _svm512_packs_epi16 => __svm_packs_w;
    /// `packus.w` (0x5a): like `packs.w` with unsigned saturation.
    _svm512_packus_epi16 => __svm_packus_w;
    /// `permd` (0x5b): `d[i] ← a.d[idx.d[i] & 15]` (first argument is `idx`).
    _svm512_permutexvar_epi32 => __svm_permd;
    /// `permq` (0x5c): `q[i] ← a.q[idx.q[i] & 7]` (first argument is `idx`).
    _svm512_permutexvar_epi64 => __svm_permq;
    /// `shufb` (0x5d): per 128-bit lane byte shuffle of `a` by `b`
    /// (index bit 7 gives 0).
    _svm512_shuffle_epi8 => __svm_shufb;
}

// ---- accumulating ops (kind `a3` / `a3i`) ----

a3! {
    /// `dpbusd` (0x50): `acc.d[i] += Σj<4 u8(a.b[4i+j]) · s8(b.b[4i+j])`, wrapping.
    _svm512_dpbusd_epi32 => __svm_dpbusd;
    /// `dpwssd` (0x51): `acc.d[i] += Σj<2 s16(a.w[2i+j]) · s16(b.w[2i+j])`, wrapping.
    _svm512_dpwssd_epi32 => __svm_dpwssd;
    /// `madd52lo` (0x54): `acc.q[i] += low52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_madd52lo_epu64 => __svm_madd52lo;
    /// `madd52hi` (0x55): `acc.q[i] += high52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_madd52hi_epu64 => __svm_madd52hi;
    /// `perm2.w` (0x5e): `w[i] ← (a ‖ b).w[idx.w[i] & 63]`, `a` is words 0..32
    /// (arguments `a, idx, b` as in `_mm512_permutex2var_epi16`).
    _svm512_permutex2var_epi16 => __svm_perm2_w;
}

unsafe extern "C" {
    fn __svm_ternlog(a: u32, b: u32, c: u32, imm: u32) -> u32;
    fn __svm_clmul(a: u32, b: u32, imm: u32) -> u32;
}

/// `ternlog` (0x2c): bitwise `IMM[(a << 2) | (b << 1) | c]` for each bit.
#[inline(always)]
pub fn _svm512_ternarylogic_epi64<const IMM: u32>(
    a: __svm512i,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    __svm512i(unsafe { __svm_ternlog(a.0, b.0, c.0, IMM) })
}

/// `clmul` (0x78): per 128-bit lane carry-less multiply of
/// `a.q[IMM & 1]` and `b.q[IMM >> 4 & 1]`. `IMM` ∈ {0x00, 0x01, 0x10, 0x11}.
#[inline(always)]
pub fn _svm512_clmulepi64_epi128<const IMM: u32>(
    a: __svm512i,
    b: __svm512i,
) -> __svm512i {
    const { assert!(IMM & !0x11 == 0, "immediate must be 0x00, 0x01, 0x10 or 0x11") };
    __svm512i(unsafe { __svm_clmul(a.0, b.0, IMM) })
}

// ---- shifts by immediate (kind `r2i`) ----

r2i! {
    /// `shli.w` (0x30): shift left by `IMM` (< 16), 32×u16.
    _svm512_slli_epi16 => __svm_shli_w < 16;
    /// `shli.d` (0x31): shift left by `IMM` (< 32), 16×u32.
    _svm512_slli_epi32 => __svm_shli_d < 32;
    /// `shli.q` (0x32): shift left by `IMM` (< 64), 8×u64.
    _svm512_slli_epi64 => __svm_shli_q < 64;
    /// `shri.w` (0x33): logical shift right by `IMM` (< 16), 32×u16.
    _svm512_srli_epi16 => __svm_shri_w < 16;
    /// `shri.d` (0x34): logical shift right by `IMM` (< 32), 16×u32.
    _svm512_srli_epi32 => __svm_shri_d < 32;
    /// `shri.q` (0x35): logical shift right by `IMM` (< 64), 8×u64.
    _svm512_srli_epi64 => __svm_shri_q < 64;
    /// `sari.w` (0x36): arithmetic shift right by `IMM` (< 16), 32×i16.
    _svm512_srai_epi16 => __svm_sari_w < 16;
    /// `sari.d` (0x37): arithmetic shift right by `IMM` (< 32), 16×i32.
    _svm512_srai_epi32 => __svm_sari_d < 32;
    /// `sari.q` (0x38): arithmetic shift right by `IMM` (< 64), 8×i64.
    _svm512_srai_epi64 => __svm_sari_q < 64;
    /// `roli.d` (0x39): rotate left by `IMM` (< 32), 16×u32.
    _svm512_rol_epi32 => __svm_roli_d < 32;
    /// `roli.q` (0x3a): rotate left by `IMM` (< 64), 8×u64.
    _svm512_rol_epi64 => __svm_roli_q < 64;
}

// ---- conversions and popcount (kind `r2`) ----

r2! {
    /// `narrowus.d.b` (0x60): `b[i] ← min(u32(a.d[i]), 255)` for i < 16, rest 0.
    _svm512_cvtusepi32_epi8 => __svm_narrowus_d_b;
    /// `narrows.d.b` (0x61): `b[i] ← sat8(i32(a.d[i]))` for i < 16, rest 0.
    _svm512_cvtsepi32_epi8 => __svm_narrows_d_b;
    /// `narrows.d.w` (0x62): `w[i] ← sat16(i32(a.d[i]))` for i < 16, rest 0.
    _svm512_cvtsepi32_epi16 => __svm_narrows_d_w;
    /// `narrow.d.b` (0x63): `b[i] ← a.d[i] as u8` for i < 16, rest 0.
    _svm512_cvtepi32_epi8 => __svm_narrow_d_b;
    /// `zx.b.d` (0x64): `d[i] ← u8(a.b[i])`, i < 16.
    _svm512_cvtepu8_epi32 => __svm_zx_b_d;
    /// `sx.b.d` (0x65): `d[i] ← i8(a.b[i])`, i < 16.
    _svm512_cvtepi8_epi32 => __svm_sx_b_d;
    /// `zx.w.d` (0x66): `d[i] ← u16(a.w[i])`, i < 16.
    _svm512_cvtepu16_epi32 => __svm_zx_w_d;
    /// `sx.w.d` (0x67): `d[i] ← i16(a.w[i])`, i < 16.
    _svm512_cvtepi16_epi32 => __svm_sx_w_d;
    /// `zx.d.q` (0x68): `q[i] ← u32(a.d[i])`, i < 8.
    _svm512_cvtepu32_epi64 => __svm_zx_d_q;
    /// `sx.d.q` (0x69): `q[i] ← i32(a.d[i])`, i < 8.
    _svm512_cvtepi32_epi64 => __svm_sx_d_q;
    /// `popcnt.b` (0x6a): population count per byte.
    _svm512_popcnt_epi8 => __svm_popcnt_b;
    /// `popcnt.q` (0x6b): population count per u64.
    _svm512_popcnt_epi64 => __svm_popcnt_q;
}

// ---- vector to scalar and scalar insert (kinds `sr`, `sri`, `ai`) ----

unsafe extern "C" {
    fn __svm_hsum_d(a: u32) -> u64;
    fn __svm_hsum_q(a: u32) -> u64;
    fn __svm_movemask_b(a: u32) -> u64;
    fn __svm_extract_q(a: u32, imm: u32) -> u64;
    fn __svm_extract_d(a: u32, imm: u32) -> u64;
    fn __svm_insert_q(a: u32, x: u64, imm: u32) -> u32;
}

/// `hsum.d` (0x70): wrapping sum of the 16 u32 lanes.
#[inline(always)]
pub fn _svm512_reduce_add_epi32(a: __svm512i) -> i32 {
    unsafe { __svm_hsum_d(a.0) as i32 }
}

/// `hsum.q` (0x71): wrapping sum of the 8 u64 lanes.
#[inline(always)]
pub fn _svm512_reduce_add_epi64(a: __svm512i) -> i64 {
    unsafe { __svm_hsum_q(a.0) as i64 }
}

/// `movemask.b` (0x75): bit `i` of the result is the sign bit of byte `i`.
#[inline(always)]
pub fn _svm512_movepi8_mask(a: __svm512i) -> u64 {
    unsafe { __svm_movemask_b(a.0) }
}

/// `extract.q` (0x72): u64 lane `IMM` (< 8).
#[inline(always)]
pub fn _svm512_extract_epi64<const IMM: u32>(a: __svm512i) -> i64 {
    const { assert!(IMM < 8, "immediate out of range") };
    unsafe { __svm_extract_q(a.0, IMM) as i64 }
}

/// `extract.d` (0x73): u32 lane `IMM` (< 16).
#[inline(always)]
pub fn _svm512_extract_epi32<const IMM: u32>(a: __svm512i) -> i32 {
    const { assert!(IMM < 16, "immediate out of range") };
    unsafe { __svm_extract_d(a.0, IMM) as i32 }
}

/// `insert.q` (0x74): `a` with u64 lane `IMM` (< 8) replaced by `x`.
#[inline(always)]
pub fn _svm512_insert_epi64<const IMM: u32>(a: __svm512i, x: i64) -> __svm512i {
    const { assert!(IMM < 8, "immediate out of range") };
    __svm512i(unsafe { __svm_insert_q(a.0, x as u64, IMM) })
}
