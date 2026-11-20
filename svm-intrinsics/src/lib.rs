//! 512-bit vector intrinsics for sBPF, in the style of `core::arch`.
//!
//! Every intrinsic is an `#[inline(always)]` wrapper around an `extern "C"`
//! marker (`__svm_<op>`). The markers are never defined: the vector pass in
//! sbpf-linker allocates `v0`–`v15` over the whole program and replaces each
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
    fn __svm_splat_d(x: u64) -> u32;
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

/// `splat.d` (0x06): broadcasts `x` to all 16 u32 lanes.
#[inline(always)]
pub fn _svm512_set1_epi32(x: i32) -> __svm512i {
    __svm512i(unsafe { __svm_splat_d(x as u64) })
}

/// `zero` (0x08): all bits zero.
#[inline(always)]
pub fn _svm512_setzero_si512() -> __svm512i {
    __svm512i(unsafe { __svm_zero() })
}

// ---- integer arithmetic, logic, permutes (kind `r3`) ----

r3! {
    /// `add.w` (0x11): wrapping add, 32×u16.
    _svm512_add_epi16 => __svm_add_w;
    /// `add.q` (0x13): wrapping add, 8×u64.
    _svm512_add_epi64 => __svm_add_q;
    /// `sub.w` (0x15): wrapping `a - b`, 32×u16.
    _svm512_sub_epi16 => __svm_sub_w;
    /// `sub.q` (0x17): wrapping `a - b`, 8×u64.
    _svm512_sub_epi64 => __svm_sub_q;
    /// `mullo.w` (0x18): low 16 bits of the product, 32×u16.
    _svm512_mullo_epi16 => __svm_mullo_w;
    /// `mullo.q` (0x1a): low 64 bits of the product, 8×u64.
    _svm512_mullo_epi64 => __svm_mullo_q;
    /// `mulhi.w` (0x1d): high 16 bits of the signed product, 32×i16.
    _svm512_mulhi_epi16 => __svm_mulhi_w;
    /// `max.d` (0x20): signed max, 16×i32.
    _svm512_max_epi32 => __svm_max_d;
    /// `min.d` (0x21): signed min, 16×i32.
    _svm512_min_epi32 => __svm_min_d;
    /// `minu.q` (0x27): unsigned min, 8×u64.
    _svm512_min_epu64 => __svm_minu_q;
    /// `and` (0x28): `a & b`.
    _svm512_and_si512 => __svm_and;
    /// `xor` (0x2a): `a ^ b`.
    _svm512_xor_si512 => __svm_xor;
    /// `minu.w` (0x2f): unsigned min, 32×u16.
    _svm512_min_epu16 => __svm_minu_w;
    /// `rolv.q` (0x40): per-lane rotate left by `b mod 64`.
    _svm512_rolv_epi64 => __svm_rolv_q;
    /// `maddwd` (0x53): `d[i] ← Σ s16(a) · s16(b)` over word pairs, wrapping.
    _svm512_madd_epi16 => __svm_maddwd;
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
    /// `perm2.w` (0x5e): `w[i] ← (a ‖ b).w[idx.w[i] & 63]`, `a` is words 0..32
    /// (arguments `a, idx, b` as in `_mm512_permutex2var_epi16`).
    _svm512_permutex2var_epi16 => __svm_perm2_w;
    /// `perm2.q` (0x5f): `q[i] ← (a ‖ b).q[idx.q[i] & 15]`, `a` is qwords 0..8
    /// (arguments `a, idx, b` as in `_mm512_permutex2var_epi64`).
    _svm512_permutex2var_epi64 => __svm_perm2_q;
}

unsafe extern "C" {
    fn __svm_ternlog(a: u32, b: u32, c: u32, imm: u32) -> u32;
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

// ---- shifts by immediate (kind `r2i`) ----

r2i! {
    /// `shli.q` (0x32): shift left by `IMM` (< 64), 8×u64.
    _svm512_slli_epi64 => __svm_shli_q < 64;
    /// `shri.q` (0x35): logical shift right by `IMM` (< 64), 8×u64.
    _svm512_srli_epi64 => __svm_shri_q < 64;
    /// `sari.d` (0x37): arithmetic shift right by `IMM` (< 32), 16×i32.
    _svm512_srai_epi32 => __svm_sari_d < 32;
    /// `sari.q` (0x38): arithmetic shift right by `IMM` (< 64), 8×i64.
    _svm512_srai_epi64 => __svm_sari_q < 64;
    /// `roli.q` (0x3a): rotate left by `IMM` (< 64), 8×u64.
    _svm512_rol_epi64 => __svm_roli_q < 64;
}

// ---- one operand (kind `r2`) ----

r2! {
    /// `narrowus.d.b` (0x60): `b[i] ← min(u32(a.d[i]), 255)` for i < 16, rest 0.
    _svm512_cvtusepi32_epi8 => __svm_narrowus_d_b;
    /// `abs.q` (0xb1): `|a.q[i]|`, 8×i64 (`i64::MIN` stays `i64::MIN`).
    _svm512_abs_epi64 => __svm_abs_q;
}
