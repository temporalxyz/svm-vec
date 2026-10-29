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
//!
//! Masked forms follow `core::arch`: `_svm512_mask_<op>(src, k, ...)` keeps
//! the lanes of `src` whose bit of `k` is clear (merge), and
//! `_svm512_maskz_<op>(k, ...)` sets them to 0. For accumulating ops the
//! accumulator is the merge source. Bit i of `k` is lane i at the op's mask
//! granularity. They call the markers `__svm_<op>_mask(src, ..., k)` and
//! `__svm_<op>_maskz(..., k)`, the mask after the other arguments and before
//! the immediate.
#![no_std]
#![allow(non_camel_case_types)]

/// A 512-bit vector value (16×u32, 8×u64, 32×u16 or 64×u8 lanes).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct __svm512i(u32);

/// A mask of 8 lanes (bit i = lane i).
pub type __mmask8 = u8;
/// A mask of 16 lanes.
pub type __mmask16 = u16;
/// A mask of 32 lanes.
pub type __mmask32 = u32;
/// A mask of 64 lanes.
pub type __mmask64 = u64;

macro_rules! merge_doc {
    ($name:ident) => {
        concat!(
            "[`",
            stringify!($name),
            "`] with merge masking: lanes whose bit of `k` is clear keep `src`."
        )
    };
}

macro_rules! zero_doc {
    ($name:ident) => {
        concat!(
            "[`",
            stringify!($name),
            "`] with zero masking: lanes whose bit of `k` is clear are 0."
        )
    };
}

/// Two-operand ops: `out ← op(a, b)` (kind `r3`), optionally with masked
/// forms.
macro_rules! r3 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident $(, $mask:ident / $maskz:ident: $k:ty)?;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, b: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __svm512i, b: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(a.0, b.0) })
        }
        $(
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, a: u32, b: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, b.0, k as u64) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(a: u32, b: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(a.0, b.0, k as u64) })
        }
        )?
    )*};
}

/// Accumulating ops: `acc ← op(acc, a, b)` (kind `a3`). The masked forms
/// merge into (`_mask`) or zero (`_maskz`) the lanes of `acc`.
macro_rules! a3 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident $(, $mask:ident / $maskz:ident: $k:ty)?;)*) => {$(
        unsafe extern "C" { fn $marker(acc: u32, a: u32, b: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(acc: __svm512i, a: __svm512i, b: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(acc.0, a.0, b.0) })
        }
        $(
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(acc: u32, a: u32, b: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, b.0, k as u64) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, acc: __svm512i, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(acc: u32, a: u32, b: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(acc.0, a.0, b.0, k as u64) })
        }
        )?
    )*};
}

/// One-operand ops: `out ← op(a)` (kind `r2`), optionally with masked
/// forms.
macro_rules! r2 {
    ($($(#[$doc:meta])* $name:ident => $marker:ident $(, $mask:ident / $maskz:ident: $k:ty)?;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __svm512i) -> __svm512i {
            __svm512i(unsafe { $marker(a.0) })
        }
        $(
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, a: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, k as u64) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(a: u32, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(a.0, k as u64) })
        }
        )?
    )*};
}

/// One-operand ops with an immediate below `$lim`: `out ← op(a, IMM)`
/// (kind `r2i`), with masked forms.
macro_rules! r2i {
    ($($(#[$doc:meta])* $name:ident => $marker:ident < $lim:literal, $mask:ident / $maskz:ident: $k:ty;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, imm: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const IMM: u32>(a: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            __svm512i(unsafe { $marker(a.0, IMM) })
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask<const IMM: u32>(src: __svm512i, k: $k, a: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, k as u64, IMM) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz<const IMM: u32>(k: $k, a: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, k as u64, IMM) })
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

macro_rules! splat_masked {
    ($($name:ident => $mask:ident / $maskz:ident: $k:ty, $t:ty, $marker:literal;)*) => {$(
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, x: $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(src: u32, x: u64, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(src.0, x as u64, k as u64) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, x: $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_maskz")]
                fn marker(x: u64, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(x as u64, k as u64) })
        }
    )*};
}

splat_masked! {
    _svm512_set1_epi8 => _svm512_mask_set1_epi8 / _svm512_maskz_set1_epi8: __mmask64, i8, "__svm_splat_b";
    _svm512_set1_epi16 => _svm512_mask_set1_epi16 / _svm512_maskz_set1_epi16: __mmask32, i16, "__svm_splat_w";
    _svm512_set1_epi32 => _svm512_mask_set1_epi32 / _svm512_maskz_set1_epi32: __mmask16, i32, "__svm_splat_d";
    _svm512_set1_epi64 => _svm512_mask_set1_epi64 / _svm512_maskz_set1_epi64: __mmask8, i64, "__svm_splat_q";
}

macro_rules! loadbc_masked {
    ($($name:ident => $mask:ident / $maskz:ident: $k:ty, $t:ty, $bytes:literal, $marker:literal;)*) => {$(
        #[doc = merge_doc!($name)]
        ///
        /// # Safety
        #[doc = concat!("`p..p+", $bytes, "` must be readable.")]
        #[inline(always)]
        pub unsafe fn $mask(src: __svm512i, k: $k, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(src: u32, p: *const u8, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(src.0, p.cast(), k as u64) })
        }
        #[doc = zero_doc!($name)]
        ///
        /// # Safety
        #[doc = concat!("`p..p+", $bytes, "` must be readable.")]
        #[inline(always)]
        pub unsafe fn $maskz(k: $k, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_maskz")]
                fn marker(p: *const u8, k: u64) -> u32;
            }
            __svm512i(unsafe { marker(p.cast(), k as u64) })
        }
    )*};
}

loadbc_masked! {
    _svm512_loadbc_epi32 => _svm512_mask_loadbc_epi32 / _svm512_maskz_loadbc_epi32: __mmask16, i32, 4, "__svm_ldbc_d";
    _svm512_loadbc_epi64 => _svm512_mask_loadbc_epi64 / _svm512_maskz_loadbc_epi64: __mmask8, i64, 8, "__svm_ldbc_q";
}

/// `zero` (0x08): all bits zero.
#[inline(always)]
pub fn _svm512_setzero_si512() -> __svm512i {
    __svm512i(unsafe { __svm_zero() })
}

// ---- integer arithmetic, logic, compares (kind `r3`) ----

r3! {
    /// `add.b` (0x10): wrapping add, 64×u8.
    _svm512_add_epi8 => __svm_add_b,
        _svm512_mask_add_epi8 / _svm512_maskz_add_epi8: __mmask64;
    /// `add.w` (0x11): wrapping add, 32×u16.
    _svm512_add_epi16 => __svm_add_w,
        _svm512_mask_add_epi16 / _svm512_maskz_add_epi16: __mmask32;
    /// `add.d` (0x12): wrapping add, 16×u32.
    _svm512_add_epi32 => __svm_add_d,
        _svm512_mask_add_epi32 / _svm512_maskz_add_epi32: __mmask16;
    /// `add.q` (0x13): wrapping add, 8×u64.
    _svm512_add_epi64 => __svm_add_q,
        _svm512_mask_add_epi64 / _svm512_maskz_add_epi64: __mmask8;
    /// `sub.b` (0x14): wrapping `a - b`, 64×u8.
    _svm512_sub_epi8 => __svm_sub_b,
        _svm512_mask_sub_epi8 / _svm512_maskz_sub_epi8: __mmask64;
    /// `sub.w` (0x15): wrapping `a - b`, 32×u16.
    _svm512_sub_epi16 => __svm_sub_w,
        _svm512_mask_sub_epi16 / _svm512_maskz_sub_epi16: __mmask32;
    /// `sub.d` (0x16): wrapping `a - b`, 16×u32.
    _svm512_sub_epi32 => __svm_sub_d,
        _svm512_mask_sub_epi32 / _svm512_maskz_sub_epi32: __mmask16;
    /// `sub.q` (0x17): wrapping `a - b`, 8×u64.
    _svm512_sub_epi64 => __svm_sub_q,
        _svm512_mask_sub_epi64 / _svm512_maskz_sub_epi64: __mmask8;
    /// `mullo.w` (0x18): low 16 bits of the product, 32×u16.
    _svm512_mullo_epi16 => __svm_mullo_w,
        _svm512_mask_mullo_epi16 / _svm512_maskz_mullo_epi16: __mmask32;
    /// `mullo.d` (0x19): low 32 bits of the product, 16×u32.
    _svm512_mullo_epi32 => __svm_mullo_d,
        _svm512_mask_mullo_epi32 / _svm512_maskz_mullo_epi32: __mmask16;
    /// `mullo.q` (0x1a): low 64 bits of the product, 8×u64.
    _svm512_mullo_epi64 => __svm_mullo_q,
        _svm512_mask_mullo_epi64 / _svm512_maskz_mullo_epi64: __mmask8;
    /// `mulwu` (0x1b): `q[i] ← u32(a.d[2i]) · u32(b.d[2i])`.
    _svm512_mul_epu32 => __svm_mulwu,
        _svm512_mask_mul_epu32 / _svm512_maskz_mul_epu32: __mmask8;
    /// `mulws` (0x1c): `q[i] ← i32(a.d[2i]) · i32(b.d[2i])`.
    _svm512_mul_epi32 => __svm_mulws,
        _svm512_mask_mul_epi32 / _svm512_maskz_mul_epi32: __mmask8;
    /// `mulhi.w` (0x1d): high 16 bits of the signed product, 32×i16.
    _svm512_mulhi_epi16 => __svm_mulhi_w,
        _svm512_mask_mulhi_epi16 / _svm512_maskz_mulhi_epi16: __mmask32;
    /// `mulhiu.w` (0x1e): high 16 bits of the unsigned product, 32×u16.
    _svm512_mulhi_epu16 => __svm_mulhiu_w,
        _svm512_mask_mulhi_epu16 / _svm512_maskz_mulhi_epu16: __mmask32;
    /// `max.d` (0x20): signed max, 16×i32.
    _svm512_max_epi32 => __svm_max_d,
        _svm512_mask_max_epi32 / _svm512_maskz_max_epi32: __mmask16;
    /// `min.d` (0x21): signed min, 16×i32.
    _svm512_min_epi32 => __svm_min_d,
        _svm512_mask_min_epi32 / _svm512_maskz_min_epi32: __mmask16;
    /// `maxu.d` (0x22): unsigned max, 16×u32.
    _svm512_max_epu32 => __svm_maxu_d,
        _svm512_mask_max_epu32 / _svm512_maskz_max_epu32: __mmask16;
    /// `minu.d` (0x23): unsigned min, 16×u32.
    _svm512_min_epu32 => __svm_minu_d,
        _svm512_mask_min_epu32 / _svm512_maskz_min_epu32: __mmask16;
    /// `max.w` (0x24): signed max, 32×i16.
    _svm512_max_epi16 => __svm_max_w,
        _svm512_mask_max_epi16 / _svm512_maskz_max_epi16: __mmask32;
    /// `min.w` (0x25): signed min, 32×i16.
    _svm512_min_epi16 => __svm_min_w,
        _svm512_mask_min_epi16 / _svm512_maskz_min_epi16: __mmask32;
    /// `maxu.q` (0x26): unsigned max, 8×u64.
    _svm512_max_epu64 => __svm_maxu_q,
        _svm512_mask_max_epu64 / _svm512_maskz_max_epu64: __mmask8;
    /// `minu.q` (0x27): unsigned min, 8×u64.
    _svm512_min_epu64 => __svm_minu_q,
        _svm512_mask_min_epu64 / _svm512_maskz_min_epu64: __mmask8;
    /// `and` (0x28): `a & b`.
    _svm512_and_si512 => __svm_and,
        _svm512_mask_and_epi64 / _svm512_maskz_and_epi64: __mmask8;
    /// `or` (0x29): `a | b`.
    _svm512_or_si512 => __svm_or,
        _svm512_mask_or_epi64 / _svm512_maskz_or_epi64: __mmask8;
    /// `xor` (0x2a): `a ^ b`.
    _svm512_xor_si512 => __svm_xor,
        _svm512_mask_xor_epi64 / _svm512_maskz_xor_epi64: __mmask8;
    /// `andn` (0x2b): `!a & b`.
    _svm512_andnot_si512 => __svm_andn,
        _svm512_mask_andnot_epi64 / _svm512_maskz_andnot_epi64: __mmask8;
    /// `maxu.w` (0x2e): unsigned max, 32×u16.
    _svm512_max_epu16 => __svm_maxu_w,
        _svm512_mask_max_epu16 / _svm512_maskz_max_epu16: __mmask32;
    /// `minu.w` (0x2f): unsigned min, 32×u16.
    _svm512_min_epu16 => __svm_minu_w,
        _svm512_mask_min_epu16 / _svm512_maskz_min_epu16: __mmask32;
    /// `shlv.d` (0x3b): per-lane `a << b`, counts ≥ 32 give 0.
    _svm512_sllv_epi32 => __svm_shlv_d,
        _svm512_mask_sllv_epi32 / _svm512_maskz_sllv_epi32: __mmask16;
    /// `shlv.q` (0x3c): per-lane `a << b`, counts ≥ 64 give 0.
    _svm512_sllv_epi64 => __svm_shlv_q,
        _svm512_mask_sllv_epi64 / _svm512_maskz_sllv_epi64: __mmask8;
    /// `shrv.d` (0x3d): per-lane logical `a >> b`, counts ≥ 32 give 0.
    _svm512_srlv_epi32 => __svm_shrv_d,
        _svm512_mask_srlv_epi32 / _svm512_maskz_srlv_epi32: __mmask16;
    /// `shrv.q` (0x3e): per-lane logical `a >> b`, counts ≥ 64 give 0.
    _svm512_srlv_epi64 => __svm_shrv_q,
        _svm512_mask_srlv_epi64 / _svm512_maskz_srlv_epi64: __mmask8;
    /// `rolv.d` (0x3f): per-lane rotate left by `b mod 32`.
    _svm512_rolv_epi32 => __svm_rolv_d,
        _svm512_mask_rolv_epi32 / _svm512_maskz_rolv_epi32: __mmask16;
    /// `rolv.q` (0x40): per-lane rotate left by `b mod 64`.
    _svm512_rolv_epi64 => __svm_rolv_q,
        _svm512_mask_rolv_epi64 / _svm512_maskz_rolv_epi64: __mmask8;
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
    _svm512_maddubs_epi16 => __svm_maddubs,
        _svm512_mask_maddubs_epi16 / _svm512_maskz_maddubs_epi16: __mmask32;
    /// `maddwd` (0x53): `d[i] ← Σ s16(a) · s16(b)` over word pairs, wrapping.
    _svm512_madd_epi16 => __svm_maddwd,
        _svm512_mask_madd_epi16 / _svm512_maskz_madd_epi16: __mmask16;
    /// `packs.d` (0x58): per 128-bit lane `[sat16(a.d×4), sat16(b.d×4)]`.
    _svm512_packs_epi32 => __svm_packs_d,
        _svm512_mask_packs_epi32 / _svm512_maskz_packs_epi32: __mmask32;
    /// `packs.w` (0x59): per 128-bit lane `[sat8(a.w×8), sat8(b.w×8)]`.
    _svm512_packs_epi16 => __svm_packs_w,
        _svm512_mask_packs_epi16 / _svm512_maskz_packs_epi16: __mmask64;
    /// `packus.w` (0x5a): like `packs.w` with unsigned saturation.
    _svm512_packus_epi16 => __svm_packus_w,
        _svm512_mask_packus_epi16 / _svm512_maskz_packus_epi16: __mmask64;
    /// `permd` (0x5b): `d[i] ← a.d[idx.d[i] & 15]` (first argument is `idx`).
    _svm512_permutexvar_epi32 => __svm_permd,
        _svm512_mask_permutexvar_epi32 / _svm512_maskz_permutexvar_epi32: __mmask16;
    /// `permq` (0x5c): `q[i] ← a.q[idx.q[i] & 7]` (first argument is `idx`).
    _svm512_permutexvar_epi64 => __svm_permq,
        _svm512_mask_permutexvar_epi64 / _svm512_maskz_permutexvar_epi64: __mmask8;
    /// `shufb` (0x5d): per 128-bit lane byte shuffle of `a` by `b`
    /// (index bit 7 gives 0).
    _svm512_shuffle_epi8 => __svm_shufb,
        _svm512_mask_shuffle_epi8 / _svm512_maskz_shuffle_epi8: __mmask64;
}

// ---- accumulating ops (kind `a3` / `a3i`) ----

a3! {
    /// `dpbusd` (0x50): `acc.d[i] += Σj<4 u8(a.b[4i+j]) · s8(b.b[4i+j])`, wrapping.
    _svm512_dpbusd_epi32 => __svm_dpbusd,
        _svm512_mask_dpbusd_epi32 / _svm512_maskz_dpbusd_epi32: __mmask16;
    /// `dpwssd` (0x51): `acc.d[i] += Σj<2 s16(a.w[2i+j]) · s16(b.w[2i+j])`, wrapping.
    _svm512_dpwssd_epi32 => __svm_dpwssd,
        _svm512_mask_dpwssd_epi32 / _svm512_maskz_dpwssd_epi32: __mmask16;
    /// `madd52lo` (0x54): `acc.q[i] += low52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_madd52lo_epu64 => __svm_madd52lo,
        _svm512_mask_madd52lo_epu64 / _svm512_maskz_madd52lo_epu64: __mmask8;
    /// `madd52hi` (0x55): `acc.q[i] += high52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_madd52hi_epu64 => __svm_madd52hi,
        _svm512_mask_madd52hi_epu64 / _svm512_maskz_madd52hi_epu64: __mmask8;
    /// `perm2.w` (0x5e): `w[i] ← (a ‖ b).w[idx.w[i] & 63]`, `a` is words 0..32
    /// (arguments `a, idx, b` as in `_mm512_permutex2var_epi16`).
    _svm512_permutex2var_epi16 => __svm_perm2_w,
        _svm512_mask_permutex2var_epi16 / _svm512_maskz_permutex2var_epi16: __mmask32;
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

/// [`_svm512_ternarylogic_epi64`] with merge masking: lanes whose bit of `k`
/// is clear keep `src` (the first operand).
#[inline(always)]
pub fn _svm512_mask_ternarylogic_epi64<const IMM: u32>(
    src: __svm512i,
    k: __mmask8,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    unsafe extern "C" {
        fn __svm_ternlog_mask(a: u32, b: u32, c: u32, k: u64, imm: u32) -> u32;
    }
    __svm512i(unsafe { __svm_ternlog_mask(src.0, b.0, c.0, k as u64, IMM) })
}

/// [`_svm512_ternarylogic_epi64`] with zero masking: lanes whose bit of `k`
/// is clear are 0.
#[inline(always)]
pub fn _svm512_maskz_ternarylogic_epi64<const IMM: u32>(
    k: __mmask8,
    a: __svm512i,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    unsafe extern "C" {
        fn __svm_ternlog_maskz(a: u32, b: u32, c: u32, k: u64, imm: u32) -> u32;
    }
    __svm512i(unsafe { __svm_ternlog_maskz(a.0, b.0, c.0, k as u64, IMM) })
}

/// `clmul` (0x78): per 128-bit lane carry-less multiply of
/// `a.q[IMM & 1]` and `b.q[IMM >> 4 & 1]`. `IMM` ∈ {0x00, 0x01, 0x10, 0x11}.
#[inline(always)]
pub fn _svm512_clmulepi64_epi128<const IMM: u32>(a: __svm512i, b: __svm512i) -> __svm512i {
    const {
        assert!(
            IMM & !0x11 == 0,
            "immediate must be 0x00, 0x01, 0x10 or 0x11"
        )
    };
    __svm512i(unsafe { __svm_clmul(a.0, b.0, IMM) })
}

// ---- shifts by immediate (kind `r2i`) ----

r2i! {
    /// `shli.w` (0x30): shift left by `IMM` (< 16), 32×u16.
    _svm512_slli_epi16 => __svm_shli_w < 16,
        _svm512_mask_slli_epi16 / _svm512_maskz_slli_epi16: __mmask32;
    /// `shli.d` (0x31): shift left by `IMM` (< 32), 16×u32.
    _svm512_slli_epi32 => __svm_shli_d < 32,
        _svm512_mask_slli_epi32 / _svm512_maskz_slli_epi32: __mmask16;
    /// `shli.q` (0x32): shift left by `IMM` (< 64), 8×u64.
    _svm512_slli_epi64 => __svm_shli_q < 64,
        _svm512_mask_slli_epi64 / _svm512_maskz_slli_epi64: __mmask8;
    /// `shri.w` (0x33): logical shift right by `IMM` (< 16), 32×u16.
    _svm512_srli_epi16 => __svm_shri_w < 16,
        _svm512_mask_srli_epi16 / _svm512_maskz_srli_epi16: __mmask32;
    /// `shri.d` (0x34): logical shift right by `IMM` (< 32), 16×u32.
    _svm512_srli_epi32 => __svm_shri_d < 32,
        _svm512_mask_srli_epi32 / _svm512_maskz_srli_epi32: __mmask16;
    /// `shri.q` (0x35): logical shift right by `IMM` (< 64), 8×u64.
    _svm512_srli_epi64 => __svm_shri_q < 64,
        _svm512_mask_srli_epi64 / _svm512_maskz_srli_epi64: __mmask8;
    /// `sari.w` (0x36): arithmetic shift right by `IMM` (< 16), 32×i16.
    _svm512_srai_epi16 => __svm_sari_w < 16,
        _svm512_mask_srai_epi16 / _svm512_maskz_srai_epi16: __mmask32;
    /// `sari.d` (0x37): arithmetic shift right by `IMM` (< 32), 16×i32.
    _svm512_srai_epi32 => __svm_sari_d < 32,
        _svm512_mask_srai_epi32 / _svm512_maskz_srai_epi32: __mmask16;
    /// `sari.q` (0x38): arithmetic shift right by `IMM` (< 64), 8×i64.
    _svm512_srai_epi64 => __svm_sari_q < 64,
        _svm512_mask_srai_epi64 / _svm512_maskz_srai_epi64: __mmask8;
    /// `roli.d` (0x39): rotate left by `IMM` (< 32), 16×u32.
    _svm512_rol_epi32 => __svm_roli_d < 32,
        _svm512_mask_rol_epi32 / _svm512_maskz_rol_epi32: __mmask16;
    /// `roli.q` (0x3a): rotate left by `IMM` (< 64), 8×u64.
    _svm512_rol_epi64 => __svm_roli_q < 64,
        _svm512_mask_rol_epi64 / _svm512_maskz_rol_epi64: __mmask8;
}

// ---- conversions and popcount (kind `r2`) ----

r2! {
    /// `narrowus.d.b` (0x60): `b[i] ← min(u32(a.d[i]), 255)` for i < 16, rest 0.
    _svm512_cvtusepi32_epi8 => __svm_narrowus_d_b,
        _svm512_mask_cvtusepi32_epi8 / _svm512_maskz_cvtusepi32_epi8: __mmask16;
    /// `narrows.d.b` (0x61): `b[i] ← sat8(i32(a.d[i]))` for i < 16, rest 0.
    _svm512_cvtsepi32_epi8 => __svm_narrows_d_b,
        _svm512_mask_cvtsepi32_epi8 / _svm512_maskz_cvtsepi32_epi8: __mmask16;
    /// `narrows.d.w` (0x62): `w[i] ← sat16(i32(a.d[i]))` for i < 16, rest 0.
    _svm512_cvtsepi32_epi16 => __svm_narrows_d_w,
        _svm512_mask_cvtsepi32_epi16 / _svm512_maskz_cvtsepi32_epi16: __mmask16;
    /// `narrow.d.b` (0x63): `b[i] ← a.d[i] as u8` for i < 16, rest 0.
    _svm512_cvtepi32_epi8 => __svm_narrow_d_b,
        _svm512_mask_cvtepi32_epi8 / _svm512_maskz_cvtepi32_epi8: __mmask16;
    /// `zx.b.d` (0x64): `d[i] ← u8(a.b[i])`, i < 16.
    _svm512_cvtepu8_epi32 => __svm_zx_b_d,
        _svm512_mask_cvtepu8_epi32 / _svm512_maskz_cvtepu8_epi32: __mmask16;
    /// `sx.b.d` (0x65): `d[i] ← i8(a.b[i])`, i < 16.
    _svm512_cvtepi8_epi32 => __svm_sx_b_d,
        _svm512_mask_cvtepi8_epi32 / _svm512_maskz_cvtepi8_epi32: __mmask16;
    /// `zx.w.d` (0x66): `d[i] ← u16(a.w[i])`, i < 16.
    _svm512_cvtepu16_epi32 => __svm_zx_w_d,
        _svm512_mask_cvtepu16_epi32 / _svm512_maskz_cvtepu16_epi32: __mmask16;
    /// `sx.w.d` (0x67): `d[i] ← i16(a.w[i])`, i < 16.
    _svm512_cvtepi16_epi32 => __svm_sx_w_d,
        _svm512_mask_cvtepi16_epi32 / _svm512_maskz_cvtepi16_epi32: __mmask16;
    /// `zx.d.q` (0x68): `q[i] ← u32(a.d[i])`, i < 8.
    _svm512_cvtepu32_epi64 => __svm_zx_d_q,
        _svm512_mask_cvtepu32_epi64 / _svm512_maskz_cvtepu32_epi64: __mmask8;
    /// `sx.d.q` (0x69): `q[i] ← i32(a.d[i])`, i < 8.
    _svm512_cvtepi32_epi64 => __svm_sx_d_q,
        _svm512_mask_cvtepi32_epi64 / _svm512_maskz_cvtepi32_epi64: __mmask8;
    /// `popcnt.b` (0x6a): population count per byte.
    _svm512_popcnt_epi8 => __svm_popcnt_b,
        _svm512_mask_popcnt_epi8 / _svm512_maskz_popcnt_epi8: __mmask64;
    /// `popcnt.q` (0x6b): population count per u64.
    _svm512_popcnt_epi64 => __svm_popcnt_q,
        _svm512_mask_popcnt_epi64 / _svm512_maskz_popcnt_epi64: __mmask8;
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

// ---- saturating, absolute and 52-bit arithmetic ----

r3! {
    /// `mul52lo` (0xaf): `q[i] ← low52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_mul52lo_epu64 => __svm_mul52lo,
        _svm512_mask_mul52lo_epu64 / _svm512_maskz_mul52lo_epu64: __mmask8;
    /// `mul52hi` (0xb0): `q[i] ← high52(a.q[i][51:0] · b.q[i][51:0])`.
    _svm512_mul52hi_epu64 => __svm_mul52hi,
        _svm512_mask_mul52hi_epu64 / _svm512_maskz_mul52hi_epu64: __mmask8;
    /// `max.q` (0xb2): signed max, 8×i64.
    _svm512_max_epi64 => __svm_max_q,
        _svm512_mask_max_epi64 / _svm512_maskz_max_epi64: __mmask8;
    /// `min.q` (0xb3): signed min, 8×i64.
    _svm512_min_epi64 => __svm_min_q,
        _svm512_mask_min_epi64 / _svm512_maskz_min_epi64: __mmask8;
    /// `max.b` (0xb7): signed max, 64×i8.
    _svm512_max_epi8 => __svm_max_b,
        _svm512_mask_max_epi8 / _svm512_maskz_max_epi8: __mmask64;
    /// `min.b` (0xb8): signed min, 64×i8.
    _svm512_min_epi8 => __svm_min_b,
        _svm512_mask_min_epi8 / _svm512_maskz_min_epi8: __mmask64;
    /// `maxu.b` (0xb9): unsigned max, 64×u8.
    _svm512_max_epu8 => __svm_maxu_b,
        _svm512_mask_max_epu8 / _svm512_maskz_max_epu8: __mmask64;
    /// `minu.b` (0xba): unsigned min, 64×u8.
    _svm512_min_epu8 => __svm_minu_b,
        _svm512_mask_min_epu8 / _svm512_maskz_min_epu8: __mmask64;
    /// `adds.b` (0xbb): signed saturating add, 64×i8.
    _svm512_adds_epi8 => __svm_adds_b,
        _svm512_mask_adds_epi8 / _svm512_maskz_adds_epi8: __mmask64;
    /// `adds.w` (0xbc): signed saturating add, 32×i16.
    _svm512_adds_epi16 => __svm_adds_w,
        _svm512_mask_adds_epi16 / _svm512_maskz_adds_epi16: __mmask32;
    /// `addus.b` (0xbd): unsigned saturating add, 64×u8.
    _svm512_adds_epu8 => __svm_addus_b,
        _svm512_mask_adds_epu8 / _svm512_maskz_adds_epu8: __mmask64;
    /// `addus.w` (0xbe): unsigned saturating add, 32×u16.
    _svm512_adds_epu16 => __svm_addus_w,
        _svm512_mask_adds_epu16 / _svm512_maskz_adds_epu16: __mmask32;
    /// `subs.b` (0xbf): signed saturating `a - b`, 64×i8.
    _svm512_subs_epi8 => __svm_subs_b,
        _svm512_mask_subs_epi8 / _svm512_maskz_subs_epi8: __mmask64;
    /// `subs.w` (0xc0): signed saturating `a - b`, 32×i16.
    _svm512_subs_epi16 => __svm_subs_w,
        _svm512_mask_subs_epi16 / _svm512_maskz_subs_epi16: __mmask32;
    /// `subus.b` (0xc1): unsigned saturating `a - b`, 64×u8.
    _svm512_subs_epu8 => __svm_subus_b,
        _svm512_mask_subs_epu8 / _svm512_maskz_subs_epu8: __mmask64;
    /// `subus.w` (0xc2): unsigned saturating `a - b`, 32×u16.
    _svm512_subs_epu16 => __svm_subus_w,
        _svm512_mask_subs_epu16 / _svm512_maskz_subs_epu16: __mmask32;
    /// `avg.b` (0xc3): `(a + b + 1) >> 1`, 64×u8.
    _svm512_avg_epu8 => __svm_avg_b,
        _svm512_mask_avg_epu8 / _svm512_maskz_avg_epu8: __mmask64;
    /// `avg.w` (0xc4): `(a + b + 1) >> 1`, 32×u16.
    _svm512_avg_epu16 => __svm_avg_w,
        _svm512_mask_avg_epu16 / _svm512_maskz_avg_epu16: __mmask32;
    /// `sad` (0xc5): `q[j] ← Σk<8 |a.b[8j+k] - b.b[8j+k]|`, unsigned bytes.
    _svm512_sad_epu8 => __svm_sad;
    /// `mulhrs.w` (0x1f): `w[i] ← (((i16 · i16) >> 14) + 1) >> 1`.
    _svm512_mulhrs_epi16 => __svm_mulhrs_w,
        _svm512_mask_mulhrs_epi16 / _svm512_maskz_mulhrs_epi16: __mmask32;
}

r2! {
    /// `abs.q` (0xb1): absolute value, 8×i64 (`abs(MIN) = MIN`).
    _svm512_abs_epi64 => __svm_abs_q,
        _svm512_mask_abs_epi64 / _svm512_maskz_abs_epi64: __mmask8;
    /// `abs.b` (0xb4): absolute value, 64×i8.
    _svm512_abs_epi8 => __svm_abs_b,
        _svm512_mask_abs_epi8 / _svm512_maskz_abs_epi8: __mmask64;
    /// `abs.w` (0xb5): absolute value, 32×i16.
    _svm512_abs_epi16 => __svm_abs_w,
        _svm512_mask_abs_epi16 / _svm512_maskz_abs_epi16: __mmask32;
    /// `abs.d` (0xb6): absolute value, 16×i32.
    _svm512_abs_epi32 => __svm_abs_d,
        _svm512_mask_abs_epi32 / _svm512_maskz_abs_epi32: __mmask16;
}

a3! {
    /// `dpbusds` (0x56): `acc.d[i] ← sat32(acc.d[i] + Σj<4 u8(a.b[4i+j]) · s8(b.b[4i+j]))`.
    _svm512_dpbusds_epi32 => __svm_dpbusds,
        _svm512_mask_dpbusds_epi32 / _svm512_maskz_dpbusds_epi32: __mmask16;
    /// `dpwssds` (0x57): `acc.d[i] ← sat32(acc.d[i] + Σj<2 s16(a.w[2i+j]) · s16(b.w[2i+j]))`.
    _svm512_dpwssds_epi32 => __svm_dpwssds,
        _svm512_mask_dpwssds_epi32 / _svm512_maskz_dpwssds_epi32: __mmask16;
}

// ---- masks (kinds `s0`, `s3`, `s3i`, blend) ----

unsafe extern "C" {
    fn __svm_mpopcnt(k: u64) -> u64;
    fn __svm_mtzcnt(k: u64) -> u64;
    fn __svm_mlzcnt(k: u64) -> u64;
    fn __svm_blendm(a: u32, b: u32, k: u64, imm: u32) -> u32;
    fn __svm_cmp(a: u32, b: u32, imm: u32) -> u64;
    fn __svm_bitshuf(a: u32, b: u32) -> u64;
}

/// `mpopcnt` (0xf6): number of set bits of `k`.
#[inline(always)]
pub fn _svm512_popcnt_mask(k: u64) -> u32 {
    unsafe { __svm_mpopcnt(k) as u32 }
}

/// `mtzcnt` (0xf7): number of trailing zero bits of `k`, 64 for 0.
#[inline(always)]
pub fn _svm512_tzcnt_mask(k: u64) -> u32 {
    unsafe { __svm_mtzcnt(k) as u32 }
}

/// `mlzcnt` (0xf8): number of leading zero bits of `k`, 64 for 0.
#[inline(always)]
pub fn _svm512_lzcnt_mask(k: u64) -> u32 {
    unsafe { __svm_mlzcnt(k) as u32 }
}

macro_rules! blend {
    ($($(#[$doc:meta])* $name:ident: $k:ty = $w:literal;)*) => {$(
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            __svm512i(unsafe { __svm_blendm(a.0, b.0, k as u64, $w) })
        }
    )*};
}

blend! {
    /// `blendm` (0x79, imm8 0): byte i is `b`'s if bit i of `k` is set, else `a`'s.
    _svm512_mask_blend_epi8: __mmask64 = 0;
    /// `blendm` (0x79, imm8 1): u16 lane i is `b`'s if bit i of `k` is set, else `a`'s.
    _svm512_mask_blend_epi16: __mmask32 = 1;
    /// `blendm` (0x79, imm8 2): u32 lane i is `b`'s if bit i of `k` is set, else `a`'s.
    _svm512_mask_blend_epi32: __mmask16 = 2;
    /// `blendm` (0x79, imm8 3): u64 lane i is `b`'s if bit i of `k` is set, else `a`'s.
    _svm512_mask_blend_epi64: __mmask8 = 3;
}

// ---- data movement ----

/// Two-operand ops with an immediate below `$lim`: `out ← op(a, b, IMM)`
/// (kind `r3i`), with masked forms.
macro_rules! r3i {
    ($($(#[$doc:meta])* $name:ident => $marker:ident < $lim:literal, $mask:ident / $maskz:ident: $k:ty;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, b: u32, imm: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const IMM: u32>(a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            __svm512i(unsafe { $marker(a.0, b.0, IMM) })
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask<const IMM: u32>(src: __svm512i, k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, b.0, k as u64, IMM) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz<const IMM: u32>(k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, b.0, k as u64, IMM) })
        }
    )*};
}

/// Ops whose imm8 is the lane width (`unpacklo`, `unpackhi`): `out ← op(a, b)`
/// with masked forms.
macro_rules! r3w {
    ($($(#[$doc:meta])* $name:ident / $mask:ident / $maskz:ident: $k:ty = $marker:literal, $w:literal;)*) => {$(
        $(#[$doc])*
        #[inline(always)]
        pub fn $name(a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = $marker]
                fn marker(a: u32, b: u32, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, b.0, $w) })
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(src: u32, a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, b.0, k as u64, $w) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_maskz")]
                fn marker(a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, b.0, k as u64, $w) })
        }
    )*};
}

r2i! {
    /// `bcast.q` (0x7a): every u64 lane is lane `IMM` (< 8) of `a`.
    _svm512_broadcast_lane_epi64 => __svm_bcast_q < 8,
        _svm512_mask_broadcast_lane_epi64 / _svm512_maskz_broadcast_lane_epi64: __mmask8;
    /// `bcast.d` (0x7b): every u32 lane is lane `IMM` (< 16) of `a`.
    _svm512_broadcast_lane_epi32 => __svm_bcast_d < 16,
        _svm512_mask_broadcast_lane_epi32 / _svm512_maskz_broadcast_lane_epi32: __mmask16;
    /// `bcast.w` (0x7c): every u16 lane is lane `IMM` (< 32) of `a`.
    _svm512_broadcast_lane_epi16 => __svm_bcast_w < 32,
        _svm512_mask_broadcast_lane_epi16 / _svm512_maskz_broadcast_lane_epi16: __mmask32;
    /// `bcast.b` (0x7d): every byte is byte `IMM` (< 64) of `a`.
    _svm512_broadcast_lane_epi8 => __svm_bcast_b < 64,
        _svm512_mask_broadcast_lane_epi8 / _svm512_maskz_broadcast_lane_epi8: __mmask64;
    /// `permqi` (0x7e): per 256-bit half `h`, `q[4h + i] ← a.q[4h + IMM[2i+1:2i]]`.
    _svm512_permutex_epi64 => __svm_permqi < 256,
        _svm512_mask_permutex_epi64 / _svm512_maskz_permutex_epi64: __mmask8;
    /// `extract256` (0x7f): 256-bit half `IMM` (< 2) of `a` in the low half,
    /// the high half 0.
    _svm512_extracti64x4_epi64 => __svm_extract256 < 2,
        _svm512_mask_extracti64x4_epi64 / _svm512_maskz_extracti64x4_epi64: __mmask8;
    /// `extract128` (0x82): 128-bit lane `IMM` (< 4) of `a` in lane 0, the
    /// rest 0.
    _svm512_extracti32x4_epi32 => __svm_extract128 < 4,
        _svm512_mask_extracti32x4_epi32 / _svm512_maskz_extracti32x4_epi32: __mmask8;
    /// `shufd` (0x88): per 128-bit lane, `d[i] ← d[IMM[2i+1:2i]]`.
    _svm512_shuffle_epi32 => __svm_shufd < 256,
        _svm512_mask_shuffle_epi32 / _svm512_maskz_shuffle_epi32: __mmask16;
    /// `shufhw` (0x89): per 128-bit lane, words 4..8 shuffled by `IMM`,
    /// words 0..4 kept.
    _svm512_shufflehi_epi16 => __svm_shufhw < 256,
        _svm512_mask_shufflehi_epi16 / _svm512_maskz_shufflehi_epi16: __mmask32;
    /// `shuflw` (0x8a): per 128-bit lane, words 0..4 shuffled by `IMM`,
    /// words 4..8 kept.
    _svm512_shufflelo_epi16 => __svm_shuflw < 256,
        _svm512_mask_shufflelo_epi16 / _svm512_maskz_shufflelo_epi16: __mmask32;
}

/// L = 0 forms of the `bcast.*` intrinsics, named as in `core::arch`.
macro_rules! broadcast_low {
    ($($name:ident / $mask:ident / $maskz:ident: $k:ty = $lane:ident / $lane_mask:ident / $lane_maskz:ident;)*) => {$(
        #[doc = concat!("[`", stringify!($lane), "`] of lane 0.")]
        #[inline(always)]
        pub fn $name(a: __svm512i) -> __svm512i {
            $lane::<0>(a)
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i) -> __svm512i {
            $lane_mask::<0>(src, k, a)
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i) -> __svm512i {
            $lane_maskz::<0>(k, a)
        }
    )*};
}

broadcast_low! {
    _svm512_broadcastq_epi64 / _svm512_mask_broadcastq_epi64 / _svm512_maskz_broadcastq_epi64: __mmask8 =
        _svm512_broadcast_lane_epi64 / _svm512_mask_broadcast_lane_epi64 / _svm512_maskz_broadcast_lane_epi64;
    _svm512_broadcastd_epi32 / _svm512_mask_broadcastd_epi32 / _svm512_maskz_broadcastd_epi32: __mmask16 =
        _svm512_broadcast_lane_epi32 / _svm512_mask_broadcast_lane_epi32 / _svm512_maskz_broadcast_lane_epi32;
    _svm512_broadcastw_epi16 / _svm512_mask_broadcastw_epi16 / _svm512_maskz_broadcastw_epi16: __mmask32 =
        _svm512_broadcast_lane_epi16 / _svm512_mask_broadcast_lane_epi16 / _svm512_maskz_broadcast_lane_epi16;
    _svm512_broadcastb_epi8 / _svm512_mask_broadcastb_epi8 / _svm512_maskz_broadcastb_epi8: __mmask64 =
        _svm512_broadcast_lane_epi8 / _svm512_mask_broadcast_lane_epi8 / _svm512_maskz_broadcast_lane_epi8;
}

r3i! {
    /// `align.q` (0x80): u64 lanes `IMM..IMM + 8` (`IMM` < 8) of `a:b`, `b`
    /// in the low half.
    _svm512_alignr_epi64 => __svm_align_q < 8,
        _svm512_mask_alignr_epi64 / _svm512_maskz_alignr_epi64: __mmask8;
    /// `align.d` (0x81): u32 lanes `IMM..IMM + 16` (`IMM` < 16) of `a:b`, `b`
    /// in the low half.
    _svm512_alignr_epi32 => __svm_align_d < 16,
        _svm512_mask_alignr_epi32 / _svm512_maskz_alignr_epi32: __mmask16;
    /// `insert256` (0x83): `a` with 256-bit half `IMM` (< 2) replaced by the low
    /// half of `b`.
    _svm512_inserti64x4 => __svm_insert256 < 2,
        _svm512_mask_inserti64x4 / _svm512_maskz_inserti64x4: __mmask8;
    /// `insert128` (0x84): `a` with 128-bit lane `IMM` (< 4) replaced by lane 0
    /// of `b`.
    _svm512_inserti32x4 => __svm_insert128 < 4,
        _svm512_mask_inserti32x4 / _svm512_maskz_inserti32x4: __mmask16;
    /// `shuf128` (0x8b): 128-bit lanes 0, 1 from `a` and 2, 3 from `b`, lane
    /// `i` picked by `IMM[2i+1:2i]`.
    _svm512_shuffle_i64x2 => __svm_shuf128 < 256,
        _svm512_mask_shuffle_i64x2 / _svm512_maskz_shuffle_i64x2: __mmask8;
    /// `shufps` (0x8c): per 128-bit lane, u32 lanes 0, 1 from `a` and 2, 3 from
    /// `b`, lane `i` picked by `IMM[2i+1:2i]` (the bits of `vshufps`).
    _svm512_shuffle_ps_epi32 => __svm_shufps < 256,
        _svm512_mask_shuffle_ps_epi32 / _svm512_maskz_shuffle_ps_epi32: __mmask16;
    /// `alignr` (0x8d): per 128-bit lane, bytes `IMM..IMM + 16` (`IMM` < 32)
    /// of `a:b`, `b` in the low half, 0 past the end.
    _svm512_alignr_epi8 => __svm_alignr < 32,
        _svm512_mask_alignr_epi8 / _svm512_maskz_alignr_epi8: __mmask64;
}

r3! {
    /// `permb` (0x86): `b[i] ← a.b[idx.b[i] & 63]` (first argument is `idx`).
    _svm512_permutexvar_epi8 => __svm_permb,
        _svm512_mask_permutexvar_epi8 / _svm512_maskz_permutexvar_epi8: __mmask64;
    /// `permw` (0x87): `w[i] ← a.w[idx.w[i] & 31]` (first argument is `idx`).
    _svm512_permutexvar_epi16 => __svm_permw,
        _svm512_mask_permutexvar_epi16 / _svm512_maskz_permutexvar_epi16: __mmask32;
    /// `packus.d` (0x94): per 128-bit lane `[usat16(a.d×4), usat16(b.d×4)]`.
    _svm512_packus_epi32 => __svm_packus_d,
        _svm512_mask_packus_epi32 / _svm512_maskz_packus_epi32: __mmask32;
}

r3w! {
    /// `unpacklo` (0x90, imm8 0): per 128-bit lane, bytes 0..8 of `a` and `b`
    /// interleaved.
    _svm512_unpacklo_epi8 / _svm512_mask_unpacklo_epi8 / _svm512_maskz_unpacklo_epi8: __mmask64 = "__svm_unpacklo", 0;
    /// `unpacklo` (0x90, imm8 1): per 128-bit lane, u16 lanes 0..4 interleaved.
    _svm512_unpacklo_epi16 / _svm512_mask_unpacklo_epi16 / _svm512_maskz_unpacklo_epi16: __mmask32 = "__svm_unpacklo", 1;
    /// `unpacklo` (0x90, imm8 2): per 128-bit lane, u32 lanes 0, 1 interleaved.
    _svm512_unpacklo_epi32 / _svm512_mask_unpacklo_epi32 / _svm512_maskz_unpacklo_epi32: __mmask16 = "__svm_unpacklo", 2;
    /// `unpacklo` (0x90, imm8 3): per 128-bit lane, u64 lane 0 of `a`, then of `b`.
    _svm512_unpacklo_epi64 / _svm512_mask_unpacklo_epi64 / _svm512_maskz_unpacklo_epi64: __mmask8 = "__svm_unpacklo", 3;
    /// `unpackhi` (0x91, imm8 0): per 128-bit lane, bytes 8..16 of `a` and `b`
    /// interleaved.
    _svm512_unpackhi_epi8 / _svm512_mask_unpackhi_epi8 / _svm512_maskz_unpackhi_epi8: __mmask64 = "__svm_unpackhi", 0;
    /// `unpackhi` (0x91, imm8 1): per 128-bit lane, u16 lanes 4..8 interleaved.
    _svm512_unpackhi_epi16 / _svm512_mask_unpackhi_epi16 / _svm512_maskz_unpackhi_epi16: __mmask32 = "__svm_unpackhi", 1;
    /// `unpackhi` (0x91, imm8 2): per 128-bit lane, u32 lanes 2, 3 interleaved.
    _svm512_unpackhi_epi32 / _svm512_mask_unpackhi_epi32 / _svm512_maskz_unpackhi_epi32: __mmask16 = "__svm_unpackhi", 2;
    /// `unpackhi` (0x91, imm8 3): per 128-bit lane, u64 lane 1 of `a`, then of `b`.
    _svm512_unpackhi_epi64 / _svm512_mask_unpackhi_epi64 / _svm512_maskz_unpackhi_epi64: __mmask8 = "__svm_unpackhi", 3;
}

unsafe extern "C" {
    fn __svm_bshli(a: u32, imm: u32) -> u32;
    fn __svm_bshri(a: u32, imm: u32) -> u32;
    fn __svm_extract_b(a: u32, imm: u32) -> u64;
    fn __svm_extract_w(a: u32, imm: u32) -> u64;
    fn __svm_insert_b(a: u32, x: u64, imm: u32) -> u32;
    fn __svm_insert_w(a: u32, x: u64, imm: u32) -> u32;
    fn __svm_insert_d(a: u32, x: u64, imm: u32) -> u32;
    fn __svm_movr_q(x: u64) -> u32;
}

/// `bshli` (0x8e): each 128-bit lane shifted left by `IMM` (< 16) bytes.
#[inline(always)]
pub fn _svm512_bslli_epi128<const IMM: u32>(a: __svm512i) -> __svm512i {
    const { assert!(IMM < 16, "immediate out of range") };
    __svm512i(unsafe { __svm_bshli(a.0, IMM) })
}

/// `bshri` (0x8f): each 128-bit lane shifted right by `IMM` (< 16) bytes.
#[inline(always)]
pub fn _svm512_bsrli_epi128<const IMM: u32>(a: __svm512i) -> __svm512i {
    const { assert!(IMM < 16, "immediate out of range") };
    __svm512i(unsafe { __svm_bshri(a.0, IMM) })
}

/// `extract.b` (0xa9): byte `IMM` (< 64), zero extended.
#[inline(always)]
pub fn _svm512_extract_epi8<const IMM: u32>(a: __svm512i) -> i32 {
    const { assert!(IMM < 64, "immediate out of range") };
    unsafe { __svm_extract_b(a.0, IMM) as i32 }
}

/// `extract.w` (0xaa): u16 lane `IMM` (< 32), zero extended.
#[inline(always)]
pub fn _svm512_extract_epi16<const IMM: u32>(a: __svm512i) -> i32 {
    const { assert!(IMM < 32, "immediate out of range") };
    unsafe { __svm_extract_w(a.0, IMM) as i32 }
}

/// `insert.b` (0xab): `a` with byte `IMM` (< 64) replaced by `x`.
#[inline(always)]
pub fn _svm512_insert_epi8<const IMM: u32>(a: __svm512i, x: i8) -> __svm512i {
    const { assert!(IMM < 64, "immediate out of range") };
    __svm512i(unsafe { __svm_insert_b(a.0, x as u64, IMM) })
}

/// `insert.w` (0xac): `a` with u16 lane `IMM` (< 32) replaced by `x`.
#[inline(always)]
pub fn _svm512_insert_epi16<const IMM: u32>(a: __svm512i, x: i16) -> __svm512i {
    const { assert!(IMM < 32, "immediate out of range") };
    __svm512i(unsafe { __svm_insert_w(a.0, x as u64, IMM) })
}

/// `insert.d` (0xad): `a` with u32 lane `IMM` (< 16) replaced by `x`.
#[inline(always)]
pub fn _svm512_insert_epi32<const IMM: u32>(a: __svm512i, x: i32) -> __svm512i {
    const { assert!(IMM < 16, "immediate out of range") };
    __svm512i(unsafe { __svm_insert_d(a.0, x as u64, IMM) })
}

/// `movr.q` (0xae): `x` in u64 lane 0, the rest 0.
#[inline(always)]
pub fn _svm512_cvtsi64_si512(x: i64) -> __svm512i {
    __svm512i(unsafe { __svm_movr_q(x as u64) })
}

macro_rules! permute2 {
    ($($w:literal: $k:ty = $mask2:ident $(, $name:ident / $mask:ident / $maskz:ident)?;)*) => {$(
        $(
        #[doc = concat!("`perm2t` (0x85, imm8 ", $w, "): lane i is lane `idx[i] mod 2L` of `a:b` (`a` the low half, L lanes each). The pass emits `perm2i` instead where that saves a copy.")]
        #[inline(always)]
        pub fn $name(a: __svm512i, idx: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                fn __svm_perm2t(a: u32, idx: u32, b: u32, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_perm2t(a.0, idx.0, b.0, $w) })
        }
        #[doc = concat!("[`", stringify!($name), "`] with merge masking: lanes whose bit of `k` is clear keep `a`.")]
        #[inline(always)]
        pub fn $mask(a: __svm512i, k: $k, idx: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                fn __svm_perm2t_mask(a: u32, idx: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_perm2t_mask(a.0, idx.0, b.0, k as u64, $w) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i, idx: __svm512i, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                fn __svm_perm2t_maskz(a: u32, idx: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_perm2t_maskz(a.0, idx.0, b.0, k as u64, $w) })
        }
        )?
        #[doc = concat!("`perm2i` (0x5f, imm8 ", $w, "): two-table permute as `_svm512_permutex2var_*` with merge masking: lanes whose bit of `k` is clear keep `idx`.")]
        #[inline(always)]
        pub fn $mask2(a: __svm512i, idx: __svm512i, k: $k, b: __svm512i) -> __svm512i {
            unsafe extern "C" {
                fn __svm_perm2i_mask(idx: u32, a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_perm2i_mask(idx.0, a.0, b.0, k as u64, $w) })
        }
    )*};
}

permute2! {
    0: __mmask64 = _svm512_mask2_permutex2var_epi8,
        _svm512_permutex2var_epi8 / _svm512_mask_permutex2var_epi8 / _svm512_maskz_permutex2var_epi8;
    1: __mmask32 = _svm512_mask2_permutex2var_epi16;
    2: __mmask16 = _svm512_mask2_permutex2var_epi32,
        _svm512_permutex2var_epi32 / _svm512_mask_permutex2var_epi32 / _svm512_maskz_permutex2var_epi32;
    3: __mmask8 = _svm512_mask2_permutex2var_epi64,
        _svm512_permutex2var_epi64 / _svm512_mask_permutex2var_epi64 / _svm512_maskz_permutex2var_epi64;
}

macro_rules! compress_expand {
    ($($mask:ident / $maskz:ident: $k:ty = $marker:literal, $w:literal, $what:literal;)*) => {$(
        #[doc = concat!("`", $what, "` with merge masking: the remaining lanes keep `src`.")]
        #[inline(always)]
        pub fn $mask(src: __svm512i, k: $k, a: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(src: u32, a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, k as u64, $w) })
        }
        #[doc = concat!("`", $what, "` with zero masking: the remaining lanes are 0.")]
        #[inline(always)]
        pub fn $maskz(k: $k, a: __svm512i) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_maskz")]
                fn marker(a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, k as u64, $w) })
        }
    )*};
}

compress_expand! {
    _svm512_mask_compress_epi8 / _svm512_maskz_compress_epi8: __mmask64 = "__svm_compress", 0,
        "compress` (0x92, imm8 0): the bytes of `a` whose bit of `k` is set, packed into the low lanes";
    _svm512_mask_compress_epi16 / _svm512_maskz_compress_epi16: __mmask32 = "__svm_compress", 1,
        "compress` (0x92, imm8 1): the u16 lanes of `a` whose bit of `k` is set, packed into the low lanes";
    _svm512_mask_compress_epi32 / _svm512_maskz_compress_epi32: __mmask16 = "__svm_compress", 2,
        "compress` (0x92, imm8 2): the u32 lanes of `a` whose bit of `k` is set, packed into the low lanes";
    _svm512_mask_compress_epi64 / _svm512_maskz_compress_epi64: __mmask8 = "__svm_compress", 3,
        "compress` (0x92, imm8 3): the u64 lanes of `a` whose bit of `k` is set, packed into the low lanes";
    _svm512_mask_expand_epi8 / _svm512_maskz_expand_epi8: __mmask64 = "__svm_expand", 0,
        "expand` (0x93, imm8 0): the low bytes of `a`, in order, to the lanes whose bit of `k` is set";
    _svm512_mask_expand_epi16 / _svm512_maskz_expand_epi16: __mmask32 = "__svm_expand", 1,
        "expand` (0x93, imm8 1): the low u16 lanes of `a`, in order, to the lanes whose bit of `k` is set";
    _svm512_mask_expand_epi32 / _svm512_maskz_expand_epi32: __mmask16 = "__svm_expand", 2,
        "expand` (0x93, imm8 2): the low u32 lanes of `a`, in order, to the lanes whose bit of `k` is set";
    _svm512_mask_expand_epi64 / _svm512_maskz_expand_epi64: __mmask8 = "__svm_expand", 3,
        "expand` (0x93, imm8 3): the low u64 lanes of `a`, in order, to the lanes whose bit of `k` is set";
}

r2! {
    /// `narrow.d.w` (0x95): `w[i] ← a.d[i] as u16` for i < 16, rest 0.
    _svm512_cvtepi32_epi16 => __svm_narrow_d_w,
        _svm512_mask_cvtepi32_epi16 / _svm512_maskz_cvtepi32_epi16: __mmask16;
    /// `narrowus.d.w` (0x96): `w[i] ← min(u32(a.d[i]), 65535)` for i < 16, rest 0.
    _svm512_cvtusepi32_epi16 => __svm_narrowus_d_w,
        _svm512_mask_cvtusepi32_epi16 / _svm512_maskz_cvtusepi32_epi16: __mmask16;
    /// `narrow.q.b` (0x97): `b[i] ← a.q[i] as u8` for i < 8, rest 0.
    _svm512_cvtepi64_epi8 => __svm_narrow_q_b,
        _svm512_mask_cvtepi64_epi8 / _svm512_maskz_cvtepi64_epi8: __mmask8;
    /// `narrows.q.b` (0x98): `b[i] ← sat8(i64(a.q[i]))` for i < 8, rest 0.
    _svm512_cvtsepi64_epi8 => __svm_narrows_q_b,
        _svm512_mask_cvtsepi64_epi8 / _svm512_maskz_cvtsepi64_epi8: __mmask8;
    /// `narrowus.q.b` (0x99): `b[i] ← min(a.q[i], 255)` for i < 8, rest 0.
    _svm512_cvtusepi64_epi8 => __svm_narrowus_q_b,
        _svm512_mask_cvtusepi64_epi8 / _svm512_maskz_cvtusepi64_epi8: __mmask8;
    /// `narrow.q.w` (0x9a): `w[i] ← a.q[i] as u16` for i < 8, rest 0.
    _svm512_cvtepi64_epi16 => __svm_narrow_q_w,
        _svm512_mask_cvtepi64_epi16 / _svm512_maskz_cvtepi64_epi16: __mmask8;
    /// `narrows.q.w` (0x9b): `w[i] ← sat16(i64(a.q[i]))` for i < 8, rest 0.
    _svm512_cvtsepi64_epi16 => __svm_narrows_q_w,
        _svm512_mask_cvtsepi64_epi16 / _svm512_maskz_cvtsepi64_epi16: __mmask8;
    /// `narrowus.q.w` (0x9c): `w[i] ← min(a.q[i], 65535)` for i < 8, rest 0.
    _svm512_cvtusepi64_epi16 => __svm_narrowus_q_w,
        _svm512_mask_cvtusepi64_epi16 / _svm512_maskz_cvtusepi64_epi16: __mmask8;
    /// `narrow.q.d` (0x9d): `d[i] ← a.q[i] as u32` for i < 8, rest 0.
    _svm512_cvtepi64_epi32 => __svm_narrow_q_d,
        _svm512_mask_cvtepi64_epi32 / _svm512_maskz_cvtepi64_epi32: __mmask8;
    /// `narrows.q.d` (0x9e): `d[i] ← sat32(i64(a.q[i]))` for i < 8, rest 0.
    _svm512_cvtsepi64_epi32 => __svm_narrows_q_d,
        _svm512_mask_cvtsepi64_epi32 / _svm512_maskz_cvtsepi64_epi32: __mmask8;
    /// `narrowus.q.d` (0x9f): `d[i] ← min(a.q[i], 2^32 - 1)` for i < 8, rest 0.
    _svm512_cvtusepi64_epi32 => __svm_narrowus_q_d,
        _svm512_mask_cvtusepi64_epi32 / _svm512_maskz_cvtusepi64_epi32: __mmask8;
    /// `narrow.w.b` (0xa0): `b[i] ← a.w[i] as u8` for i < 32, rest 0.
    _svm512_cvtepi16_epi8 => __svm_narrow_w_b,
        _svm512_mask_cvtepi16_epi8 / _svm512_maskz_cvtepi16_epi8: __mmask32;
    /// `narrows.w.b` (0xa1): `b[i] ← sat8(i16(a.w[i]))` for i < 32, rest 0.
    _svm512_cvtsepi16_epi8 => __svm_narrows_w_b,
        _svm512_mask_cvtsepi16_epi8 / _svm512_maskz_cvtsepi16_epi8: __mmask32;
    /// `narrowus.w.b` (0xa2): `b[i] ← min(a.w[i], 255)` for i < 32, rest 0.
    _svm512_cvtusepi16_epi8 => __svm_narrowus_w_b,
        _svm512_mask_cvtusepi16_epi8 / _svm512_maskz_cvtusepi16_epi8: __mmask32;
    /// `zx.b.w` (0xa3): `w[i] ← u8(a.b[i])`, i < 32.
    _svm512_cvtepu8_epi16 => __svm_zx_b_w,
        _svm512_mask_cvtepu8_epi16 / _svm512_maskz_cvtepu8_epi16: __mmask32;
    /// `sx.b.w` (0xa4): `w[i] ← i8(a.b[i])`, i < 32.
    _svm512_cvtepi8_epi16 => __svm_sx_b_w,
        _svm512_mask_cvtepi8_epi16 / _svm512_maskz_cvtepi8_epi16: __mmask32;
    /// `zx.b.q` (0xa5): `q[i] ← u8(a.b[i])`, i < 8.
    _svm512_cvtepu8_epi64 => __svm_zx_b_q,
        _svm512_mask_cvtepu8_epi64 / _svm512_maskz_cvtepu8_epi64: __mmask8;
    /// `sx.b.q` (0xa6): `q[i] ← i8(a.b[i])`, i < 8.
    _svm512_cvtepi8_epi64 => __svm_sx_b_q,
        _svm512_mask_cvtepi8_epi64 / _svm512_maskz_cvtepi8_epi64: __mmask8;
    /// `zx.w.q` (0xa7): `q[i] ← u16(a.w[i])`, i < 8.
    _svm512_cvtepu16_epi64 => __svm_zx_w_q,
        _svm512_mask_cvtepu16_epi64 / _svm512_maskz_cvtepu16_epi64: __mmask8;
    /// `sx.w.q` (0xa8): `q[i] ← i16(a.w[i])`, i < 8.
    _svm512_cvtepi16_epi64 => __svm_sx_w_q,
        _svm512_mask_cvtepi16_epi64 / _svm512_maskz_cvtepi16_epi64: __mmask8;
}

/// `cmp` predicate: equal.
pub const _SVM_CMPINT_EQ: u32 = 0;
/// `cmp` predicate: less than.
pub const _SVM_CMPINT_LT: u32 = 1;
/// `cmp` predicate: less than or equal.
pub const _SVM_CMPINT_LE: u32 = 2;
/// `cmp` predicate: not equal.
pub const _SVM_CMPINT_NE: u32 = 4;
/// `cmp` predicate: not less than (greater than or equal).
pub const _SVM_CMPINT_NLT: u32 = 5;
/// `cmp` predicate: not less than or equal (greater than).
pub const _SVM_CMPINT_NLE: u32 = 6;

macro_rules! cmp {
    ($($name:ident: $k:ty = $flags:literal, $what:literal [$($wrapper:ident = $p:ident),*];)*) => {$(
        #[doc = concat!("`cmp` (0x46): bit i of the result is `P(a, b)` of ", $what, " lane i. `P` is one of the `_SVM_CMPINT_*` predicates.")]
        #[inline(always)]
        pub fn $name<const P: u32>(a: __svm512i, b: __svm512i) -> $k {
            const { assert!(P < 7 && P != 3, "unknown predicate") };
            unsafe { __svm_cmp(a.0, b.0, P | $flags) as $k }
        }
        $(
        #[doc = concat!("[`", stringify!($name), "`] with `", stringify!($p), "`.")]
        #[inline(always)]
        pub fn $wrapper(a: __svm512i, b: __svm512i) -> $k {
            $name::<$p>(a, b)
        }
        )*
    )*};
}

cmp! {
    _svm512_cmp_epi8_mask: __mmask64 = 0x00, "the signed u8" [
        _svm512_cmpeq_epi8_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epi8_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epi8_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epi8_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epi8_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epi8_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epu8_mask: __mmask64 = 0x08, "the unsigned u8" [
        _svm512_cmpeq_epu8_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epu8_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epu8_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epu8_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epu8_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epu8_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epi16_mask: __mmask32 = 0x10, "the signed u16" [
        _svm512_cmpeq_epi16_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epi16_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epi16_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epi16_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epi16_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epi16_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epu16_mask: __mmask32 = 0x18, "the unsigned u16" [
        _svm512_cmpeq_epu16_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epu16_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epu16_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epu16_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epu16_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epu16_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epi32_mask: __mmask16 = 0x20, "the signed u32" [
        _svm512_cmpeq_epi32_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epi32_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epi32_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epi32_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epi32_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epi32_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epu32_mask: __mmask16 = 0x28, "the unsigned u32" [
        _svm512_cmpeq_epu32_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epu32_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epu32_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epu32_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epu32_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epu32_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epi64_mask: __mmask8 = 0x30, "the signed u64" [
        _svm512_cmpeq_epi64_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epi64_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epi64_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epi64_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epi64_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epi64_mask = _SVM_CMPINT_NLE];
    _svm512_cmp_epu64_mask: __mmask8 = 0x38, "the unsigned u64" [
        _svm512_cmpeq_epu64_mask = _SVM_CMPINT_EQ, _svm512_cmplt_epu64_mask = _SVM_CMPINT_LT,
        _svm512_cmple_epu64_mask = _SVM_CMPINT_LE, _svm512_cmpneq_epu64_mask = _SVM_CMPINT_NE,
        _svm512_cmpge_epu64_mask = _SVM_CMPINT_NLT, _svm512_cmpgt_epu64_mask = _SVM_CMPINT_NLE];
}

/// `bitshuf` (0x4f): bit `8j + k` of the result is bit `c.b[8j+k] & 63` of
/// `b.q[j]`.
#[inline(always)]
pub fn _svm512_bitshuffle_epi64_mask(b: __svm512i, c: __svm512i) -> __mmask64 {
    unsafe { __svm_bitshuf(b.0, c.0) }
}

unsafe extern "C" {
    fn __svm_test(a: u32, b: u32, imm: u32) -> u64;
    fn __svm_testn(a: u32, b: u32, imm: u32) -> u64;
    fn __svm_movemask_w(a: u32) -> u64;
    fn __svm_movemask_d(a: u32) -> u64;
    fn __svm_movemask_q(a: u32) -> u64;
    fn __svm_movm_b(k: u64) -> u32;
    fn __svm_movm_w(k: u64) -> u32;
    fn __svm_movm_d(k: u64) -> u32;
    fn __svm_movm_q(k: u64) -> u32;
}

macro_rules! test_mask {
    ($($test:ident / $testn:ident: $k:ty = $w:literal, $what:literal;)*) => {$(
        #[doc = concat!("`test` (0x47, imm8 ", $w, "): bit i of the result is set if ", $what, " lane i of `a & b` is not 0.")]
        #[inline(always)]
        pub fn $test(a: __svm512i, b: __svm512i) -> $k {
            unsafe { __svm_test(a.0, b.0, $w) as $k }
        }
        #[doc = concat!("`testn` (0x4e, imm8 ", $w, "): bit i of the result is set if ", $what, " lane i of `a & b` is 0.")]
        #[inline(always)]
        pub fn $testn(a: __svm512i, b: __svm512i) -> $k {
            unsafe { __svm_testn(a.0, b.0, $w) as $k }
        }
    )*};
}

test_mask! {
    _svm512_test_epi8_mask / _svm512_testn_epi8_mask: __mmask64 = 0, "u8";
    _svm512_test_epi16_mask / _svm512_testn_epi16_mask: __mmask32 = 1, "u16";
    _svm512_test_epi32_mask / _svm512_testn_epi32_mask: __mmask16 = 2, "u32";
    _svm512_test_epi64_mask / _svm512_testn_epi64_mask: __mmask8 = 3, "u64";
}

/// `movemask.w` (0x76): bit `i` of the result is the sign bit of u16 `i`.
#[inline(always)]
pub fn _svm512_movepi16_mask(a: __svm512i) -> __mmask32 {
    unsafe { __svm_movemask_w(a.0) as __mmask32 }
}

/// `movemask.d` (0x77): bit `i` of the result is the sign bit of u32 `i`.
#[inline(always)]
pub fn _svm512_movepi32_mask(a: __svm512i) -> __mmask16 {
    unsafe { __svm_movemask_d(a.0) as __mmask16 }
}

/// `movemask.q` (0xc6): bit `i` of the result is the sign bit of u64 `i`.
#[inline(always)]
pub fn _svm512_movepi64_mask(a: __svm512i) -> __mmask8 {
    unsafe { __svm_movemask_q(a.0) as __mmask8 }
}

/// `movm.b` (0xc7): byte `i` is all ones if bit `i` of `k` is set, else 0.
#[inline(always)]
pub fn _svm512_movm_epi8(k: __mmask64) -> __svm512i {
    __svm512i(unsafe { __svm_movm_b(k) })
}

/// `movm.w` (0xc8): u16 `i` is all ones if bit `i` of `k` is set, else 0.
#[inline(always)]
pub fn _svm512_movm_epi16(k: __mmask32) -> __svm512i {
    __svm512i(unsafe { __svm_movm_w(k as u64) })
}

/// `movm.d` (0xc9): u32 `i` is all ones if bit `i` of `k` is set, else 0.
#[inline(always)]
pub fn _svm512_movm_epi32(k: __mmask16) -> __svm512i {
    __svm512i(unsafe { __svm_movm_d(k as u64) })
}

/// `movm.q` (0xca): u64 `i` is all ones if bit `i` of `k` is set, else 0.
#[inline(always)]
pub fn _svm512_movm_epi64(k: __mmask8) -> __svm512i {
    __svm512i(unsafe { __svm_movm_q(k as u64) })
}

// ---- bit ops: shifts, double shifts, conflict, counts ----

r3! {
    /// `shlv.w` (0x41): per-lane shift left, count ≥ 16 gives 0.
    _svm512_sllv_epi16 => __svm_shlv_w,
        _svm512_mask_sllv_epi16 / _svm512_maskz_sllv_epi16: __mmask32;
    /// `shrv.w` (0x42): per-lane logical shift right, count ≥ 16 gives 0.
    _svm512_srlv_epi16 => __svm_shrv_w,
        _svm512_mask_srlv_epi16 / _svm512_maskz_srlv_epi16: __mmask32;
    /// `sarv.w` (0x43): per-lane arithmetic shift right, count ≥ 16 fills with the sign.
    _svm512_srav_epi16 => __svm_sarv_w,
        _svm512_mask_srav_epi16 / _svm512_maskz_srav_epi16: __mmask32;
    /// `sarv.d` (0x44): per-lane arithmetic shift right, count ≥ 32 fills with the sign.
    _svm512_srav_epi32 => __svm_sarv_d,
        _svm512_mask_srav_epi32 / _svm512_maskz_srav_epi32: __mmask16;
    /// `sarv.q` (0x45): per-lane arithmetic shift right, count ≥ 64 fills with the sign.
    _svm512_srav_epi64 => __svm_sarv_q,
        _svm512_mask_srav_epi64 / _svm512_maskz_srav_epi64: __mmask8;
    /// `multishift` (0xcd): byte `8j+k` is the low byte of `b.q[j]` rotated right by
    /// `a.b[8j+k] & 63`.
    _svm512_multishift_epi64_epi8 => __svm_multishift,
        _svm512_mask_multishift_epi64_epi8 / _svm512_maskz_multishift_epi64_epi8: __mmask64;
}

a3! {
    /// `shldv.w` (0xd4): the high u16 of `(a:b) << (c mod 16)` per lane.
    _svm512_shldv_epi16 => __svm_shldv_w,
        _svm512_mask_shldv_epi16 / _svm512_maskz_shldv_epi16: __mmask32;
    /// `shldv.d` (0xd5): the high u32 of `(a:b) << (c mod 32)` per lane.
    _svm512_shldv_epi32 => __svm_shldv_d,
        _svm512_mask_shldv_epi32 / _svm512_maskz_shldv_epi32: __mmask16;
    /// `shldv.q` (0xd6): the high u64 of `(a:b) << (c mod 64)` per lane.
    _svm512_shldv_epi64 => __svm_shldv_q,
        _svm512_mask_shldv_epi64 / _svm512_maskz_shldv_epi64: __mmask8;
    /// `shrdv.w` (0xd7): the low u16 of `(b:a) >> (c mod 16)` per lane.
    _svm512_shrdv_epi16 => __svm_shrdv_w,
        _svm512_mask_shrdv_epi16 / _svm512_maskz_shrdv_epi16: __mmask32;
    /// `shrdv.d` (0xd8): the low u32 of `(b:a) >> (c mod 32)` per lane.
    _svm512_shrdv_epi32 => __svm_shrdv_d,
        _svm512_mask_shrdv_epi32 / _svm512_maskz_shrdv_epi32: __mmask16;
    /// `shrdv.q` (0xd9): the low u64 of `(b:a) >> (c mod 64)` per lane.
    _svm512_shrdv_epi64 => __svm_shrdv_q,
        _svm512_mask_shrdv_epi64 / _svm512_maskz_shrdv_epi64: __mmask8;
}

r2! {
    /// `conflict.d` (0xcb): bit `j` of u32 `i` is set if lane `j < i` equals lane `i`.
    _svm512_conflict_epi32 => __svm_conflict_d,
        _svm512_mask_conflict_epi32 / _svm512_maskz_conflict_epi32: __mmask16;
    /// `conflict.q` (0xcc): bit `j` of u64 `i` is set if lane `j < i` equals lane `i`.
    _svm512_conflict_epi64 => __svm_conflict_q,
        _svm512_mask_conflict_epi64 / _svm512_maskz_conflict_epi64: __mmask8;
    /// `lzcnt.d` (0x6e): leading zeros per u32, 32 for 0.
    _svm512_lzcnt_epi32 => __svm_lzcnt_d,
        _svm512_mask_lzcnt_epi32 / _svm512_maskz_lzcnt_epi32: __mmask16;
    /// `lzcnt.q` (0x6f): leading zeros per u64, 64 for 0.
    _svm512_lzcnt_epi64 => __svm_lzcnt_q,
        _svm512_mask_lzcnt_epi64 / _svm512_maskz_lzcnt_epi64: __mmask8;
    /// `popcnt.w` (0x6c): population count per u16.
    _svm512_popcnt_epi16 => __svm_popcnt_w,
        _svm512_mask_popcnt_epi16 / _svm512_maskz_popcnt_epi16: __mmask32;
    /// `popcnt.d` (0x6d): population count per u32.
    _svm512_popcnt_epi32 => __svm_popcnt_d,
        _svm512_mask_popcnt_epi32 / _svm512_maskz_popcnt_epi32: __mmask16;
}

macro_rules! shift_double {
    ($($(#[$doc:meta])* $name:ident => $marker:ident < $lim:literal, $mask:ident / $maskz:ident: $k:ty;)*) => {$(
        unsafe extern "C" { fn $marker(a: u32, b: u32, imm: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const IMM: u32>(a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            __svm512i(unsafe { $marker(a.0, b.0, IMM) })
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask<const IMM: u32>(src: __svm512i, k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, a.0, b.0, k as u64, IMM) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz<const IMM: u32>(k: $k, a: __svm512i, b: __svm512i) -> __svm512i {
            const { assert!(IMM < $lim, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(a: u32, b: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(a.0, b.0, k as u64, IMM) })
        }
    )*};
}

shift_double! {
    /// `shldi.w` (0xce): the high u16 of `(a:b) << IMM` per lane.
    _svm512_shldi_epi16 => __svm_shldi_w < 16,
        _svm512_mask_shldi_epi16 / _svm512_maskz_shldi_epi16: __mmask32;
    /// `shldi.d` (0xcf): the high u32 of `(a:b) << IMM` per lane.
    _svm512_shldi_epi32 => __svm_shldi_d < 32,
        _svm512_mask_shldi_epi32 / _svm512_maskz_shldi_epi32: __mmask16;
    /// `shldi.q` (0xd0): the high u64 of `(a:b) << IMM` per lane.
    _svm512_shldi_epi64 => __svm_shldi_q < 64,
        _svm512_mask_shldi_epi64 / _svm512_maskz_shldi_epi64: __mmask8;
    /// `shrdi.w` (0xd1): the low u16 of `(b:a) >> IMM` per lane.
    _svm512_shrdi_epi16 => __svm_shrdi_w < 16,
        _svm512_mask_shrdi_epi16 / _svm512_maskz_shrdi_epi16: __mmask32;
    /// `shrdi.d` (0xd2): the low u32 of `(b:a) >> IMM` per lane.
    _svm512_shrdi_epi32 => __svm_shrdi_d < 32,
        _svm512_mask_shrdi_epi32 / _svm512_maskz_shrdi_epi32: __mmask16;
    /// `shrdi.q` (0xd3): the low u64 of `(b:a) >> IMM` per lane.
    _svm512_shrdi_epi64 => __svm_shrdi_q < 64,
        _svm512_mask_shrdi_epi64 / _svm512_maskz_shrdi_epi64: __mmask8;
}

/// `ternlog.d` (0x2d): `ternlog` with u32 mask lanes.
#[inline(always)]
pub fn _svm512_ternarylogic_epi32<const IMM: u32>(
    a: __svm512i,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    unsafe extern "C" {
        fn __svm_ternlog_d(a: u32, b: u32, c: u32, imm: u32) -> u32;
    }
    __svm512i(unsafe { __svm_ternlog_d(a.0, b.0, c.0, IMM) })
}

#[doc = merge_doc!(_svm512_ternarylogic_epi32)]
#[inline(always)]
pub fn _svm512_mask_ternarylogic_epi32<const IMM: u32>(
    src: __svm512i,
    k: __mmask16,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    unsafe extern "C" {
        fn __svm_ternlog_d_mask(a: u32, b: u32, c: u32, k: u64, imm: u32) -> u32;
    }
    __svm512i(unsafe { __svm_ternlog_d_mask(src.0, b.0, c.0, k as u64, IMM) })
}

#[doc = zero_doc!(_svm512_ternarylogic_epi32)]
#[inline(always)]
pub fn _svm512_maskz_ternarylogic_epi32<const IMM: u32>(
    k: __mmask16,
    a: __svm512i,
    b: __svm512i,
    c: __svm512i,
) -> __svm512i {
    const { assert!(IMM < 256, "immediate out of range") };
    unsafe extern "C" {
        fn __svm_ternlog_d_maskz(a: u32, b: u32, c: u32, k: u64, imm: u32) -> u32;
    }
    __svm512i(unsafe { __svm_ternlog_d_maskz(a.0, b.0, c.0, k as u64, IMM) })
}

// ---- SHA (lane 0 only, bits 511:128 of the first operand are kept) ----

unsafe extern "C" {
    fn __svm_sha256rnds2(cdgh: u32, abef: u32, wk: u32) -> u32;
    fn __svm_sha256msg1(a: u32, b: u32) -> u32;
    fn __svm_sha1rnds4(a: u32, b: u32, imm: u32) -> u32;
}

/// `sha256rnds2` (0xe1): two SHA-256 rounds on the low 128 bits, with the
/// state `cdgh` and `abef` and the two round inputs `wk.d[0..2]`. Returns the
/// new `abef`.
#[inline(always)]
pub fn _svm512_sha256rnds2_epu32(cdgh: __svm512i, abef: __svm512i, wk: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha256rnds2(cdgh.0, abef.0, wk.0) })
}

/// `sha256msg1` (0xe2): the σ0 part of the SHA-256 message schedule on the
/// low 128 bits.
#[inline(always)]
pub fn _svm512_sha256msg1_epu32(a: __svm512i, b: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha256msg1(a.0, b.0) })
}

/// `sha1rnds4` (0xe4): four SHA-1 rounds on the low 128 bits, `F` (< 4)
/// selects the round function and constant.
#[inline(always)]
pub fn _svm512_sha1rnds4_epu32<const F: u32>(a: __svm512i, b: __svm512i) -> __svm512i {
    const { assert!(F < 4, "immediate out of range") };
    __svm512i(unsafe { __svm_sha1rnds4(a.0, b.0, F) })
}

unsafe extern "C" {
    fn __svm_sha256msg2(a: u32, b: u32) -> u32;
    fn __svm_sha1nexte(a: u32, b: u32) -> u32;
    fn __svm_sha1msg1(a: u32, b: u32) -> u32;
    fn __svm_sha1msg2(a: u32, b: u32) -> u32;
}

/// `sha256msg2` (0xe3): the σ1 part of the SHA-256 message schedule on the
/// low 128 bits.
#[inline(always)]
pub fn _svm512_sha256msg2_epu32(a: __svm512i, b: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha256msg2(a.0, b.0) })
}

/// `sha1nexte` (0xe5): the next SHA-1 state variable E, `rol(a.d[3], 30)`,
/// added to `b.d[3]`, on the low 128 bits.
#[inline(always)]
pub fn _svm512_sha1nexte_epu32(a: __svm512i, b: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha1nexte(a.0, b.0) })
}

/// `sha1msg1` (0xe6): the XOR part of the SHA-1 message schedule on the low
/// 128 bits.
#[inline(always)]
pub fn _svm512_sha1msg1_epu32(a: __svm512i, b: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha1msg1(a.0, b.0) })
}

/// `sha1msg2` (0xe7): the final SHA-1 message schedule step on the low 128
/// bits.
#[inline(always)]
pub fn _svm512_sha1msg2_epu32(a: __svm512i, b: __svm512i) -> __svm512i {
    __svm512i(unsafe { __svm_sha1msg2(a.0, b.0) })
}

// ---- GF(2^8) and AES (per 128-bit lane) ----

r3! {
    /// `gf2p8mul` (0xda): bytewise product in GF(2^8) modulo
    /// x^8 + x^4 + x^3 + x + 1.
    _svm512_gf2p8mul_epi8 => __svm_gf2p8mul,
        _svm512_mask_gf2p8mul_epi8 / _svm512_maskz_gf2p8mul_epi8: __mmask64;
    /// `aesenc` (0xdd): one AES round on each 128-bit lane.
    _svm512_aesenc_epi128 => __svm_aesenc;
    /// `aesenclast` (0xde): the last AES round (no MixColumns).
    _svm512_aesenclast_epi128 => __svm_aesenclast;
    /// `aesdec` (0xdf): one round of the AES equivalent inverse cipher.
    _svm512_aesdec_epi128 => __svm_aesdec;
    /// `aesdeclast` (0xe0): the last round of the inverse cipher (no
    /// InvMixColumns).
    _svm512_aesdeclast_epi128 => __svm_aesdeclast;
}

/// Byte affine maps over GF(2): `x` is the byte operand, `a.q[j]` the matrix
/// of the bytes of qword j, `B` the added constant.
macro_rules! gf2p8affine {
    ($($(#[$doc:meta])* $name:ident, $mask:ident, $maskz:ident => $marker:ident;)*) => {$(
        unsafe extern "C" { fn $marker(x: u32, a: u32, imm: u32) -> u32; }
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<const B: u32>(x: __svm512i, a: __svm512i) -> __svm512i {
            const { assert!(B < 256, "immediate out of range") };
            __svm512i(unsafe { $marker(x.0, a.0, B) })
        }
        #[doc = merge_doc!($name)]
        #[inline(always)]
        pub fn $mask<const B: u32>(
            src: __svm512i,
            k: __mmask64,
            x: __svm512i,
            a: __svm512i,
        ) -> __svm512i {
            const { assert!(B < 256, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_mask")]
                fn marker(src: u32, x: u32, a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(src.0, x.0, a.0, k, B) })
        }
        #[doc = zero_doc!($name)]
        #[inline(always)]
        pub fn $maskz<const B: u32>(k: __mmask64, x: __svm512i, a: __svm512i) -> __svm512i {
            const { assert!(B < 256, "immediate out of range") };
            unsafe extern "C" {
                #[link_name = concat!(stringify!($marker), "_maskz")]
                fn marker(x: u32, a: u32, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { marker(x.0, a.0, k, B) })
        }
    )*};
}

gf2p8affine! {
    /// `gf2p8affine` (0xdb): bit t of byte i of the result is the parity of
    /// `x.b[i]` and byte 7 − t of `a.q[i / 8]`, XOR bit t of `B`.
    _svm512_gf2p8affine_epi64_epi8, _svm512_mask_gf2p8affine_epi64_epi8,
        _svm512_maskz_gf2p8affine_epi64_epi8 => __svm_gf2p8affine;
    /// `gf2p8affineinv` (0xdc): [`_svm512_gf2p8affine_epi64_epi8`] of the
    /// inverses of `x` in GF(2^8) (0 for 0).
    _svm512_gf2p8affineinv_epi64_epi8, _svm512_mask_gf2p8affineinv_epi64_epi8,
        _svm512_maskz_gf2p8affineinv_epi64_epi8 => __svm_gf2p8affineinv;
}

// ---- masked memory, gather and scatter (kinds `lsi`, `sti`, `ga`, `sc`) ----

macro_rules! masked_memory {
    ($($load:ident / $loadz:ident / $store:ident: $k:ty, $t:ty = $w:literal, $what:literal;)*) => {$(
        #[doc = concat!("`ld.m` (0x0a, imm8 ", $w, ") with merge masking: ", $what, " lane i is loaded from `p` if bit i of `k` is set, else taken from `src`.")]
        ///
        /// # Safety
        /// `p..p+64` must be readable, whatever the mask.
        #[inline(always)]
        pub unsafe fn $load(src: __svm512i, k: $k, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                fn __svm_ld_m_mask(src: u32, p: *const u8, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_ld_m_mask(src.0, p.cast(), k as u64, $w) })
        }
        #[doc = concat!("`ld.m` (0x0a, imm8 ", $w, ") with zero masking: ", $what, " lane i is loaded from `p` if bit i of `k` is set, else 0.")]
        ///
        /// # Safety
        /// `p..p+64` must be readable, whatever the mask.
        #[inline(always)]
        pub unsafe fn $loadz(k: $k, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                fn __svm_ld_m_maskz(p: *const u8, k: u64, imm: u32) -> u32;
            }
            __svm512i(unsafe { __svm_ld_m_maskz(p.cast(), k as u64, $w) })
        }
        #[doc = concat!("`st.m` (0x0b, imm8 ", $w, "): stores ", $what, " lane i of `a` to `p` if bit i of `k` is set. The other bytes are not written.")]
        ///
        /// # Safety
        /// `p..p+64` must be writable, whatever the mask.
        #[inline(always)]
        pub unsafe fn $store(p: *mut $t, k: $k, a: __svm512i) {
            unsafe extern "C" {
                fn __svm_st_m_mask(a: u32, p: *mut u8, k: u64, imm: u32);
            }
            unsafe { __svm_st_m_mask(a.0, p.cast(), k as u64, $w) }
        }
    )*};
}

masked_memory! {
    _svm512_mask_loadu_epi8 / _svm512_maskz_loadu_epi8 / _svm512_mask_storeu_epi8: __mmask64, i8 = 0, "u8";
    _svm512_mask_loadu_epi16 / _svm512_maskz_loadu_epi16 / _svm512_mask_storeu_epi16: __mmask32, i16 = 1, "u16";
    _svm512_mask_loadu_epi32 / _svm512_maskz_loadu_epi32 / _svm512_mask_storeu_epi32: __mmask16, i32 = 2, "u32";
    _svm512_mask_loadu_epi64 / _svm512_maskz_loadu_epi64 / _svm512_mask_storeu_epi64: __mmask8, i64 = 3, "u64";
}

/// log2 of a gather or scatter scale (1, 2, 4 or 8).
const fn log2_scale(scale: i32) -> u32 {
    match scale {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => panic!("scale must be 1, 2, 4 or 8"),
    }
}

unsafe extern "C" {
    fn __svm_gather_qq(idx: u32, p: *const u8, imm: u32) -> u32;
    fn __svm_gather_qq_mask(src: u32, idx: u32, p: *const u8, k: u64, imm: u32) -> u32;
    fn __svm_gather_qq_maskz(idx: u32, p: *const u8, k: u64, imm: u32) -> u32;
    fn __svm_scatter_qq(idx: u32, a: u32, p: *mut u8, imm: u32);
    fn __svm_scatter_qq_mask(idx: u32, a: u32, p: *mut u8, k: u64, imm: u32);
}

/// `gather.qq` (0xe8): lane i is the u64 at `p + idx.q[i] · SCALE`
/// (wrapping).
///
/// # Safety
/// Every lane's 8 bytes must be readable. They may lie in different regions.
#[inline(always)]
pub unsafe fn _svm512_i64gather_epi64<const SCALE: i32>(
    idx: __svm512i,
    p: *const i64,
) -> __svm512i {
    let imm = const { log2_scale(SCALE) };
    __svm512i(unsafe { __svm_gather_qq(idx.0, p.cast(), imm) })
}

/// [`_svm512_i64gather_epi64`] with merge masking: only the lanes whose bit
/// of `k` is set are loaded, the others keep `src`.
///
/// # Safety
/// The 8 bytes of every active lane must be readable.
#[inline(always)]
pub unsafe fn _svm512_mask_i64gather_epi64<const SCALE: i32>(
    src: __svm512i,
    k: __mmask8,
    idx: __svm512i,
    p: *const i64,
) -> __svm512i {
    let imm = const { log2_scale(SCALE) };
    __svm512i(unsafe { __svm_gather_qq_mask(src.0, idx.0, p.cast(), k as u64, imm) })
}

/// [`_svm512_i64gather_epi64`] with zero masking: only the lanes whose bit
/// of `k` is set are loaded, the others are 0.
///
/// # Safety
/// The 8 bytes of every active lane must be readable.
#[inline(always)]
pub unsafe fn _svm512_maskz_i64gather_epi64<const SCALE: i32>(
    k: __mmask8,
    idx: __svm512i,
    p: *const i64,
) -> __svm512i {
    let imm = const { log2_scale(SCALE) };
    __svm512i(unsafe { __svm_gather_qq_maskz(idx.0, p.cast(), k as u64, imm) })
}

/// `scatter.qq` (0xec): stores lane i of `a` to `p + idx.q[i] · SCALE`
/// (wrapping), in ascending lane order: a later lane wins where two overlap.
///
/// # Safety
/// Every lane's 8 bytes must be writable.
#[inline(always)]
pub unsafe fn _svm512_i64scatter_epi64<const SCALE: i32>(
    p: *mut i64,
    idx: __svm512i,
    a: __svm512i,
) {
    let imm = const { log2_scale(SCALE) };
    unsafe { __svm_scatter_qq(idx.0, a.0, p.cast(), imm) }
}

/// [`_svm512_i64scatter_epi64`] for the lanes whose bit of `k` is set.
///
/// # Safety
/// The 8 bytes of every active lane must be writable.
#[inline(always)]
pub unsafe fn _svm512_mask_i64scatter_epi64<const SCALE: i32>(
    p: *mut i64,
    k: __mmask8,
    idx: __svm512i,
    a: __svm512i,
) {
    let imm = const { log2_scale(SCALE) };
    unsafe { __svm_scatter_qq_mask(idx.0, a.0, p.cast(), k as u64, imm) }
}

macro_rules! gather {
    ($($name:ident / $mask:ident / $maskz:ident: $k:ty, $t:ty, $bytes:literal, $marker:literal, $doc:literal;)*) => {$(
        #[doc = $doc]
        ///
        /// # Safety
        #[doc = concat!("Every lane's ", $bytes, " bytes must be readable. They may lie in different regions.")]
        #[inline(always)]
        pub unsafe fn $name<const SCALE: i32>(idx: __svm512i, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = $marker]
                fn marker(idx: u32, p: *const u8, imm: u32) -> u32;
            }
            let imm = const { log2_scale(SCALE) };
            __svm512i(unsafe { marker(idx.0, p.cast(), imm) })
        }
        #[doc = concat!("[`", stringify!($name), "`] with merge masking: only the lanes whose bit of `k` is set are loaded, the others keep `src`.")]
        ///
        /// # Safety
        #[doc = concat!("The ", $bytes, " bytes of every active lane must be readable.")]
        #[inline(always)]
        pub unsafe fn $mask<const SCALE: i32>(
            src: __svm512i,
            k: $k,
            idx: __svm512i,
            p: *const $t,
        ) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(src: u32, idx: u32, p: *const u8, k: u64, imm: u32) -> u32;
            }
            let imm = const { log2_scale(SCALE) };
            __svm512i(unsafe { marker(src.0, idx.0, p.cast(), k as u64, imm) })
        }
        #[doc = concat!("[`", stringify!($name), "`] with zero masking: only the lanes whose bit of `k` is set are loaded, the others are 0.")]
        ///
        /// # Safety
        #[doc = concat!("The ", $bytes, " bytes of every active lane must be readable.")]
        #[inline(always)]
        pub unsafe fn $maskz<const SCALE: i32>(k: $k, idx: __svm512i, p: *const $t) -> __svm512i {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_maskz")]
                fn marker(idx: u32, p: *const u8, k: u64, imm: u32) -> u32;
            }
            let imm = const { log2_scale(SCALE) };
            __svm512i(unsafe { marker(idx.0, p.cast(), k as u64, imm) })
        }
    )*};
}

gather! {
    _svm512_i32gather_epi32 / _svm512_mask_i32gather_epi32 / _svm512_maskz_i32gather_epi32: __mmask16, i32, 4, "__svm_gather_dd",
        "`gather.dd` (0xe9): lane i (< 16) is the u32 at `p + idx.d[i] · SCALE` (signed index, wrapping).";
    _svm512_i32gather_epi64 / _svm512_mask_i32gather_epi64 / _svm512_maskz_i32gather_epi64: __mmask8, i64, 8, "__svm_gather_dq",
        "`gather.dq` (0xea): lane i (< 8) is the u64 at `p + idx.d[i] · SCALE` (signed index, wrapping). The indices are the low 256 bits of `idx`.";
    _svm512_i64gather_epi32 / _svm512_mask_i64gather_epi32 / _svm512_maskz_i64gather_epi32: __mmask8, i32, 4, "__svm_gather_qd",
        "`gather.qd` (0xeb): u32 lane i (< 8) is the u32 at `p + idx.q[i] · SCALE` (wrapping). The upper 256 bits of the result are 0.";
}

macro_rules! scatter {
    ($($name:ident / $mask:ident: $k:ty, $t:ty, $bytes:literal, $marker:literal, $doc:literal;)*) => {$(
        #[doc = $doc]
        ///
        /// # Safety
        #[doc = concat!("Every lane's ", $bytes, " bytes must be writable.")]
        #[inline(always)]
        pub unsafe fn $name<const SCALE: i32>(p: *mut $t, idx: __svm512i, a: __svm512i) {
            unsafe extern "C" {
                #[link_name = $marker]
                fn marker(idx: u32, a: u32, p: *mut u8, imm: u32);
            }
            let imm = const { log2_scale(SCALE) };
            unsafe { marker(idx.0, a.0, p.cast(), imm) }
        }
        #[doc = concat!("[`", stringify!($name), "`] for the lanes whose bit of `k` is set.")]
        ///
        /// # Safety
        #[doc = concat!("The ", $bytes, " bytes of every active lane must be writable.")]
        #[inline(always)]
        pub unsafe fn $mask<const SCALE: i32>(p: *mut $t, k: $k, idx: __svm512i, a: __svm512i) {
            unsafe extern "C" {
                #[link_name = concat!($marker, "_mask")]
                fn marker(idx: u32, a: u32, p: *mut u8, k: u64, imm: u32);
            }
            let imm = const { log2_scale(SCALE) };
            unsafe { marker(idx.0, a.0, p.cast(), k as u64, imm) }
        }
    )*};
}

scatter! {
    _svm512_i32scatter_epi32 / _svm512_mask_i32scatter_epi32: __mmask16, i32, 4, "__svm_scatter_dd",
        "`scatter.dd` (0xed): stores u32 lane i (< 16) of `a` to `p + idx.d[i] · SCALE` (signed index, wrapping), in ascending lane order: a later lane wins where two overlap.";
    _svm512_i32scatter_epi64 / _svm512_mask_i32scatter_epi64: __mmask8, i64, 8, "__svm_scatter_dq",
        "`scatter.dq` (0xee): stores u64 lane i (< 8) of `a` to `p + idx.d[i] · SCALE` (signed index, wrapping), in ascending lane order. The indices are the low 256 bits of `idx`.";
    _svm512_i64scatter_epi32 / _svm512_mask_i64scatter_epi32: __mmask8, i32, 4, "__svm_scatter_qd",
        "`scatter.qd` (0xef): stores u32 lane i (< 8) of `a` to `p + idx.q[i] · SCALE` (wrapping), in ascending lane order.";
}

// ---- 16 and 32 byte accesses, broadcasts from memory ----

unsafe extern "C" {
    fn __svm_ld_y(p: *const u8) -> u32;
    fn __svm_st_y(a: u32, p: *mut u8);
    fn __svm_ld_x(p: *const u8) -> u32;
    fn __svm_st_x(a: u32, p: *mut u8);
    fn __svm_ldbc_b(p: *const u8) -> u32;
    fn __svm_ldbc_w(p: *const u8) -> u32;
    fn __svm_ldbc_x(p: *const u8) -> u32;
    fn __svm_ldbc_y(p: *const u8) -> u32;
}

/// `ld.y` (0x0e): loads 32 bytes from `p` into the low half, the upper half
/// is 0.
///
/// # Safety
/// `p..p+32` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadu_si256(p: *const u8) -> __svm512i {
    __svm512i(unsafe { __svm_ld_y(p) })
}

/// `st.y` (0x0f): stores the low 32 bytes of `a` to `p`.
///
/// # Safety
/// `p..p+32` must be writable.
#[inline(always)]
pub unsafe fn _svm512_storeu_si256(p: *mut u8, a: __svm512i) {
    unsafe { __svm_st_y(a.0, p) }
}

/// `ld.x` (0xf0): loads 16 bytes from `p` into the low 128 bits, the rest
/// is 0.
///
/// # Safety
/// `p..p+16` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadu_si128(p: *const u8) -> __svm512i {
    __svm512i(unsafe { __svm_ld_x(p) })
}

/// `st.x` (0xf1): stores the low 16 bytes of `a` to `p`.
///
/// # Safety
/// `p..p+16` must be writable.
#[inline(always)]
pub unsafe fn _svm512_storeu_si128(p: *mut u8, a: __svm512i) {
    unsafe { __svm_st_x(a.0, p) }
}

/// `ldbc.b` (0xf2): broadcasts the byte at `p` to all 64 lanes.
///
/// # Safety
/// `p` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_epi8(p: *const i8) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_b(p.cast()) })
}

/// `ldbc.w` (0xf3): broadcasts the u16 at `p` to all 32 lanes.
///
/// # Safety
/// `p..p+2` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_epi16(p: *const i16) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_w(p.cast()) })
}

/// `ldbc.x` (0xf4): broadcasts the 16 bytes at `p` to all four 128-bit
/// lanes.
///
/// # Safety
/// `p..p+16` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_i32x4(p: *const i32) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_x(p.cast()) })
}

/// `ldbc.y` (0xf5): broadcasts the 32 bytes at `p` to both 256-bit halves.
///
/// # Safety
/// `p..p+32` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadbc_i64x4(p: *const i64) -> __svm512i {
    __svm512i(unsafe { __svm_ldbc_y(p.cast()) })
}

loadbc_masked! {
    _svm512_loadbc_epi8 => _svm512_mask_loadbc_epi8 / _svm512_maskz_loadbc_epi8: __mmask64, i8, 1, "__svm_ldbc_b";
    _svm512_loadbc_epi16 => _svm512_mask_loadbc_epi16 / _svm512_maskz_loadbc_epi16: __mmask32, i16, 2, "__svm_ldbc_w";
    _svm512_loadbc_i32x4 => _svm512_mask_loadbc_i32x4 / _svm512_maskz_loadbc_i32x4: __mmask16, i32, 16, "__svm_ldbc_x";
    _svm512_loadbc_i64x4 => _svm512_mask_loadbc_i64x4 / _svm512_maskz_loadbc_i64x4: __mmask8, i64, 32, "__svm_ldbc_y";
}

// ---- several registers (`ldn`, `stn`) ----

unsafe extern "C" {
    fn __svm_ldn(p: *const u8, k: u32, n: u32) -> u32;
    fn __svm_stn(a: u32, p: *mut u8, k: u32, n: u32);
}

/// `ldn` (0x0c): loads `N` vectors (2 ≤ N ≤ 30) from `p..p+64N`, exactly
/// like `N` single loads in ascending order. Where the pass cannot give the
/// results consecutive registers, it emits those single loads.
///
/// # Safety
/// `p..p+64N` must be readable.
#[inline(always)]
pub unsafe fn _svm512_loadu_n<const N: usize>(p: *const u8) -> [__svm512i; N] {
    const { assert!(2 <= N && N <= 30, "N must be 2..=30") };
    let mut out = [__svm512i(0); N];
    let mut k = 0;
    while k < N {
        out[k] = __svm512i(unsafe { __svm_ldn(p, k as u32, N as u32) });
        k += 1;
    }
    out
}

/// `stn` (0x0d): stores the `N` vectors of `a` (2 ≤ N ≤ 30) to `p..p+64N`,
/// exactly like `N` single stores in ascending order. Where the pass cannot
/// place them in consecutive registers, it emits those single stores.
///
/// # Safety
/// `p..p+64N` must be writable.
#[inline(always)]
pub unsafe fn _svm512_storeu_n<const N: usize>(p: *mut u8, a: [__svm512i; N]) {
    const { assert!(2 <= N && N <= 30, "N must be 2..=30") };
    let mut k = 0;
    while k < N {
        unsafe { __svm_stn(a[k].0, p, k as u32, N as u32) };
        k += 1;
    }
}
