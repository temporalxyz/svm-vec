//! 8 lanes of GF(2^255 - 19): r = a, then 256 times r = r * b, output canonical.
//!
//! region (1024 bytes): lane l has a at 64 l and b at 64 l + 32 (32 byte little endian, < p).
//! the program writes r at 512 + 64 l and b again at 512 + 64 l + 32.
//!
//! a field element is 5 vectors of radix 2^52 limbs, one lane per u64 lane. the multiplier is
//! fixed, so the program first computes b_i = b * 2^(52 i) mod p (5 limbs each, i = 0..5). then
//! r * b = sum_i r_i * b_i lands in columns 0..5 only: column j collects lo52(r_i b_i[j]) and
//! hi52(r_i b_i[j - 1]), column 5 (weight 2^260 = 608 mod p) is folded into columns 0 and 1, and
//! bit 255 and up of column 4 into column 0 (times 19). madd52 reads only the low 52 bits of its
//! inputs, so limbs 0..3 of r keep the bits that were already carried out of them.
//!
//! the 25 limbs of b_0..b_4 need not be distinct: with lo_m, hi_m the low and high 52 bits of
//! 608 b[m], b_i = [lo_(5-i), c_(6-i), .., c_4, x, b[1], .., b[4-i]] where c_m = lo_m + hi_(m-1)
//! and x = b[0] + hi_4. that is 13 vectors, which stay in registers next to r, the columns and the
//! constants. it holds exactly if c_2, c_3, c_4 and x are below 2^52 in every lane (for uniform
//! random b the chance that they are not is below 2^-38). otherwise the program takes the general
//! path: b_i reduced limb by limb, 16 vectors in registers and 9 reloaded from the stack in every
//! product.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

type V = __svm512i;

const ITERS: usize = 256;
const UNROLL: usize = 8; // general path
const OUT: usize = 512;
const M52: i64 = (1 << 52) - 1;
const M47: i64 = (1 << 47) - 1;

#[repr(C, align(64))]
struct Tab<const N: usize>([[u16; 32]; N]);

// perm2.w indices of the 8 x 8 u64 transpose: for d = 4, 2, 1, the rows i and i + d become
// [x_i[j] if j & d == 0 else x_{i+d}[j - d]] and [x_i[j + d] if j & d == 0 else x_{i+d}[j]]
const fn transpose_idx() -> Tab<6> {
    let mut t = [[0u16; 32]; 6];
    let mut s = 0;
    while s < 3 {
        let d = 4 >> s;
        let mut j = 0;
        while j < 8 {
            let (lo, hi) = if j & d == 0 { (j, j + d) } else { (8 + j - d, 8 + j) };
            let mut w = 0;
            while w < 4 {
                t[2 * s][4 * j + w] = (4 * lo + w) as u16;
                t[2 * s + 1][4 * j + w] = (4 * hi + w) as u16;
                w += 1;
            }
            j += 1;
        }
        s += 1;
    }
    Tab(t)
}
static TIDX: Tab<6> = transpose_idx();

#[inline(always)]
unsafe fn transpose(x: [V; 8]) -> [V; 8] {
    let t = TIDX.0.as_ptr() as *const u8;
    let mut x = x;
    let mut s = 0;
    while s < 3 {
        let d = 4 >> s;
        let lo = _svm512_loadu_si512(t.add(128 * s));
        let hi = _svm512_loadu_si512(t.add(128 * s + 64));
        let mut y = x;
        let mut i = 0;
        while i < 8 {
            if i & d == 0 {
                y[i] = _svm512_permutex2var_epi16(x[i], lo, x[i + d]);
                y[i + d] = _svm512_permutex2var_epi16(x[i], hi, x[i + d]);
            }
            i += 1;
        }
        x = y;
        s += 1;
    }
    x
}

#[inline(always)]
fn shr52(x: V) -> V {
    _svm512_srli_epi64::<52>(x)
}

/// exact limbs: 0..4 below 2^52, 4 below 2^47 + 2 (any u64 limbs with limb 4 below 2^58)
#[inline(always)]
fn normalize(n: [V; 5], m52: V, m47: V, c19: V) -> [V; 5] {
    let [mut n0, mut n1, mut n2, mut n3, mut n4] = n;
    for pass in 0..2 {
        if pass == 1 {
            let c = _svm512_srli_epi64::<47>(n4);
            n4 = _svm512_and_si512(n4, m47);
            n0 = _svm512_madd52lo_epu64(n0, c, c19);
        }
        n1 = _svm512_add_epi64(n1, shr52(n0));
        n0 = _svm512_and_si512(n0, m52);
        n2 = _svm512_add_epi64(n2, shr52(n1));
        n1 = _svm512_and_si512(n1, m52);
        n3 = _svm512_add_epi64(n3, shr52(n2));
        n2 = _svm512_and_si512(n2, m52);
        n4 = _svm512_add_epi64(n4, shr52(n3));
        n3 = _svm512_and_si512(n3, m52);
    }
    [n0, n1, n2, n3, n4]
}

/// radix 2^52 limbs of 4 u64 words (limbs 0..3 unmasked above bit 52)
#[inline(always)]
fn limbs(w: [V; 4]) -> [V; 5] {
    let f = |lo: V, hi: V, s: u32| match s {
        52 => _svm512_or_si512(_svm512_srli_epi64::<52>(lo), _svm512_slli_epi64::<12>(hi)),
        40 => _svm512_or_si512(_svm512_srli_epi64::<40>(lo), _svm512_slli_epi64::<24>(hi)),
        _ => _svm512_or_si512(_svm512_srli_epi64::<28>(lo), _svm512_slli_epi64::<36>(hi)),
    };
    [w[0], f(w[0], w[1], 52), f(w[1], w[2], 40), f(w[2], w[3], 28), _svm512_srli_epi64::<16>(w[3])]
}

/// r * b mod p (not canonical). `b(5 i + j)` is limb j of b_i; `wide` if column 5 can reach
/// 2^52 (the shared limbs are not reduced).
#[inline(always)]
fn fmul(r: [V; 5], b: impl Fn(usize) -> V, wide: bool, c608: V, c19: V, m47: V) -> [V; 5] {
    let mut z = [_svm512_setzero_si512(); 6];
    for i in 0..5 {
        for j in 0..5 {
            let bij = b(5 * i + j);
            z[j] = _svm512_madd52lo_epu64(z[j], r[i], bij);
            z[j + 1] = _svm512_madd52hi_epu64(z[j + 1], r[i], bij);
        }
    }
    let [mut z0, mut z1, mut z2, mut z3, mut z4, z5] = z;
    z0 = _svm512_madd52lo_epu64(z0, z5, c608);
    z1 = _svm512_madd52hi_epu64(z1, z5, c608);
    if wide {
        z1 = _svm512_madd52lo_epu64(z1, shr52(z5), c608);
    }
    let c = _svm512_srli_epi64::<47>(z4);
    z4 = _svm512_and_si512(z4, m47);
    z0 = _svm512_madd52lo_epu64(z0, c, c19);
    z1 = _svm512_add_epi64(z1, shr52(z0));
    z2 = _svm512_add_epi64(z2, shr52(z1));
    z3 = _svm512_add_epi64(z3, shr52(z2));
    z4 = _svm512_add_epi64(z4, shr52(z3));
    [z0, z1, z2, z3, z4]
}

macro_rules! x4 {
    ($($e:stmt;)*) => { $($e; $e; $e; $e;)* };
}
macro_rules! x8 {
    ($($e:stmt;)*) => { $($e; $e; $e; $e; $e; $e; $e; $e;)* };
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    // the fast path is straight-line code and always runs; its output is overwritten if the
    // shared limbs do not hold
    if !run(input, true) {
        general(input);
    }
    0
}

#[inline(never)]
unsafe fn general(input: *mut u8) {
    run(input, false);
}

/// writes the output; with `shared`, returns whether the shared limbs held
#[inline(always)]
unsafe fn run(input: *mut u8, shared: bool) -> bool {
    let m52 = _svm512_set1_epi64(M52);
    let m47 = _svm512_set1_epi64(M47);
    let c19 = _svm512_set1_epi64(19);
    let c608 = _svm512_set1_epi64(608);

    let mut x = [_svm512_setzero_si512(); 8];
    for l in 0..8 {
        x[l] = _svm512_loadu_si512(input.add(64 * l));
    }
    let t = transpose(x);
    let a = limbs([t[0], t[1], t[2], t[3]]);
    let mut bb = limbs([t[4], t[5], t[6], t[7]]);
    for j in 0..4 {
        bb[j] = _svm512_and_si512(bb[j], m52);
    }

    let mut spill = core::mem::MaybeUninit::<Tab<{ 13 * 2 }>>::uninit();
    let sp = spill.as_mut_ptr() as *mut u8;
    // b's words for the output, kept out of registers during the loop
    for k in 0..4 {
        _svm512_storeu_si512(sp.add(64 * (9 + k)), t[4 + k]);
    }

    let zero = _svm512_setzero_si512();
    let mut r = a;
    let mut held = true;
    if shared {
        let lo = |m: usize| _svm512_madd52lo_epu64(zero, bb[m], c608);
        let (lo1, lo2, lo3, lo4) = (lo(1), lo(2), lo(3), lo(4));
        let c2 = _svm512_madd52hi_epu64(lo2, bb[1], c608);
        let c3 = _svm512_madd52hi_epu64(lo3, bb[2], c608);
        let c4 = _svm512_madd52hi_epu64(lo4, bb[3], c608);
        let x = _svm512_madd52hi_epu64(bb[0], bb[4], c608);
        let over = _svm512_or_si512(_svm512_or_si512(c2, c3), _svm512_or_si512(c4, x));
        held = _svm512_reduce_add_epi64(_svm512_srli_epi64::<52>(over)) == 0;
        let [b0, b1, b2, b3, b4] = bb;
        let tab = [
            b0, b1, b2, b3, b4, //
            lo4, x, b1, b2, b3, //
            lo3, c4, x, b1, b2, //
            lo2, c3, c4, x, b1, //
            lo1, c2, c3, c4, x,
        ];
        x8! { x8! { x4! { r = fmul(r, |k| tab[k], true, c608, c19, m47); }; }; }
    } else {
        // b_i = b_{i-1} * 2^52 mod p, limbs reduced
        let mut b = [zero; 16];
        let mut cur = bb;
        for i in 0..5 {
            if i > 0 {
                cur = normalize([_svm512_mullo_epi64(cur[4], c608), cur[0], cur[1], cur[2], cur[3]], m52, m47, c19);
            }
            for j in 0..5 {
                let k = 5 * i + j;
                if k < 16 {
                    b[k] = cur[j];
                } else {
                    _svm512_storeu_si512(sp.add(64 * (k - 16)), cur[j]);
                }
            }
        }
        let b = |k: usize| if k < 16 { b[k] } else { _svm512_loadu_si512(sp.add(64 * (k - 16))) };
        for _ in 0..ITERS / UNROLL {
            x8! { r = fmul(r, b, false, c608, c19, m47); }
        }
    }

    // canonical: v = r < 2p, v - p = v + 19 - 2^255 if that is not negative
    let m52 = _svm512_set1_epi64(M52);
    let mut f = [_svm512_and_si512(r[0], m52), _svm512_and_si512(r[1], m52), _svm512_and_si512(r[2], m52),
        _svm512_and_si512(r[3], m52), r[4]];
    let mut s = _svm512_add_epi64(f[0], c19);
    for j in 1..5 {
        s = _svm512_add_epi64(f[j], shr52(s));
    }
    let c = _svm512_srli_epi64::<47>(s);
    f[0] = _svm512_madd52lo_epu64(f[0], c, c19);
    for j in 1..5 {
        f[j] = _svm512_add_epi64(f[j], shr52(f[j - 1]));
        f[j - 1] = _svm512_and_si512(f[j - 1], m52);
    }
    f[4] = _svm512_and_si512(f[4], m47);
    let w = [
        _svm512_or_si512(f[0], _svm512_slli_epi64::<52>(f[1])),
        _svm512_or_si512(_svm512_srli_epi64::<12>(f[1]), _svm512_slli_epi64::<40>(f[2])),
        _svm512_or_si512(_svm512_srli_epi64::<24>(f[2]), _svm512_slli_epi64::<28>(f[3])),
        _svm512_or_si512(_svm512_srli_epi64::<36>(f[3]), _svm512_slli_epi64::<16>(f[4])),
    ];
    let tb = |k: usize| _svm512_loadu_si512(sp.add(64 * (9 + k)));
    let o = transpose([w[0], w[1], w[2], w[3], tb(0), tb(1), tb(2), tb(3)]);
    for l in 0..8 {
        _svm512_storeu_si512(input.add(OUT + 64 * l), o[l]);
    }
    held
}
