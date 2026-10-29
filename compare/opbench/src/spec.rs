//! What every op benchmark computes, shared by both sides (ours: compare/opbench, Dean's:
//! compare/ops_dean.rs includes this file) and the host reference for it, written with the native
//! AVX-512 intrinsics.
//!
//! A program loads vectors 0..NV from the input region (vector k at byte 64 k), runs a loop of L
//! iterations whose body is K instances of one op, stores vectors 0..NV back and returns r0. The
//! baseline program is the same with an empty body. In the dependent variant every instance updates
//! vector 0 (one chain), in the independent one instance j updates vector 3 + j % 8. Sources are
//! vectors 1 and 2 (and 11), scalar operands are the constants S and M.
#![allow(clippy::missing_safety_doc, unused_unsafe)]

use std::arch::x86_64::*;
#[allow(unused_imports)]
use std::convert::TryInto;

pub const NV: usize = 12;
pub const NS: usize = 16;
pub const SCRATCH: usize = 1024;
pub const INPUT_LEN: usize = 4096;
pub const K: usize = 256;
pub const L: usize = 16;
pub const N: usize = K * L;
pub const S: u64 = 0x8bad_f00d_1234_5678;
pub const M: u64 = 0xa5c3_5a3c_96e1_0ff0;

pub fn chain(dep: bool, j: usize) -> usize {
    if dep {
        0
    } else {
        3 + j % 8
    }
}
pub fn mem_off(j: usize) -> usize {
    SCRATCH + 64 * (j % 32)
}
pub fn bulk_off(j: usize) -> usize {
    SCRATCH + 1024 * (j % 2)
}
pub fn bc_off(j: usize) -> usize {
    SCRATCH + 8 * (j % 32)
}
pub fn ro_off(j: usize) -> usize {
    8 * (j % 8)
}

pub fn input(seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..INPUT_LEN)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

pub struct St {
    pub v: [__m512i; NS],
    pub mem: Vec<u8>,
    pub r0: u64,
    pub ro: Vec<u8>,
}

pub type VF = unsafe fn(__m512i, __m512i, __m512i, __m512i) -> __m512i;
pub type SF = unsafe fn(__m512i, __m512i) -> u64;
pub type CF = unsafe fn(&mut St, bool, usize);

pub enum Ref {
    /// v[c] = f(v[c], v[1], v[2], v[11]), r0 = 0
    V(VF),
    /// r0 = f(v[3 + j % 8], v[1]), independent only
    S(SF),
    C(CF),
}

pub struct Bench {
    pub name: &'static str,
    pub dep: bool,
    pub ind: bool,
    pub r: Ref,
}

macro_rules! feat {
    ($($item:item)*) => {$(
        #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx512ifma,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,gfni,vpclmulqdq")]
        #[allow(unused_variables)]
        $item
    )*};
}
macro_rules! vf {
    (|$c:ident, $x:ident, $y:ident, $z:ident| $e:expr) => {{
        feat! { unsafe fn f($c: __m512i, $x: __m512i, $y: __m512i, $z: __m512i) -> __m512i { unsafe { $e } } }
        Ref::V(f)
    }};
}
macro_rules! sf {
    (|$c:ident, $x:ident| $e:expr) => {{
        feat! { unsafe fn f($c: __m512i, $x: __m512i) -> u64 { unsafe { $e } } }
        Ref::S(f)
    }};
}
macro_rules! cf {
    (|$s:ident, $dep:ident, $j:ident| $e:expr) => {{
        feat! { unsafe fn f($s: &mut St, $dep: bool, $j: usize) { unsafe { $e } } }
        Ref::C(f)
    }};
}

pub fn ld(p: &[u8]) -> __m512i {
    unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }
}
pub fn st(p: &mut [u8], v: __m512i) {
    unsafe { _mm512_storeu_si512(p.as_mut_ptr().cast(), v) }
}
pub fn q(v: __m512i) -> [u64; 8] {
    unsafe { std::mem::transmute(v) }
}
pub fn from_q(x: [u64; 8]) -> __m512i {
    unsafe { std::mem::transmute(x) }
}
fn zx128(x: __m128i) -> __m512i {
    unsafe { _mm512_zextsi128_si512(x) }
}
fn zx256(x: __m256i) -> __m512i {
    unsafe { _mm512_zextsi256_si512(x) }
}
fn lo128(x: __m512i) -> __m128i {
    unsafe { _mm512_castsi512_si128(x) }
}
fn lo256(x: __m512i) -> __m256i {
    unsafe { _mm512_castsi512_si256(x) }
}
fn movm8(k: u64) -> __m512i {
    unsafe { _mm512_movm_epi8(k) }
}
fn movm32(k: u16) -> __m512i {
    unsafe { _mm512_movm_epi32(k) }
}
fn movm64(k: u8) -> __m512i {
    unsafe { _mm512_movm_epi64(k) }
}
const M16: u16 = M as u16;

fn b(name: &'static str, r: Ref) -> Bench {
    Bench { name, dep: true, ind: true, r }
}
fn ind(name: &'static str, r: Ref) -> Bench {
    Bench { name, dep: false, ind: true, r }
}
fn dep(name: &'static str, r: Ref) -> Bench {
    Bench { name, dep: true, ind: false, r }
}

/// Every benchmark of either side. Names: `ours=dean` for the same function on both, `ours` or
/// `=dean` for an op only one design has. `(composite)` marks a sequence of the other design's ops.
pub fn benches() -> Vec<Bench> {
    vec![
        // Same function on both
        b("add.b=add8x64", vf!(|c, x, y, z| _mm512_add_epi8(c, x))),
        b("add.w=add16x32", vf!(|c, x, y, z| _mm512_add_epi16(c, x))),
        b("add.d=add32x16", vf!(|c, x, y, z| _mm512_add_epi32(c, x))),
        b("add.q=add64x8", vf!(|c, x, y, z| _mm512_add_epi64(c, x))),
        b("sub.d=sub32x16", vf!(|c, x, y, z| _mm512_sub_epi32(c, x))),
        b("sub.q=sub64x8", vf!(|c, x, y, z| _mm512_sub_epi64(c, x))),
        b("mullo.w=mullo16x32", vf!(|c, x, y, z| _mm512_mullo_epi16(c, x))),
        b("mullo.d=mullo32x16", vf!(|c, x, y, z| _mm512_mullo_epi32(c, x))),
        b("mullo.q=mullo64x8", vf!(|c, x, y, z| _mm512_mullo_epi64(c, x))),
        b("mulwu=mulu32x8", vf!(|c, x, y, z| _mm512_mul_epu32(c, x))),
        b("mulws=muls32x8", vf!(|c, x, y, z| _mm512_mul_epi32(c, x))),
        b("mulhi.w=mulhi16x32", vf!(|c, x, y, z| _mm512_mulhi_epi16(c, x))),
        b("mulhiu.w=mulhu16x32", vf!(|c, x, y, z| _mm512_mulhi_epu16(c, x))),
        b("max.d=max512.d", vf!(|c, x, y, z| _mm512_max_epi32(c, x))),
        b("max.w=max512.w", vf!(|c, x, y, z| _mm512_max_epi16(c, x))),
        b("minu.d=minu32x16", vf!(|c, x, y, z| _mm512_min_epu32(c, x))),
        b("minu.w=minu16x32", vf!(|c, x, y, z| _mm512_min_epu16(c, x))),
        b("minu.q=min512.uq", vf!(|c, x, y, z| _mm512_min_epu64(c, x))),
        b("and=and512", vf!(|c, x, y, z| _mm512_and_si512(c, x))),
        b("or=or512", vf!(|c, x, y, z| _mm512_or_si512(c, x))),
        b("xor=xor512", vf!(|c, x, y, z| _mm512_xor_si512(c, x))),
        b("andn=andn512", vf!(|c, x, y, z| _mm512_andnot_si512(c, x))),
        b("ternlog=ternlog512", vf!(|c, x, y, z| _mm512_ternarylogic_epi64::<0x96>(c, x, y))),
        b("shli.w=sll16x32", vf!(|c, x, y, z| _mm512_slli_epi16::<3>(c))),
        b("shli.d=sll32x16", vf!(|c, x, y, z| _mm512_slli_epi32::<7>(c))),
        b("shri.q=srl64x8", vf!(|c, x, y, z| _mm512_srli_epi64::<13>(c))),
        b("sari.q=sra64x8", vf!(|c, x, y, z| _mm512_srai_epi64::<13>(c))),
        b("roli.d=ror32x16", vf!(|c, x, y, z| _mm512_rol_epi32::<7>(c))),
        b("roli.q=ror64x8", vf!(|c, x, y, z| _mm512_rol_epi64::<13>(c))),
        b("shlv.d=sllv32x16", vf!(|c, x, y, z| _mm512_sllv_epi32(c, x))),
        b("shrv.q=srlv64x8", vf!(|c, x, y, z| _mm512_srlv_epi64(c, x))),
        b("rolv.d=rolv32x16", vf!(|c, x, y, z| _mm512_rolv_epi32(c, x))),
        b("rolv.q=rolv64x8", vf!(|c, x, y, z| _mm512_rolv_epi64(c, x))),
        b("dpbusd=dpbusd512", vf!(|c, x, y, z| _mm512_dpbusd_epi32(c, x, y))),
        b("dpwssd=dpwssd512", vf!(|c, x, y, z| _mm512_dpwssd_epi32(c, x, y))),
        b("maddubs=maddubs16x32", vf!(|c, x, y, z| _mm512_maddubs_epi16(c, x))),
        b("maddwd=madd16x32", vf!(|c, x, y, z| _mm512_madd_epi16(c, x))),
        b("madd52lo=madd52lo64x8", vf!(|c, x, y, z| _mm512_madd52lo_epu64(c, x, y))),
        b("madd52hi=madd52hi64x8", vf!(|c, x, y, z| _mm512_madd52hi_epu64(c, x, y))),
        b("packs.d=packss512.d", vf!(|c, x, y, z| _mm512_packs_epi32(c, x))),
        b("packus.w=packus512.w", vf!(|c, x, y, z| _mm512_packus_epi16(c, x))),
        b("permd=permd512", vf!(|c, x, y, z| _mm512_permutexvar_epi32(c, x))),
        b("shufb=shufb512", vf!(|c, x, y, z| _mm512_shuffle_epi8(c, x))),
        b("zx.b.d=zext512.bd", vf!(|c, x, y, z| _mm512_cvtepu8_epi32(lo128(c)))),
        b("sx.w.d=sext512.wd", vf!(|c, x, y, z| _mm512_cvtepi16_epi32(lo256(c)))),
        b("narrow.d.b=trunc512.db", vf!(|c, x, y, z| zx128(_mm512_cvtepi32_epi8(c)))),
        b("popcnt.b=popcnt512.b", vf!(|c, x, y, z| _mm512_popcnt_epi8(c))),
        b("popcnt.q=popcnt512.q", vf!(|c, x, y, z| _mm512_popcnt_epi64(c))),
        b("clmul=clmul512", vf!(|c, x, y, z| _mm512_clmulepi64_epi128::<0x01>(c, x))),
        ind("mov=mov512", vf!(|c, x, y, z| x)),
        ind("splat.d=bcast32x16", vf!(|c, x, y, z| _mm512_set1_epi32(S as i32))),
        ind("splat.q=bcast64x8", vf!(|c, x, y, z| _mm512_set1_epi64(S as i64))),
        b("insert.q=ins64", cf!(|s, dep, j| {
            let c = chain(dep, j);
            let mut l = q(s.v[c]);
            l[j % 8] = S;
            s.v[c] = from_q(l);
        })),
        ind("extract.q=ext64", sf!(|c, x| q(c)[5])),
        ind("movemask.b=movmsk8x64", sf!(|c, x| _mm512_movepi8_mask(c))),
        dep("insert.q+extract.q=ins64+ext64", cf!(|s, dep, j| {
            let mut l = q(s.v[0]);
            l[j % 8] = s.r0;
            s.r0 = l[(j + 1) % 8];
            s.v[0] = from_q(l);
        })),
        ind("ld=load256x2", cf!(|s, dep, j| {
            let c = chain(false, j);
            s.v[c] = ld(&s.mem[mem_off(j)..]);
        })),
        ind("st=store256x2", cf!(|s, dep, j| {
            let c = chain(false, j);
            st(&mut s.mem[mem_off(j)..], s.v[c]);
        })),
        ind("ld*16=load256x32", cf!(|s, dep, j| {
            for k in 0..16 {
                s.v[k] = ld(&s.mem[bulk_off(j) + 64 * k..]);
            }
        })),
        ind("st*16=store256x32", cf!(|s, dep, j| {
            for k in 0..16 {
                st(&mut s.mem[bulk_off(j) + 64 * k..], s.v[k]);
            }
        })),
        ind("hsum.d=(composite)", sf!(|c, x| _mm512_reduce_add_epi32(c) as u32 as u64)),
        ind("hsum.q=(composite)", sf!(|c, x| _mm512_reduce_add_epi64(c) as u64)),
        // Ours only
        ind("zero", vf!(|c, x, y, z| _mm512_setzero_si512())),
        ind("extract.d", sf!(|c, x| q(c)[5] >> 32)),
        ind("ldbc.d", cf!(|s, dep, j| {
            let o = bc_off(j);
            s.v[chain(false, j)] = _mm512_set1_epi32(i32::from_le_bytes(s.mem[o..o + 4].try_into().unwrap()));
        })),
        ind("ldbc.q", cf!(|s, dep, j| {
            let o = bc_off(j);
            s.v[chain(false, j)] = _mm512_set1_epi64(i64::from_le_bytes(s.mem[o..o + 8].try_into().unwrap()));
        })),
        b("cmpeq.b", vf!(|c, x, y, z| movm8(_mm512_cmpeq_epi8_mask(c, x)))),
        b("cmpeq.d", vf!(|c, x, y, z| movm32(_mm512_cmpeq_epi32_mask(c, x)))),
        b("cmpeq.q", vf!(|c, x, y, z| movm64(_mm512_cmpeq_epi64_mask(c, x)))),
        b("cmpgt.d", vf!(|c, x, y, z| movm32(_mm512_cmpgt_epi32_mask(c, x)))),
        b("cmpgt.q", vf!(|c, x, y, z| movm64(_mm512_cmpgt_epi64_mask(c, x)))),
        b("cmpgtu.q", vf!(|c, x, y, z| movm64(_mm512_cmpgt_epu64_mask(c, x)))),
        b("perm2.w", vf!(|c, x, y, z| _mm512_permutex2var_epi16(c, x, y))),
        b("permq", vf!(|c, x, y, z| _mm512_permutexvar_epi64(c, x))),
        b("narrowus.d.b", vf!(|c, x, y, z| zx128(_mm512_cvtusepi32_epi8(c)))),
        b("narrows.d.b", vf!(|c, x, y, z| zx128(_mm512_cvtsepi32_epi8(c)))),
        b("narrows.d.w", vf!(|c, x, y, z| zx256(_mm512_cvtsepi32_epi16(c)))),
        // Dean's only
        ind("=load256x2.rodata", cf!(|s, dep, j| {
            s.v[chain(false, j)] = ld(&s.ro[ro_off(j)..]);
        })),
        ind("=loadm512.d", cf!(|s, dep, j| {
            s.v[chain(false, j)] = _mm512_maskz_loadu_epi32(M16, s.mem[mem_off(j)..].as_ptr().cast());
        })),
        ind("=storem512.d", cf!(|s, dep, j| {
            let v = s.v[chain(false, j)];
            _mm512_mask_storeu_epi32(s.mem[mem_off(j)..].as_mut_ptr().cast(), M16, v);
        })),
        b("=compress512.d", cf!(|s, dep, j| {
            let c = chain(dep, j);
            s.v[c] = _mm512_maskz_compress_epi32(M16, s.v[c]);
            s.r0 = M16.count_ones() as u64;
        })),
        b("=compress512.b", cf!(|s, dep, j| {
            let c = chain(dep, j);
            s.v[c] = _mm512_maskz_compress_epi8(M, s.v[c]);
            s.r0 = M.count_ones() as u64;
        })),
        b("=expand512.d", vf!(|c, x, y, z| _mm512_maskz_expand_epi32(M16, c))),
        b("=blendm512.d", vf!(|c, x, y, z| _mm512_mask_blend_epi32(M16, x, c))),
        b("=conflict512.d", vf!(|c, x, y, z| _mm512_conflict_epi32(c))),
        b("=conflict512.q", vf!(|c, x, y, z| _mm512_conflict_epi64(c))),
        b("=lzcnt512.d", vf!(|c, x, y, z| _mm512_lzcnt_epi32(c))),
        dep("=transpose8x8q", cf!(|s, dep, j| {
            let m: Vec<[u64; 8]> = (0..8).map(|r| q(s.v[3 + r])).collect();
            for r in 0..8 {
                s.v[3 + r] = from_q(std::array::from_fn(|c| m[c][r]));
            }
        })),
        dep("=transpose8x8", cf!(|s, dep, j| {
            let mut m = [[0u32; 8]; 8];
            for r in 0..8 {
                let w: [u32; 16] = std::mem::transmute(s.v[3 + r / 2]);
                m[r].copy_from_slice(&w[8 * (r % 2)..8 * (r % 2) + 8]);
            }
            for v in 0..4 {
                let w: [u32; 16] = std::array::from_fn(|i| m[i % 8][2 * v + i / 8]);
                s.v[3 + v] = std::mem::transmute(w);
            }
        })),
        b("=mont16x32", vf!(|c, x, y, z| _mm512_sub_epi16(
            _mm512_mulhi_epi16(c, x),
            _mm512_mulhi_epi16(_mm512_mullo_epi16(c, y), z)
        ))),
        ind("=cmp512.gt.d", sf!(|c, x| _mm512_cmpgt_epi32_mask(c, x) as u64)),
        ind("=testm512.d", sf!(|c, x| _mm512_test_epi32_mask(c, x) as u64)),
        b("=permb512", vf!(|c, x, y, z| _mm512_permutexvar_epi8(c, x))),
        b("=permi2b512", vf!(|c, x, y, z| _mm512_permutex2var_epi8(x, c, y))),
        b("=permw512", vf!(|c, x, y, z| _mm512_permutexvar_epi16(c, x))),
        b("=permi2d512", vf!(|c, x, y, z| _mm512_permutex2var_epi32(x, c, y))),
        b("=permt2d512", vf!(|c, x, y, z| _mm512_permutex2var_epi32(c, x, y))),
        b("=multishift512", vf!(|c, x, y, z| _mm512_multishift_epi64_epi8(x, c))),
        b("=gf2p8affine512", vf!(|c, x, y, z| _mm512_gf2p8affine_epi64_epi8::<0x5a>(c, x))),
        b("=gf2p8mul512", vf!(|c, x, y, z| _mm512_gf2p8mul_epi8(c, x))),
        b("=sadbw512", vf!(|c, x, y, z| _mm512_sad_epu8(c, x))),
        b("=shldv512.d", vf!(|c, x, y, z| _mm512_shldv_epi32(c, x, y))),
        b("=alignr512", vf!(|c, x, y, z| _mm512_alignr_epi8::<5>(c, x))),
        b("=abs512.d", vf!(|c, x, y, z| _mm512_abs_epi32(c))),
        b("=adds512.w", vf!(|c, x, y, z| _mm512_adds_epi16(c, x))),
        b("=avg512.b", vf!(|c, x, y, z| _mm512_avg_epu8(c, x))),
        b("=permq512", vf!(|c, x, y, z| _mm512_permutex_epi64::<0x1b>(c))),
        b("=shufd512", vf!(|c, x, y, z| _mm512_shuffle_epi32::<0x1b>(c))),
        b("=blendd512", vf!(|c, x, y, z| _mm512_mask_blend_epi32(0xa5a5, c, x))),
        b("=unpacklo512.d", vf!(|c, x, y, z| _mm512_unpacklo_epi32(c, x))),
        b("=sllv16x32", vf!(|c, x, y, z| _mm512_sllv_epi16(c, x))),
        b("=srav16x32", vf!(|c, x, y, z| _mm512_srav_epi16(c, x))),
        b("=mulhrs16x32", vf!(|c, x, y, z| _mm512_mulhrs_epi16(c, x))),
    ]
}

/// Final input region and r0 of a benchmark program (`ro`: the read-only region)
pub fn reference(bench: &Bench, dep: bool, input: &[u8], ro: &[u8], k: usize, iterations: usize) -> (Vec<u8>, u64) {
    let mut s = St {
        v: [ld(&[0u8; 64]); NS],
        mem: input.to_vec(),
        r0: 0,
        ro: ro.to_vec(),
    };
    for k in 0..NV {
        s.v[k] = ld(&input[64 * k..]);
    }
    for _ in 0..iterations {
        for j in 0..k {
            unsafe {
                match bench.r {
                    Ref::V(f) => {
                        let c = chain(dep, j);
                        s.v[c] = f(s.v[c], s.v[1], s.v[2], s.v[11]);
                        s.r0 = 0;
                    }
                    Ref::S(f) => s.r0 = f(s.v[chain(false, j)], s.v[1]),
                    Ref::C(f) => f(&mut s, dep, j),
                }
            }
        }
    }
    for k in 0..NV {
        let v = s.v[k];
        st(&mut s.mem[64 * k..], v);
    }
    (s.mem, s.r0)
}

/// (instances per body, iterations) of the programs checked against the reference: the timed one,
/// and short ones in which each chain sees the op once or a few times (long chains of some ops
/// converge to values that would hide a wrong result)
pub const CHECKS: [(usize, usize); 3] = [(K, L), (1, 1), (8, 1)];

/// `name/dep` or `name/ind`
pub fn label(bench: &Bench, dep: bool) -> String {
    format!("{}/{}", bench.name, if dep { "dep" } else { "ind" })
}

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Back-to-back runs per batch so that a batch takes about 2 ms, from one run's time
pub fn repeat_for(single_ns: f64) -> usize {
    ((2e6 / single_ns.max(1.0)) as usize).clamp(1, 1_000_000)
}

pub const BATCHES: usize = 9;

pub struct Row {
    pub label: String,
    pub cu: f64,
    pub jit_ns: f64,
    pub interp_ns: f64,
}

/// The output line, and the ops above `budget` ns per CU on the JIT (to stderr)
pub fn report(rows: &[Row], ok: bool, budget: f64) -> String {
    let ops: Vec<String> = rows
        .iter()
        .map(|r| {
            let cu = r.cu.round().max(1.0);
            format!(
                "{{\"op\": \"{}\", \"cu\": {}, \"jit_ns_per_op\": {:.3}, \"interp_ns_per_op\": {:.3}, \"jit_ns_per_cu\": {:.3}, \"interp_ns_per_cu\": {:.3}}}",
                r.label, cu as u64, r.jit_ns, r.interp_ns, r.jit_ns / cu, r.interp_ns / cu
            )
        })
        .collect();
    for r in rows {
        let cu = r.cu.round().max(1.0);
        if r.jit_ns / cu > budget {
            eprintln!("above {budget} ns/CU on the JIT: {} ({:.2} ns/CU)", r.label, r.jit_ns / cu);
        }
    }
    format!("{{\"ops\": [{}], \"ok\": {}}}", ops.join(", "), ok)
}
