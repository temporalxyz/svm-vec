// generates programs/fieldmul's algorithm as straight-line code on Dean's intrinsics (slot numbers
// must be compile time constants for the JIT to inline the ops). see ../../programs/fieldmul for the
// arithmetic: radix 2^52, b_i = b * 2^(52 i) mod p precomputed, columns 0..5, fold, carry chain.
// here all 25 b_i limbs live in slots, and madd52 reads them as memory operands.
use std::{fmt::Write, fs, path::Path};

const ITERS: usize = 256;
// slots (32 bytes each); 512 bit operands are even slot pairs, shrd / ternlog need slots below 128
const X: u8 = 0; // 8 pairs: the input lanes, transposed: pairs 0..4 words of a, 4..8 words of b
const M52: u8 = 16;
const M47: u8 = 18;
const C19: u8 = 20;
const C608: u8 = 22;
const S: [u8; 2] = [24, 36]; // two sets of 6 pairs: r of one product, columns of the next
const T: u8 = 48;
const B: u8 = 64; // 25 pairs, b_i limb j at B + 2 (5 i + j)

struct G {
    s: String,
    ops: usize,
}

impl G {
    fn call(&mut self, f: &str, ops: u32) {
        self.ops += 1;
        writeln!(self.s, "    {f}({ops});").unwrap();
    }
    fn call2(&mut self, f: &str, ops: u32, r2: &str) {
        self.ops += 1;
        writeln!(self.s, "    {f}({ops}, {r2});").unwrap();
    }
    fn o3(&mut self, f: &str, d: u8, a: u8, b: u8) {
        self.call(f, d as u32 | (a as u32) << 8 | (b as u32) << 16);
    }
    fn lo(&mut self, d: u8, a: u8, b: u8) {
        self.o3("sol_avx512_madd52lo64x8", d, a, b);
    }
    fn hi(&mut self, d: u8, a: u8, b: u8) {
        self.o3("sol_avx512_madd52hi64x8", d, a, b);
    }
    fn add(&mut self, d: u8, a: u8, b: u8) {
        self.o3("sol_avx512_add64x8", d, a, b);
    }
    fn and(&mut self, d: u8, a: u8, b: u8) {
        self.o3("sol_avx512_and512", d, a, b);
    }
    fn or(&mut self, d: u8, a: u8, b: u8) {
        self.o3("sol_avx512_or512", d, a, b);
    }
    fn srl(&mut self, d: u8, a: u8, n: u8) {
        self.o3("sol_avx512_srl64x8", d, a, n);
    }
    fn sll(&mut self, d: u8, a: u8, n: u8) {
        self.o3("sol_avx512_sll64x8", d, a, n);
    }
    // (lo >> n) | (hi << (64 - n))
    fn shrd(&mut self, d: u8, lo: u8, hi: u8, n: u8) {
        let imm = 3 << 6 | n as u32;
        self.call("sol_avx512_shrd512", d as u32 | (lo as u32) << 7 | (hi as u32) << 14 | imm << 21);
    }
    fn bcast(&mut self, d: u8, v: u64) {
        self.call2("sol_avx512_bcast64x8", d as u32, &format!("{v}"));
    }
    // hi += lo >> 52
    fn carry(&mut self, lo: u8, hi: u8) {
        self.srl(T, lo, 52);
        self.add(hi, hi, T);
    }
    /// exact limbs (limbs 0..4 below 2^52, limb 4 below 2^47 + 2) in place
    fn normalize(&mut self, n: [u8; 5]) {
        for pass in 0..2 {
            if pass == 1 {
                self.srl(T + 2, n[4], 47);
                self.and(n[4], n[4], M47);
                self.lo(n[0], T + 2, C19);
            }
            for j in 0..4 {
                self.carry(n[j], n[j + 1]);
                self.and(n[j], n[j], M52);
            }
        }
    }
}

fn main() {
    let mut g = G { s: String::new(), ops: 0 };
    let b = |i: usize, j: usize| B + 2 * (5 * i + j) as u8;
    g.s += "#[inline(never)]\nunsafe fn vector(data: *const u8, out: *mut u8) {\n";
    g.call2("sol_avx_load256", X as u32 | 15 << 24, "data as u64");
    g.call("sol_avx512_transpose8x8q", X as u32);
    g.bcast(M52, (1 << 52) - 1);
    g.bcast(M47, (1 << 47) - 1);
    g.bcast(C19, 19);
    g.bcast(C608, 608);

    // limbs of a into S[0] (unmasked above bit 52, madd52 ignores those bits), of b into b_0
    let w = |k: u8| X + 2 * k;
    for (base, dst, mask) in [(0, [0, 1, 2, 3, 4].map(|j| S[0] + 2 * j), false), (4, [0, 1, 2, 3, 4].map(|j| b(0, j as usize)), true)] {
        g.o3("sol_avx512_mov512", dst[0], w(base), 0);
        g.shrd(dst[1], w(base), w(base + 1), 52);
        g.shrd(dst[2], w(base + 1), w(base + 2), 40);
        g.shrd(dst[3], w(base + 2), w(base + 3), 28);
        g.srl(dst[4], w(base + 3), 16);
        if mask {
            for j in 0..4 {
                g.and(dst[j], dst[j], M52);
            }
        }
    }
    // b_i = b_{i-1} * 2^52 mod p
    for i in 1..5 {
        g.o3("sol_avx512_mullo64x8", b(i, 0), b(i - 1, 4), C608);
        for j in 1..5 {
            g.o3("sol_avx512_mov512", b(i, j), b(i - 1, j - 1), 0);
        }
        g.normalize([0, 1, 2, 3, 4].map(|j| b(i, j)));
    }
    let setup = g.ops;

    for n in 0..ITERS {
        let r = |j: usize| S[n % 2] + 2 * j as u8;
        let z = |j: usize| S[(n + 1) % 2] + 2 * j as u8;
        // zero idiom on a cached pair. a bulk load of 6 zero pairs from .rodata (one op) costs
        // fewer CUs but makes the JIT write the pairs back and reload them at 512 bit width
        for j in 0..6 {
            g.o3("sol_avx512_sub64x8", z(j), r(0), r(0));
        }
        for i in 0..5 {
            for j in 0..5 {
                g.lo(z(j), r(i), b(i, j));
                g.hi(z(j + 1), r(i), b(i, j));
            }
        }
        g.lo(z(0), z(5), C608);
        g.hi(z(1), z(5), C608);
        g.srl(T, z(4), 47);
        g.and(z(4), z(4), M47);
        g.lo(z(0), T, C19);
        for j in 0..4 {
            g.carry(z(j), z(j + 1));
        }
    }
    let per_mul = (g.ops - setup) / ITERS;

    // canonical: v = r < 2p, v - p = v + 19 - 2^255 if that is not negative
    let r = |j: usize| S[ITERS % 2] + 2 * j as u8;
    for j in 0..4 {
        g.and(r(j), r(j), M52);
    }
    g.add(T + 2, r(0), C19);
    for j in 1..5 {
        g.srl(T + 2, T + 2, 52);
        g.add(T + 2, r(j), T + 2);
    }
    g.srl(T + 2, T + 2, 47);
    g.lo(r(0), T + 2, C19);
    for j in 0..4 {
        g.carry(r(j), r(j + 1));
        g.and(r(j), r(j), M52);
    }
    g.and(r(4), r(4), M47);
    // words over the pairs of a, then back to one lane per pair: (r, b) words of lane l in pair l
    for (k, (lo, hi, right, left)) in [(0, 1, 0, 52), (1, 2, 12, 40), (2, 3, 24, 28), (3, 4, 36, 16)].into_iter().enumerate() {
        g.sll(T, r(hi), left);
        if right == 0 {
            g.or(w(k as u8), r(lo), T);
        } else {
            g.srl(w(k as u8), r(lo), right);
            g.or(w(k as u8), w(k as u8), T);
        }
    }
    g.call("sol_avx512_transpose8x8q", X as u32);
    g.call2("sol_avx_store256", X as u32 | 15 << 24, "out as u64");
    g.s += "}\n";
    writeln!(g.s, "pub const VECTOR_OPS: usize = {};", g.ops).unwrap();
    writeln!(g.s, "pub const OPS_PER_PRODUCT: usize = {per_mul};").unwrap();
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), g.s).unwrap();
}
