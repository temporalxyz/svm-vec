// generates the vector part of the Falcon-512 verifier (OUT_DIR/core.rs): ||s2||^2, the forward
// NTT of s2, the product with the prepared public key, the inverse NTT, s1 = c - s2 h and
// ||s1||^2, as straight-line intrinsics. the op stream is simulated and checked against schoolbook
// arithmetic (with a 16 bit overflow and bound check of every NTT op) before the program is
// built. also emits the Keccak-f[1600] of ../keccak (OUT_DIR/keccak.rs).
//
// adapted from simd-falcon/program/build.rs in Blueshift's sbpf-simd experiment (Apache-2.0 / MIT):
// Firedancer's NTT over Z/12289 (Cooley-Tukey forward, Gentleman-Sande inverse, NTT(h) / N
// prepared at compile time), Montgomery products from 16 bit multiplies
// (mulhi(a, w) - mulhi(mullo(a, w q^-1), q)), lazy reduction where a tracked bound requires it,
// and the c - m stage with unsigned minima. differences: 32 coefficients per vector (16 vectors),
// SSA values instead of slots, and the 5 levels inside a vector run between whole vectors after a
// 2 x 2 transpose that swaps one word bit of the index with one vector bit (two perm2.w per pair).
use std::{collections::HashMap, fmt::Write, path::Path};

#[allow(dead_code)]
#[path = "../keccak/build.rs"]
mod keccak_gen;

const Q: i64 = 12289;
const N: usize = 512;
const I16_MAX: i64 = i16::MAX as i64;
/// bound of values that are not checked (products, the final stage)
const ANY: i64 = i64::MAX;

type Row = [i64; 32];

fn pow_mod(mut b: i64, mut e: i64) -> i64 {
    let mut r = 1;
    b %= Q;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % Q;
        }
        b = b * b % Q;
        e >>= 1;
    }
    r
}

fn bitrev9(x: usize) -> usize {
    (0..9).fold(0, |r, i| r | (((x >> i) & 1) << (8 - i)))
}

/// Firedancer's twiddles: bit-reversed powers of psi (a primitive 2N-th root of unity) and psi^-1
fn twiddles() -> ([i64; N], [i64; N]) {
    let exp = (Q - 1) / (2 * N as i64);
    let psi = (1..Q).map(|i| pow_mod(i, exp)).find(|g| pow_mod(*g, N as i64) != 1).unwrap();
    let powers = |x: i64| {
        let mut out = [1; N];
        let mut p = 1;
        for i in 1..N {
            p = p * x % Q;
            out[bitrev9(i)] = p;
        }
        out
    };
    (powers(psi), powers(pow_mod(psi, Q - 2)))
}

/// reference forward NTT, exact reduction
fn ntt_ref(a: &mut [i64; N], psi: &[i64; N]) {
    let (mut t, mut m) = (N, 1);
    while m < N {
        t >>= 1;
        for i in 0..m {
            for j in 2 * i * t..2 * i * t + t {
                let (u, v) = (a[j], a[j + t] * psi[m + i] % Q);
                a[j] = (u + v) % Q;
                a[j + t] = (u + Q - v) % Q;
            }
        }
        m <<= 1;
    }
}

fn qinv() -> i64 {
    (1..1 << 16).find(|x| (x * Q) & 0xffff == 1).unwrap()
}

/// Montgomery form (x 2^16 mod q), centered
fn mont_centered(x: i64) -> i64 {
    let m = (x.rem_euclid(Q) << 16) % Q;
    if m > Q / 2 { m - Q } else { m }
}

/// w q^-1 mod 2^16 as i16
fn mont_prime(w: i64) -> i64 {
    ((w * qinv()) & 0xffff) as i16 as i64
}

/// |mont(a, w)| for |a| <= a_bound: |a w| / 2^16 + 1 + 2^15 q / 2^16 (rounded up)
fn mont_bound(a_bound: i64, w_abs: i64) -> i64 {
    assert!(a_bound <= I16_MAX && w_abs <= Q / 2 + 1);
    a_bound * w_abs / 65536 + 1 + 6145
}

/// twiddle of block `i` at the level with distance `t`
fn twiddle(psi: &[i64; N], t: usize, i: usize) -> i64 {
    mont_centered(psi[N / (2 * t) + i])
}

#[derive(Clone, Copy, Debug)]
enum Src {
    Table(usize),
    H(usize),
    S2(usize),
    C(usize),
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Ld(usize, Src),
    Add(usize, usize, usize),
    Sub(usize, usize, usize),
    Mullo(usize, usize, usize),
    Mulhi(usize, usize, usize),
    Minu(usize, usize, usize),
    Madd(usize, usize, usize),
    /// (d, acc, a, b): dpwssd
    Dp(usize, usize, usize, usize),
    /// (d, a, idx, b): perm2.w
    Perm2(usize, usize, usize, usize),
}

#[derive(Default)]
struct Gen {
    ops: Vec<Op>,
    /// per value: max |x| of its i16 lanes (ANY: not checked)
    bound: Vec<i64>,
    table: Vec<Row>,
    rows: HashMap<Row, usize>,
    /// table row -> value, reset per level so constants are not live across the whole function
    cache: HashMap<usize, usize>,
}

/// data vectors and the coefficient index held by each of their words
struct Poly {
    v: [usize; 16],
    lay: [[usize; 32]; 16],
}

impl Gen {
    fn op(&mut self, bound: i64, f: impl FnOnce(usize) -> Op) -> usize {
        let d = self.bound.len();
        self.bound.push(bound);
        self.ops.push(f(d));
        d
    }

    fn row(&mut self, r: Row) -> usize {
        let index = match self.rows.get(&r) {
            Some(&i) => i,
            None => {
                self.table.push(r);
                self.rows.insert(r, self.table.len() - 1);
                self.table.len() - 1
            }
        };
        if let Some(&v) = self.cache.get(&index) {
            return v;
        }
        let v = self.op(r.iter().map(|x| x.abs()).max().unwrap(), |d| Op::Ld(d, Src::Table(index)));
        self.cache.insert(index, v);
        v
    }

    fn bcast(&mut self, x: i64) -> usize {
        self.row([x; 32])
    }

    /// mont(a, w) with w' = w q^-1 in `wp`
    fn mont(&mut self, a: usize, w: usize, wp: usize, w_abs: i64) -> usize {
        let b = mont_bound(self.bound[a], w_abs);
        let q = self.bcast(Q);
        let lo = self.op(ANY, |d| Op::Mullo(d, a, wp));
        let hi = self.op(ANY, |d| Op::Mulhi(d, a, w));
        let lo = self.op(ANY, |d| Op::Mulhi(d, lo, q));
        self.op(b, |d| Op::Sub(d, hi, lo))
    }

    fn reduce(&mut self, x: usize) -> usize {
        let one = mont_centered(1);
        let (w, wp) = (self.bcast(one), self.bcast(mont_prime(one)));
        self.mont(x, w, wp, one.abs())
    }

    fn reduce_if(&mut self, x: usize, limit: i64) -> usize {
        if self.bound[x] > limit { self.reduce(x) } else { x }
    }

    /// Cooley-Tukey: (a + b w, a - b w)
    fn ct(&mut self, a: usize, b: usize, w: (usize, usize, i64)) -> (usize, usize) {
        let bw = mont_bound(self.bound[b], w.2);
        let a = self.reduce_if(a, I16_MAX - bw);
        let out = self.bound[a] + bw;
        assert!(out <= I16_MAX);
        let t = self.mont(b, w.0, w.1, w.2);
        let b = self.op(out, |d| Op::Sub(d, a, t));
        (self.op(out, |d| Op::Add(d, a, t)), b)
    }

    /// Gentleman-Sande: (a + b, (a - b) w)
    fn gs(&mut self, mut a: usize, mut b: usize, w: (usize, usize, i64)) -> (usize, usize) {
        if self.bound[a] + self.bound[b] > I16_MAX {
            if self.bound[a] >= self.bound[b] { a = self.reduce(a) } else { b = self.reduce(b) }
            if self.bound[a] + self.bound[b] > I16_MAX {
                if self.bound[a] >= self.bound[b] { a = self.reduce(a) } else { b = self.reduce(b) }
            }
        }
        let sum = self.bound[a] + self.bound[b];
        assert!(sum <= I16_MAX);
        let t = self.op(sum, |d| Op::Sub(d, a, b));
        let a = self.op(sum, |d| Op::Add(d, a, b));
        (a, self.mont(t, w.0, w.1, w.2))
    }

    /// 2 x 2 transpose of word bit `wb` and vector bit `vb`: X (vb = 0) keeps its words with
    /// wb = 0 and gets those of Y (vb = 1) in the wb = 1 places, Y the rest
    fn swap(&mut self, p: &mut Poly, vb: usize, wb: usize) {
        let m = 1 << wb;
        let ia = self.row(std::array::from_fn(|w| if w & m == 0 { w as i64 } else { 32 + (w ^ m) as i64 }));
        let ib = self.row(std::array::from_fn(|w| if w & m == 0 { (w | m) as i64 } else { 32 + w as i64 }));
        for x in (0..16).filter(|x| x >> vb & 1 == 0) {
            let y = x | 1 << vb;
            let b = self.bound[p.v[x]].max(self.bound[p.v[y]]);
            let (vx, vy) = (p.v[x], p.v[y]);
            p.v[x] = self.op(b, |d| Op::Perm2(d, vx, ia, vy));
            p.v[y] = self.op(b, |d| Op::Perm2(d, vx, ib, vy));
            let (lx, ly) = (p.lay[x], p.lay[y]);
            p.lay[x] = std::array::from_fn(|w| if w & m == 0 { lx[w] } else { ly[w ^ m] });
            p.lay[y] = std::array::from_fn(|w| if w & m == 0 { lx[w | m] } else { ly[w] });
        }
    }

    /// one NTT level on coefficient bit `k`: forward (CT, psi) or inverse (GS, psi^-1)
    fn level(&mut self, p: &mut Poly, k: usize, psi: &[i64; N], forward: bool) {
        self.cache.clear();
        let vbit = |p: &Poly, b: usize| p.lay[1 << b][0] ^ p.lay[0][0];
        let vb = match (0..4).find(|&b| vbit(p, b) == 1 << k) {
            Some(vb) => vb,
            None => {
                let wb = (0..5).find(|&b| p.lay[0][1 << b] ^ p.lay[0][0] == 1 << k).unwrap();
                // swap out a bit whose level is done: the highest going forward, the lowest back
                let order = (0..4).map(|b| (vbit(p, b), b));
                let vb = if forward { order.max().unwrap().1 } else { order.min().unwrap().1 };
                self.swap(p, vb, wb);
                vb
            }
        };
        for x in (0..16).filter(|x| x >> vb & 1 == 0) {
            let y = x | 1 << vb;
            let w: Row = std::array::from_fn(|w| {
                let j = p.lay[x][w];
                assert!(j >> k & 1 == 0 && p.lay[y][w] == j | 1 << k);
                twiddle(psi, 1 << k, j >> (k + 1))
            });
            let tw = (self.row(w), self.row(w.map(mont_prime)), w.iter().map(|x| x.abs()).max().unwrap());
            let (a, b) = if forward { self.ct(p.v[x], p.v[y], tw) } else { self.gs(p.v[x], p.v[y], tw) };
            (p.v[x], p.v[y]) = (a, b);
        }
    }
}

struct Core {
    g: Gen,
    /// layout of the forward NTT output (H rows follow it)
    h_lay: [[usize; 32]; 16],
    /// m = s2 h (natural order) per vector, the accumulator, first op of the final stage
    m: [usize; 16],
    acc: usize,
    final_stage: usize,
}

fn generate(psi_pos: &[i64; N], psi_neg: &[i64; N]) -> Core {
    let mut g = Gen::default();
    let natural: [[usize; 32]; 16] = std::array::from_fn(|v| std::array::from_fn(|w| 32 * v + w));
    // decoded s2: |s2| <= 2047
    let mut p = Poly { v: std::array::from_fn(|k| g.op(2047, |d| Op::Ld(d, Src::S2(k)))), lay: natural };
    // ||s2||^2 (u32 lanes, at most 32 * 2047^2 + 32 * 6144^2 < 2^31 with ||s1||^2)
    let mut acc = g.op(ANY, |d| Op::Madd(d, p.v[0], p.v[0]));
    for k in 1..16 {
        let x = p.v[k];
        acc = g.op(ANY, |d| Op::Dp(d, acc, x, x));
    }
    for k in (0..9).rev() {
        g.level(&mut p, k, psi_pos, true);
    }
    // pointwise product with NTT(h) / N (Montgomery form, rows (H, H q^-1) per vector)
    g.cache.clear();
    for k in 0..16 {
        let h = g.op(Q / 2, |d| Op::Ld(d, Src::H(2 * k)));
        let hp = g.op(ANY, |d| Op::Ld(d, Src::H(2 * k + 1)));
        p.v[k] = g.mont(p.v[k], h, hp, Q / 2 + 1);
    }
    let h_lay = p.lay;
    for k in 0..9 {
        g.level(&mut p, k, psi_neg, false);
    }
    assert!(p.lay == natural);
    // s1 = c - m: c (u16 < 5q) to [0, 2q), u = c + q - m in [0, 4q) (m reduced below q), u to
    // [0, q), |s1| = min(u, q - u), squares summed in pairs into the u32 lanes
    g.cache.clear();
    let final_stage = g.ops.len();
    let mut m = [0; 16];
    for k in 0..16 {
        m[k] = g.reduce_if(p.v[k], Q);
        assert!(g.bound[m[k]] <= Q);
        let (q, q2, q4) = (g.bcast(Q), g.bcast(2 * Q), g.bcast(4 * Q));
        let c = g.op(ANY, |d| Op::Ld(d, Src::C(k)));
        let mut op = |f: fn(usize, usize, usize) -> Op, a: usize, b: usize| g.op(ANY, |d| f(d, a, b));
        let t = op(Op::Sub, c, q4);
        let u = op(Op::Minu, c, t); // [0, 4q) (5q > 4q values wrapped below q)
        let t = op(Op::Sub, u, q2);
        let u = op(Op::Minu, u, t); // [0, 2q)
        let u = op(Op::Add, u, q);
        let u = op(Op::Sub, u, m[k]); // [0, 4q)
        let t = op(Op::Sub, u, q2);
        let u = op(Op::Minu, u, t); // [0, 2q)
        let t = op(Op::Sub, u, q);
        let u = op(Op::Minu, u, t); // [0, q)
        let t = op(Op::Sub, q, u);
        let u = op(Op::Minu, u, t); // [0, q / 2]
        acc = g.op(ANY, |d| Op::Dp(d, acc, u, u));
    }
    Core { g, h_lay, m, acc, final_stage }
}

type V = [u16; 32];

/// runs the op stream, checking bounds and (before the final stage) 16 bit overflow
fn simulate(core: &Core, s2: &[i64; N], c: &[i64; N], h: &[Row]) -> Vec<V> {
    let g = &core.g;
    let mut v: Vec<V> = vec![[0; 32]; g.bound.len()];
    let row = |r: &Row| -> V { r.map(|x| x as u16) };
    let s = |x: u16| x as i16 as i64;
    let w2 = |a: &V, b: &V, f: &dyn Fn(u16, u16) -> u16| -> V { std::array::from_fn(|i| f(a[i], b[i])) };
    let dot = |a: &V, b: &V, i: usize| (s(a[2 * i]) * s(b[2 * i]) + s(a[2 * i + 1]) * s(b[2 * i + 1])) as u32;
    let from_d = |d: [u32; 16]| -> V { std::array::from_fn(|i| (d[i / 2] >> (16 * (i % 2))) as u16) };
    let to_d = |x: &V, i: usize| x[2 * i] as u32 | (x[2 * i + 1] as u32) << 16;
    for (n, op) in g.ops.iter().enumerate() {
        let d = match *op {
            Op::Ld(d, src) => {
                v[d] = match src {
                    Src::Table(r) => row(&g.table[r]),
                    Src::H(r) => row(&h[r]),
                    Src::S2(k) => std::array::from_fn(|w| s2[32 * k + w] as u16),
                    Src::C(k) => std::array::from_fn(|w| c[32 * k + w] as u16),
                };
                d
            }
            Op::Add(d, a, b) | Op::Sub(d, a, b) => {
                let sub = matches!(op, Op::Sub(..));
                if n < core.final_stage {
                    for i in 0..32 {
                        let r = if sub { s(v[a][i]) - s(v[b][i]) } else { s(v[a][i]) + s(v[b][i]) };
                        assert!(r.abs() <= I16_MAX, "overflow at op {n} {op:?}");
                    }
                }
                v[d] = w2(&v[a], &v[b], if sub { &|x, y| x.wrapping_sub(y) } else { &|x, y| x.wrapping_add(y) });
                d
            }
            Op::Mullo(d, a, b) => {
                v[d] = w2(&v[a], &v[b], &|x, y| x.wrapping_mul(y));
                d
            }
            Op::Mulhi(d, a, b) => {
                v[d] = w2(&v[a], &v[b], &|x, y| ((x as i16 as i32 * y as i16 as i32) >> 16) as u16);
                d
            }
            Op::Minu(d, a, b) => {
                v[d] = w2(&v[a], &v[b], &|x, y| x.min(y));
                d
            }
            Op::Madd(d, a, b) => {
                v[d] = from_d(std::array::from_fn(|i| dot(&v[a], &v[b], i)));
                d
            }
            Op::Dp(d, acc, a, b) => {
                v[d] = from_d(std::array::from_fn(|i| to_d(&v[acc], i).wrapping_add(dot(&v[a], &v[b], i))));
                d
            }
            Op::Perm2(d, a, idx, b) => {
                v[d] = std::array::from_fn(|i| {
                    let k = (v[idx][i] & 63) as usize;
                    if k < 32 { v[a][k] } else { v[b][k - 32] }
                });
                d
            }
        };
        if g.bound[d] != ANY {
            assert!(v[d].iter().all(|&x| s(x).abs() <= g.bound[d]), "value {d} exceeds {} after op {n} {op:?}", g.bound[d]);
        }
    }
    v
}

fn decode_pubkey(pk: &[u8]) -> [i64; N] {
    assert!(pk.len() == 897 && pk[0] == 0x09);
    let mut h = [0; N];
    let (mut acc, mut bits, mut idx) = (0u64, 0, 0);
    for byte in &pk[1..] {
        acc = acc << 8 | *byte as u64;
        bits += 8;
        if bits >= 14 {
            bits -= 14;
            h[idx] = ((acc >> bits) & 0x3fff) as i64;
            assert!(h[idx] < Q);
            idx += 1;
        }
    }
    h
}

/// NTT(h) / N in Montgomery form in the layout of the forward NTT output: rows (H, H q^-1)
fn prepare(h: &[i64; N], psi_pos: &[i64; N], lay: &[[usize; 32]; 16]) -> Vec<Row> {
    let mut hn = *h;
    ntt_ref(&mut hn, psi_pos);
    let n_inv = pow_mod(N as i64, Q - 2);
    lay.iter()
        .flat_map(|l| {
            let w: Row = l.map(|j| mont_centered(hn[j] * n_inv));
            [w, w.map(mont_prime)]
        })
        .collect()
}

fn self_test(core: &Core, psi_pos: &[i64; N]) {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rand = |m: i64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % m as u64) as i64
    };
    for round in 0..6 {
        let h: [i64; N] = std::array::from_fn(|_| rand(Q));
        // s2 with |s2| <= 2047: sparse, the extremes, random
        let s2: [i64; N] = std::array::from_fn(|i| match round {
            0 if i > 3 => 0,
            1 => 2047,
            2 => if i % 2 == 0 { -2047 } else { 2047 },
            _ => rand(4095) - 2047,
        });
        let c: [i64; N] = std::array::from_fn(|i| if round == 1 { 5 * Q - 1 - (i as i64 % 3) } else { rand(5 * Q) });
        let v = simulate(core, &s2, &c, &prepare(&h, psi_pos, &core.h_lay));
        let mut prod = [0i64; N];
        for i in 0..N {
            for j in 0..N {
                let x = s2[i] * h[j];
                if i + j < N { prod[i + j] += x } else { prod[i + j - N] -= x }
            }
        }
        let centered = |x: i64| {
            let x = x.rem_euclid(Q);
            if x > Q / 2 { x - Q } else { x }
        };
        for i in 0..N {
            let m = v[core.m[i / 32]][i % 32] as i16 as i64;
            assert_eq!((m - prod[i]).rem_euclid(Q), 0, "m[{i}] round {round}");
        }
        let norm: i64 = (0..N).map(|i| centered(c[i] - prod[i]).pow(2) + s2[i].pow(2)).sum();
        let acc = v[core.acc];
        let got: i64 = (0..16).map(|i| (acc[2 * i] as u32 | (acc[2 * i + 1] as u32) << 16) as i64).sum();
        assert_eq!(got, norm, "norm round {round}");
    }
}

fn emit_rows(s: &mut String, name: &str, rows: &[Row]) {
    writeln!(s, "pub static {name}: Rows<{}> = Rows([", rows.len()).unwrap();
    for r in rows {
        let r: Vec<String> = r.iter().map(|x| format!("{:#x}", *x as u16)).collect();
        writeln!(s, "    [{}],", r.join(", ")).unwrap();
    }
    writeln!(s, "]);").unwrap();
}

fn emit(core: &Core, h_rows: &[Row]) -> String {
    let g = &core.g;
    let mut s = String::new();
    writeln!(s, "// generated by build.rs: {} vector ops", g.ops.len()).unwrap();
    emit_rows(&mut s, "NT", &g.table);
    emit_rows(&mut s, "H", h_rows);
    writeln!(s, "/// sums of squares of s1 and s2 (u32 lanes, see build.rs)").unwrap();
    writeln!(s, "#[inline(always)]\nunsafe fn ntt_core(s2: *const u8, c: *const u8) -> __svm512i {{").unwrap();
    writeln!(s, "    let t = NT.0.as_ptr() as *const u8;\n    let h = H.0.as_ptr() as *const u8;").unwrap();
    for op in &g.ops {
        let bin = |name: &str, d: usize, a: usize, b: usize| format!("    let v{d} = _svm512_{name}(v{a}, v{b});");
        let line = match *op {
            Op::Ld(d, src) => {
                let (base, off) = match src {
                    Src::Table(r) => ("t", r),
                    Src::H(r) => ("h", r),
                    Src::S2(k) => ("s2", k),
                    Src::C(k) => ("c", k),
                };
                format!("    let v{d} = _svm512_loadu_si512({base}.add({}));", 64 * off)
            }
            Op::Add(d, a, b) => bin("add_epi16", d, a, b),
            Op::Sub(d, a, b) => bin("sub_epi16", d, a, b),
            Op::Mullo(d, a, b) => bin("mullo_epi16", d, a, b),
            Op::Mulhi(d, a, b) => bin("mulhi_epi16", d, a, b),
            Op::Minu(d, a, b) => bin("min_epu16", d, a, b),
            Op::Madd(d, a, b) => bin("madd_epi16", d, a, b),
            Op::Dp(d, acc, a, b) => format!("    let v{d} = _svm512_dpwssd_epi32(v{acc}, v{a}, v{b});"),
            Op::Perm2(d, a, i, b) => format!("    let v{d} = _svm512_permutex2var_epi16(v{a}, v{i}, v{b});"),
        };
        writeln!(s, "{line}").unwrap();
    }
    writeln!(s, "    v{}\n}}", core.acc).unwrap();
    s
}

/// straight-line sample taking (as in Blueshift's experiment): the rare rejection is a cold call,
/// so the common path is ldxh, stxh, add and a not taken branch
fn emit_take() -> String {
    let mut s = String::from("#[cold]\n#[inline(never)]\nfn rejected(p: *mut u16) -> *mut u16 {\n    p.wrapping_sub(1)\n}\n");
    s += "/// appends the samples of `o` (68 u16) below 5q at `p`, may write one more\n";
    s += "#[inline(always)]\nunsafe fn take(o: *const u16, mut p: *mut u16) -> *mut u16 {\n    unsafe {\n";
    for j in 0..68 {
        writeln!(s, "        let v = o.add({j}).read_volatile(); p.write_volatile(v); p = p.add(1); if v >= C_LIMIT {{ p = rejected(p); }}").unwrap();
    }
    s + "        p\n    }\n}\n"
}

fn main() {
    let pk_path = std::env::var("FALCON_PK").unwrap_or_else(|_| "fixtures/falcon.pk".to_string());
    for path in [&pk_path, "build.rs", "../keccak/build.rs", "../keccak/keccak.rs"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let (psi_pos, psi_neg) = twiddles();
    let core = generate(&psi_pos, &psi_neg);
    self_test(&core, &psi_pos);
    let h = decode_pubkey(&std::fs::read(&pk_path).expect("public key"));
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    std::fs::write(out.join("core.rs"), emit(&core, &prepare(&h, &psi_pos, &core.h_lay))).unwrap();
    std::fs::write(out.join("keccak.rs"), keccak_gen::emit() + &emit_take()).unwrap();
}
