// SHA-256 compression of 16 blocks at once (lane i = block i) as SSA over u32x16 vectors, shared by
// programs/sha256/build.rs and compare/dean-sha256/build.rs. Each backend supplies the 16 message
// words W_t (big-endian words, lane = block) and maps the ops to its own instructions.
//
// constants are folded at generation time (rounds 0..3 start from the IV), the message schedule is
// computed one word ahead, and h + K + W of the next round is added before ch so that ch can
// overwrite g. ternlog operands are reordered so that the accumulator is a dying value where one
// exists (otherwise a copy is needed on both designs).
#![allow(dead_code)]

pub const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
pub const IV: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

pub const XOR3: u8 = 0x96;
pub const CH: u8 = 0xca; // acc ? a : b
pub const MAJ: u8 = 0xe8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    In(usize),                     // W_t, t < 16
    Const(u32),                    // broadcast
    Add(usize, usize),
    Ror(usize, u32),
    Shr(usize, u32),
    Tern(u8, usize, usize, usize), // bit = imm[acc << 2 | a << 1 | b]
}

pub struct Prog {
    pub ops: Vec<Op>,
    pub out: [usize; 8], // H_j = state_j + IV_j, each an Add(_, Const) used nowhere else
    pub last: Vec<usize>, // index of the last op reading each value (usize::MAX: an output)
}

#[derive(Clone, Copy)]
enum Val {
    C(u32),
    V(usize, u32), // vector + a constant not yet added
}

pub fn tern_eval(imm: u8, x: u32, y: u32, z: u32) -> u32 {
    (0..32).fold(0, |r, bit| {
        let i = (x >> bit & 1) << 2 | (y >> bit & 1) << 1 | (z >> bit & 1);
        r | ((imm as u32 >> i) & 1) << bit
    })
}

/// the same function with the operands in the order `perm` (new operand k = old operand perm[k])
fn tern_permute(imm: u8, perm: [usize; 3]) -> u8 {
    (0..8).fold(0, |r, j| {
        let mut old = [0; 3];
        for k in 0..3 {
            old[perm[k]] = (j >> (2 - k)) & 1;
        }
        r | ((imm >> (old[0] << 2 | old[1] << 1 | old[2])) & 1) << j
    })
}

#[derive(Default)]
struct B {
    ops: Vec<Op>,
}

impl B {
    fn op(&mut self, o: Op) -> usize {
        self.ops.push(o);
        self.ops.len() - 1
    }
    fn mat(&mut self, v: Val) -> usize {
        match v {
            Val::C(c) => self.op(Op::Const(c)),
            Val::V(i, 0) => i,
            Val::V(i, c) => {
                let k = self.op(Op::Const(c));
                self.op(Op::Add(i, k))
            }
        }
    }
    fn add(&mut self, x: Val, y: Val) -> Val {
        match (x, y) {
            (Val::C(a), Val::C(b)) => Val::C(a.wrapping_add(b)),
            (Val::C(a), Val::V(i, b)) | (Val::V(i, b), Val::C(a)) => Val::V(i, a.wrapping_add(b)),
            (Val::V(i, a), Val::V(j, b)) => Val::V(self.op(Op::Add(i, j)), a.wrapping_add(b)),
        }
    }
    fn ror(&mut self, x: Val, n: u32) -> Val {
        match x {
            Val::C(c) => Val::C(c.rotate_right(n)),
            v => {
                let i = self.mat(v);
                Val::V(self.op(Op::Ror(i, n)), 0)
            }
        }
    }
    fn shr(&mut self, x: Val, n: u32) -> Val {
        match x {
            Val::C(c) => Val::C(c >> n),
            v => {
                let i = self.mat(v);
                Val::V(self.op(Op::Shr(i, n)), 0)
            }
        }
    }
    fn tern(&mut self, imm: u8, x: Val, y: Val, z: Val) -> Val {
        if let (Val::C(a), Val::C(b), Val::C(c)) = (x, y, z) {
            return Val::C(tern_eval(imm, a, b, c));
        }
        let (a, b, c) = (self.mat(x), self.mat(y), self.mat(z));
        Val::V(self.op(Op::Tern(imm, a, b, c)), 0)
    }
    fn sigma(&mut self, x: Val, r: [u32; 2], s: u32) -> Val {
        let a = self.ror(x, r[0]);
        let b = self.ror(x, r[1]);
        let c = self.shr(x, s);
        self.tern(XOR3, a, b, c)
    }
    fn big_sigma(&mut self, x: Val, r: [u32; 3]) -> Val {
        let a = self.ror(x, r[0]);
        let b = self.ror(x, r[1]);
        let c = self.ror(x, r[2]);
        self.tern(XOR3, a, b, c)
    }
    fn fixed(&mut self, v: Val) -> Val {
        match v {
            Val::V(_, c) if c != 0 => Val::V(self.mat(v), 0),
            v => v,
        }
    }
}

pub fn generate() -> Prog {
    let mut b = B::default();
    let mut w: Vec<Val> = (0..16).map(|t| Val::V(b.op(Op::In(t)), 0)).collect();
    let sched = |b: &mut B, w: &mut Vec<Val>, t: usize| {
        while w.len() <= t {
            let n = w.len();
            let s1 = b.sigma(w[n - 2], [17, 19], 10);
            let s0 = b.sigma(w[n - 15], [7, 18], 3);
            let x = b.add(s1, w[n - 7]);
            let x = b.add(x, s0);
            let x = b.add(x, w[n - 16]);
            let x = b.fixed(x);
            w.push(x);
        }
    };
    let mut s: [Val; 8] = IV.map(Val::C);
    // x = h + K_t + W_t for the current round
    let kw = b.add(Val::C(K[0]), w[0]);
    let mut x = b.add(s[7], kw);
    for t in 0..64 {
        let [a, bb, c, d, e, f, g, _] = s;
        let next = if t + 1 < 64 {
            sched(&mut b, &mut w, t + 1);
            let kw = b.add(Val::C(K[t + 1]), w[t + 1]);
            Some(b.add(g, kw))
        } else {
            None
        };
        let s1 = b.big_sigma(e, [6, 11, 25]);
        let ch = b.tern(CH, e, f, g);
        let t1 = b.add(x, s1);
        let t1 = b.add(t1, ch);
        let s0 = b.big_sigma(a, [2, 13, 22]);
        let maj = b.tern(MAJ, a, bb, c);
        let t2 = b.add(s0, maj);
        // add T1's constant once unless both uses are constants (round 0)
        let t1 = if matches!((t2, d), (Val::C(_), Val::C(_))) { t1 } else { b.fixed(t1) };
        let na = b.add(t1, t2);
        let ne = b.add(d, t1);
        let na = if t == 63 { na } else { b.fixed(na) };
        let ne = if t == 63 { ne } else { b.fixed(ne) };
        s = [na, a, bb, c, ne, e, f, g];
        if let Some(n) = next {
            x = n;
        }
    }
    let mut out = [0; 8];
    for j in 0..8 {
        let v = b.add(s[j], Val::C(IV[j]));
        let Val::V(i, c) = v else { panic!("constant output") };
        let k = b.op(Op::Const(c));
        out[j] = b.op(Op::Add(i, k));
    }
    let mut ops = b.ops;
    // liveness, then put a dying operand in the accumulator position of each ternlog
    let mut last = vec![0usize; ops.len()];
    let uses = |o: &Op| -> Vec<usize> {
        match *o {
            Op::In(_) | Op::Const(_) => vec![],
            Op::Add(a, b) => vec![a, b],
            Op::Ror(a, _) | Op::Shr(a, _) => vec![a],
            Op::Tern(_, a, b, c) => vec![a, b, c],
        }
    };
    for (i, o) in ops.iter().enumerate() {
        for u in uses(o) {
            last[u] = i;
        }
    }
    for &o in &out {
        last[o] = usize::MAX;
    }
    for i in 0..ops.len() {
        if let Op::Tern(imm, x, y, z) = ops[i] {
            let v = [x, y, z];
            if last[x] != i {
                if let Some(k) = (1..3).find(|&k| last[v[k]] == i) {
                    let perm = if k == 1 { [1, 0, 2] } else { [2, 1, 0] };
                    ops[i] = Op::Tern(tern_permute(imm, perm), v[perm[0]], v[perm[1]], v[perm[2]]);
                }
            }
        }
    }
    let p = Prog { ops, out, last };
    check(&p);
    p
}

pub fn eval(p: &Prog, w: &[[u32; 16]; 16]) -> [[u32; 16]; 8] {
    let mut v: Vec<[u32; 16]> = Vec::with_capacity(p.ops.len());
    for o in &p.ops {
        let r = match *o {
            Op::In(t) => w[t],
            Op::Const(c) => [c; 16],
            Op::Add(a, b) => std::array::from_fn(|l| v[a][l].wrapping_add(v[b][l])),
            Op::Ror(a, n) => v[a].map(|x| x.rotate_right(n)),
            Op::Shr(a, n) => v[a].map(|x| x >> n),
            Op::Tern(imm, a, b, c) => std::array::from_fn(|l| tern_eval(imm, v[a][l], v[b][l], v[c][l])),
        };
        v.push(r);
    }
    p.out.map(|o| v[o])
}

pub fn compress(block: &[u32; 16]) -> [u32; 8] {
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(block);
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
    }
    let mut s = IV;
    for t in 0..64 {
        let [a, b, c, d, e, f, g, h] = s;
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        s = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
    }
    std::array::from_fn(|j| s[j].wrapping_add(IV[j]))
}

fn check(p: &Prog) {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..4 {
        let w: [[u32; 16]; 16] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u32
            })
        });
        let got = eval(p, &w);
        for l in 0..16 {
            let want = compress(&std::array::from_fn(|t| w[t][l]));
            for j in 0..8 {
                assert_eq!(got[j][l], want[j], "generated rounds disagree with the reference");
            }
        }
    }
}

/// ops per kind, for the generated-code header
pub fn summary(p: &Prog) -> String {
    let n = |f: fn(&Op) -> bool| p.ops.iter().filter(|o| f(o)).count();
    let copies = p.ops.iter().enumerate().filter(|(i, o)| matches!(o, Op::Tern(_, a, _, _) if p.last[*a] != *i)).count();
    format!(
        "add {} ror {} shr {} tern {} const {} (ternlog copies needed: {copies})",
        n(|o| matches!(o, Op::Add(..))),
        n(|o| matches!(o, Op::Ror(..))),
        n(|o| matches!(o, Op::Shr(..))),
        n(|o| matches!(o, Op::Tern(..))),
        n(|o| matches!(o, Op::Const(_))),
    )
}
