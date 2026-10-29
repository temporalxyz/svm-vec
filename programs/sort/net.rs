// Sorting network for 256 u32 keys held as 16 vectors x 16 lanes, shared by the build scripts of
// programs/sort (ours) and compare/dean-sort (Dean's). Both get the same network: Green's 60
// comparator 16-sorter across the 16 registers (every lane column sorted), then bitonic merges of
// 16 -> 32 -> ... -> 256 whose first step is mirrored, so every comparator is ascending.
//
// A key index k (its position in the sorted output) has 8 bits. The layout says which k each
// register lane holds. A comparator step on bit j is min/max of register pairs, so the layout must
// pair the registers lane for lane on bit j ("register bit"). Only 4 of the 8 bits can be register
// bits, so exchanges move a lane bit into a register: two two-source permutes per register pair.
// A beam search picks, per ISA, how each exchange is done (any permute, or the cheaper unpacks
// where the layout allows them) and which lane order it leaves behind.

use std::collections::HashMap;

pub type V = u32;
pub type Layout = [[u8; 16]; 16];

#[derive(Clone, Debug)]
pub enum Op {
    Load { d: V, k: usize },
    Min { d: V, a: V, b: V },
    Max { d: V, a: V, b: V },
    /// d[i] = (a ++ b)[idx[i]]
    Perm2 { d: V, a: V, b: V, idx: [u8; 16] },
    /// d[i] = a[idx[i]]
    Perm1 { d: V, a: V, idx: [u8; 16] },
    /// d[i] = mask bit i ? b[i] : a[i]
    Select { d: V, a: V, b: V, mask: u16 },
    Store { a: V, k: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Ours,
    Dean,
}

pub const NET16: [(usize, usize); 60] = [
    (0, 13), (1, 12), (2, 15), (3, 14), (4, 8), (5, 6), (7, 11), (9, 10),
    (0, 5), (1, 7), (2, 9), (3, 4), (6, 13), (8, 14), (10, 15), (11, 12),
    (0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 10), (5, 11), (6, 7), (8, 9), (12, 14), (13, 15),
    (1, 2), (3, 12), (4, 6), (5, 7), (8, 10), (9, 11), (13, 14),
    (1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14),
    (2, 4), (3, 6), (9, 12), (11, 13),
    (3, 5), (6, 8), (7, 9), (10, 12),
    (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
    (6, 7), (8, 9),
];

/// unpacklo / unpackhi (per 128 bit lane) of 32 bit (`q` false) or 64 bit lanes, as Perm2 indices
pub fn unpack(hi: bool, q: bool) -> [u8; 16] {
    let mut idx = [0u8; 16];
    for lane in 0..4 {
        let base = 4 * lane + if hi { 2 } else { 0 };
        for i in 0..4 {
            let (k, from_b) = if q { (i % 2, i >= 2) } else { (i / 2, i % 2 == 1) };
            idx[4 * lane + i] = (base + k) as u8 + if from_b { 16 } else { 0 };
        }
    }
    idx
}

/// (hi, q, operands swapped) if `idx` is an unpack
pub fn as_unpack(idx: &[u8; 16]) -> Option<(bool, bool, bool)> {
    for hi in [false, true] {
        for q in [false, true] {
            let u = unpack(hi, q);
            if *idx == u {
                return Some((hi, q, false));
            }
            if idx.iter().zip(u.iter()).all(|(a, b)| *a == b ^ 16) {
                return Some((hi, q, true));
            }
        }
    }
    None
}

#[derive(Clone)]
pub struct St {
    pub at: Layout,
    val: [V; 16],
    next: V,
    pub ops: Vec<Op>,
    pub cost: u32,
    isa: Isa,
}

#[derive(Clone, Copy, Debug)]
enum Act {
    /// move key bit `enter` into the register of key bit `leave`; `flip`: the next step is the
    /// mirrored first step of stage s, fold its lane permutation into this exchange if possible
    Round(u8, u8, Option<u8>),
    Step(u8),
    Flip(u8),
    InLane(u8),
}

impl St {
    fn new(isa: Isa, lanes: [u8; 4]) -> St {
        let mut s = St { at: [[0; 16]; 16], val: [0; 16], next: 0, ops: vec![], cost: 0, isa };
        for r in 0..16 {
            for l in 0..16 {
                let mut k = r as u8;
                for q in 0..4 {
                    if l >> q & 1 == 1 {
                        k |= 1 << lanes[q];
                    }
                }
                s.at[r][l] = k;
            }
            let d = s.fresh();
            s.ops.push(Op::Load { d, k: r });
            s.val[r] = d;
        }
        s
    }

    fn fresh(&mut self) -> V {
        self.next += 1;
        self.next - 1
    }

    fn pos(&self) -> [(usize, usize); 256] {
        let mut p = [(0, 0); 256];
        for r in 0..16 {
            for l in 0..16 {
                p[self.at[r][l] as usize] = (r, l);
            }
        }
        p
    }

    fn minmax(&mut self, lo: usize, hi: usize) {
        let (a, b) = (self.val[lo], self.val[hi]);
        let (d0, d1) = (self.fresh(), self.fresh());
        self.ops.push(Op::Min { d: d0, a, b });
        self.ops.push(Op::Max { d: d1, a, b });
        self.val[lo] = d0;
        self.val[hi] = d1;
        self.cost += 2;
    }

    fn perm1(&mut self, r: usize, idx: [u8; 16]) {
        if (0..16).all(|l| idx[l] as usize == l) {
            return;
        }
        let d = self.fresh();
        self.ops.push(Op::Perm1 { d, a: self.val[r], idx });
        let old = self.at[r];
        for l in 0..16 {
            self.at[r][l] = old[idx[l] as usize];
        }
        self.val[r] = d;
        self.cost += 1;
    }

    fn column(&mut self) {
        for (i, j) in NET16 {
            self.minmax(i, j);
        }
    }

    /// pair bit of the registers if key bit `x` is a register bit
    fn reg_bit(&self, x: u8) -> Option<usize> {
        let bit = |r: usize| self.at[r][0] >> x & 1;
        if (0..16).any(|r| (0..16).any(|l| self.at[r][l] >> x & 1 != bit(r))) {
            return None;
        }
        [1, 2, 4, 8].into_iter().find(|&p| (0..16).all(|r| bit(r) != bit(r ^ p)))
    }

    fn lane_bits(&self) -> Vec<u8> {
        (0..8).filter(|&b| (0..16).any(|l| (self.at[0][l] ^ self.at[0][0]) >> b & 1 == 1)).collect()
    }

    /// comparators k vs k ^ m, min to the one with bit `low` clear; aligns partners with Perm1
    fn step(&mut self, m: u8, low: u8) -> bool {
        let pos = self.pos();
        for r in 0..16 {
            let lower = self.at[r][0] >> low & 1 == 0;
            if (0..16).any(|l| (self.at[r][l] >> low & 1 == 0) != lower) {
                return false;
            }
            if !lower {
                continue;
            }
            let r2 = pos[(self.at[r][0] ^ m) as usize].0;
            let mut idx = [0u8; 16];
            for l in 0..16 {
                let (rr, ll) = pos[(self.at[r][l] ^ m) as usize];
                if rr != r2 {
                    return false;
                }
                idx[l] = ll as u8;
            }
            self.perm1(r2, idx);
            self.minmax(r, r2);
        }
        true
    }

    /// comparators on a lane bit: partner by Perm1, min, max, select
    fn inlane(&mut self, j: u8) -> bool {
        for r in 0..16 {
            let mut idx = [0u8; 16];
            let mut mask = 0u16;
            for l in 0..16 {
                let x = self.at[r][l];
                match (0..16).position(|ll| self.at[r][ll] == x ^ (1 << j)) {
                    Some(p) => idx[l] = p as u8,
                    None => return false,
                }
                if x >> j & 1 == 1 {
                    mask |= 1 << l;
                }
            }
            let a = self.val[r];
            let (p, lo, hi, d) = (self.fresh(), self.fresh(), self.fresh(), self.fresh());
            self.ops.push(Op::Perm1 { d: p, a, idx });
            self.ops.push(Op::Min { d: lo, a, b: p });
            self.ops.push(Op::Max { d: hi, a, b: p });
            self.ops.push(Op::Select { d, a: lo, b: hi, mask });
            self.val[r] = d;
            self.cost += 4;
        }
        true
    }

    /// rearranges each register pair (r, r | p) into `t`
    fn round(&mut self, p: usize, t: &Layout) -> bool {
        for r in (0..16).filter(|r| r & p == 0) {
            let r2 = r | p;
            let mut idx = [[0u8; 16]; 2];
            for (o, rr) in [r, r2].into_iter().enumerate() {
                for l in 0..16 {
                    let x = t[rr][l];
                    let src = (0..16)
                        .position(|ll| self.at[r][ll] == x)
                        .map(|ll| ll as u8)
                        .or_else(|| (0..16).position(|ll| self.at[r2][ll] == x).map(|ll| ll as u8 + 16));
                    match src {
                        Some(s) => idx[o][l] = s,
                        None => return false,
                    }
                }
            }
            let (a, b) = (self.val[r], self.val[r2]);
            let (d0, d1) = (self.fresh(), self.fresh());
            let u = [as_unpack(&idx[0]).is_some(), as_unpack(&idx[1]).is_some()];
            // a cheap (non destructive) one first, so the other can overwrite a dead input
            if !u[0] && u[1] {
                self.ops.push(Op::Perm2 { d: d1, a, b, idx: idx[1] });
                self.ops.push(Op::Perm2 { d: d0, a, b, idx: idx[0] });
            } else {
                self.ops.push(Op::Perm2 { d: d0, a, b, idx: idx[0] });
                self.ops.push(Op::Perm2 { d: d1, a, b, idx: idx[1] });
            }
            self.val[r] = d0;
            self.val[r2] = d1;
            self.cost += match self.isa {
                // perm2.w overwrites its first table: one mov per pair
                Isa::Ours => 3,
                // unpacks are non destructive, permt2d overwrites its first table
                Isa::Dean => 2 + (!u[0] && !u[1]) as u32,
            };
        }
        self.at = *t;
        true
    }

    /// target of an exchange done with arbitrary permutes: registers keep their other bits, key bit
    /// `e` picks the register of the pair, lane bits in `order`
    fn canon(&self, p: usize, e: u8, order: &[u8], flip: Option<u8>) -> Option<Layout> {
        let mut t = [[0u8; 16]; 16];
        let mut used = [[false; 16]; 16];
        for r in 0..16 {
            for l in 0..16 {
                let x = self.at[r][l];
                let nr = (r & !p) | if x >> e & 1 == 1 { p } else { 0 };
                let mut nl = 0;
                for (q, &b) in order.iter().enumerate() {
                    nl |= ((x >> b & 1) as usize) << q;
                }
                if let Some(s) = flip {
                    if x >> s & 1 == 1 {
                        for (q, &b) in order.iter().enumerate() {
                            if b < s {
                                nl ^= 1 << q;
                            }
                        }
                    }
                }
                if used[nr][nl] {
                    return None;
                }
                used[nr][nl] = true;
                t[nr][nl] = x;
            }
        }
        Some(t)
    }

    /// each register a block of 16 consecutive keys in lane order, then the stores
    fn finish(&mut self) -> bool {
        for r in 0..16 {
            let mn = *self.at[r].iter().min().unwrap();
            if mn % 16 != 0 {
                return false;
            }
            let mut idx = [0u8; 16];
            for l in 0..16 {
                match self.at[r].iter().position(|&x| x == mn + l as u8) {
                    Some(p) => idx[l] = p as u8,
                    None => return false,
                }
            }
            self.perm1(r, idx);
        }
        for r in 0..16 {
            self.ops.push(Op::Store { a: self.val[r], k: self.at[r][0] as usize / 16 });
        }
        true
    }

    fn apply(&self, act: Act) -> Vec<St> {
        let one = |f: &dyn Fn(&mut St) -> bool| {
            let mut s = self.clone();
            if f(&mut s) {
                vec![s]
            } else {
                vec![]
            }
        };
        match act {
            Act::Step(j) => one(&|s| s.step(1 << j, j)),
            Act::Flip(s0) => one(&|s| s.step(((2u16 << s0) - 1) as u8, s0)),
            Act::InLane(j) => one(&|s| s.inlane(j)),
            Act::Round(e, x, flip) => {
                let Some(p) = self.reg_bit(x) else { return vec![] };
                let mut lanes = self.lane_bits();
                let Some(i) = lanes.iter().position(|&b| b == e) else { return vec![] };
                lanes[i] = x;
                let mut out = vec![];
                for order in perms(&lanes) {
                    for f in [flip, None] {
                        if let Some(t) = self.canon(p, e, &order, f) {
                            let mut s = self.clone();
                            if s.round(p, &t) {
                                out.push(s);
                            }
                        }
                        if flip.is_none() {
                            break;
                        }
                    }
                }
                if self.isa == Isa::Dean {
                    for q in [false, true] {
                        for sw in [false, true] {
                            let x16 = if sw { 16 } else { 0 };
                            let (lo, hi) = (unpack(false, q).map(|i| i ^ x16), unpack(true, q).map(|i| i ^ x16));
                            let mut t = self.at;
                            for r in (0..16).filter(|r| r & p == 0) {
                                let src = |i: u8| {
                                    if i < 16 {
                                        self.at[r][i as usize]
                                    } else {
                                        self.at[r | p][i as usize - 16]
                                    }
                                };
                                for l in 0..16 {
                                    t[r][l] = src(lo[l]);
                                    t[r | p][l] = src(hi[l]);
                                }
                            }
                            let mut s = self.clone();
                            if s.round(p, &t) && s.reg_bit(e).is_some() {
                                out.push(s);
                            }
                        }
                    }
                }
                out
            }
        }
    }
}

fn perms(v: &[u8]) -> Vec<Vec<u8>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = vec![];
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in perms(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

fn schedule(end: usize) -> Vec<Act> {
    use Act::*;
    let mut v = vec![];
    // merges 16 -> 32 -> 64 -> 128: registers start and end each stage holding key bits 0..3
    for s in 4..7 {
        v.push(Round(s, 0, Some(s)));
        v.push(Flip(s));
        for b in (4..s).rev() {
            v.push(Round(b, b + 1, None));
            v.push(Step(b));
        }
        v.extend([Step(3), Step(2), Step(1), Round(0, 4, None), Step(0)]);
    }
    // 128 -> 256, ending with registers holding key bits 4..7 (the store layout)
    v.extend([Round(7, 0, Some(7)), Flip(7)]);
    for b in (4..7).rev() {
        v.push(Round(b, b + 1, None));
        v.push(Step(b));
    }
    v.extend([Step(3), Round(5, 3, None), Step(2), Round(6, 2, None), Step(1), Round(7, 1, None)]);
    match end {
        0 => v.push(InLane(0)),
        _ => v.extend([Round(0, 4, None), Step(0), Round(4, 0, None)]),
    }
    v
}

const BEAM: usize = 48;

/// the cheapest network found for `isa`, checked on random inputs
pub fn build(isa: Isa) -> St {
    let mut best: Option<St> = None;
    for end in 0..2 {
        let mut beam: Vec<St> = perms(&[4, 5, 6, 7])
            .into_iter()
            .map(|o| {
                let mut s = St::new(isa, [o[0], o[1], o[2], o[3]]);
                s.column();
                s
            })
            .collect();
        if isa == Isa::Ours {
            beam.truncate(1);
        }
        for act in schedule(end) {
            let mut next: HashMap<Layout, St> = HashMap::new();
            for st in &beam {
                for s in st.apply(act) {
                    if next.get(&s.at).map_or(true, |o| s.cost < o.cost) {
                        next.insert(s.at, s);
                    }
                }
            }
            let mut v: Vec<St> = next.into_values().collect();
            v.sort_by(|a, b| (a.cost, a.at).cmp(&(b.cost, b.at)));
            v.truncate(BEAM);
            assert!(!v.is_empty(), "no layout for {act:?}");
            beam = v;
        }
        for mut s in beam {
            if s.finish() && best.as_ref().map_or(true, |b| s.cost < b.cost) {
                best = Some(s);
            }
        }
    }
    let best = best.expect("no network");
    check(&best.ops);
    best
}

/// the abstract ops on concrete lanes
pub fn simulate(ops: &[Op], keys: &[u32; 256]) -> [u32; 256] {
    let mut v: HashMap<V, [u32; 16]> = HashMap::new();
    let mut out = [0u32; 256];
    for op in ops {
        match op {
            Op::Load { d, k } => {
                v.insert(*d, keys[16 * k..16 * k + 16].try_into().unwrap());
            }
            Op::Min { d, a, b } => {
                let r = std::array::from_fn(|i| v[a][i].min(v[b][i]));
                v.insert(*d, r);
            }
            Op::Max { d, a, b } => {
                let r = std::array::from_fn(|i| v[a][i].max(v[b][i]));
                v.insert(*d, r);
            }
            Op::Perm2 { d, a, b, idx } => {
                let r = std::array::from_fn(|i| {
                    let j = idx[i] as usize;
                    if j < 16 { v[a][j] } else { v[b][j - 16] }
                });
                v.insert(*d, r);
            }
            Op::Perm1 { d, a, idx } => {
                let r = std::array::from_fn(|i| v[a][idx[i] as usize]);
                v.insert(*d, r);
            }
            Op::Select { d, a, b, mask } => {
                let r = std::array::from_fn(|i| if mask >> i & 1 == 1 { v[b][i] } else { v[a][i] });
                v.insert(*d, r);
            }
            Op::Store { a, k } => out[16 * k..16 * k + 16].copy_from_slice(&v[a]),
        }
    }
    out
}

fn check(ops: &[Op]) {
    let mut ks: Vec<usize> = ops.iter().filter_map(|o| if let Op::Store { k, .. } = o { Some(*k) } else { None }).collect();
    ks.sort();
    assert_eq!(ks, (0..16).collect::<Vec<_>>());
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for t in 0..2000 {
        let keys: [u32; 256] = std::array::from_fn(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let k = (x >> 16) as u32;
            match t % 4 {
                0 => k,
                1 => k % 3,
                2 => k & 1,
                _ => k | 0x8000_0000,
            }
        });
        let mut want = keys;
        want.sort();
        assert_eq!(simulate(ops, &keys), want, "network does not sort");
    }
}

/// vector ops excluding loads and stores, by kind
pub fn histogram(ops: &[Op]) -> String {
    let mut h = [0usize; 5];
    for op in ops {
        match op {
            Op::Min { .. } | Op::Max { .. } => h[0] += 1,
            Op::Perm2 { idx, .. } if as_unpack(idx).is_some() => h[1] += 1,
            Op::Perm2 { .. } => h[2] += 1,
            Op::Perm1 { .. } => h[3] += 1,
            Op::Select { .. } => h[4] += 1,
            _ => {}
        }
    }
    format!("min/max {} unpack {} perm2 {} perm1 {} select {}", h[0], h[1], h[2], h[3], h[4])
}
