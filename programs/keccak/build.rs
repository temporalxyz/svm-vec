// generates the vector Keccak-f[1600] as straight-line intrinsics (OUT_DIR/vector.rs) and checks
// it against keccak.rs by simulating the generated ops before the program is built.
//
// layouts: a vector holds 5 state lanes (q-lanes 0..5, 5..8 stay zero). theta needs every vector
// to hold one lane per column x with the lane index a function of x ("x layout"); chi needs the
// x + 1 and x + 2 neighbours in the same lane of other vectors ("y layout", lane index a function
// of y), or, when a vector is a whole row, lane rotations of itself. pi only moves lanes inside a
// vector (one permq each), so the sets of positions per vector follow pi: rows -> columns ->
// lines of slope 3 -> 2 -> 4 -> 1 -> rows (period 6). on columns theta needs horizontal xors.
// the planner picks, per round, lane orders (lane = s*x + o or t*y + o mod 5) so that every lane
// permutation is a rotation of lanes 0..5: four index vectors cover all of them, and rotation 0
// costs nothing.
use std::{collections::BTreeMap, fmt::Write, fs, path::Path};

#[path = "keccak.rs"]
mod keccak;

type Pos = (u8, u8); // (x, y)
type Layout = Vec<[Pos; 5]>; // 5 vectors x lanes 0..5

const XOR3: u8 = 0x96; // acc ^ a ^ b
const CHI: u8 = 0xd2; // acc ^ (!a & b)

fn pi(p: Pos) -> Pos {
    (p.1, (2 * p.0 + 3 * p.1) % 5)
}

/// rotation `a` with `dst[j] = src[(j + a) % 5]` (permq index (j + a) % 5), if there is one
fn rot(src: &[Pos; 5], dst: &[Pos; 5]) -> Option<usize> {
    (0..5).find(|&a| (0..5).all(|j| dst[j] == src[(j + a) % 5]))
}

/// the same sets of positions, lane `f(p)` for position `p`
fn place(l: &Layout, f: impl Fn(Pos) -> usize) -> Option<Layout> {
    l.iter()
        .map(|v| {
            let mut out = [(9, 9); 5];
            for &p in v {
                out[f(p)] = p;
            }
            out.iter().all(|p| p.0 < 9).then_some(out)
        })
        .collect()
}

fn x_layout(l: &Layout) -> bool {
    (0..5).all(|j| l.iter().all(|v| v[j].0 == l[0][j].0)) && (0..5).all(|j| (0..j).all(|i| l[0][i].0 != l[0][j].0))
}

fn distinct(v: &[Pos; 5], f: impl Fn(Pos) -> u8) -> usize {
    let mut seen = [false; 5];
    v.iter().for_each(|&p| seen[f(p) as usize] = true);
    seen.iter().filter(|&&b| b).count()
}

#[derive(Clone)]
enum Chi {
    /// relabel to the y layout, chi across vectors, then relabel to the x layout (if any)
    Cross(Layout, Option<Layout>),
    /// every vector is a row: chi on lane rotations of each vector, output in this x layout
    Rows(Layout),
}

fn params() -> impl Iterator<Item = (usize, usize)> {
    (1..5).flat_map(|s| (0..5).map(move |o| (s, o)))
}

fn perms(src: &Layout, dst: &Layout) -> Option<usize> {
    src.iter().zip(dst).map(|(a, b)| rot(a, b).map(|r| (r != 0) as usize)).sum()
}

/// (estimated ops, chi plan, next layout) for each way to run one round from `l`
fn options(l: &Layout, last: bool) -> Vec<(usize, Chi, Layout)> {
    let theta = if x_layout(l) {
        11
    } else if l.iter().all(|v| distinct(v, |p| p.0) == 1) {
        35
    } else {
        return vec![];
    };
    let base = theta + 5 + 6; // + rho + 6 table loads
    let post: Layout = l.iter().map(|v| v.map(pi)).collect();
    let natural = |x: &Layout| !last || x.iter().all(|v| (0..5).all(|j| v[j].0 as usize == j));
    let mut out = vec![];
    if post.iter().all(|v| distinct(v, |p| p.1) == 1) {
        for (s, o) in params() {
            let Some(x) = place(&post, |p| (s * p.0 as usize + o) % 5) else { continue };
            let n: Option<usize> = (0..5)
                .flat_map(|k| (0..3).map(move |sh| (k, sh)))
                .map(|(k, sh)| rot(&post[k], &x[k].map(|q| ((q.0 + sh) % 5, q.1))).map(|r| (r != 0) as usize))
                .sum();
            if let (Some(n), true) = (n, natural(&x)) {
                out.push((base + n + 6, Chi::Rows(x.clone()), x));
            }
        }
    } else if post.iter().all(|v| distinct(v, |p| p.1) == 5) {
        let columns = post.iter().all(|v| distinct(v, |p| p.0) == 1);
        for (t, o) in params() {
            let y = place(&post, |p| (t * p.1 as usize + o) % 5).unwrap();
            let Some(n) = perms(&post, &y) else { continue };
            if columns {
                if !last {
                    out.push((base + n + 7, Chi::Cross(y.clone(), None), y));
                }
                continue;
            }
            for (s, o) in params() {
                let Some(x) = place(&post, |p| (s * p.0 as usize + o) % 5) else { continue };
                if let (Some(m), true) = (perms(&y, &x), natural(&x)) {
                    out.push((base + n + 7 + m, Chi::Cross(y.clone(), Some(x.clone())), x));
                }
            }
        }
    }
    out
}

fn plan(start: &Layout) -> Vec<Chi> {
    let mut dp: BTreeMap<Layout, (usize, Vec<Chi>)> = BTreeMap::from([(start.clone(), (0, vec![]))]);
    for r in 0..24 {
        let mut next: BTreeMap<Layout, (usize, Vec<Chi>)> = BTreeMap::new();
        for (l, (c, hist)) in &dp {
            for (cost, chi, l2) in options(l, r == 23) {
                if next.get(&l2).is_none_or(|e| e.0 > c + cost) {
                    let mut h = hist.clone();
                    h.push(chi);
                    next.insert(l2, (c + cost, h));
                }
            }
        }
        dp = next;
    }
    dp.into_values().min_by_key(|e| e.0).expect("no plan").1
}

#[derive(Clone, Copy)]
enum Op {
    Ld(usize, usize),                  // d <- table row
    Xor(usize, usize, usize),          // d <- a ^ b
    Tern(usize, u8, usize, usize, usize), // d <- ternlog(imm, acc, a, b)
    Perm(usize, usize, usize),         // d <- permq(ROT[r - 1], a)
    Roli(usize, usize),                // d <- rol(a, 1)
    Rolv(usize, usize, usize),         // d <- rolv(a, b)
}

#[derive(Default)]
struct Gen {
    ops: Vec<Op>,
    n: usize,
    table: Vec<[u64; 8]>,
}

impl Gen {
    fn op(&mut self, f: impl Fn(usize) -> Op) -> usize {
        self.n += 1;
        self.ops.push(f(self.n - 1));
        self.n - 1
    }
    fn ld(&mut self, row: [u64; 8]) -> usize {
        self.table.push(row);
        let i = self.table.len() - 1;
        self.op(|d| Op::Ld(d, i))
    }
    fn perm(&mut self, r: usize, a: usize) -> usize {
        if r == 0 { a } else { self.op(|d| Op::Perm(d, r, a)) }
    }
    fn xor(&mut self, a: usize, b: usize) -> usize {
        self.op(|d| Op::Xor(d, a, b))
    }
    fn tern(&mut self, imm: u8, acc: usize, a: usize, b: usize) -> usize {
        self.op(|d| Op::Tern(d, imm, acc, a, b))
    }
}

fn rc_row(v: &[Pos; 5], round: usize) -> Option<[u64; 8]> {
    let j = v.iter().position(|&p| p == (0, 0))?;
    let mut row = [0; 8];
    row[j] = keccak::RC[round];
    Some(row)
}

/// emits one permutation: input/output ids are the rows y = 0..5 in natural order (lane x)
fn generate(g: &mut Gen, plan: &[Chi], mut cur: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut l: Layout = (0..5).map(|y| std::array::from_fn(|x| (x as u8, y))).collect();
    let mut per_round = vec![];
    for (round, chi) in plan.iter().enumerate() {
        let start = g.ops.len();
        // theta
        if x_layout(&l) {
            let c = g.xor(cur[0], cur[1]);
            let c = g.tern(XOR3, c, cur[2], cur[3]);
            let c = g.xor(c, cur[4]);
            let xs: [Pos; 5] = l[0].map(|p| (p.0, 0));
            let m1 = g.perm(rot(&xs, &xs.map(|p| ((p.0 + 4) % 5, 0))).unwrap(), c);
            let p1 = g.perm(rot(&xs, &xs.map(|p| ((p.0 + 1) % 5, 0))).unwrap(), c);
            let p1 = g.op(|d| Op::Roli(d, p1));
            for v in cur.iter_mut() {
                *v = g.tern(XOR3, *v, m1, p1);
            }
        } else {
            // columns: broadcast column parities h = v ^ v>>1 ^ v>>2 ^ v>>3 ^ v>>4 (lane rotations)
            let h: Vec<usize> = cur
                .clone()
                .into_iter()
                .map(|v| {
                    let a = g.perm(1, v);
                    let a = g.xor(v, a);
                    let b = g.perm(2, a);
                    let c = g.perm(4, v);
                    g.tern(XOR3, a, b, c)
                })
                .collect();
            let col = |x: u8| l.iter().position(|v| v[0].0 == x % 5).unwrap();
            for k in 0..5 {
                let x = l[k][0].0;
                let p1 = g.op(|d| Op::Roli(d, h[col(x + 1)]));
                cur[k] = g.tern(XOR3, cur[k], h[col(x + 4)], p1);
            }
        }
        // rho
        for k in 0..5 {
            let mut row = [0u64; 8];
            for j in 0..5 {
                row[j] = keccak::RHO[(l[k][j].0 + 5 * l[k][j].1) as usize] as u64;
            }
            let r = g.ld(row);
            cur[k] = g.op(|d| Op::Rolv(d, cur[k], r));
        }
        // pi (lanes stay) + chi + iota
        let post: Layout = l.iter().map(|v| v.map(pi)).collect();
        match chi {
            Chi::Cross(y, x) => {
                let u: Vec<usize> = (0..5).map(|k| g.perm(rot(&post[k], &y[k]).unwrap(), cur[k])).collect();
                // vector holding the x + 1 neighbours, same lanes
                let nb = |k: usize| {
                    let n = y.iter().position(|v| v[0] == ((y[k][0].0 + 1) % 5, y[k][0].1)).unwrap();
                    assert!((0..5).all(|j| y[n][j] == ((y[k][j].0 + 1) % 5, y[k][j].1)));
                    n
                };
                let a = (0..5).find(|&k| rc_row(&y[k], round).is_some()).unwrap();
                let order: Vec<usize> = std::iter::successors(Some(a), |&k| Some(nb(k))).take(7).collect();
                let rc = g.ld(rc_row(&y[a], round).unwrap());
                // iota first, on a copy of u[a] (the copy chi needs anyway)
                let w = g.xor(u[a], rc);
                let mut out = u.clone();
                for i in 0..5 {
                    let acc = if i == 0 { w } else { u[order[i]] };
                    out[order[i]] = g.tern(CHI, acc, u[order[i + 1]], u[order[i + 2]]);
                }
                cur = out;
                l = y.clone();
                if let Some(x) = x {
                    for k in 0..5 {
                        cur[k] = g.perm(rot(&y[k], &x[k]).unwrap(), cur[k]);
                    }
                    l = x.clone();
                }
            }
            Chi::Rows(x) => {
                for k in 0..5 {
                    let sh = |s: u8| x[k].map(|q| ((q.0 + s) % 5, q.1));
                    let n1 = g.perm(rot(&post[k], &sh(1)).unwrap(), cur[k]);
                    let n2 = g.perm(rot(&post[k], &sh(2)).unwrap(), cur[k]);
                    let v = g.perm(rot(&post[k], &x[k]).unwrap(), cur[k]);
                    cur[k] = g.tern(CHI, v, n1, n2);
                    if let Some(row) = rc_row(&x[k], round) {
                        let rc = g.ld(row);
                        cur[k] = g.xor(cur[k], rc);
                    }
                }
                l = x.clone();
            }
        }
        // predicted movs inserted by the allocator: one per cross-vector chi
        per_round.push(g.ops.len() - start + matches!(chi, Chi::Cross(..)) as usize);
    }
    // back to row order
    let out = (0..5).map(|y| cur[l.iter().position(|v| v[0].1 == y as u8).unwrap()]).collect();
    assert!(l.iter().all(|v| (0..5).all(|j| v[j].0 as usize == j && v[j].1 == v[0].1)));
    (out, per_round)
}

fn simulate(g: &Gen, input: &[u64; 25], out: &[usize]) -> [u64; 25] {
    let mut v: Vec<[u64; 8]> = (0..5).map(|y| std::array::from_fn(|x| if x < 5 { input[x + 5 * y] } else { 0 })).collect();
    v.resize(g.n, [0; 8]);
    for op in &g.ops {
        match *op {
            Op::Ld(d, i) => v[d] = g.table[i],
            Op::Xor(d, a, b) => v[d] = std::array::from_fn(|j| v[a][j] ^ v[b][j]),
            Op::Tern(d, imm, acc, a, b) => {
                v[d] = std::array::from_fn(|j| {
                    (0..64).fold(0, |r, bit| {
                        let i = (v[acc][j] >> bit & 1) << 2 | (v[a][j] >> bit & 1) << 1 | v[b][j] >> bit & 1;
                        r | ((imm as u64 >> i) & 1) << bit
                    })
                })
            }
            Op::Perm(d, r, a) => v[d] = std::array::from_fn(|j| if j < 5 { v[a][(j + r) % 5] } else { v[a][j] }),
            Op::Roli(d, a) => v[d] = v[a].map(|x| x.rotate_left(1)),
            Op::Rolv(d, a, b) => v[d] = std::array::from_fn(|j| v[a][j].rotate_left(v[b][j] as u32 % 64)),
        }
    }
    for &o in out {
        assert!(v[o][5..].iter().all(|&x| x == 0), "lanes 5..8 must stay zero");
    }
    std::array::from_fn(|i| v[out[i / 5]][i % 5])
}

fn row(v: &[u64; 8]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:#x}")).collect::<Vec<_>>().join(", "))
}

/// the generated code (also used by ../falcon/build.rs)
pub fn emit() -> String {
    let start: Layout = (0..5).map(|y| std::array::from_fn(|x| (x as u8, y))).collect();
    let plan = plan(&start);
    let mut g = Gen { n: 5, ..Default::default() };
    let (out, per_round) = generate(&mut g, &plan, (0..5).collect());

    // check against the reference: zero state (standard vector) and pseudo-random states
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for t in 0..4 {
        let input: [u64; 25] = std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if t == 0 { 0 } else { seed }
        });
        let mut want = input;
        keccak::keccak_f(&mut want);
        assert_eq!(simulate(&g, &input, &out), want, "generated Keccak-f disagrees with keccak.rs");
    }

    let mut s = String::new();
    let total: usize = per_round.iter().sum();
    writeln!(s, "// generated by build.rs. vector ops per round (incl. table loads and predicted movs):").unwrap();
    writeln!(s, "// {per_round:?}, total {total}").unwrap();
    let rot: Vec<[u64; 8]> = (1..5).map(|r| std::array::from_fn(|j| if j < 5 { ((j + r) % 5) as u64 } else { j as u64 })).collect();
    writeln!(s, "pub static ROT: Tab<4> = Tab([{}]);", rot.iter().map(row).collect::<Vec<_>>().join(", ")).unwrap();
    writeln!(s, "pub static KT: Tab<{}> = Tab([", g.table.len()).unwrap();
    for r in &g.table {
        writeln!(s, "    {},", row(r)).unwrap();
    }
    writeln!(s, "]);").unwrap();
    writeln!(s, "#[inline(always)]").unwrap();
    writeln!(s, "unsafe fn keccak_f(s: [__svm512i; 5], r: &[__svm512i; 4]) -> [__svm512i; 5] {{").unwrap();
    writeln!(s, "    let t = KT.0.as_ptr() as *const u8;").unwrap();
    writeln!(s, "    let [v0, v1, v2, v3, v4] = s;").unwrap();
    for op in &g.ops {
        match *op {
            Op::Ld(d, i) => writeln!(s, "    let v{d} = _svm512_loadu_si512(t.add({}));", 64 * i),
            Op::Xor(d, a, b) => writeln!(s, "    let v{d} = _svm512_xor_si512(v{a}, v{b});"),
            Op::Tern(d, imm, acc, a, b) => {
                writeln!(s, "    let v{d} = _svm512_ternarylogic_epi64::<{imm:#04x}>(v{acc}, v{a}, v{b});")
            }
            Op::Perm(d, r, a) => writeln!(s, "    let v{d} = _svm512_permutexvar_epi64(r[{}], v{a});", r - 1),
            Op::Roli(d, a) => writeln!(s, "    let v{d} = _svm512_rol_epi64::<1>(v{a});"),
            Op::Rolv(d, a, b) => writeln!(s, "    let v{d} = _svm512_rolv_epi64(v{a}, v{b});"),
        }
        .unwrap();
    }
    writeln!(s, "    [{}]", out.iter().map(|o| format!("v{o}")).collect::<Vec<_>>().join(", ")).unwrap();
    writeln!(s, "}}").unwrap();
    s
}

#[allow(dead_code)]
fn main() {
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), emit()).unwrap();
    println!("cargo:rerun-if-changed=keccak.rs");
}
