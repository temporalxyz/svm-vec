// generates the straight-line vector code (OUT_DIR/sha.rs): 16 loads, byte swaps and a 16 x 16 u32
// transpose (4 stages of 2-source word permutes) to get W_t with lane = block, the rounds from
// gen.rs, then the 8 x 16 -> 16 x 8 transpose back (3 stages) and 8 stores.
use std::{fmt::Write, fs, path::Path};

#[path = "gen.rs"]
mod gen;
use gen::Op;

type Sym = (usize, usize); // (original vector, original lane)

/// one stage of 2-source permutes: each (a, b) pair of vectors produces outputs (lo, hi) whose
/// lanes are `target(out, lane)`, which must come from a or b. returns the dword indices per output.
fn stage(cur: &[[Sym; 16]], pairs: &[(usize, usize)], target: impl Fn(usize, usize) -> Sym) -> (Vec<[[Sym; 16]; 2]>, Vec<[[u32; 16]; 2]>) {
    let mut content = vec![];
    let mut idx = vec![];
    for &(a, b) in pairs {
        let mut c = [[(0, 0); 16]; 2];
        let mut ix = [[0u32; 16]; 2];
        for h in 0..2 {
            for l in 0..16 {
                let s = target(if h == 0 { a } else { b }, l);
                c[h][l] = s;
                ix[h][l] = if let Some(j) = cur[a].iter().position(|&x| x == s) {
                    j as u32
                } else {
                    16 + cur[b].iter().position(|&x| x == s).expect("element not in either source") as u32
                };
            }
        }
        content.push(c);
        idx.push(ix);
    }
    (content, idx)
}

struct Em {
    s: String,
    rows: Vec<[u8; 64]>,
}

impl Em {
    fn row(&mut self, r: [u8; 64]) -> usize {
        if let Some(i) = self.rows.iter().position(|&x| x == r) {
            return i;
        }
        self.rows.push(r);
        self.rows.len() - 1
    }
    fn line(&mut self, l: String) {
        self.s += "    ";
        self.s += &l;
        self.s += ";\n";
    }
    fn idx(&mut self, name: &str, d: &[u32; 16]) {
        let mut r = [0u8; 64];
        for (l, &x) in d.iter().enumerate() {
            for h in 0..2 {
                r[4 * l + 2 * h..4 * l + 2 * h + 2].copy_from_slice(&((2 * x + h as u32) as u16).to_le_bytes());
            }
        }
        let i = self.row(r);
        self.line(format!("let {name} = _svm512_loadu_si512(ix.add({}))", 64 * i));
    }
    /// runs the stages on vectors named `names`, returns the new names
    fn transpose(&mut self, tag: &str, mut names: Vec<String>, mut cur: Vec<[Sym; 16]>, stages: &[(Vec<(usize, usize)>, Box<dyn Fn(usize, usize) -> Sym>)]) -> Vec<String> {
        for (k, (pairs, target)) in stages.iter().enumerate() {
            let (content, idx) = stage(&cur, pairs, target);
            let mut nn = vec![String::new(); names.len()];
            let mut nc = cur.clone();
            let mut loaded: Vec<([u32; 16], String)> = vec![];
            for (p, &(a, b)) in pairs.iter().enumerate() {
                for h in 0..2 {
                    let o = if h == 0 { a } else { b };
                    let iname = match loaded.iter().find(|x| x.0 == idx[p][h]) {
                        Some(x) => x.1.clone(),
                        None => {
                            let n = format!("i{tag}{k}_{}", loaded.len());
                            self.idx(&n, &idx[p][h]);
                            loaded.push((idx[p][h], n.clone()));
                            n
                        }
                    };
                    let name = format!("{tag}{}_{o}", k + 1);
                    self.line(format!("let {name} = _svm512_permutex2var_epi16({}, {iname}, {})", names[a], names[b]));
                    nn[o] = name;
                    nc[o] = content[p][h];
                }
            }
            names = nn;
            cur = nc;
        }
        names
    }
}

// exchanges bit `vbit` of the vector index with bit `lbit` of the lane index
fn swap_bits(vbit: usize, lbit: usize, v: usize, l: usize) -> Sym {
    let vb = v >> vbit & 1;
    let lb = l >> lbit & 1;
    ((v & !(1 << vbit)) | lb << vbit, (l & !(1 << lbit)) | vb << lbit)
}

fn main() {
    let p = gen::generate();
    let mut e = Em { s: String::new(), rows: vec![] };
    let bswap: [u8; 64] = std::array::from_fn(|i| ((i & !3) | (3 - (i & 3))) as u8 & 15);
    e.s += "#[inline(always)]\nunsafe fn sha256x16(inp: *const u8, out: *mut u8) {\n";
    e.line("let kc = KC.0.as_ptr() as *const i32".into());
    e.line("let ix = IX.0.as_ptr() as *const u8".into());
    let bs = e.row(bswap);
    e.line(format!("let bs = _svm512_loadu_si512(ix.add({}))", 64 * bs));
    for i in 0..16 {
        e.line(format!("let m{i} = _svm512_shuffle_epi8(_svm512_loadu_si512(inp.add({})), bs)", 64 * i));
    }

    // input: vector = block, lane = word -> vector = word, lane = block (swap all 4 bits)
    let cur: Vec<[Sym; 16]> = (0..16).map(|v| std::array::from_fn(|l| (v, l))).collect();
    let mut stages: Vec<(Vec<(usize, usize)>, Box<dyn Fn(usize, usize) -> Sym>)> = vec![];
    // symbol at (v, l) after stages 0..=k: positions with bits 0..=k of v and l exchanged
    for k in 0..4 {
        let pairs: Vec<(usize, usize)> = (0..16).filter(|v| v >> k & 1 == 0).map(|v| (v, v | 1 << k)).collect();
        stages.push((
            pairs,
            Box::new(move |o, l| {
                let mut s = (o, l);
                for j in (0..=k).rev() {
                    s = swap_bits(j, j, s.0, s.1);
                }
                s
            }),
        ));
    }
    let w = e.transpose("a", (0..16).map(|i| format!("m{i}")).collect(), cur, &stages);

    // rounds
    let mut kc: Vec<u32> = vec![];
    let var = |i: usize| format!("v{i}");
    for (i, o) in p.ops.iter().enumerate() {
        let l = match *o {
            Op::In(t) => format!("let v{i} = {}", w[t]),
            Op::Const(c) => {
                kc.push(c);
                format!("let v{i} = _svm512_loadbc_epi32(kc.add({}))", kc.len() - 1)
            }
            Op::Add(a, b) => format!("let v{i} = _svm512_add_epi32({}, {})", var(a), var(b)),
            Op::Ror(a, n) => format!("let v{i} = _svm512_rol_epi32::<{}>({})", 32 - n, var(a)),
            Op::Shr(a, n) => format!("let v{i} = _svm512_srli_epi32::<{n}>({})", var(a)),
            Op::Tern(imm, a, b, c) => format!("let v{i} = _svm512_ternarylogic_epi64::<{imm:#04x}>({}, {}, {})", var(a), var(b), var(c)),
        };
        e.line(l);
    }

    // output: H_j (lane = block) -> 8 rows of 2 digests: row r lane 8 m + j = H_j lane 2 r + m
    for j in 0..8 {
        e.line(format!("let h{j} = _svm512_shuffle_epi8(v{}, bs)", p.out[j]));
    }
    let cur: Vec<[Sym; 16]> = (0..8).map(|v| std::array::from_fn(|l| (v, l))).collect();
    let mut stages: Vec<(Vec<(usize, usize)>, Box<dyn Fn(usize, usize) -> Sym>)> = vec![];
    for k in 0..2 {
        let pairs: Vec<(usize, usize)> = (0..8).filter(|v| v >> k & 1 == 0).map(|v| (v, v | 1 << k)).collect();
        stages.push((
            pairs,
            Box::new(move |o, l| {
                let mut s = (o, l);
                for j in (0..=k).rev() {
                    s = swap_bits(j, j + 1, s.0, s.1);
                }
                s
            }),
        ));
    }
    let pairs: Vec<(usize, usize)> = (0..8).filter(|v| v >> 2 & 1 == 0).map(|v| (v, v | 4)).collect();
    stages.push((pairs, Box::new(|r, l| (l & 7, 2 * r + (l >> 3)))));
    let rows = e.transpose("b", (0..8).map(|j| format!("h{j}")).collect(), cur, &stages);
    for (r, name) in rows.iter().enumerate() {
        e.line(format!("_svm512_storeu_si512(out.add({}), {name})", 64 * r));
    }
    e.s += "}\n";

    let mut s = format!("// generated by build.rs. rounds: {}\n", gen::summary(&p));
    writeln!(s, "static KC: Tab32<{}> = Tab32({kc:?});", kc.len()).unwrap();
    writeln!(s, "static IX: Tab8<{}> = Tab8([", e.rows.len() * 64).unwrap();
    for r in &e.rows {
        writeln!(s, "    {},", r.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")).unwrap();
    }
    writeln!(s, "]);").unwrap();
    s += &e.s;
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("sha.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=gen.rs");
}
