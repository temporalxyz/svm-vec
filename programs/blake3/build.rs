// generates the 16-way BLAKE3 compression (7 rounds straight-line, lane i = one chunk or parent),
// the 16x16 u32 transpose and the index tables (OUT_DIR/gen.rs). known-zero state words (the
// counter's high half, and the whole counter for parents) are folded away.
use std::{fmt::Write, fs, path::Path};

const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

#[derive(Clone)]
enum V {
    Zero,
    Var(String),
}

/// message words kept in memory (at `s + 64 j`) from round 2 on: 16 state words, 13 message words
/// and the reloaded one fit in the 30 registers without spills
const MEM: [usize; 3] = [0, 2, 4];

#[derive(Default)]
struct Gen {
    s: String,
    n: usize,
    /// values emitted at their first use (constants, the counter)
    lazy: Vec<(String, String)>,
}

impl Gen {
    fn emit(&mut self, e: String) -> V {
        let name = format!("t{}", self.n);
        self.n += 1;
        writeln!(self.s, "    let {name} = {e};").unwrap();
        V::Var(name)
    }
    fn touch(&mut self, v: &V) {
        if let V::Var(x) = v {
            if let Some(i) = self.lazy.iter().position(|(n, _)| n == x) {
                let (n, e) = self.lazy.remove(i);
                writeln!(self.s, "    let {n} = {e};").unwrap();
            }
        }
    }
    fn bin(&mut self, f: &str, a: &V, b: &V) -> V {
        self.touch(a);
        self.touch(b);
        match (a, b) {
            (V::Zero, x) | (x, V::Zero) => x.clone(),
            (V::Var(x), V::Var(y)) => self.emit(format!("{f}({x}, {y})")),
        }
    }
    fn rol(&mut self, a: &V, r: u32) -> V {
        match a {
            V::Zero => V::Zero,
            V::Var(x) => self.emit(format!("_svm512_rol_epi32::<{r}>({x})")),
        }
    }
}

fn rounds(g: &mut Gen, mut s: Vec<V>) -> String {
    let mut sched: [usize; 16] = std::array::from_fn(|i| i);
    for round in 0..7 {
        for half in 0..2 {
            let cols: [[usize; 4]; 4] = if half == 0 {
                [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
            } else {
                [[0, 5, 10, 15], [1, 6, 11, 12], [2, 7, 8, 13], [3, 4, 9, 14]]
            };
            for (step, (rd, rb)) in [(16, 12), (8, 7)].into_iter().enumerate() {
                let add = "_svm512_add_epi32";
                let xor = "_svm512_xor_si512";
                for k in 0..4 {
                    let a = cols[k][0];
                    let w = sched[8 * half + 2 * k + step];
                    let m = match MEM.iter().position(|&x| x == w) {
                        Some(j) if round > 0 => g.emit(format!("_svm512_loadu_si512(s.add({}))", 64 * j)),
                        _ => V::Var(format!("m{w}")),
                    };
                    s[a] = g.bin(add, &s[a], &m);
                }
                for [a, b, _, _] in cols {
                    s[a] = g.bin(add, &s[a], &s[b]);
                }
                for [a, _, _, d] in cols {
                    s[d] = g.bin(xor, &s[d], &s[a]);
                }
                for [_, _, _, d] in cols {
                    s[d] = g.rol(&s[d], 32 - rd);
                }
                for [_, _, c, d] in cols {
                    s[c] = g.bin(add, &s[c], &s[d]);
                }
                for [_, b, c, _] in cols {
                    s[b] = g.bin(xor, &s[b], &s[c]);
                }
                for [_, b, _, _] in cols {
                    s[b] = g.rol(&s[b], 32 - rb);
                }
            }
        }
        sched = std::array::from_fn(|i| sched[PERM[i]]);
    }
    let out: Vec<String> = (0..8)
        .map(|i| match g.bin("_svm512_xor_si512", &s[i], &s[i + 8]) {
            V::Var(x) => x,
            V::Zero => unreachable!(),
        })
        .collect();
    format!("    [{}]\n", out.join(", "))
}

fn var(s: &str) -> V {
    V::Var(s.into())
}

fn row(v: &[u16]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

/// perm2.w index (u16 words) of 16 dwords, each (table 0 = a / 1 = b, dword)
fn idx(f: impl Fn(usize) -> (usize, usize)) -> Vec<u16> {
    (0..32).map(|w| {
        let (t, d) = f(w / 2);
        (32 * t + 2 * d + w % 2) as u16
    })
    .collect()
}

fn main() {
    let mut s = String::new();
    let ms: String = (0..16).map(|i| format!("m{i}, ")).collect();
    let stores: String = MEM.iter().enumerate().map(|(j, w)| format!("    _svm512_storeu_si512(s.add({}), m{w});\n", 64 * j)).collect();
    let lazy = |n: usize| -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = (0..n).map(|i| (format!("iv{i}"), format!("_svm512_loadbc_epi32(K.as_ptr().add({i}) as *const i32)"))).collect();
        v.push(("ctr".into(), "_svm512_loadu_si512(s.add(CTR))".into()));
        v.push(("bl".into(), "bl()".into()));
        v.push(("fl".into(), "fl()".into()));
        v
    };
    let sig = "s: *mut u8, bl: impl Fn() -> __svm512i, fl: impl Fn() -> __svm512i) -> [__svm512i; 8] {";

    // chunk blocks: cv, counter (per lane at s + 128, high half 0), block length, flags
    let mut g = Gen { lazy: lazy(4), ..Default::default() };
    let mut st: Vec<V> = (0..8).map(|i| var(&format!("c{i}"))).collect();
    st.extend((0..4).map(|i| var(&format!("iv{i}"))));
    st.extend([var("ctr"), V::Zero, var("bl"), var("fl")]);
    let body = rounds(&mut g, st);
    writeln!(s, "#[inline(always)]\nunsafe fn compress_chunk(cv: [__svm512i; 8], m: [__svm512i; 16], {sig}").unwrap();
    writeln!(s, "    let [c0, c1, c2, c3, c4, c5, c6, c7] = cv;\n    let [{ms}] = m;").unwrap();
    s += &stores;
    s += &g.s;
    s += &body;
    s += "}\n";

    // parents: cv = IV, counter 0
    let mut g = Gen { lazy: lazy(8), ..Default::default() };
    let mut st: Vec<V> = (0..8).map(|i| var(&format!("iv{i}"))).collect();
    st.extend((0..4).map(|i| var(&format!("iv{i}"))));
    st.extend([V::Zero, V::Zero, var("bl"), var("fl")]);
    let body = rounds(&mut g, st);
    writeln!(s, "#[inline(always)]\nunsafe fn compress_parent(m: [__svm512i; 16], {sig}").unwrap();
    writeln!(s, "    let [{ms}] = m;").unwrap();
    s += &stores;
    s += &g.s;
    s += &body;
    s += "}\n";

    // transpose: stage k swaps bit k of the vector index with bit k of the dword lane
    let mut tab = vec![];
    writeln!(s, "#[inline(always)]\nunsafe fn transpose(r: [__svm512i; 16]) -> [__svm512i; 16] {{").unwrap();
    writeln!(s, "    let t = TAB.0.as_ptr() as *const u8;").unwrap();
    writeln!(s, "    let [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15] = r;").unwrap();
    for k in 0..4 {
        let h = 1 << k;
        tab.push(idx(|j| if j & h == 0 { (0, j) } else { (1, j - h) }));
        tab.push(idx(|j| if j & h == 0 { (0, j + h) } else { (1, j) }));
        writeln!(s, "    let lo = _svm512_loadu_si512(t.add({}));", 128 * k).unwrap();
        writeln!(s, "    let hi = _svm512_loadu_si512(t.add({}));", 128 * k + 64).unwrap();
        for i in (0..16).filter(|i| i & h == 0) {
            let j = i | h;
            writeln!(s, "    let (r{i}, r{j}) = (_svm512_permutex2var_epi16(r{i}, lo, r{j}), _svm512_permutex2var_epi16(r{i}, hi, r{j}));").unwrap();
        }
    }
    writeln!(s, "    [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15]\n}}").unwrap();
    // parents of 32 nodes (a = nodes 0..16, b = 16..32): left and right children
    tab.push(idx(|i| if i < 8 { (0, 2 * i) } else { (1, 2 * i - 16) }));
    tab.push(idx(|i| if i < 8 { (0, 2 * i + 1) } else { (1, 2 * i - 15) }));
    tab.push((0..32).map(|w| if w % 2 == 0 { w as u16 / 2 } else { 0 }).collect()); // iota
    writeln!(s, "static TAB: Tab<{}> = Tab([", tab.len()).unwrap();
    for r in &tab {
        writeln!(s, "    {},", row(r)).unwrap();
    }
    writeln!(s, "]);").unwrap();
    writeln!(s, "const CTR: usize = {};", 64 * MEM.len()).unwrap();
    writeln!(s, "const T_EVEN: usize = 512;\nconst T_ODD: usize = 576;\nconst T_IOTA: usize = 640;").unwrap();
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("gen.rs"), s).unwrap();
}
