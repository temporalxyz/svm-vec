// emits net.rs's sorting network for our ops as straight-line intrinsics (OUT_DIR/sort.rs).
// Perm2 -> perm2.w (dword indices as word pairs), Perm1 -> roli.q 32 for adjacent swaps, else
// permd, Select -> ternlog with a lane mask. Index and mask vectors are loaded from .rodata and
// kept live while they are among the last CONSTS used.
use std::{collections::HashMap, fmt::Write, fs, path::Path};

#[path = "net.rs"]
#[allow(dead_code)]
mod net;
use net::Op;

const CONSTS: usize = 14;

#[derive(Default)]
struct Consts {
    table: Vec<[u8; 64]>,
    ids: HashMap<[u8; 64], usize>,
    live: Vec<(usize, String)>,
    loads: usize,
}

impl Consts {
    fn get(&mut self, s: &mut String, bytes: [u8; 64]) -> String {
        let n = self.table.len();
        let id = *self.ids.entry(bytes).or_insert(n);
        if id == n {
            self.table.push(bytes);
        }
        if let Some(i) = self.live.iter().position(|(c, _)| *c == id) {
            let e = self.live.remove(i);
            self.live.push(e.clone());
            return e.1;
        }
        let name = format!("c{}", self.loads);
        self.loads += 1;
        writeln!(s, "    let {name} = _svm512_loadu_si512(K.0[{id}].as_ptr());").unwrap();
        self.live.push((id, name.clone()));
        if self.live.len() > CONSTS {
            self.live.remove(0);
        }
        name
    }
}

fn dwords(f: impl Fn(usize) -> u32) -> [u8; 64] {
    let mut b = [0u8; 64];
    for i in 0..16 {
        b[4 * i..4 * i + 4].copy_from_slice(&f(i).to_le_bytes());
    }
    b
}

fn main() {
    let st = net::build(net::Isa::Ours);
    let mut s = String::new();
    let mut c = Consts::default();
    let mut ops = 0;
    // select: ternlog(lo, hi, mask) = mask ? hi : lo
    let sel: u32 = (0..8).map(|i| (if i & 1 == 1 { i >> 1 & 1 } else { i >> 2 & 1 }) << i).sum();
    s += "#[inline(always)]\n#[allow(unused_variables)]\nunsafe fn sort(inp: *const u8, out: *mut u8) {\n";
    for op in &st.ops {
        ops += 1;
        match op {
            Op::Load { d, k } => writeln!(s, "    let v{d} = _svm512_loadu_si512(inp.add({}));", 64 * k).unwrap(),
            Op::Store { a, k } => writeln!(s, "    _svm512_storeu_si512(out.add({}), v{a});", 64 * k).unwrap(),
            Op::Min { d, a, b } => writeln!(s, "    let v{d} = _svm512_min_epu32(v{a}, v{b});").unwrap(),
            Op::Max { d, a, b } => writeln!(s, "    let v{d} = _svm512_max_epu32(v{a}, v{b});").unwrap(),
            Op::Perm2 { d, a, b, idx } => {
                let w = dwords(|i| 2 * idx[i] as u32 | (2 * idx[i] as u32 + 1) << 16);
                let k = c.get(&mut s, w);
                writeln!(s, "    let v{d} = _svm512_permutex2var_epi16(v{a}, {k}, v{b});").unwrap();
            }
            Op::Perm1 { d, a, idx } => {
                if (0..16).all(|i| idx[i] as usize == i ^ 1) {
                    writeln!(s, "    let v{d} = _svm512_rol_epi64::<32>(v{a});").unwrap();
                } else {
                    let k = c.get(&mut s, dwords(|i| idx[i] as u32));
                    writeln!(s, "    let v{d} = _svm512_permutexvar_epi32({k}, v{a});").unwrap();
                }
            }
            Op::Select { d, a, b, mask } => {
                let k = c.get(&mut s, dwords(|i| if mask >> i & 1 == 1 { !0 } else { 0 }));
                writeln!(s, "    let v{d} = _svm512_ternarylogic_epi64::<{sel:#x}>(v{a}, v{b}, {k});").unwrap();
            }
        }
    }
    s += "}\n";
    writeln!(s, "static K: Tab<{}> = Tab({:?});", c.table.len(), c.table).unwrap();
    println!(
        "cargo:warning=sort: {} ops + {} constant loads (+ one mov per perm2.w pair), {}",
        ops,
        c.loads,
        net::histogram(&st.ops)
    );
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("sort.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=net.rs");
}
