// emits programs/sort's sorting network (net.rs, searched for this ISA: unpacks where the layout
// allows them) in Dean's intrinsic form, straight-line, with slots picked here (slot numbers must
// be compile time constants for the JIT to inline them). 512 bit values live in slot pairs:
// pairs 0..16 the keys (one bulk load), 16..48 temporaries (most recently freed reused first, for
// his register cache), 48..64 the sorted output (one bulk store), constants from slot 128 (bulk
// loads from .rodata at known addresses). min / max / unpack / blendd take 7 bit slots, so
// everything they touch is below 128.
use std::{collections::HashMap, fmt::Write, fs, path::Path};

#[path = "../../programs/sort/net.rs"]
#[allow(dead_code)]
mod net;
use net::{as_unpack, Op, V};

const OUT: u8 = 48;
const CONST: u8 = 64;

fn ops3(d: u8, a: u8, b: u8) -> u32 {
    d as u32 * 2 | (a as u32 * 2) << 8 | (b as u32 * 2) << 16
}
fn ops7(d: u8, a: u8, b: u8, imm: u8) -> u32 {
    d as u32 * 2 | (a as u32 * 2) << 7 | (b as u32 * 2) << 14 | (imm as u32) << 21
}
fn ops1(d: u8, a: u8, imm: u8) -> u32 {
    d as u32 * 2 | (a as u32 * 2) << 8 | (imm as u32) << 16
}

/// shufd imm if `idx` is the same in-lane shuffle in every 128 bit lane
fn shufd(idx: &[u8; 16]) -> Option<u8> {
    let c: Vec<u8> = (0..4).map(|i| idx[i]).collect();
    (0..16).all(|i| idx[i] as usize == (i & !3) + c[i & 3] as usize).then(|| (0..4).map(|i| c[i] << (2 * i)).sum())
}

/// permq imm if `idx` moves u64 lanes, the same way in both 256 bit halves
fn permq(idx: &[u8; 16]) -> Option<u8> {
    let mut c = [0u8; 4];
    for i in 0..8 {
        let (lo, hi) = (idx[2 * i] as usize, idx[2 * i + 1] as usize);
        if lo % 2 != 0 || hi != lo + 1 || lo / 8 != i / 4 {
            return None;
        }
        let k = (lo / 2 % 4) as u8;
        if i >= 4 && c[i % 4] != k {
            return None;
        }
        c[i % 4] = k;
    }
    Some((0..4).map(|i| c[i] << (2 * i)).sum())
}

fn inputs(op: &Op) -> Vec<V> {
    match op {
        Op::Min { a, b, .. } | Op::Max { a, b, .. } | Op::Perm2 { a, b, .. } | Op::Select { a, b, .. } => vec![*a, *b],
        Op::Perm1 { a, .. } | Op::Store { a, .. } => vec![*a],
        Op::Load { .. } => vec![],
    }
}

fn def(op: &Op) -> Option<V> {
    match op {
        Op::Load { d, .. } | Op::Min { d, .. } | Op::Max { d, .. } | Op::Perm2 { d, .. } | Op::Perm1 { d, .. } | Op::Select { d, .. } => Some(*d),
        Op::Store { .. } => None,
    }
}

/// list schedule for his register cache (15 LRU zmm pairs, results allocate, uncached sources
/// are memory operands): prefer ops whose sources are among the 15 most recent results and that
/// end live ranges
fn schedule(ops: &[Op]) -> Vec<Op> {
    let mut uses: HashMap<V, usize> = HashMap::new();
    for op in ops {
        for v in inputs(op) {
            *uses.entry(v).or_default() += 1;
        }
    }
    let mut defined: HashMap<V, bool> = HashMap::new();
    let mut done = vec![false; ops.len()];
    let mut out = vec![];
    let mut cache: Vec<V> = vec![];
    for (i, op) in ops.iter().enumerate() {
        if let Op::Load { d, .. } = op {
            defined.insert(*d, true);
            done[i] = true;
            out.push(op.clone());
        }
    }
    let mut first = 0;
    while out.len() < ops.len() {
        while done[first] {
            first += 1;
        }
        let mut best: Option<(i64, usize)> = None;
        for i in first..ops.len().min(first + 200) {
            if done[i] || matches!(ops[i], Op::Store { .. }) && i != first {
                continue;
            }
            let ins = inputs(&ops[i]);
            if !ins.iter().all(|v| defined.contains_key(v)) {
                continue;
            }
            let hits = ins.iter().filter(|v| cache.contains(v)).count() as i64;
            let ends = ins.iter().filter(|v| uses[v] == 1).count() as i64;
            let score = 4 * hits + 3 * ends - (i - first) as i64 / 64;
            if best.map_or(true, |(b, _)| score > b) {
                best = Some((score, i));
            }
        }
        let i = best.unwrap().1;
        done[i] = true;
        for v in inputs(&ops[i]) {
            let u = uses.get_mut(&v).unwrap();
            *u -= 1;
            cache.retain(|&c| c != v);
            if *u > 0 {
                cache.push(v);
            }
        }
        if let Some(d) = def(&ops[i]) {
            defined.insert(d, true);
            cache.push(d);
            if cache.len() > 15 {
                cache.remove(0);
            }
        }
        out.push(ops[i].clone());
    }
    out
}

fn main() {
    let st = net::build(net::Isa::Dean);
    let ops = &schedule(&st.ops);
    let mut last: HashMap<V, usize> = HashMap::new();
    let mut fixed: HashMap<V, u8> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Store { a, k } => _ = fixed.insert(*a, OUT + *k as u8),
            Op::Load { d, k } => _ = fixed.insert(*d, *k as u8),
            _ => {}
        }
        for u in inputs(op) {
            last.insert(u, i);
        }
    }
    let mut consts: Vec<[u32; 16]> = vec![];
    let mut konst = |c: [u32; 16]| -> u8 {
        let i = consts.iter().position(|x| *x == c).unwrap_or_else(|| {
            consts.push(c);
            consts.len() - 1
        });
        CONST + i as u8
    };
    let mut free: Vec<u8> = (16..OUT).rev().collect();
    let mut slot: HashMap<V, u8> = HashMap::new();
    let mut body = String::new();
    let mut n = 0;
    let mut emit = |s: String| {
        n += 1;
        writeln!(body, "    {s};").unwrap();
    };
    for (i, op) in ops.iter().enumerate() {
        let dies = |v: &V| last[v] == i;
        let (d, srcs): (V, Vec<V>) = match op {
            Op::Load { d, .. } => {
                slot.insert(*d, fixed[d]);
                continue;
            }
            Op::Store { .. } => continue,
            Op::Min { d, a, b } | Op::Max { d, a, b } | Op::Perm2 { d, a, b, .. } | Op::Select { d, a, b, .. } => (*d, vec![*a, *b]),
            Op::Perm1 { d, a, .. } => (*d, vec![*a]),
        };
        let s: Vec<u8> = srcs.iter().map(|v| slot[v]).collect();
        // permt2d overwrites its first table: in place on an input that dies here, else a copy
        let mut inplace = None;
        if let Op::Perm2 { a, b, idx, .. } = op {
            if as_unpack(idx).is_none() && !fixed.contains_key(&d) {
                if dies(a) {
                    inplace = Some((s[0], *idx, s[1]));
                } else if dies(b) {
                    inplace = Some((s[1], idx.map(|x| x ^ 16), s[0]));
                }
            }
        }
        let mut seen = vec![];
        for v in &srcs {
            if dies(v) && !seen.contains(v) && slot[v] < OUT {
                seen.push(*v);
                free.push(slot[v]);
            }
        }
        let t = match (fixed.get(&d), inplace) {
            (Some(&f), _) => f,
            (None, Some((x, _, _))) => {
                free.retain(|&p| p != x);
                x
            }
            (None, None) => free.pop().expect("out of slots"),
        };
        slot.insert(d, t);
        match op {
            Op::Min { .. } => emit(format!("sol_avx512_minu32x16({})", ops3(t, s[0], s[1]))),
            Op::Max { .. } => emit(format!("sol_avx512_max512({})", ops7(t, s[0], s[1], 6))),
            Op::Perm2 { idx, .. } => {
                if let Some((hi, q, sw)) = as_unpack(idx) {
                    let (x, y) = if sw { (s[1], s[0]) } else { (s[0], s[1]) };
                    let name = if hi { "unpackhi" } else { "unpacklo" };
                    emit(format!("sol_avx512_{name}512({})", ops7(t, x, y, if q { 3 } else { 2 })));
                } else if let Some((_, idx, other)) = inplace {
                    let k = konst(idx.map(|x| x as u32));
                    emit(format!("sol_avx512_permt2d512({})", ops3(t, k, other)));
                } else {
                    let k = konst(idx.map(|x| x as u32));
                    emit(format!("sol_avx512_mov512({})", t as u32 * 2 | (s[0] as u32 * 2) << 8));
                    emit(format!("sol_avx512_permt2d512({})", ops3(t, k, s[1])));
                }
            }
            Op::Perm1 { idx, .. } => {
                if let Some(imm) = shufd(idx) {
                    emit(format!("sol_avx512_shufd512({})", ops1(t, s[0], imm)));
                } else if let Some(imm) = permq(idx) {
                    emit(format!("sol_avx512_permq512({})", ops1(t, s[0], imm)));
                } else {
                    let k = konst(idx.map(|x| x as u32));
                    emit(format!("sol_avx512_permd512({})", ops3(t, s[0], k)));
                }
            }
            Op::Select { mask, .. } => {
                if mask & 0xff == mask >> 8 {
                    emit(format!("sol_avx512_blendd512({})", ops7(t, s[0], s[1], *mask as u8)));
                } else {
                    // blendm: mask bit ? va : vb
                    let p = t as u32 * 2 | (s[1] as u32 * 2) << 8 | (s[0] as u32 * 2) << 16 | 2 << 24;
                    emit(format!("sol_avx512_blendm512({p}, {mask:#x})"));
                }
            }
            _ => unreachable!(),
        }
    }
    assert!(consts.len() <= 64, "too many constants");
    let mut s = String::new();
    writeln!(s, "static C: Tab<{}> = Tab({:?});", consts.len().max(1), if consts.is_empty() { vec![[0u32; 16]] } else { consts.clone() }).unwrap();
    s += "#[inline(always)]\nunsafe fn vector(keys: *const u8, out: *mut u8) {\n";
    writeln!(s, "    sol_avx_load256({}, keys as u64);", 31 << 24).unwrap();
    for (j, chunk) in consts.chunks(16).enumerate() {
        writeln!(s, "    sol_avx_load256({}, C.0.as_ptr().add({}) as u64);", (CONST as u32 + 16 * j as u32) * 2 | (2 * chunk.len() as u32 - 1) << 24, 16 * j).unwrap();
    }
    s += &body;
    writeln!(s, "    sol_avx_store256({}, out as u64);", OUT as u32 * 2 | 31 << 24).unwrap();
    s += "}\n";
    let loads = 2 + consts.len().div_ceil(16);
    println!("cargo:warning=sort-dean: {n} vector ops + {loads} bulk loads/stores ({} constants), {}", consts.len(), net::histogram(ops));
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=../../programs/sort/net.rs");
}
