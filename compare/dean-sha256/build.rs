// programs/sha256 on Dean's vector intrinsics: the same rounds (programs/sha256/gen.rs), with slots
// assigned here (slot numbers must be compile time constants for the JIT to inline them).
//
// input: one 32-slot load (block i = pair 32 + 2i), two transpose8x8q (u64 lane k of row j = words
// 2k, 2k + 1 of block j), then per k two permi2b512 that pick words 2k / 2k + 1 of all 16 blocks
// from the two transposed rows and byte swap them; their 16 index pairs come in with one 32-slot
// rodata load. constants: 16 per 32-slot rodata load, loaded before their first use. output: the
// IV adds as 256 bit halves into 16 consecutive slots, two transpose8x8 (slot = one digest), one
// shufb512 byte swap per pair and one 16-slot store.
use std::{fmt::Write, fs, path::Path};

#[path = "../../programs/sha256/gen.rs"]
mod gen;
use gen::Op;

fn ops3(d: u8, a: u8, b: u8) -> u32 {
    d as u32 | (a as u32) << 8 | (b as u32) << 16
}
fn ops7(d: u8, a: u8, b: u8, imm: u8) -> u32 {
    assert!(d < 128 && a < 128 && b < 128, "ternlog slots must be below 128");
    d as u32 | (a as u32) << 7 | (b as u32) << 14 | (imm as u32) << 21
}

const W: u8 = 0; // W_t at pair 2t, also the permi2b indices
const DATA: u8 = 32; // the 16 blocks, then the transposed rows
const LOW_CHUNK: u8 = 96; // first constant chunk, released to the working pool when dead
const RING: u8 = 128; // 4 x 32 slots of later constant chunks

struct Out {
    s: String,
    n: usize,
}
impl Out {
    fn op(&mut self, call: String) {
        self.n += 1;
        writeln!(self.s, "    {call};").unwrap();
    }
}

fn main() {
    let p = gen::generate();
    let last = &p.last;
    let mut o = Out { s: String::new(), n: 0 };

    // constants in order of first use, 16 per chunk
    let consts: Vec<usize> = (0..p.ops.len()).filter(|&i| matches!(p.ops[i], Op::Const(_))).collect();
    let chunks: Vec<&[usize]> = consts.chunks(16).collect();
    let chunk_base = |c: usize| if c == 0 { LOW_CHUNK } else { RING + 32 * ((c - 1) % 4) as u8 };
    let mut slot: Vec<Option<u8>> = vec![None; p.ops.len()];
    for (c, ch) in chunks.iter().enumerate() {
        for (j, &i) in ch.iter().enumerate() {
            slot[i] = Some(chunk_base(c) + 2 * j as u8);
        }
        if c >= 5 {
            let first = ch[0];
            assert!(chunks[c - 4].iter().all(|&i| last[i] < first), "constant ring overwritten while live");
        }
    }
    let ctab: Vec<u32> = consts.iter().map(|&i| if let Op::Const(c) = p.ops[i] { c } else { unreachable!() }).collect();

    // input
    let idx = |odd: usize| -> [u8; 64] {
        std::array::from_fn(|i| {
            let (blk, b) = (i / 4, i % 4);
            ((blk / 8) * 64 + 8 * (blk % 8) + 4 * odd + 3 - b) as u8
        })
    };
    let mut itab: Vec<u8> = vec![];
    for _ in 0..8 {
        itab.extend_from_slice(&idx(0));
        itab.extend_from_slice(&idx(1));
    }
    o.op(format!("sol_avx_load256({}, inp as u64)", DATA as u32 | 31 << 24));
    o.op(format!("sol_avx_load256({}, IDX.0.as_ptr() as u64)", W as u32 | 31 << 24));
    o.op(format!("sol_avx512_transpose8x8q({DATA})"));
    o.op(format!("sol_avx512_transpose8x8q({})", DATA + 16));
    for k in 0..8u8 {
        for h in 0..2 {
            o.op(format!("sol_avx512_permi2b512({:#x})", ops3(W + 4 * k + 2 * h, DATA + 2 * k, DATA + 16 + 2 * k)));
        }
    }

    // working pairs: LIFO, so a result usually takes the pair an operand just released
    let mut free: Vec<u8> = (16..48).rev().map(|p| 2 * p).collect();
    let mut low_released = false;
    let mut loaded = 0;
    let free_if_dead = |free: &mut Vec<u8>, slot: &[Option<u8>], v: usize, i: usize, ops: &[Op]| {
        if last[v] == i && !matches!(ops[v], Op::Const(_)) {
            free.push(slot[v].unwrap());
        }
    };
    let mut min_free = usize::MAX;
    for i in 0..p.ops.len() {
        if loaded < chunks.len() && chunks[loaded][0] == i {
            o.op(format!(
                "sol_avx_load256({}, CK.0.as_ptr().add({}) as u64)",
                chunk_base(loaded) as u32 | ((2 * chunks[loaded].len() - 1) as u32) << 24,
                16 * 16 * loaded
            ));
            loaded += 1;
        }
        if !low_released && chunks[0].iter().all(|&c| last[c] < i) {
            free.extend((LOW_CHUNK / 2..64).rev().map(|p| 2 * p));
            low_released = true;
        }
        macro_rules! s {
            ($v:expr) => {
                slot[$v].unwrap()
            };
        }
        match p.ops[i] {
            Op::In(t) => slot[i] = Some(W + 2 * t as u8),
            Op::Const(_) => {}
            Op::Add(a, b) if p.out.contains(&i) => {
                // output: handled below
                let _ = (a, b);
            }
            Op::Add(a, b) => {
                free_if_dead(&mut free, &slot, a, i, &p.ops);
                free_if_dead(&mut free, &slot, b, i, &p.ops);
                let d = free.pop().expect("out of slots");
                slot[i] = Some(d);
                o.op(format!("sol_avx512_add32x16({:#x})", ops3(d, s!(a), s!(b))));
            }
            Op::Ror(a, n) | Op::Shr(a, n) => {
                free_if_dead(&mut free, &slot, a, i, &p.ops);
                let d = free.pop().expect("out of slots");
                slot[i] = Some(d);
                let name = if matches!(p.ops[i], Op::Ror(..)) { "ror32x16" } else { "srl32x16" };
                o.op(format!("sol_avx512_{name}({:#x})", ops3(d, s!(a), n as u8)));
            }
            Op::Tern(imm, x, y, z) => {
                let d = if last[x] == i && !matches!(p.ops[x], Op::Const(_)) {
                    s!(x)
                } else {
                    let d = free.pop().expect("out of slots");
                    o.op(format!("sol_avx512_mov512({:#x})", ops3(d, s!(x), 0)));
                    d
                };
                free_if_dead(&mut free, &slot, y, i, &p.ops);
                free_if_dead(&mut free, &slot, z, i, &p.ops);
                slot[i] = Some(d);
                o.op(format!("sol_avx512_ternlog512({:#x})", ops7(d, s!(y), s!(z), imm)));
            }
        }
        min_free = min_free.min(free.len());
    }
    assert_eq!(loaded, chunks.len());

    // output: 16 slots in a ring chunk whose constants are dead
    let first_out = *p.out.iter().min().unwrap();
    let out_base = (0..4)
        .map(|r| RING + 32 * r)
        .find(|&b| (0..chunks.len()).filter(|&c| c > 0 && chunk_base(c) == b).all(|c| chunks[c].iter().all(|&i| last[i] < first_out)))
        .expect("no free output region");
    for (j, &i) in p.out.iter().enumerate() {
        let Op::Add(a, b) = p.ops[i] else { unreachable!() };
        let (a, b) = (slot[a].unwrap(), slot[b].unwrap());
        o.op(format!("sol_avx2_add32x8({:#x})", ops3(out_base + j as u8, a, b)));
        o.op(format!("sol_avx2_add32x8({:#x})", ops3(out_base + 8 + j as u8, a + 1, b + 1)));
    }
    o.op(format!("sol_avx2_transpose8x8({out_base})"));
    o.op(format!("sol_avx2_transpose8x8({})", out_base + 8));
    let bs = free.pop().unwrap();
    o.op(format!("sol_avx_load256({}, BSWAP.0.as_ptr() as u64)", bs as u32 | 1 << 24));
    for k in 0..8u8 {
        let d = out_base + 2 * k;
        o.op(format!("sol_avx512_shufb512({:#x})", ops3(d, d, bs)));
    }
    o.op(format!("sol_avx_store256({}, out as u64)", out_base as u32 | 15 << 24));

    let mut s = String::new();
    writeln!(s, "// generated by build.rs. rounds: {}", gen::summary(&p)).unwrap();
    writeln!(s, "// {} constant chunks, at least {min_free} working pairs always free", chunks.len()).unwrap();
    let ck: Vec<u32> = ctab.iter().flat_map(|&c| [c; 16]).collect();
    writeln!(s, "static CK: Tab32<{}> = Tab32({ck:?});", ck.len()).unwrap();
    writeln!(s, "static IDX: Tab8<{}> = Tab8({itab:?});", itab.len()).unwrap();
    let bswap: Vec<u8> = (0..64).map(|i| ((i & !3) | (3 - (i & 3))) as u8 & 15).collect();
    writeln!(s, "static BSWAP: Tab8<64> = Tab8({bswap:?});").unwrap();
    s += "#[inline(always)]\nunsafe fn sha256x16(inp: *const u8, out: *mut u8) {\n";
    s += &o.s;
    s += "}\n";
    writeln!(s, "pub const VECTOR_OPS: usize = {};", o.n).unwrap();
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=../../programs/sha256/gen.rs");
}
