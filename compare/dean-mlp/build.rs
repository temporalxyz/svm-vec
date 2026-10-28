// emits the packed weights and the straight-line vector code of programs/mlp's vector path in
// Dean's intrinsic form (slot numbers must be compile time constants for the JIT to inline them).
use std::{fmt::Write, fs, path::Path};

#[path = "../../programs/mlp/model.rs"]
#[allow(dead_code)]
mod model;
use model::{HID, IN, OUT, SHIFT};

// slots (32 bytes each), 512 bit operands are even slot pairs; min / max need slots below 128
const ACC: u8 = 0; // 8 pairs, layer 1 accumulators
const ZERO: u8 = 16;
const MAX: u8 = 18;
const ACC2: u8 = 20; // layer 2 accumulator
const XB: u8 = 22; // 2 pairs, broadcast inputs (alternating)
const H: u8 = 26; // 8 pairs, requantized hidden bytes in the low 16 bytes of each pair
const X: u8 = 64; // 4 pairs, the 256 input bytes
const IDX: u8 = 96; // 16 pairs, pair i = [i; 16] (permd index)
const W: u8 = 128; // 32 slots, layer 1 weights of 2 groups
const W2: u8 = 160; // 64 slots, layer 2 weights

fn ops3(d: u8, a: u8, b: u8) -> u32 {
    d as u32 | (a as u32) << 8 | (b as u32) << 16
}
fn ops7(d: u8, a: u8, b: u8, imm: u8) -> u32 {
    d as u32 | (a as u32) << 7 | (b as u32) << 14 | (imm as u32) << 21
}

fn main() {
    let m = model::model();
    let mut s = String::new();
    let w1p = model::pack(&m.w1);
    let w2p = model::pack(&m.w2);
    let idx: Vec<u32> = (0..16u32).flat_map(|i| [i; 16]).collect();
    writeln!(s, "static W1P: [u8; {}] = {:?};", w1p.len(), w1p).unwrap();
    writeln!(s, "static W2P: [u8; {}] = {:?};", w2p.len(), w2p).unwrap();
    writeln!(s, "static B1: [i32; {HID}] = {:?};", m.b1).unwrap();
    writeln!(s, "static B2: [i32; {OUT}] = {:?};", m.b2).unwrap();
    writeln!(s, "static IDX: [u32; 256] = {idx:?};").unwrap();

    let mut n = 0;
    let mut op = |s: &mut String, call: String| {
        n += 1;
        writeln!(s, "    {call};").unwrap();
    };
    let load = |d: u8, slots: u32, src: String| {
        format!("sol_avx_load256({}, {src} as u64)", d as u32 | (slots - 1) << 24)
    };
    s += "#[inline(always)]\nunsafe fn vector(x: *const u8, out: *mut u8) {\n";
    op(&mut s, load(ACC, 16, "B1.as_ptr()".into()));
    op(&mut s, load(ACC2, 2, "B2.as_ptr()".into()));
    op(&mut s, load(X, 8, "x".into()));
    op(&mut s, load(IDX, 32, "IDX.as_ptr()".into()));
    op(&mut s, format!("sol_avx512_bcast32x16({ZERO}, 0)"));
    op(&mut s, format!("sol_avx512_bcast32x16({MAX}, 127)"));

    // layer 1: per group of 4 inputs, broadcast them (permd of lane g) and dpbusd 8 accumulators
    for g in 0..IN / 4 {
        let j = (g % 2) as u8;
        if j == 0 {
            op(&mut s, load(W, 32, format!("W1P.as_ptr().add({})", g * HID * 4)));
        }
        let xb = XB + 2 * j;
        op(&mut s, format!("sol_avx512_permd512({})", ops3(xb, X + 2 * (g / 16) as u8, IDX + 2 * (g % 16) as u8)));
        for k in 0..8 {
            op(&mut s, format!("sol_avx512_dpbusd512({})", ops3(ACC + 2 * k, xb, W + 16 * j + 2 * k)));
        }
    }
    // relu + requantize: (acc >> 10) clamped to 0..=127 (signed 32 bit min / max), truncated to bytes
    for k in 0..8 {
        let a = ACC + 2 * k;
        op(&mut s, format!("sol_avx512_sra32x16({})", ops3(a, a, SHIFT as u8)));
        op(&mut s, format!("sol_avx512_max512({})", ops7(a, a, ZERO, 2)));
        op(&mut s, format!("sol_avx512_min512({})", ops7(a, a, MAX, 2)));
        op(&mut s, format!("sol_avx512_trunc512({})", ops3(H + 2 * k, a, 2)));
    }
    // layer 2: hidden group g is u32 lane g % 4 of H pair g / 4
    op(&mut s, load(W2, 32, "W2P.as_ptr()".into()));
    op(&mut s, load(W2 + 32, 32, "W2P.as_ptr().add(1024)".into()));
    for g in 0..HID / 4 {
        let xb = XB + 2 * (g % 2) as u8;
        op(&mut s, format!("sol_avx512_permd512({})", ops3(xb, H + 2 * (g / 4) as u8, IDX + 2 * (g % 4) as u8)));
        op(&mut s, format!("sol_avx512_dpbusd512({})", ops3(ACC2, xb, W2 + 2 * g as u8)));
    }
    op(&mut s, format!("sol_avx_store256({}, out as u64)", ACC2 as u32 | 1 << 24));
    s += "}\n";
    writeln!(s, "pub const VECTOR_OPS: usize = {n};").unwrap();

    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=../../programs/mlp/model.rs");
}
