// emits programs/matmul's GEMM as straight-line code in Dean's intrinsic form (slot numbers must
// be compile time constants for the JIT to inline them), and B packed per column block.
//
// Per block of 4 rows: one bulk load of A (32 slots), accumulators for the 4 x 128 block of C in
// 64 slots laid out like C (two bulk stores). Within it, 4 x 2 tiles (4 rows x 32 columns): per
// group of 4 k and row, one permd broadcasts the row's 4 A bytes (index pair g % 16) and two
// dpbusd use it. The JIT keeps 512-bit pairs in 15 LRU registers and needs every dpbusd operand
// in one: 8 accumulators, the broadcast, the index and 2 B pairs leave room, so nothing live is
// evicted. B comes in bulk loads of 32 slots (16 groups per refill of 64 slots). Their address is
// not a compile time constant on purpose: a constant .rodata load fills the 256-bit cache, which
// writes every slot back before its first 512-bit use (measured: 33 us against 16 us).
use std::{fmt::Write, fs, path::Path};

#[path = "../../programs/matmul/model.rs"]
#[allow(dead_code)]
mod model;
use model::{C_OFF, K, M, N};

const MB: usize = 4;
const NB: usize = 2;
const G: usize = K / 4;
const V: usize = N / 16;
/// 32-slot B loads per refill
const BT: usize = 2;

// slots (32 bytes each), 512 bit operands are even slot pairs
const IDX: usize = 0; // 16 pairs, pair i = [i; 16] (permd index)
const A: usize = 32; // 4 rows x 4 pairs
const ACC: usize = 64; // 4 rows x 8 pairs, like C
const XB: usize = 128; // the broadcast
const B: usize = 136; // BT x 16 pairs

fn ops3(d: usize, a: usize, b: usize) -> u32 {
    (d | a << 8 | b << 16) as u32
}

fn main() {
    let b = model::weights();
    // [column block][group][column vector j][16 columns][4 k]
    let mut bp = Vec::with_capacity(K * N);
    for cb in 0..V / NB {
        for g in 0..G {
            for n in 16 * NB * cb..16 * NB * (cb + 1) {
                bp.extend((0..4).map(|t| b[4 * g + t][n] as u8));
            }
        }
    }
    let idx: Vec<u32> = (0..16u32).flat_map(|i| [i; 16]).collect();
    let mut s = String::new();
    writeln!(s, "#[repr(C, align(64))]\nstruct Packed([u8; {}]);", bp.len()).unwrap();
    writeln!(s, "static BP: Packed = Packed({bp:?});").unwrap();
    writeln!(s, "static IDX: [u32; 256] = {idx:?};").unwrap();
    writeln!(s, "static ZERO: [u8; 1024] = [0; 1024];").unwrap();

    let mut n = 0;
    let mut op = |s: &mut String, call: String| {
        n += 1;
        writeln!(s, "    {call};").unwrap();
    };
    let load = |d: usize, src: String| format!("sol_avx_load256({}, {src} as u64)", d | 31 << 24);
    let store = |a: usize, dst: String| format!("sol_avx_store256({}, {dst} as u64)", a | 31 << 24);
    let acc = |r: usize, v: usize| ACC + 16 * r + 2 * v;
    // one function per tile, so LLVM keeps no addresses live across the straight-line code (it
    // would spill them to the stack: scalar memory accesses, which flush the JIT's cache), and a
    // branch around the code would exceed the 16 bit jump offset. `z` is 0 at run time.
    let mut main = String::from("#[inline(always)]\nunsafe fn vector(data: *mut u8, z: usize) {\n");
    main += "    let bp = BP.0.as_ptr().add(z);\n    let zp = ZERO.as_ptr().add(z);\n";
    op(&mut main, load(IDX, "IDX.as_ptr()".into()));
    for rb in 0..M / 4 {
        for r0 in (0..4).step_by(MB) {
            for cb in 0..V / NB {
                let name = format!("t{rb}_{r0}_{cb}");
                writeln!(main, "    {name}(data, bp, zp);").unwrap();
                writeln!(s, "#[inline(never)]\n#[allow(unused_variables)]\nunsafe fn {name}(data: *mut u8, bp: *const u8, zp: *const u8) {{").unwrap();
                if r0 == 0 && cb == 0 {
                    op(&mut s, load(A, format!("data.add({})", rb * 4 * K)));
                    // the vector file starts zeroed
                    if rb > 0 {
                        op(&mut s, load(ACC, "zp".into()));
                        op(&mut s, load(ACC + 32, "zp".into()));
                    }
                }
                let q = 16 * BT / NB;
                for g in 0..G {
                    if g % q == 0 {
                        for t in 0..BT {
                            let off = (cb * G + g) * NB * 64 + 1024 * t;
                            op(&mut s, load(B + 32 * t, format!("bp.add({off})")));
                        }
                    }
                    for r in 0..MB {
                        let a = A + 8 * (r0 + r) + 2 * (g / 16);
                        op(&mut s, format!("sol_avx512_permd512({})", ops3(XB, a, IDX + 2 * (g % 16))));
                        for j in 0..NB {
                            let d = acc(r0 + r, NB * cb + j);
                            let w = B + 2 * ((g % q) * NB + j);
                            op(&mut s, format!("sol_avx512_dpbusd512({})", ops3(d, XB, w)));
                        }
                    }
                }
                if r0 + MB == 4 && cb == V / NB - 1 {
                    for h in 0..2 {
                        op(&mut s, store(ACC + 32 * h, format!("data.add({})", C_OFF + 4 * N * (4 * rb + 2 * h))));
                    }
                }
                s += "}\n";
            }
        }
    }
    main += "}\n";
    s += &main;
    writeln!(s, "pub const VECTOR_OPS: usize = {n};").unwrap();
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=../../programs/matmul/model.rs");
}
