// emits programs/perps's kernel as straight-line code in Dean's intrinsic form (slot numbers must
// be compile time constants for the JIT to inline them). Two chunks of 128 positions: each field
// of a chunk is 32 slots (one bulk load), results overwrite their inputs' slots in place.
use std::{fmt::Write, fs, path::Path};

const N: u32 = 256;
const CHUNK: u32 = 128;
// slot bases, 32 slots (16 pairs) each
const SIZE: u32 = 0;
const ENTRY: u32 = 32; // entry, then mark - entry, then pnl
const MARK: u32 = 64; // mark, then healthy
const FUND: u32 = 96; // funding_delta, then funding
const COLL: u32 = 128; // collateral, then equity
const ONE: u32 = 160;
const ZERO: u32 = 162;
const REQ: u32 = 164; // |size|, then the requirement; one pair reused, so it stays cached

fn ops3(d: u32, a: u32, b: u32) -> u32 {
    d | a << 8 | b << 16
}

fn main() {
    let mut s = String::new();
    let mut n = 0;
    let mut op = |s: &mut String, call: String| {
        n += 1;
        writeln!(s, "    {call};").unwrap();
    };
    let field = |k: u32, c: u32| format!("p.add({})", 8 * (N * k + CHUNK * c));
    s += "#[inline(always)]\nunsafe fn vector(p: *mut u8) {\n";
    op(&mut s, format!("sol_avx512_bcast64x8({ONE}, 1)"));
    op(&mut s, format!("sol_avx512_bcast64x8({ZERO}, 0)"));
    for c in 0..N / CHUNK {
        for (k, base) in [SIZE, ENTRY, MARK, FUND, COLL].into_iter().enumerate() {
            op(&mut s, format!("sol_avx_load256({}, {} as u64)", base | 31 << 24, field(k as u32, c)));
        }
        for j in (0..32).step_by(2) {
            let (sz, e, m, f, cl) = (SIZE + j, ENTRY + j, MARK + j, FUND + j, COLL + j);
            op(&mut s, format!("sol_avx512_sub64x8({})", ops3(e, m, e)));
            op(&mut s, format!("sol_avx512_mullo64x8({})", ops3(e, sz, e)));
            op(&mut s, format!("sol_avx512_mullo64x8({})", ops3(f, sz, f)));
            op(&mut s, format!("sol_avx512_sra64x8({})", ops3(f, f, 32)));
            op(&mut s, format!("sol_avx512_add64x8({})", ops3(cl, cl, e)));
            op(&mut s, format!("sol_avx512_sub64x8({})", ops3(cl, cl, f)));
            op(&mut s, format!("sol_avx512_abs512({})", ops3(REQ, sz, 3)));
            op(&mut s, format!("sol_avx512_mullo64x8({})", ops3(REQ, REQ, m)));
            op(&mut s, format!("sol_avx512_sra64x8({})", ops3(REQ, REQ, 36)));
            // signed 64 bit equity >= requirement, then 1 / 0 per lane
            op(&mut s, format!("let k = sol_avx512_cmp512({})", ops3(cl, REQ, 5 | 3 << 4)));
            op(&mut s, format!("sol_avx512_blendm512({}, k)", ops3(m, ONE, ZERO) | 3 << 24));
        }
        for (k, base) in [ENTRY, FUND, COLL, MARK].into_iter().enumerate() {
            op(&mut s, format!("sol_avx_store256({}, {} as u64)", base | 31 << 24, field(5 + k as u32, c)));
        }
    }
    s += "}\n";
    writeln!(s, "pub const VECTOR_OPS: usize = {n};").unwrap();
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
}
