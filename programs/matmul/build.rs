// emits B packed for dpbusd: [k group g][column vector j][16 columns][4 k], so the 64 bytes at
// 512 g + 64 j hold columns 16j..16j+16 of rows 4g..4g+4 and line up with a broadcast of 4 A
// bytes. Also emits the tile kernels with one named variable per vector (arrays of vectors in a
// loop do not stay in SSA).
use std::{env, fmt::Write, fs, path::Path};

#[path = "model.rs"]
#[allow(dead_code)]
mod model;
use model::{K, N};

/// column vectors per tile
const NB: usize = 4;
/// k groups per loop iteration
const U: usize = 16;

/// `tile<MB>`: an MB x NB block of accumulators in registers across the k loop. Per group: MB
/// broadcasts of A, NB loads of B, MB x NB dpbusd. Memory ops address through r1, so each switch
/// between A and B costs a scalar mov. With B first, NB vectors are live (MB NB + NB + 1
/// registers); with A first, MB (MB NB + MB + 1). If both fit, groups alternate between the two
/// orders and switch once per group instead of twice.
fn tile(s: &mut String, mb: usize) {
    let alternate = mb * NB + mb + 1 <= 30 && mb * NB + NB + 1 <= 30;
    writeln!(s, "#[inline(never)]\nunsafe fn tile{mb}(mut a: *const u8, mut b: *const u8, c: *mut u8) {{").unwrap();
    for r in 0..mb {
        for j in 0..NB {
            writeln!(s, "let mut c{r}_{j} = _svm512_setzero_si512();").unwrap();
        }
    }
    s.push_str("let end = a.add(K);\nwhile a < end {\n");
    for u in 0..U {
        let bc = |r: usize| format!("_svm512_loadbc_epi32(a.add({}) as _)", K * r + 4 * u);
        let bv = |j: usize| format!("_svm512_loadu_si512(b.add({}))", 4 * N * u + 64 * j);
        if alternate && u % 2 == 0 {
            for r in 0..mb {
                writeln!(s, "let x{r} = {};", bc(r)).unwrap();
            }
            for j in 0..NB {
                writeln!(s, "let w = {};", bv(j)).unwrap();
                for r in 0..mb {
                    writeln!(s, "c{r}_{j} = _svm512_dpbusd_epi32(c{r}_{j}, x{r}, w);").unwrap();
                }
            }
        } else {
            for j in 0..NB {
                writeln!(s, "let w{j} = {};", bv(j)).unwrap();
            }
            for r in 0..mb {
                writeln!(s, "let x = {};", bc(r)).unwrap();
                for j in 0..NB {
                    writeln!(s, "c{r}_{j} = _svm512_dpbusd_epi32(c{r}_{j}, x, w{j});").unwrap();
                }
            }
        }
    }
    writeln!(s, "a = a.add({});\nb = b.add({});\n}}", 4 * U, 4 * N * U).unwrap();
    for r in 0..mb {
        for j in 0..NB {
            writeln!(s, "_svm512_storeu_si512(c.add({}), c{r}_{j});", 4 * N * r + 64 * j).unwrap();
        }
    }
    s.push_str("}\n");
}

fn main() {
    let b = model::weights();
    let mut p = Vec::with_capacity(K * N);
    for g in 0..K / 4 {
        for n in 0..N {
            p.extend((0..4).map(|t| b[4 * g + t][n] as u8));
        }
    }
    let mut s = format!(
        "#[repr(C, align(64))]\nstruct Packed([u8; {}]);\nstatic BP: Packed = Packed({p:?});\nconst NB: usize = {NB};\n",
        p.len()
    );
    tile(&mut s, 6);
    tile(&mut s, 5);
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("kernel.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=model.rs");
}
