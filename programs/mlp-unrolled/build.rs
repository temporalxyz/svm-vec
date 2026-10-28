// emits programs/mlp's model as one rodata blob (b1, packed w1, b2, packed w2, argmax indices)
// and the vector path as straight-line code addressing it at constant offsets from one base.
use std::{fmt::Write, fs, path::Path};

#[path = "../mlp/model.rs"]
#[allow(dead_code)]
mod model;
use model::{HID, IN};

// every offset from the base must fit the 16-bit `off` field
const BASE: usize = 32768;
// broadcasts per batch: 8 accumulators + B1 + 1 weight, 1 accumulator + B2 + 1 weight <= 30
const B1: usize = 16;
const B2: usize = 16;

fn main() {
    let m = model::model();
    let mut blob: Vec<u8> = m.b1.iter().flat_map(|b| b.to_le_bytes()).collect();
    let w1 = blob.len() as isize - BASE as isize;
    blob.extend(model::pack(&m.w1));
    let b2 = blob.len() as isize - BASE as isize;
    blob.extend(m.b2.iter().flat_map(|b| b.to_le_bytes()));
    let w2 = blob.len() as isize - BASE as isize;
    blob.extend(model::pack(&m.w2));
    // permq indices for argmax: swap qwords within 128 bits, 128-bit lanes within 256, 256 halves
    let perm = blob.len() as isize - BASE as isize;
    for idx in [[1u64, 0, 3, 2, 5, 4, 7, 6], [2, 3, 0, 1, 6, 7, 4, 5], [4, 5, 6, 7, 0, 1, 2, 3]] {
        blob.extend(idx.iter().flat_map(|i| i.to_le_bytes()));
    }
    assert!(blob.len() - 64 - BASE <= i16::MAX as usize);

    let mut s = String::new();
    writeln!(s, "#[repr(C, align(64))]\nstruct Blob([u8; {}]);", blob.len()).unwrap();
    writeln!(s, "static BLOB: Blob = Blob({blob:?});\nconst BASE: usize = {BASE};\nconst PERM: isize = {perm};").unwrap();
    s += "#[inline(always)]\nunsafe fn vector(x: *const u8, out: *mut u8, p: *const u8, h: *mut u8) -> __svm512i {\n";
    let ld = |o: isize| format!("_svm512_loadu_si512(p.offset({o}))");

    // layer 1: acc k holds outputs 16k..16k+15; per group of 4 inputs one broadcast, 8 dpbusd
    for k in 0..HID / 16 {
        writeln!(s, "let mut a{k} = {};", ld(-(BASE as isize) + 64 * k as isize)).unwrap();
    }
    // memory ops all address through r1, so broadcasts are loaded in batches to switch r1
    // between x and the weights once per batch instead of once per group
    for batch in (0..IN / 4).collect::<Vec<_>>().chunks(B1) {
        for &g in batch {
            writeln!(s, "let x{g} = _svm512_loadbc_epi32(x.add({}) as _);", 4 * g).unwrap();
        }
        for &g in batch {
            for k in 0..HID / 16 {
                let o = w1 + (g * HID * 4 + 64 * k) as isize;
                writeln!(s, "a{k} = _svm512_dpbusd_epi32(a{k}, x{g}, {});", ld(o)).unwrap();
            }
        }
    }
    // (acc >> 10) clamped to 0..=127: sat16 pack, max 0, sat8 pack. per 128-bit lane L the packs
    // put 4-byte group L of a0, a1, a2, a3 in order, so hidden group 4m + L of vector v lands at
    // byte 64v + 16L + 4m.
    s += "let z = _svm512_setzero_si512();\n";
    for v in 0..2 {
        let a = |j: usize| format!("_svm512_srai_epi32::<10>(a{})", 4 * v + j);
        writeln!(s, "let lo = _svm512_max_epi16(_svm512_packs_epi32({}, {}), z);", a(0), a(1)).unwrap();
        writeln!(s, "let hi = _svm512_max_epi16(_svm512_packs_epi32({}, {}), z);", a(2), a(3)).unwrap();
        writeln!(s, "_svm512_storeu_si512(h.add({}), _svm512_packs_epi16(lo, hi));", 64 * v).unwrap();
    }
    // layer 2
    writeln!(s, "let mut acc = {};", ld(b2)).unwrap();
    for batch in (0..HID / 4).collect::<Vec<_>>().chunks(B2) {
        for &g in batch {
            let (v, r) = (g / 16, g % 16);
            let off = 64 * v + 16 * (r % 4) + 4 * (r / 4);
            writeln!(s, "let h{g} = _svm512_loadbc_epi32(h.add({off}) as _);").unwrap();
        }
        for &g in batch {
            writeln!(s, "acc = _svm512_dpbusd_epi32(acc, h{g}, {});", ld(w2 + 64 * g as isize)).unwrap();
        }
    }
    s += "_svm512_storeu_si512(out, acc);\nacc\n}\n";
    fs::write(Path::new(&std::env::var("OUT_DIR").unwrap()).join("vector.rs"), s).unwrap();
    println!("cargo:rerun-if-changed=../mlp/model.rs");
}
