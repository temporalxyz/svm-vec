// emits the model weights as .rodata statics, row-major for the scalar path and packed for dpbusd.
use std::{fmt::Write, fs, path::Path};

#[path = "model.rs"]
#[allow(dead_code)]
mod model;

fn bytes(name: &str, v: &[u8]) -> String {
    format!("pub static {name}: [u8; {}] = {v:?};\n", v.len())
}

fn main() {
    let m = model::model();
    let mut s = String::new();
    let w1: Vec<u8> = m.w1.iter().flatten().map(|&w| w as u8).collect();
    let w2: Vec<u8> = m.w2.iter().flatten().map(|&w| w as u8).collect();
    s += &bytes("W1", &w1);
    s += &bytes("W2", &w2);
    s += &bytes("W1P", &model::pack(&m.w1));
    s += &bytes("W2P", &model::pack(&m.w2));
    writeln!(s, "pub static B1: [i32; {}] = {:?};", m.b1.len(), m.b1).unwrap();
    writeln!(s, "pub static B2: [i32; {}] = {:?};", m.b2.len(), m.b2).unwrap();
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("weights.rs");
    fs::write(out, s).unwrap();
    println!("cargo:rerun-if-changed=model.rs");
}
