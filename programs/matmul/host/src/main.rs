// writes input-<seed>.bin (A, C zeroed) and expected-<seed>.bin (A, C = A x B) for each seed.
// usage: matmul-host <out_dir> <seed>...
use std::{fs, path::Path};

#[path = "../../model.rs"]
mod model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    fs::create_dir_all(dir).unwrap();
    let b = model::weights();
    for seed in &args[2..] {
        let a = model::input(seed.parse().unwrap());
        let mut buf = vec![0u8; model::LEN];
        buf[..model::C_OFF].copy_from_slice(&a);
        fs::write(dir.join(format!("input-{seed}.bin")), &buf).unwrap();
        for (i, c) in model::matmul(&a, &b).iter().enumerate() {
            buf[model::C_OFF + 4 * i..][..4].copy_from_slice(&c.to_le_bytes());
        }
        fs::write(dir.join(format!("expected-{seed}.bin")), &buf).unwrap();
    }
}
