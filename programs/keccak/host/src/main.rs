// writes keccak input files (scalar and vector mode) and the expected final regions.
// usage: keccak-host <out_dir> <k> [seed]   (seed 0 = zero state)
use std::{fs, path::Path};

#[path = "../../keccak.rs"]
mod keccak;

fn main() {
    let mut z = [0u64; 25];
    keccak::keccak_f(&mut z);
    assert_eq!(z[0], 0xF1258F7940E1DDE7, "standard test vector");

    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let k: u64 = args.get(2).map_or(1, |s| s.parse().unwrap());
    let mut seed: u64 = args.get(3).map_or(1, |s| s.parse().unwrap());
    let zero = seed == 0;
    let state: [u64; 25] = std::array::from_fn(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if zero { 0 } else { seed ^ seed >> 29 }
    });
    let mut out = state;
    for _ in 0..k {
        keccak::keccak_f(&mut out);
    }

    fs::create_dir_all(dir).unwrap();
    for (name, mode) in [("scalar", 0u64), ("vector", 1)] {
        let mut buf = vec![0u8; 288];
        buf[..8].copy_from_slice(&mode.to_le_bytes());
        buf[8..16].copy_from_slice(&k.to_le_bytes());
        let mut exp = buf.clone();
        for i in 0..25 {
            buf[64 + 8 * i..72 + 8 * i].copy_from_slice(&state[i].to_le_bytes());
            exp[64 + 8 * i..72 + 8 * i].copy_from_slice(&out[i].to_le_bytes());
        }
        fs::write(dir.join(format!("input-{name}.bin")), &buf).unwrap();
        fs::write(dir.join(format!("expected-{name}.bin")), &exp).unwrap();
    }
    println!("{{\"k\":{k},\"lane0\":\"{:#018x}\"}}", out[0]);
}
