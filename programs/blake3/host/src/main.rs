// writes the blake3 test cases: msg-<name>.bin (the message), digest-<name>.bin (blake3::hash),
// input-<name>.bin / expected-<name>.bin (programs/blake3's input region before and after).
// usage: blake3-host <out_dir>. prints the case names.
use std::{fs, path::Path};

const DATA: usize = 64;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or(".".into());
    let dir = Path::new(&dir);
    fs::create_dir_all(dir).unwrap();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rand = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect()
    };
    // "1m" is the workload of Dean's benches/simd_blake3.rs at its largest size: 1 MiB of i % 251
    let mut cases: Vec<(String, Vec<u8>)> = vec![("1m".into(), (0..1 << 20).map(|i| (i % 251) as u8).collect())];
    for n in [0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 3073, 8192, 15 * 1024 + 7, 16384, 17 * 1024, 33 * 1024 + 1, 65536, 100_003] {
        cases.push((format!("p{n}"), (0..n).map(|i| (i % 251) as u8).collect()));
    }
    for n in [1000, 31 * 1024, 48 * 1024 + 999, 1 << 20] {
        cases.push((format!("r{n}"), rand(n)));
    }
    for (name, msg) in &cases {
        let digest = blake3::hash(msg);
        let mut input = vec![0u8; DATA + msg.len()];
        input[..8].copy_from_slice(&(msg.len() as u64).to_le_bytes());
        input[DATA..].copy_from_slice(msg);
        let mut expected = input.clone();
        expected[8..40].copy_from_slice(digest.as_bytes());
        fs::write(dir.join(format!("msg-{name}.bin")), msg).unwrap();
        fs::write(dir.join(format!("digest-{name}.bin")), digest.as_bytes()).unwrap();
        fs::write(dir.join(format!("input-{name}.bin")), &input).unwrap();
        fs::write(dir.join(format!("expected-{name}.bin")), &expected).unwrap();
        println!("{name}");
    }
}
