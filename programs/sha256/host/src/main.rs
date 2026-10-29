// writes sha256 input files and the expected final input regions, from sha2::compress256.
// usage: sha256-host <out_dir> [case...]   (cases: zero, ones, or a numeric seed; default all of
// zero ones 1 2 3 4). input-<case>.bin = 16 blocks of 64 bytes + 512 zero bytes;
// expected-<case>.bin = the same blocks + 16 x 32-byte big-endian states after
// compress256(IV, block) (one block each, no padding).
use sha2::digest::generic_array::GenericArray;
use std::{fs, path::Path};

const IV: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];

fn blocks(case: &str) -> Vec<u8> {
    match case {
        "zero" => vec![0; 1024],
        "ones" => vec![0xff; 1024],
        s => {
            let mut x: u64 = s.parse::<u64>().unwrap().wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            (0..1024)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 24) as u8
                })
                .collect()
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let mut cases: Vec<String> = args.iter().skip(2).cloned().collect();
    if cases.is_empty() {
        cases = ["zero", "ones", "1", "2", "3", "4"].map(String::from).to_vec();
    }
    fs::create_dir_all(dir).unwrap();
    for case in &cases {
        let b = blocks(case);
        let mut input = b.clone();
        input.resize(1536, 0);
        let mut expected = b.clone();
        for i in 0..16 {
            let mut st = IV;
            sha2::compress256(&mut st, &[*GenericArray::from_slice(&b[64 * i..64 * i + 64])]);
            for w in st {
                expected.extend_from_slice(&w.to_be_bytes());
            }
        }
        fs::write(dir.join(format!("input-{case}.bin")), &input).unwrap();
        fs::write(dir.join(format!("expected-{case}.bin")), &expected).unwrap();
    }
    // sanity: compress256 of the padded empty message is sha256("")
    let mut st = IV;
    let mut pad = [0u8; 64];
    pad[0] = 0x80;
    sha2::compress256(&mut st, &[pad.into()]);
    assert_eq!(st[0], 0xe3b0c442);
}
