// writes fieldmul input and expected files: 8 lanes, r = a, then 256 times r = r * b mod 2^255 - 19.
// region (1024 bytes): lane l has a at 64 l, b at 64 l + 32; the program writes r at 512 + 64 l and
// b again at 512 + 64 l + 32. all numbers are 32 byte little endian, r canonical.
// usage: fieldmul-host <out_dir> <case>...   (case = seed, "edge" or "fallback")
// "edge" keeps programs/fieldmul on its shared-limb path with limbs at the bound, "fallback" has two
// lanes (b = 2^254 + 2^52 - 1 and p - 2^52 + 1) that make it take its general path.
use num_bigint::BigUint;
use std::{fs, path::Path};

const LANES: usize = 8;
const ITERS: usize = 256;

fn p() -> BigUint {
    (BigUint::from(1u8) << 255u32) - 19u32
}

fn bytes(x: &BigUint) -> [u8; 32] {
    let mut out = [0u8; 32];
    let b = x.to_bytes_le();
    out[..b.len()].copy_from_slice(&b);
    out
}

fn lanes(case: &str) -> Vec<(BigUint, BigUint)> {
    let p = p();
    if case == "edge" || case == "fallback" {
        let one = BigUint::from(1u8);
        let low = (BigUint::from(1u64) << 52u32) - 1u32;
        let big = (one.clone() << 254u32) + &low;
        return vec![
            (BigUint::from(0u8), &p - 1u32),
            (one.clone(), one.clone()),
            (&p - 1u32, &p - 1u32),
            (&p - 1u32, BigUint::from(2u8)),
            (&p - 2u32, low.clone()),
            (big.clone(), if case == "edge" { &p - &big } else { big.clone() }),
            (&p - 1u32, if case == "edge" { &p - &low - 1u32 } else { &p - &low }),
            (BigUint::from(19u8), (&p - 1u32) >> 1u32),
        ];
    }
    let mut s: u64 = case.parse().expect("case: a seed, \"edge\" or \"fallback\"");
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ z >> 30).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ z >> 27).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ z >> 31
    };
    let mut field = || loop {
        let mut b = [0u8; 32];
        for c in b.chunks_mut(8) {
            c.copy_from_slice(&next().to_le_bytes());
        }
        b[31] &= 0x7f;
        let x = BigUint::from_bytes_le(&b);
        if x < p {
            return x;
        }
    };
    (0..LANES).map(|_| (field(), field())).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    fs::create_dir_all(dir).unwrap();
    let p = p();
    for case in &args[2..] {
        let mut input = vec![0u8; 1024];
        let mut out = vec![0u8; 1024];
        for (l, (a, b)) in lanes(case).iter().enumerate() {
            let mut r = a.clone();
            for _ in 0..ITERS {
                r = r * b % &p;
            }
            input[64 * l..64 * l + 32].copy_from_slice(&bytes(a));
            input[64 * l + 32..64 * l + 64].copy_from_slice(&bytes(b));
            out[512 + 64 * l..512 + 64 * l + 32].copy_from_slice(&bytes(&r));
            out[512 + 64 * l + 32..512 + 64 * l + 64].copy_from_slice(&bytes(b));
        }
        out[..512].copy_from_slice(&input[..512]);
        fs::write(dir.join(format!("input-{case}.bin")), &input).unwrap();
        fs::write(dir.join(format!("expected-{case}.bin")), &out).unwrap();
    }
}
