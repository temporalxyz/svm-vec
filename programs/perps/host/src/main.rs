// writes a perps input region from a seed and the expected region after the program has run.
// usage: perps-host <out_dir> [seed...]   (writes input-<seed>.bin, expected-<seed>.bin)
use std::{fs, path::Path};

const N: usize = 256;

struct Rng(u64);
impl Rng {
    // splitmix64
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn signed(&mut self, magnitude: u64) -> i64 {
        let m = self.below(magnitude) as i64;
        if self.next() & 1 == 0 { m } else { -m }
    }
}

struct Pos {
    size: i64,
    entry: i64,
    mark: i64,
    funding_delta: i64,
    collateral: i64,
}

// the reference semantics
fn risk(p: &Pos) -> [u64; 4] {
    let pnl = p.size.wrapping_mul(p.mark.wrapping_sub(p.entry));
    let funding = p.size.wrapping_mul(p.funding_delta) >> 32;
    let equity = p.collateral.wrapping_add(pnl).wrapping_sub(funding);
    let req = p.size.wrapping_abs().wrapping_mul(p.mark) >> 36;
    [pnl as u64, funding as u64, equity as u64, (equity >= req) as u64]
}

fn positions(seed: u64) -> Vec<Pos> {
    let mut r = Rng(seed);
    (0..N)
        .map(|_| {
            let bits = 1 + r.below(40);
            let size = r.signed(1 << bits) | 1;
            let entry = (1 << 20) + r.below((1 << 32) - (1 << 20)) as i64;
            let mark = entry + r.signed(entry as u64 / 4);
            let funding_delta = r.signed(1 << 34);
            let mut p = Pos { size, entry, mark, funding_delta, collateral: 0 };
            // half near the maintenance threshold, half with random collateral
            p.collateral = if r.next() & 1 == 0 {
                let [pnl, funding, _, _] = risk(&p);
                let req = p.size.wrapping_abs().wrapping_mul(p.mark) >> 36;
                req.wrapping_sub(pnl as i64).wrapping_add(funding as i64).wrapping_add(r.signed(1 + (req.unsigned_abs() >> 3)))
            } else {
                r.below(1 << 50) as i64
            };
            p
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let seeds: Vec<u64> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![1] };
    fs::create_dir_all(dir).unwrap();
    for seed in seeds {
        let ps = positions(seed);
        let mut buf = vec![0u8; 9 * 8 * N];
        let put = |buf: &mut [u8], field: usize, i: usize, v: u64| {
            buf[8 * (field * N + i)..][..8].copy_from_slice(&v.to_le_bytes())
        };
        for (i, p) in ps.iter().enumerate() {
            for (f, v) in [p.size, p.entry, p.mark, p.funding_delta, p.collateral].into_iter().enumerate() {
                put(&mut buf, f, i, v as u64);
            }
        }
        fs::write(dir.join(format!("input-{seed}.bin")), &buf).unwrap();
        let mut healthy = 0;
        for (i, p) in ps.iter().enumerate() {
            let out = risk(p);
            healthy += out[3];
            for (f, v) in out.into_iter().enumerate() {
                put(&mut buf, 5 + f, i, v);
            }
        }
        fs::write(dir.join(format!("expected-{seed}.bin")), &buf).unwrap();
        eprintln!("seed {seed}: {healthy}/{N} healthy");
    }
}
