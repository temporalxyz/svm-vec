//! Keccak-f only: the falcon program with a keccak-only entry (8 bytes of instruction data = k,
//! keccak_init then k x keccak_f). CU and wall time per permutation = (k=100 - k=0) / 100.
use solana_sbpf::{simd::harness::{config, Engine, Runner}, vm::SimdJitStrategy};
use std::time::Instant;

fn input(k: u64) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&0u64.to_le_bytes());
    d.extend_from_slice(&8u64.to_le_bytes());
    d.extend_from_slice(&k.to_le_bytes());
    d.extend_from_slice(&[7u8; 32]);
    d
}

fn main() {
    let elf = std::fs::read(std::env::args().nth(1).expect("path to .so")).unwrap();
    let mut res = vec![];
    for k in [0u64, 100] {
        let mut r = Runner::from_elf(&elf, &input(k), 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
        r.jit_compile().unwrap();
        let (cu, result) = r.run(Engine::Jit);
        assert!(matches!(result, solana_sbpf::error::ProgramResult::Ok(0)), "{result:?}");
        let mut t: Vec<f64> = (0..15)
            .map(|_| {
                let s = Instant::now();
                for _ in 0..2000 {
                    r.run(Engine::Jit);
                }
                s.elapsed().as_nanos() as f64 / 2000.0
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        res.push((cu, t[t.len() / 2]));
    }
    println!(
        "per permutation {:.1} CU {:.1} ns",
        (res[1].0 - res[0].0) as f64 / 100.0,
        (res[1].1 - res[0].1) / 100.0
    );
}
