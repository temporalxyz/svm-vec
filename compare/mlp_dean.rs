//! programs/mlp's vector path ported to Dean's intrinsics (compare/dean-mlp). The instruction data
//! is mlp-host's 384 byte input-vector.bin unchanged; the region after the run must equal
//! expected-vector.bin. Prints CU and ns per inference (mean of back-to-back runs, median of 15).
//!
//!     mlp_dean <mlp_dean.so> <input-vector.bin> <expected-vector.bin> [repeat]
use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let elf = std::fs::read(&a[1]).unwrap();
    let data = std::fs::read(&a[2]).unwrap();
    let expected = std::fs::read(&a[3]).unwrap();
    let repeat: usize = a.get(4).map_or(2000, |s| s.parse().unwrap());
    let mut input = Vec::new();
    input.extend_from_slice(&0u64.to_le_bytes());
    input.extend_from_slice(&(data.len() as u64).to_le_bytes());
    input.extend_from_slice(&data);
    input.extend_from_slice(&[7u8; 32]);

    let mut res = vec![];
    for (engine, n) in [(Engine::Interpreter, repeat / 20 + 1), (Engine::Jit, repeat)] {
        let mut r = Runner::from_elf(&elf, &input, 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
        if matches!(engine, Engine::Jit) {
            r.jit_compile().unwrap();
        }
        let (cu, result) = r.run(engine);
        let ProgramResult::Ok(r0) = result else { panic!("{:?}", result) };
        assert!(r.input.as_slice()[16..16 + data.len()] == expected[..], "output mismatch");
        let mut t: Vec<f64> = (0..15)
            .map(|_| {
                let s = Instant::now();
                for _ in 0..n {
                    r.run(engine);
                }
                s.elapsed().as_nanos() as f64 / n as f64
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        res.push((cu, r0, t[t.len() / 2]));
        if matches!(engine, Engine::Jit) {
            eprintln!("{:?}", r.executable.get_compiled_program().unwrap().simd_stats);
        }
    }
    assert_eq!(res[0].0, res[1].0, "CU differ between engines");
    println!("ok argmax {} CU {} jit_ns {:.1} interp_ns {:.1}", res[1].1, res[1].0, res[1].2, res[0].2);
}
