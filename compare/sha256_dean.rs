//! programs/sha256 ported to Dean's intrinsics (compare/dean-sha256). The instruction data is
//! sha256-host's 1536 byte input-*.bin unchanged; the region after the run must equal the matching
//! expected-*.bin. Every case runs on the interpreter and the JIT (same CU, same output). Timing on
//! the first case: mean of back-to-back runs in batches of about 2 ms, median of 15 batches.
//!
//!     sha256_dean <sha256_dean.so> <input.bin> <expected.bin> [<input.bin> <expected.bin>]...
use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let elf = std::fs::read(&a[1]).unwrap();
    let mut res = vec![];
    for (c, pair) in a[2..].chunks(2).enumerate() {
        let data = std::fs::read(&pair[0]).unwrap();
        let expected = std::fs::read(&pair[1]).unwrap();
        let mut input = Vec::new();
        input.extend_from_slice(&0u64.to_le_bytes());
        input.extend_from_slice(&(data.len() as u64).to_le_bytes());
        input.extend_from_slice(&data);
        input.extend_from_slice(&[7u8; 32]);
        let mut cus = vec![];
        for engine in [Engine::Interpreter, Engine::Jit] {
            let mut r = Runner::from_elf(&elf, &input, 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
            if matches!(engine, Engine::Jit) {
                r.jit_compile().unwrap();
            }
            let (cu, result) = r.run(engine);
            assert!(matches!(result, ProgramResult::Ok(0)), "{}: {result:?}", pair[0]);
            assert!(r.input.as_slice()[16..16 + data.len()] == expected[..], "{}: output mismatch", pair[0]);
            cus.push(cu);
            if c == 0 {
                let s = Instant::now();
                let mut n = 0;
                while s.elapsed().as_micros() < 2000 {
                    r.run(engine);
                    n += 1;
                }
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
                res.push(t[t.len() / 2]);
                if matches!(engine, Engine::Jit) {
                    eprintln!("{:?}", r.executable.get_compiled_program().unwrap().simd_stats);
                }
                // the timed runs left the same output
                assert!(r.input.as_slice()[16..16 + data.len()] == expected[..], "output changed");
            }
        }
        assert_eq!(cus[0], cus[1], "{}: CU differ between engines", pair[0]);
        if c == 0 {
            res.insert(0, cus[0] as f64);
        }
    }
    println!("ok CU {} jit_ns {:.1} interp_ns {:.1}", res[0], res[2], res[1]);
}
