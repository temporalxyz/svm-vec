//! programs/matmul ported to Dean's intrinsics (compare/dean-matmul). Each input is matmul-host's
//! input-<seed>.bin unchanged as instruction data; the data after the run must equal the matching
//! expected-<seed>.bin on both engines. Times the first input per engine: mean of back-to-back
//! runs in batches of about 2 ms (calibrated), median of `batches`. Prints one JSON line, the JIT's
//! register cache statistics on stderr.
//!
//!     matmul_dean <matmul_dean.so> <batches> <input.bin> <expected.bin> [<input> <expected>]...
use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::time::Instant;

fn wrap(data: &[u8]) -> Vec<u8> {
    let mut input = Vec::new();
    input.extend_from_slice(&0u64.to_le_bytes());
    input.extend_from_slice(&(data.len() as u64).to_le_bytes());
    input.extend_from_slice(data);
    input.extend_from_slice(&[7u8; 32]);
    input
}

fn batch(r: &mut Runner, engine: Engine, n: usize) -> f64 {
    let s = Instant::now();
    for _ in 0..n {
        r.run(engine);
    }
    s.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let elf = std::fs::read(&a[1]).unwrap();
    let batches: usize = a[2].parse().unwrap();
    let cases: Vec<(Vec<u8>, Vec<u8>)> = a[3..]
        .chunks(2)
        .map(|c| (std::fs::read(&c[0]).unwrap(), std::fs::read(&c[1]).unwrap()))
        .collect();

    let mut ok = true;
    let mut cus = vec![];
    let mut times = vec![];
    for engine in [Engine::Interpreter, Engine::Jit] {
        for (k, (data, expected)) in cases.iter().enumerate() {
            let mut r = Runner::from_elf(&elf, &wrap(data), 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
            if matches!(engine, Engine::Jit) {
                r.jit_compile().unwrap();
            }
            let (cu, result) = r.run(engine);
            ok &= matches!(result, ProgramResult::Ok(0));
            ok &= r.input.as_slice()[16..16 + data.len()] == expected[..];
            cus.push(cu);
            if k == 0 {
                let n = (2e6 / batch(&mut r, engine, 3)).round().max(1.0) as usize;
                let mut t: Vec<f64> = (0..batches).map(|_| batch(&mut r, engine, n)).collect();
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times.push(t[t.len() / 2]);
                ok &= r.input.as_slice()[16..16 + data.len()] == expected[..];
                if matches!(engine, Engine::Jit) {
                    eprintln!("{:?}", r.executable.get_compiled_program().unwrap().simd_stats);
                }
            }
        }
    }
    ok &= cus.iter().all(|&c| c == cus[0]);
    println!(
        "{{\"cu\": {}, \"jit_ns\": {:.1}, \"interp_ns\": {:.1}, \"ok\": {ok}}}",
        cus[0], times[1], times[0]
    );
}
