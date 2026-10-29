//! programs/fieldmul ported to Dean's intrinsics (compare/dean-fieldmul). The instruction data is
//! fieldmul-host's 1024 byte input-*.bin unchanged; the region after the run must equal the
//! matching expected-*.bin. Checks every pair on the interpreter and the JIT, then times the first:
//! mean of back-to-back runs in batches of ~2 ms, median of 9 batches per engine. Prints
//! {"cu", "jit_ns", "interp_ns", "ok"}.
//!
//!     fieldmul_dean <fieldmul_dean.so> <input.bin> <expected.bin> [<input.bin> <expected.bin>]...
use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::time::Instant;

fn runner(elf: &[u8], data: &[u8]) -> Runner {
    let mut input = Vec::new();
    input.extend_from_slice(&0u64.to_le_bytes());
    input.extend_from_slice(&(data.len() as u64).to_le_bytes());
    input.extend_from_slice(data);
    input.extend_from_slice(&[7u8; 32]);
    let r = Runner::from_elf(elf, &input, 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
    r.jit_compile().unwrap();
    r
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let elf = std::fs::read(&a[1]).unwrap();
    let mut ok = true;
    let mut cus = vec![];
    for pair in a[2..].chunks(2) {
        let data = std::fs::read(&pair[0]).unwrap();
        let expected = std::fs::read(&pair[1]).unwrap();
        for engine in [Engine::Interpreter, Engine::Jit] {
            let mut r = runner(&elf, &data);
            let (cu, result) = r.run(engine);
            let good = matches!(result, ProgramResult::Ok(0)) && r.input.as_slice()[16..16 + data.len()] == expected[..];
            if !good {
                eprintln!("{} {engine:?}: {result:?}, output matches: {}", pair[0], r.input.as_slice()[16..16 + data.len()] == expected[..]);
            }
            ok &= good;
            cus.push(cu);
        }
    }
    ok &= cus.iter().all(|&c| c == cus[0]);

    let data = std::fs::read(&a[2]).unwrap();
    let mut times = vec![];
    for engine in [Engine::Interpreter, Engine::Jit] {
        let mut r = runner(&elf, &data);
        let s = Instant::now();
        r.run(engine);
        let n = (2e6 / s.elapsed().as_nanos() as f64).round().max(1.0) as usize;
        let mut t: Vec<f64> = (0..9)
            .map(|_| {
                let s = Instant::now();
                for _ in 0..n {
                    r.run(engine);
                }
                s.elapsed().as_nanos() as f64 / n as f64
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.push(t[t.len() / 2]);
        if matches!(engine, Engine::Jit) {
            eprintln!("repeat {n}: {:?}", r.executable.get_compiled_program().unwrap().simd_stats);
        }
    }
    println!(
        "{{\"cu\": {}, \"jit_ns\": {:.1}, \"interp_ns\": {:.1}, \"ok\": {ok}}}",
        cus[0], times[1], times[0]
    );
}
