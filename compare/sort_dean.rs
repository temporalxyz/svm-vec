//! programs/sort on Dean's intrinsics (compare/dean-sort). The instruction data is sort-host's
//! 2048 byte input-<seed>.bin unchanged; the region after the run must equal expected-<seed>.bin.
//! Every case runs on the interpreter and the JIT. The first case is timed like svm-run: the mean
//! of `repeat` back-to-back runs (fresh VM each), median of `batches`; repeat 0 calibrates it to
//! ~2 ms of JIT runs per batch, and the interpreter uses the same repeat. Prints one JSON line.
//!
//!     sort_dean <sort_dean.so> <repeat> <batches> <input.bin> <expected.bin> [<input> <expected>]...
use solana_sbpf::{
    error::ProgramResult,
    simd::harness::{config, Engine, Runner},
    vm::SimdJitStrategy,
};
use std::time::Instant;

fn runner(elf: &[u8], data: &[u8], engine: Engine) -> Runner {
    let mut input = Vec::new();
    input.extend_from_slice(&0u64.to_le_bytes());
    input.extend_from_slice(&(data.len() as u64).to_le_bytes());
    input.extend_from_slice(data);
    input.extend_from_slice(&[7u8; 32]);
    let r = Runner::from_elf(elf, &input, 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
    if matches!(engine, Engine::Jit) {
        r.jit_compile().unwrap();
    }
    r
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
    let mut repeat: usize = a[2].parse().unwrap();
    let batches: usize = a[3].parse().unwrap();
    let mut ok = true;
    let mut cus = vec![];
    for case in a[4..].chunks(2) {
        let data = std::fs::read(&case[0]).unwrap();
        let expected = std::fs::read(&case[1]).unwrap();
        for engine in [Engine::Interpreter, Engine::Jit] {
            let mut r = runner(&elf, &data, engine);
            let (cu, result) = r.run(engine);
            let good = matches!(result, ProgramResult::Ok(0)) && r.input.as_slice()[16..16 + data.len()] == expected[..];
            if !good {
                eprintln!("{} {:?}: {:?} output mismatch or error", case[0], engine, result);
            }
            ok &= good;
            cus.push(cu);
        }
    }
    ok &= cus.iter().all(|&c| c == cus[0]);
    let data = std::fs::read(&a[4]).unwrap();
    let mut t = vec![];
    for engine in [Engine::Interpreter, Engine::Jit] {
        let mut r = runner(&elf, &data, engine);
        r.run(engine);
        if repeat == 0 {
            let mut j = runner(&elf, &data, Engine::Jit);
            batch(&mut j, Engine::Jit, 200);
            repeat = ((2e6 / batch(&mut j, Engine::Jit, 2000)).round() as usize).max(1);
        }
        let mut s: Vec<f64> = (0..batches).map(|_| batch(&mut r, engine, repeat)).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t.push(s[s.len() / 2]);
        if matches!(engine, Engine::Jit) {
            eprintln!("{:?} repeat {repeat}", r.executable.get_compiled_program().unwrap().simd_stats);
        }
    }
    println!("{{\"cu\": {}, \"jit_ns\": {:.1}, \"interp_ns\": {:.1}, \"ok\": {ok}}}", cus[0], t[1], t[0]);
}
