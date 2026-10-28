//! Dean's BLAKE3 hash program (src/simd/blake3_program.rs, Simd8 kernel, the one his
//! benches/simd_blake3.rs times) on his harness, register-cached JIT.
//!
//!     blake3_dean <call|slot> <dir> <bench> [check...]
//!
//! <dir>/msg-<name>.bin and <dir>/digest-<name>.bin come from programs/blake3/host. Every check
//! and the bench case are run on the interpreter and the JIT and the heap output is compared with
//! the digest. The bench case is then timed: the mean of back-to-back runs (a fresh VM each, same
//! mapping) in batches of >= 2 ms, median over 15 batches. Prints one JSON line.
use solana_sbpf::{
    error::ProgramResult,
    simd::{
        asm::VecMode,
        blake3_program::{build_input, hash_program, Kernel, HEAP_OUT, HEAP_SIZE},
        harness::{config, Engine, Runner},
    },
    vm::SimdJitStrategy,
};
use std::{path::Path, time::Instant};

fn runner(program: &[u8], dir: &Path, name: &str) -> (Runner, Vec<u8>) {
    let msg = std::fs::read(dir.join(format!("msg-{name}.bin"))).unwrap();
    let digest = std::fs::read(dir.join(format!("digest-{name}.bin"))).unwrap();
    let r = Runner::new(program, &build_input(&msg), HEAP_SIZE, config(SimdJitStrategy::RegisterCached)).unwrap();
    r.jit_compile().unwrap();
    (r, digest)
}

fn check(r: &mut Runner, digest: &[u8], engine: Engine) -> (u64, bool) {
    r.heap.as_slice_mut()[HEAP_OUT..HEAP_OUT + 32].fill(0);
    let (cu, result) = r.run(engine);
    (cu, matches!(result, ProgramResult::Ok(0)) && r.heap()[HEAP_OUT..HEAP_OUT + 32] == *digest)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = match a[1].as_str() {
        "call" => VecMode::Call,
        "slot" => VecMode::Slot,
        m => panic!("mode {}", m),
    };
    let dir = Path::new(&a[2]);
    let program = hash_program(Kernel::Simd8, mode);
    let mut ok = true;
    for name in &a[4..] {
        let (mut r, d) = runner(&program, dir, name);
        let (ci, oi) = check(&mut r, &d, Engine::Interpreter);
        let (cj, oj) = check(&mut r, &d, Engine::Jit);
        if !(oi && oj && ci == cj) {
            eprintln!("check {name} failed: interp {oi} jit {oj} cu {ci} {cj}");
            ok = false;
        }
    }
    let (mut r, d) = runner(&program, dir, &a[3]);
    let mut cu = vec![];
    let mut ns = vec![];
    for engine in [Engine::Interpreter, Engine::Jit] {
        let (c, o) = check(&mut r, &d, engine);
        ok &= o;
        cu.push(c);
        let s = Instant::now();
        r.run(engine);
        let n = (2e6 / s.elapsed().as_nanos() as f64).ceil().max(1.0) as usize;
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
        ns.push(t[t.len() / 2]);
        ok &= check(&mut r, &d, engine).1;
    }
    ok &= cu[0] == cu[1];
    println!(r#"{{"cu": {}, "jit_ns": {:.1}, "interp_ns": {:.1}, "ok": {}}}"#, cu[1], ns[1], ns[0], ok);
}
