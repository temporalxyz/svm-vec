//! Cost per op of Dean's vector intrinsics (static call form: `mov64 r1, operands` + `call`, 512
//! bit forms on slot pairs, vector k of the spec = slots 2k, 2k + 1). The programs come from his
//! macro assembler (`simd::asm`, `VecMode::Call`); what they compute and the host reference are in
//! compare/opbench/src/spec.rs, shared with our side. Each program is checked against the reference
//! on three inputs in the interpreter and the JIT, then program and baseline are timed with a loop
//! around Runner::run (fresh VM per run on one memory mapping, batches of about 2 ms, median of 9).
//!
//!     ops_dean [filter]   -> one JSON line {"ops": [...], "ok": bool}
#[allow(dead_code)]
#[path = "ops_spec.rs"]
mod spec;

use solana_sbpf::{
    error::ProgramResult,
    simd::{
        asm::{Asm, VecMode},
        harness::{config, Engine, Runner},
        *,
    },
    vm::SimdJitStrategy,
};
use spec::*;
use std::time::Instant;

#[derive(Clone, Copy)]
enum R {
    C,
    X,
    Y,
    No,
}

/// (sub-operation, off, va, vb) of the 512 bit ops whose destination is the chain vector
fn wide(name: &str) -> Option<(u8, i16, R, R)> {
    use R::*;
    let dean = name.split('=').nth(1)?;
    Some(match dean {
        "add8x64" => (VADD8, 0, C, X),
        "add16x32" => (VADD16, 0, C, X),
        "add32x16" => (VADD32, 0, C, X),
        "add64x8" => (VADD64, 0, C, X),
        "sub32x16" => (VSUB32, 0, C, X),
        "sub64x8" => (VSUB64, 0, C, X),
        "mullo16x32" => (VMULLO16, 0, C, X),
        "mullo32x16" => (VMULLO32, 0, C, X),
        "mullo64x8" => (VMULLO64, 0, C, X),
        "mulu32x8" => (VMULU32, 0, C, X),
        "muls32x8" => (VMULS32, 0, C, X),
        "mulhi16x32" => (VMULHI16, 0, C, X),
        "mulhu16x32" => (VMULHU16, 0, C, X),
        "max512.d" => (VMAX, 2, C, X),
        "max512.w" => (VMAX, 1, C, X),
        "minu32x16" => (VMINU32, 0, C, X),
        "minu16x32" => (VMINU16, 0, C, X),
        "min512.uq" => (VMIN, 7, C, X),
        "and512" => (VAND, 0, C, X),
        "or512" => (VOR, 0, C, X),
        "xor512" => (VXOR, 0, C, X),
        "andn512" => (VANDN, 0, C, X),
        "ternlog512" => (VTERNLOG, 0x96, X, Y),
        "sll16x32" => (VSLL16, 3, C, No),
        "sll32x16" => (VSLL32, 7, C, No),
        "srl64x8" => (VSRL64, 13, C, No),
        "sra64x8" => (VSRA64, 13, C, No),
        "ror32x16" => (VROR32, 25, C, No),
        "ror64x8" => (VROR64, 51, C, No),
        "sllv32x16" => (VSLLV32, 0, C, X),
        "srlv64x8" => (VSRLV64, 0, C, X),
        "rolv32x16" => (VROLV32, 0, C, X),
        "rolv64x8" => (VROLV64, 0, C, X),
        "dpbusd512" => (VDPBUSD, 0, X, Y),
        "dpwssd512" => (VDPWSSD, 0, X, Y),
        "maddubs16x32" => (VMADDUBS16, 0, C, X),
        "madd16x32" => (VMADD16, 0, C, X),
        "madd52lo64x8" => (VMADD52LO, 0, X, Y),
        "madd52hi64x8" => (VMADD52HI, 0, X, Y),
        "packss512.d" => (VPACKSS, 1, C, X),
        "packus512.w" => (VPACKUS, 0, C, X),
        "permd512" => (VPERMD, 0, X, C),
        "shufb512" => (VSHUFB, 0, C, X),
        "zext512.bd" => (VZEXT, 8, C, No),
        "sext512.wd" => (VSEXT, 9, C, No),
        "trunc512.db" => (VTRUNC, 2, C, No),
        "popcnt512.b" => (VPOPCNT, 0, C, No),
        "popcnt512.q" => (VPOPCNT, 3, C, No),
        "clmul512" => (VCLMUL, 0x01, C, X),
        "mov512" => (VMOV, 0, X, No),
        "bcast32x16" => (VBCAST32, 0, No, No),
        "bcast64x8" => (VBCAST64, 0, No, No),
        "expand512.d" => (VEXPAND, 2, C, No),
        "blendm512.d" => (VBLENDM, 2, C, X),
        "conflict512.d" => (VCONFLICT, 2, C, No),
        "conflict512.q" => (VCONFLICT, 3, C, No),
        "lzcnt512.d" => (VLZCNT, 2, C, No),
        "mont16x32" => (VMONT16, 22, C, X),
        "permb512" => (VPERMB, 0, X, C),
        "permi2b512" => (VPERMI2B, 0, X, Y),
        "permw512" => (VPERMW, 0, X, C),
        "permi2d512" => (VPERMI2D, 0, X, Y),
        "permt2d512" => (VPERMT2D, 0, X, Y),
        "multishift512" => (VMULTISHIFT, 0, C, X),
        "gf2p8affine512" => (VGF2P8AFFINE, 0x5a, C, X),
        "gf2p8mul512" => (VGF2P8MUL, 0, C, X),
        "sadbw512" => (VSADBW, 0, C, X),
        "shldv512.d" => (VSHLDV, 2, X, Y),
        "alignr512" => (VALIGNR, 5, C, X),
        "abs512.d" => (VABS, 2, C, No),
        "adds512.w" => (VADDS, 1, C, X),
        "avg512.b" => (VAVG, 0, C, X),
        "permq512" => (VPERMQ, 0x1b, C, No),
        "shufd512" => (VSHUFD, 0x1b, C, No),
        "blendd512" => (VBLENDD, 0xa5, C, X),
        "unpacklo512.d" => (VUNPCKLO, 2, C, X),
        "sllv16x32" => (VSLLV16, 0, C, X),
        "srav16x32" => (VSRAV16, 0, C, X),
        "mulhrs16x32" => (VMULHRS16, 0, C, X),
        _ => return None,
    })
}

const CUSTOM: &[&str] = &[
    "ins64", "ext64", "movmsk8x64", "ins64+ext64", "load256x2", "store256x2", "load256x32", "store256x32",
    "(composite)", "load256x2.rodata", "loadm512.d", "storem512.d", "compress512.d", "compress512.b",
    "transpose8x8q", "transpose8x8", "cmp512.gt.d", "testm512.d",
];

fn supported(name: &str) -> bool {
    wide(name).is_some() || name.split('=').nth(1).is_some_and(|d| CUSTOM.contains(&d))
}

/// Scalar operand in r2 / r3 before the loop (r2 is the input base otherwise)
fn setup(a: &mut Asm, name: &str) {
    let dean = name.split('=').nth(1).unwrap();
    match dean {
        "bcast32x16" | "bcast64x8" | "ins64" => a.lddw(2, S),
        "expand512.d" | "blendm512.d" | "compress512.d" | "compress512.b" => a.lddw(2, M),
        "loadm512.d" | "storem512.d" => a.lddw(3, M),
        _ => {}
    }
}

fn instance(a: &mut Asm, name: &str, dep: bool, j: usize) {
    let c = chain(dep, j) as u8;
    let slot = |r: R| match r {
        R::C => 2 * c,
        R::X => 2,
        R::Y => 4,
        R::No => 0,
    };
    if let Some((subop, off, va, vb)) = wide(name) {
        let src = if uses_src(subop) { 2 } else { 0 };
        return a.vop_full(subop, true, 2 * c, slot(va), slot(vb), 0, src, off);
    }
    let lane = |k: usize| ((k % 8) / 4) as u8;
    let sub = |k: usize| (k % 4) as i16;
    match name.split('=').nth(1).unwrap() {
        "ins64" => a.vop_full(VINS64, false, 2 * c + lane(j), 0, 0, 0, 2, sub(j)),
        "ext64" => a.vext64(0, 2 * c + 1, 1),
        "movmsk8x64" => a.vmovmsk8(0, 2 * c, true),
        "ins64+ext64" => {
            a.vins64(lane(j), 0, sub(j));
            a.vext64(0, lane(j + 1), sub(j + 1));
        }
        "load256x2" => a.vload_n(2 * c, 2, 2, mem_off(j) as i16),
        "store256x2" => a.vstore_n(2, mem_off(j) as i16, 2 * c, 2),
        "load256x32" => a.vload_n(0, 32, 2, bulk_off(j) as i16),
        "store256x32" => a.vstore_n(2, bulk_off(j) as i16, 0, 32),
        "load256x2.rodata" => {
            if j == 0 {
                a.lddw(2, solana_sbpf::ebpf::MM_RODATA_START);
            }
            a.vload_n(2 * c, 2, 2, ro_off(j) as i16)
        }
        "loadm512.d" => a.vop_full(VLOADM, true, 2 * c, 0, 2, 3, 2, mem_off(j) as i16),
        "storem512.d" => a.vop_full(VSTOREM, true, 0, 2 * c, 2, 3, 2, mem_off(j) as i16),
        "compress512.d" => a.vop_full(VCOMPRESS, true, 2 * c, 2 * c, 0, 0, 2, 2),
        "compress512.b" => a.vop_full(VCOMPRESS, true, 2 * c, 2 * c, 0, 0, 2, 0),
        "transpose8x8q" => a.vop_full(VTRANSPOSE64, true, 6, 0, 0, 0, 0, 0),
        "transpose8x8" => a.vop_full(VTRANSPOSE8, false, 6, 0, 0, 0, 0, 0),
        "cmp512.gt.d" => a.vop_full(VCMP, true, 0, 2 * c, 2, 0, 0, 0x26),
        "testm512.d" => a.vop_full(VTESTM, true, 0, 2 * c, 2, 0, 0, 2),
        // No horizontal sum: fold the halves, then within 256 bits (temporaries in slots 24, 25)
        "(composite)" if name.starts_with("hsum.d") => {
            a.vadd(24, 2 * c, 2 * c + 1);
            a.vop_full(VPERMQ, false, 25, 24, 0, 0, 0, 0x4e);
            a.vadd(24, 24, 25);
            a.vshufd(25, 24, 0x4e);
            a.vadd(24, 24, 25);
            a.vext64(0, 24, 0);
            a.mov64(3, 0);
            a.rsh64_imm(3, 32);
            a.add64(0, 3);
            a.mov32(0, 0);
        }
        "(composite)" if name.starts_with("hsum.q") => {
            a.vop_full(VADD64, false, 24, 2 * c, 2 * c + 1, 0, 0, 0);
            a.vop_full(VPERMQ, false, 25, 24, 0, 0, 0, 0x4e);
            a.vop_full(VADD64, false, 24, 24, 25, 0, 0, 0);
            a.vshufd(25, 24, 0x4e);
            a.vop_full(VADD64, false, 24, 24, 25, 0, 0, 0);
            a.vext64(0, 24, 0);
        }
        other => panic!("no lowering for {}", other),
    }
}

/// Whole program with k instances per body and l iterations; `bench` None is the baseline (empty
/// loop body)
fn program(bench: Option<(&str, bool)>, k: usize, l: usize) -> Vec<u8> {
    let mut a = Asm::new(VecMode::Call);
    a.mov64(6, 1);
    a.mov64_imm(0, 0);
    a.mov64(2, 6);
    a.vload_n(0, 2 * NV as u8, 2, 0);
    if let Some((name, _)) = bench {
        setup(&mut a, name);
    }
    a.mov64_imm(8, l as i64);
    let top = a.new_label();
    a.bind(top);
    if let Some((name, dep)) = bench {
        for j in 0..k {
            instance(&mut a, name, dep, j);
        }
    }
    a.add64_imm(8, -1);
    a.jne_imm(8, 0, top);
    a.mov64(7, 0);
    a.mov64(2, 6);
    a.vstore_n(2, 0, 0, 2 * NV as u8);
    a.mov64(0, 7);
    a.exit();
    a.finish()
}

fn runner(text: &[u8], input: &[u8]) -> Runner {
    let r = Runner::new(text, input, 32 * 1024, config(SimdJitStrategy::RegisterCached)).unwrap();
    r.jit_compile().unwrap();
    r
}

fn batch(r: &mut Runner, engine: Engine, repeat: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..repeat {
        std::hint::black_box(r.run(engine));
    }
    start.elapsed().as_nanos() as f64 / repeat as f64
}

/// Median batch means of the program and the baseline, alternating
fn time(p: &mut Runner, b: &mut Runner, engine: Engine) -> (f64, f64) {
    let single = |r: &mut Runner| (0..3).map(|_| batch(r, engine, 1)).fold(f64::MAX, f64::min);
    let (rp, rb) = (repeat_for(single(p)), repeat_for(single(b)));
    let (mut tp, mut tb) = (vec![], vec![]);
    for _ in 0..BATCHES {
        tb.push(batch(b, engine, rb));
        tp.push(batch(p, engine, rp));
    }
    (median(tp), median(tb))
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let base_text = program(None, K, L);
    let mut ok = true;
    let mut rows = vec![];
    for bench in benches() {
        if !supported(bench.name) || !bench.name.contains(filter.as_str()) {
            continue;
        }
        for dep in [true, false] {
            if !(if dep { bench.dep } else { bench.ind }) {
                continue;
            }
            let label = label(&bench, dep);
            let text = program(Some((bench.name, dep)), K, L);
            let mut cus = vec![];
            for (k, l) in CHECKS {
                let text = program(Some((bench.name, dep)), k, l);
                for seed in [1, 2, 3] {
                    let input = input(seed);
                    let (want, want_r0) = reference(&bench, dep, &input, &text, k, l);
                    for engine in [Engine::Interpreter, Engine::Jit] {
                        let mut r = runner(&text, &input);
                        let (cu, result) = r.run(engine);
                        let good = matches!(result, ProgramResult::Ok(r0) if r0 == want_r0) && r.input.as_slice() == &want[..];
                        if !good {
                            eprintln!("{label} ({k}x{l}): seed {seed} {engine:?}: mismatch ({result:?}, want r0 {want_r0:#x})");
                            ok = false;
                        }
                        if k == K {
                            cus.push(cu);
                        }
                    }
                }
            }
            let input = input(1);
            let (mut p, mut b) = (runner(&text, &input), runner(&base_text, &input));
            let cu = (p.run(Engine::Interpreter).0 - b.run(Engine::Interpreter).0) as f64 / N as f64;
            if cus.iter().any(|c| *c != cus[0]) {
                eprintln!("{label}: CU differ between runs {cus:?}");
                ok = false;
            }
            let stats = p.executable.get_compiled_program().unwrap().simd_stats;
            let (ji, jb) = time(&mut p, &mut b, Engine::Jit);
            let (ii, ib) = time(&mut p, &mut b, Engine::Interpreter);
            let row = Row { label, cu, jit_ns: (ji - jb) / N as f64, interp_ns: (ii - ib) / N as f64 };
            eprintln!(
                "{:40} cu {:5.2} jit {:7.3} ns interp {:8.3} ns  slow_paths {} translations {} static_loads {} flush_stores {} spills {}",
                row.label, row.cu, row.jit_ns, row.interp_ns, stats.intrinsic_slow_paths, stats.translations,
                stats.static_loads, stats.flush_stores, stats.spills
            );
            rows.push(row);
        }
    }
    println!("{}", report(&rows, ok, 4.0));
}
