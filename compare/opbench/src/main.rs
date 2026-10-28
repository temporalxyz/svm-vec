//! Cost per op of our vector ops (see spec.rs for what each program computes). Generates the
//! programs as raw bytecode, checks every one against the host reference on three inputs in the
//! interpreter and the JIT, then times program and baseline the way svm-run --repeat does (fresh
//! VM per run on one memory mapping, batches of about 2 ms, median of 9).
//!
//!     opbench [filter]   -> one JSON line {"ops": [...], "ok": bool}

#[allow(dead_code)]
mod spec;

use solana_sbpf::{
    aligned_memory::AlignedMemory,
    ebpf,
    elf::Executable,
    error::ProgramResult,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    vector::{encode, encode_with_offset, op},
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ContextObject, EbpfVm, ExecutionMode},
};
use spec::*;
use std::{ptr::NonNull, sync::Arc, time::Instant};

struct Ctx {
    remaining: u64,
    mapping: MemoryMapping,
}
impl ContextObject for Ctx {
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }
    fn get_remaining(&self) -> u64 {
        self.remaining
    }
    fn active_mapping_ptr(&mut self) -> NonNull<MemoryMapping> {
        NonNull::from_mut(&mut self.mapping)
    }
}

fn insn(opc: u8, dst: u8, src: u8, off: i16, imm: i32) -> u64 {
    opc as u64 | ((dst | src << 4) as u64) << 8 | (off as u16 as u64) << 16 | (imm as u32 as u64) << 32
}
fn mov(dst: u8, src: u8) -> u64 {
    insn(ebpf::MOV64_REG, dst, src, 0, 0)
}
fn movi(dst: u8, imm: i32) -> u64 {
    insn(ebpf::MOV64_IMM, dst, 0, 0, imm)
}
fn lddw(dst: u8, imm: u64) -> [u64; 2] {
    [insn(ebpf::LD_DW_IMM, dst, 0, 0, imm as u32 as i32), (imm >> 32) << 32]
}
fn vop(o: u8, imm8: u8, out: usize, in0: usize, in1: usize) -> u64 {
    encode(o, imm8, out as u8, in0 as u8, in1 as u8)
}
fn vmem(o: u8, reg: usize, off: usize) -> u64 {
    if o == op::ST {
        encode_with_offset(o, 0, 0, reg as u8, 0, off as i16)
    } else {
        encode_with_offset(o, 0, reg as u8, 0, 0, off as i16)
    }
}

/// How an op's fields map to the chain vector c and the sources x = v1, y = v2
enum Shape {
    /// out = c, in0 = c, in1 = x
    R3(u8, u8),
    /// out = c, in0 = x, in1 = y (accumulators and ternlog)
    A3(u8, u8),
    /// out = c, in0 = c
    R2(u8, u8),
    /// out = c, in0 = x
    Mov,
    /// out = c
    Out(u8),
    /// r0 = op(c)
    Sr(u8, u8),
    Custom,
}

fn shape(name: &str) -> Option<Shape> {
    use Shape::*;
    let base = name.split('=').next().unwrap();
    Some(match base {
        "add.b" => R3(op::ADD_B, 0),
        "add.w" => R3(op::ADD_W, 0),
        "add.d" => R3(op::ADD_D, 0),
        "add.q" => R3(op::ADD_Q, 0),
        "sub.d" => R3(op::SUB_D, 0),
        "sub.q" => R3(op::SUB_Q, 0),
        "mullo.w" => R3(op::MULLO_W, 0),
        "mullo.d" => R3(op::MULLO_D, 0),
        "mullo.q" => R3(op::MULLO_Q, 0),
        "mulwu" => R3(op::MULWU, 0),
        "mulws" => R3(op::MULWS, 0),
        "mulhi.w" => R3(op::MULHI_W, 0),
        "mulhiu.w" => R3(op::MULHIU_W, 0),
        "max.d" => R3(op::MAX_D, 0),
        "max.w" => R3(op::MAX_W, 0),
        "minu.d" => R3(op::MINU_D, 0),
        "minu.w" => R3(op::MINU_W, 0),
        "minu.q" => R3(op::MINU_Q, 0),
        "and" => R3(op::AND, 0),
        "or" => R3(op::OR, 0),
        "xor" => R3(op::XOR, 0),
        "andn" => R3(op::ANDN, 0),
        "ternlog" => A3(op::TERNLOG, 0x96),
        "shli.w" => R2(op::SHLI_W, 3),
        "shli.d" => R2(op::SHLI_D, 7),
        "shri.q" => R2(op::SHRI_Q, 13),
        "sari.q" => R2(op::SARI_Q, 13),
        "roli.d" => R2(op::ROLI_D, 7),
        "roli.q" => R2(op::ROLI_Q, 13),
        "shlv.d" => R3(op::SHLV_D, 0),
        "shrv.q" => R3(op::SHRV_Q, 0),
        "rolv.d" => R3(op::ROLV_D, 0),
        "rolv.q" => R3(op::ROLV_Q, 0),
        "dpbusd" => A3(op::DPBUSD, 0),
        "dpwssd" => A3(op::DPWSSD, 0),
        "maddubs" => R3(op::MADDUBS, 0),
        "maddwd" => R3(op::MADDWD, 0),
        "madd52lo" => A3(op::MADD52LO, 0),
        "madd52hi" => A3(op::MADD52HI, 0),
        "packs.d" => R3(op::PACKS_D, 0),
        "packus.w" => R3(op::PACKUS_W, 0),
        "permd" => R3(op::PERMD, 0),
        "shufb" => R3(op::SHUFB, 0),
        "zx.b.d" => R2(op::ZX_B_D, 0),
        "sx.w.d" => R2(op::SX_W_D, 0),
        "narrow.d.b" => R2(op::NARROW_D_B, 0),
        "popcnt.b" => R2(op::POPCNT_B, 0),
        "popcnt.q" => R2(op::POPCNT_Q, 0),
        "clmul" => R3(op::CLMUL, 0x01),
        "mov" => Mov,
        "splat.d" => Out(op::SPLAT_D),
        "splat.q" => Out(op::SPLAT_Q),
        "zero" => Out(op::ZERO),
        "extract.q" => Sr(op::EXTRACT_Q, 5),
        "extract.d" => Sr(op::EXTRACT_D, 11),
        "movemask.b" => Sr(op::MOVEMASK_B, 0),
        "hsum.d" => Sr(op::HSUM_D, 0),
        "hsum.q" => Sr(op::HSUM_Q, 0),
        "cmpeq.b" => R3(op::CMPEQ_B, 0),
        "cmpeq.d" => R3(op::CMPEQ_D, 0),
        "cmpeq.q" => R3(op::CMPEQ_Q, 0),
        "cmpgt.d" => R3(op::CMPGT_D, 0),
        "cmpgt.q" => R3(op::CMPGT_Q, 0),
        "cmpgtu.q" => R3(op::CMPGTU_Q, 0),
        "perm2.w" => A3(op::PERM2_W, 0),
        "permq" => R3(op::PERMQ, 0),
        "narrowus.d.b" => R2(op::NARROWUS_D_B, 0),
        "narrows.d.b" => R2(op::NARROWS_D_B, 0),
        "narrows.d.w" => R2(op::NARROWS_D_W, 0),
        "insert.q" | "insert.q+extract.q" | "ld" | "st" | "ld*16" | "st*16" | "ldbc.d" | "ldbc.q" => {
            Custom
        }
        _ => return None,
    })
}

/// Instance j of the loop body
fn instance(name: &str, dep: bool, j: usize) -> Vec<u64> {
    let c = chain(dep, j);
    let base = name.split('=').next().unwrap();
    match shape(name).unwrap() {
        Shape::R3(o, i) => vec![vop(o, i, c, c, 1)],
        Shape::A3(o, i) => vec![vop(o, i, c, 1, 2)],
        Shape::R2(o, i) => vec![vop(o, i, c, c, 0)],
        Shape::Mov => vec![vop(op::MOV, 0, c, 1, 0)],
        Shape::Out(o) => vec![vop(o, 0, c, 0, 0)],
        Shape::Sr(o, i) => vec![vop(o, i, 0, c, 0)],
        Shape::Custom => match base {
            "insert.q" => vec![vop(op::INSERT_Q, (j % 8) as u8, c, 0, 0)],
            "insert.q+extract.q" => vec![
                mov(1, 0),
                vop(op::INSERT_Q, (j % 8) as u8, 0, 0, 0),
                vop(op::EXTRACT_Q, ((j + 1) % 8) as u8, 0, 0, 0),
            ],
            "ld" => vec![vmem(op::LD, c, mem_off(j))],
            "st" => vec![vmem(op::ST, c, mem_off(j))],
            "ld*16" => (0..16).map(|k| vmem(op::LD, k, bulk_off(j) + 64 * k)).collect(),
            "st*16" => (0..16).map(|k| vmem(op::ST, k, bulk_off(j) + 64 * k)).collect(),
            "ldbc.d" => vec![vmem(op::LDBC_D, c, bc_off(j))],
            "ldbc.q" => vec![vmem(op::LDBC_Q, c, bc_off(j))],
            _ => unreachable!(),
        },
    }
}

/// Whole program with k instances per body and l iterations; `bench` None is the baseline (empty
/// loop body)
fn program(bench: Option<(&str, bool)>, k: usize, l: usize) -> Vec<u8> {
    let mut w = vec![mov(6, 1), movi(0, 0)];
    w.extend((0..NV).map(|k| vmem(op::LD, k, 64 * k)));
    let mut body = vec![];
    if let Some((name, dep)) = bench {
        let base = name.split('=').next().unwrap();
        if matches!(base, "splat.d" | "splat.q" | "insert.q") {
            w.extend(lddw(1, S));
        }
        for j in 0..k {
            body.extend(instance(name, dep, j));
        }
    }
    w.push(movi(8, l as i32));
    let len = body.len() as i32;
    w.extend(body);
    w.push(insn(ebpf::ADD64_IMM, 8, 0, 0, -1));
    w.push(insn(ebpf::JNE64_IMM, 8, 0, (-(len + 2)) as i16, 0));
    w.extend([mov(7, 0), mov(1, 6)]);
    w.extend((0..NV).map(|k| vmem(op::ST, k, 64 * k)));
    w.extend([mov(0, 7), insn(ebpf::EXIT, 0, 0, 0, 0)]);
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Runner {
    exe: Executable<Ctx>,
    input: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    _stack: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    _heap: AlignedMemory<{ ebpf::HOST_ALIGN }>,
    ctx: Box<Ctx>,
    frames: Vec<CallFrame>,
    stack_len: usize,
}

impl Runner {
    fn new(text: &[u8], input: &[u8]) -> Runner {
        let config = Config {
            enable_vector_ops: true,
            aligned_memory_mapping: true,
            enabled_sbpf_versions: SBPFVersion::V3..=SBPFVersion::V3,
            ..Config::default()
        };
        let loader = Arc::new(BuiltinProgram::new_loader(config));
        let exe = Executable::from_text_bytes(text, loader, SBPFVersion::V3, FunctionRegistry::default()).unwrap();
        exe.verify::<RequisiteVerifier>().unwrap();
        exe.jit_compile().unwrap();
        let config = exe.get_config().clone();
        let mut input = AlignedMemory::from_slice(input);
        let mut stack = AlignedMemory::zero_filled(config.stack_size());
        let mut heap = AlignedMemory::zero_filled(32 * 1024);
        let gap = if exe.get_sbpf_version().stack_frame_gaps() && config.enable_stack_frame_gaps {
            config.stack_frame_size as u64
        } else {
            0
        };
        let regions = vec![
            exe.get_ro_region(),
            MemoryRegion::new_gapped(&raw mut *stack.as_slice_mut(), ebpf::MM_STACK_START, gap),
            MemoryRegion::new(&raw mut *heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new(&raw mut *input.as_slice_mut(), ebpf::MM_INPUT_START),
        ];
        let mapping = unsafe { MemoryMapping::new(regions, &config, exe.get_sbpf_version()) }.unwrap();
        Runner {
            stack_len: stack.len(),
            exe,
            input,
            _stack: stack,
            _heap: heap,
            ctx: Box::new(Ctx { remaining: 0, mapping }),
            frames: vec![CallFrame::default(); config.max_call_depth],
        }
    }

    fn run(&mut self, jit: bool) -> (u64, ProgramResult) {
        self.ctx.remaining = 1 << 40;
        let mut vm = EbpfVm::new(self.exe.get_loader().clone(), self.exe.get_sbpf_version(), &mut *self.ctx, self.stack_len);
        vm.registers[1] = ebpf::MM_INPUT_START;
        let mut mode = if jit { ExecutionMode::Jit } else { ExecutionMode::Interpreted };
        vm.execute_program(&self.exe, &mut mode, &mut self.frames)
    }

    fn batch(&mut self, jit: bool, repeat: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..repeat {
            std::hint::black_box(self.run(jit));
        }
        start.elapsed().as_nanos() as f64 / repeat as f64
    }
}

/// Median batch means of the program and the baseline, alternating
fn time(p: &mut Runner, b: &mut Runner, jit: bool) -> (f64, f64) {
    let single = (0..3).map(|_| p.batch(jit, 1)).fold(f64::MAX, f64::min);
    let (rp, rb) = (repeat_for(single), repeat_for((0..3).map(|_| b.batch(jit, 1)).fold(f64::MAX, f64::min)));
    let (mut tp, mut tb) = (vec![], vec![]);
    for _ in 0..BATCHES {
        tb.push(b.batch(jit, rb));
        tp.push(p.batch(jit, rp));
    }
    (median(tp), median(tb))
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let base_text = program(None, K, L);
    let mut ok = true;
    let mut rows = vec![];
    for bench in benches() {
        if shape(bench.name).is_none() || !bench.name.contains(filter.as_str()) {
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
                    for jit in [false, true] {
                        let mut r = Runner::new(&text, &input);
                        let (cu, result) = r.run(jit);
                        let good = matches!(result, ProgramResult::Ok(r0) if r0 == want_r0) && r.input.as_slice() == &want[..];
                        if !good {
                            eprintln!("{label} ({k}x{l}): seed {seed} jit {jit}: mismatch ({result:?}, want r0 {want_r0:#x})");
                            ok = false;
                        }
                        if k == K {
                            cus.push(cu);
                        }
                    }
                }
            }
            let input = input(1);
            let (mut p, mut b) = (Runner::new(&text, &input), Runner::new(&base_text, &input));
            let cu = (p.run(false).0 - b.run(false).0) as f64 / N as f64;
            if cus.iter().any(|c| *c != cus[0]) {
                eprintln!("{label}: CU differ between runs {cus:?}");
                ok = false;
            }
            let (ji, jb) = time(&mut p, &mut b, true);
            let (ii, ib) = time(&mut p, &mut b, false);
            let row = Row { label, cu, jit_ns: (ji - jb) / N as f64, interp_ns: (ii - ib) / N as f64 };
            eprintln!("{:40} cu {:5.2} jit {:7.3} ns interp {:8.3} ns", row.label, row.cu, row.jit_ns, row.interp_ns);
            rows.push(row);
        }
    }
    println!("{}", report(&rows, ok, 4.0));
}
