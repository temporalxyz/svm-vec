//! `svm-run`: runs an SBPF v3 program with vector ops enabled in the interpreter and the JIT,
//! checks that both engines agree and reports the result as JSON on stdout.
//!
//! The program gets one writable input region at `MM_INPUT_START`, initialized from a file, and
//! is entered with `r1 = MM_INPUT_START`. Every combination of `--entrypoint` and `--input` is one
//! case. Diagnostics go to stderr, the exit code is 1 if anything disagrees or a program fails.

use sha2::{Digest, Sha256};
use solana_sbpf::{
    aligned_memory::AlignedMemory,
    declare_builtin_function, ebpf,
    elf::Executable,
    memory_region::{AccessType, MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, SBPFVersion},
    verifier::RequisiteVerifier,
    vm::{CallFrame, Config, ContextObject, EbpfVm, ExecutionMode},
};
use std::{error::Error, ptr::NonNull, sync::Arc, time::Instant};

const USAGE: &str =
    "usage: svm-run PROGRAM.so [--input FILE]... [--entrypoint SYMBOL|0xADDR|pc:N]...
               [--runs N] [--repeat N] [--budget CU] [--heap BYTES] [--expect FILE]... [--hex] [--allow-error]

  --input FILE        initial contents of the writable input region (default: empty)
  --entrypoint E      function to enter instead of the ELF entry point: a symbol name (needs a
                      symbol table), a virtual address or an instruction index (default: ELF entry)
  --runs N            timed runs per engine, the median is reported (default: 10)
  --repeat N          time N back-to-back executions per run on the same memory mapping, with
                      a fresh VM each, and report the mean (default: 1, a single execution)
  --budget CU         compute unit budget (default: 1400000)
  --heap BYTES        size of the zeroed heap region at MM_HEAP_START (default: 32768)
  --expect FILE       expected final input region, one per case or one for all cases
  --hex               include the final input region as hex
  --allow-error       do not fail if the program returns an error (both engines must still agree)

Every combination of entrypoint and input is a case. With exactly two cases a comparison of
them is added. The exit code is 1 if the engines disagree, a run is not deterministic, an
expectation does not hold or (without --allow-error) a program fails.";

/// Execution context: compute meter, memory mapping and the program log
struct Context {
    remaining: u64,
    memory_mapping: MemoryMapping,
    logs: Vec<String>,
}

impl ContextObject for Context {
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }

    fn get_remaining(&self) -> u64 {
        self.remaining
    }

    fn active_mapping_ptr(&mut self) -> NonNull<MemoryMapping> {
        NonNull::from_mut(&mut self.memory_mapping)
    }
}

impl Context {
    fn slice(
        &mut self,
        access: AccessType,
        vm_addr: u64,
        len: u64,
    ) -> Result<&mut [u8], Box<dyn Error>> {
        if len == 0 {
            return Ok(&mut []);
        }
        let buffer = Result::from(self.memory_mapping.map(access, vm_addr, len))?;
        let ptr = match buffer {
            solana_sbpf::memory_region::HostBuffer::Immutable(ptr) => ptr.cast_mut(),
            solana_sbpf::memory_region::HostBuffer::Mutable(ptr) => ptr,
        };
        // SAFETY: the mapping checked the bounds and the regions outlive the execution
        Ok(unsafe { &mut *ptr })
    }
}

declare_builtin_function!(
    /// `abort()`
    SyscallAbort,
    fn rust(
        _context: &mut Context,
        _a: u64,
        _b: u64,
        _c: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        Err("program aborted".into())
    }
);

declare_builtin_function!(
    /// `sol_panic_(file, len, line, column)`
    SyscallPanic,
    fn rust(
        context: &mut Context,
        file: u64,
        len: u64,
        line: u64,
        column: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        let file = String::from_utf8_lossy(context.slice(AccessType::Load, file, len)?).into_owned();
        Err(format!("program panicked at {file}:{line}:{column}").into())
    }
);

declare_builtin_function!(
    /// `sol_log_(message, len)`
    SyscallLog,
    fn rust(
        context: &mut Context,
        message: u64,
        len: u64,
        _c: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        let message = String::from_utf8_lossy(context.slice(AccessType::Load, message, len)?).into_owned();
        context.logs.push(message);
        Ok(0)
    }
);

declare_builtin_function!(
    /// `sol_log_64_(a, b, c, d, e)`
    SyscallLog64,
    fn rust(
        context: &mut Context,
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        context.logs.push(format!("{a:#x}, {b:#x}, {c:#x}, {d:#x}, {e:#x}"));
        Ok(0)
    }
);

declare_builtin_function!(
    /// `sol_memcpy_(dst, src, n)`, the ranges must not overlap
    SyscallMemcpy,
    fn rust(
        context: &mut Context,
        dst: u64,
        src: u64,
        n: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        if dst.saturating_sub(src) < n || src.saturating_sub(dst) < n {
            return Err("sol_memcpy_: overlapping ranges".into());
        }
        let src = context.slice(AccessType::Load, src, n)?.to_vec();
        context.slice(AccessType::Store, dst, n)?.copy_from_slice(&src);
        Ok(0)
    }
);

declare_builtin_function!(
    /// `sol_memmove_(dst, src, n)`
    SyscallMemmove,
    fn rust(
        context: &mut Context,
        dst: u64,
        src: u64,
        n: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        let src = context.slice(AccessType::Load, src, n)?.to_vec();
        context.slice(AccessType::Store, dst, n)?.copy_from_slice(&src);
        Ok(0)
    }
);

declare_builtin_function!(
    /// `sol_memset_(dst, c, n)`
    SyscallMemset,
    fn rust(
        context: &mut Context,
        dst: u64,
        c: u64,
        n: u64,
        _d: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        context.slice(AccessType::Store, dst, n)?.fill(c as u8);
        Ok(0)
    }
);

declare_builtin_function!(
    /// `sol_memcmp_(a, b, n, result)`, stores the difference of the first unequal bytes as `i32`
    SyscallMemcmp,
    fn rust(
        context: &mut Context,
        a: u64,
        b: u64,
        n: u64,
        result: u64,
        _e: u64,
    ) -> Result<u64, Box<dyn Error>> {
        let a = context.slice(AccessType::Load, a, n)?.to_vec();
        let b = context.slice(AccessType::Load, b, n)?.to_vec();
        let difference = a
            .iter()
            .zip(b.iter())
            .find(|(a, b)| a != b)
            .map_or(0, |(a, b)| *a as i32 - *b as i32);
        context
            .slice(AccessType::Store, result, 4)?
            .copy_from_slice(&difference.to_le_bytes());
        Ok(0)
    }
);

fn loader(enable_symbol_and_section_labels: bool) -> Arc<BuiltinProgram<Context>> {
    let mut loader = BuiltinProgram::new_loader(Config {
        enable_vector_ops: true,
        aligned_memory_mapping: true,
        enable_symbol_and_section_labels,
        enabled_sbpf_versions: SBPFVersion::V3..=SBPFVersion::V3,
        ..Config::default()
    });
    loader.register_definition::<SyscallAbort>("abort").unwrap();
    loader
        .register_definition::<SyscallPanic>("sol_panic_")
        .unwrap();
    loader
        .register_definition::<SyscallLog>("sol_log_")
        .unwrap();
    loader
        .register_definition::<SyscallLog64>("sol_log_64_")
        .unwrap();
    loader
        .register_definition::<SyscallMemcpy>("sol_memcpy_")
        .unwrap();
    loader
        .register_definition::<SyscallMemmove>("sol_memmove_")
        .unwrap();
    loader
        .register_definition::<SyscallMemset>("sol_memset_")
        .unwrap();
    loader
        .register_definition::<SyscallMemcmp>("sol_memcmp_")
        .unwrap();
    Arc::new(loader)
}

/// Loads, verifies and JIT compiles the program, entered at `entrypoint` if given
fn load(elf: &[u8], entrypoint: Option<&str>) -> Result<(Executable<Context>, u64), String> {
    let mut elf = elf.to_vec();
    if let Some(entrypoint) = entrypoint {
        let executable =
            Executable::from_elf(&elf, loader(true)).map_err(|err| format!("loading: {err}"))?;
        let text_vaddr = executable.get_text_bytes().0;
        let pc = if let Some(address) = entrypoint.strip_prefix("0x") {
            let address = u64::from_str_radix(address, 16)
                .map_err(|err| format!("entrypoint {entrypoint}: {err}"))?;
            address.wrapping_sub(text_vaddr) / ebpf::INSN_SIZE as u64
        } else if let Some(pc) = entrypoint.strip_prefix("pc:") {
            pc.parse()
                .map_err(|err| format!("entrypoint {entrypoint}: {err}"))?
        } else {
            executable
                .get_function_registry()
                .lookup_by_name(entrypoint.as_bytes())
                .ok_or_else(|| format!("entrypoint {entrypoint}: no such symbol (stripped ELF? use 0xADDR or pc:N)"))?
                .1 as u64
        };
        let e_entry = text_vaddr.wrapping_add(pc.wrapping_mul(ebpf::INSN_SIZE as u64));
        elf.get_mut(24..32)
            .ok_or("ELF too short")?
            .copy_from_slice(&e_entry.to_le_bytes());
    }
    let executable =
        Executable::from_elf(&elf, loader(false)).map_err(|err| format!("loading: {err}"))?;
    if executable.get_sbpf_version() != SBPFVersion::V3 {
        return Err(format!(
            "expected an SBPF v3 ELF, got {:?}",
            executable.get_sbpf_version()
        ));
    }
    executable
        .verify::<RequisiteVerifier>()
        .map_err(|err| format!("verification: {err}"))?;
    let start = Instant::now();
    executable
        .jit_compile()
        .map_err(|err| format!("JIT compilation: {err}"))?;
    Ok((executable, start.elapsed().as_nanos() as u64))
}

/// Everything observable about one execution
#[derive(Clone, PartialEq, Eq)]
struct Outcome {
    result: Result<u64, String>,
    compute_units: u64,
    output: Vec<u8>,
    logs: Vec<String>,
}

/// Executes the program once on a fresh copy of the input, returns the outcome and wall time
fn execute(
    executable: &Executable<Context>,
    input: &[u8],
    options: &Options,
    jit: bool,
) -> (Outcome, u64) {
    let (budget, heap_size) = (options.budget, options.heap_size);
    let config = executable.get_config();
    let mut input = input.to_vec();
    let mut stack = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(config.stack_size());
    let mut heap = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(heap_size);
    let regions = vec![
        executable.get_ro_region(),
        MemoryRegion::new_gapped(
            &raw mut *stack.as_slice_mut(),
            ebpf::MM_STACK_START,
            if executable.get_sbpf_version().stack_frame_gaps() && config.enable_stack_frame_gaps {
                config.stack_frame_size as u64
            } else {
                0
            },
        ),
        MemoryRegion::new(&raw mut *heap.as_slice_mut(), ebpf::MM_HEAP_START),
        MemoryRegion::new(&raw mut input[..], ebpf::MM_INPUT_START),
    ];
    let mut context = Context {
        remaining: budget,
        // SAFETY: the regions outlive the mapping
        memory_mapping: unsafe {
            MemoryMapping::new(regions, config, executable.get_sbpf_version())
        }
        .unwrap(),
        logs: vec![],
    };
    let mut call_frames = vec![CallFrame::default(); config.max_call_depth];
    let (compute_units, result, elapsed, output) = {
        let mut vm = EbpfVm::new(
            executable.get_loader().clone(),
            executable.get_sbpf_version(),
            &mut context,
            stack.len(),
        );
        vm.registers[1] = ebpf::MM_INPUT_START;
        let mut mode = if jit {
            ExecutionMode::Jit
        } else {
            ExecutionMode::Interpreted
        };
        let start = Instant::now();
        let (compute_units, result) = vm.execute_program(executable, &mut mode, &mut call_frames);
        let mut elapsed = start.elapsed().as_nanos() as u64;
        drop(vm);
        // Outcome of the first execution. Repeated ones see the input it left, only their time counts
        let (output, log_count) = (input.clone(), context.logs.len());
        if options.repeat > 1 {
            let start = Instant::now();
            for _ in 0..options.repeat {
                context.remaining = budget;
                let mut vm = EbpfVm::new(
                    executable.get_loader().clone(),
                    executable.get_sbpf_version(),
                    &mut context,
                    stack.len(),
                );
                vm.registers[1] = ebpf::MM_INPUT_START;
                std::hint::black_box(vm.execute_program(executable, &mut mode, &mut call_frames));
            }
            elapsed = start.elapsed().as_nanos() as u64 / options.repeat as u64;
        }
        context.logs.truncate(log_count);
        (
            compute_units,
            Result::from(result).map_err(|err| format!("{err:?}")),
            elapsed,
            output,
        )
    };
    drop(context.memory_mapping);
    let outcome = Outcome {
        result,
        compute_units,
        output,
        logs: context.logs,
    };
    (outcome, elapsed)
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Options {
    program: String,
    inputs: Vec<String>,
    entrypoints: Vec<String>,
    expects: Vec<String>,
    runs: usize,
    repeat: u32,
    budget: u64,
    heap_size: usize,
    hex: bool,
    allow_error: bool,
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        program: String::new(),
        inputs: vec![],
        entrypoints: vec![],
        expects: vec![],
        runs: 10,
        repeat: 1,
        budget: 1_400_000,
        heap_size: 32 * 1024,
        hex: false,
        allow_error: false,
    };
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--input" => options.inputs.push(value()?),
            "--entrypoint" => options.entrypoints.push(value()?),
            "--expect" => options.expects.push(value()?),
            "--runs" => options.runs = value()?.parse().map_err(|err| format!("--runs: {err}"))?,
            "--repeat" => {
                options.repeat = value()?.parse().map_err(|err| format!("--repeat: {err}"))?
            }
            "--budget" => {
                options.budget = value()?.parse().map_err(|err| format!("--budget: {err}"))?
            }
            "--heap" => {
                options.heap_size = value()?.parse().map_err(|err| format!("--heap: {err}"))?
            }
            "--hex" => options.hex = true,
            "--allow-error" => options.allow_error = true,
            "-h" | "--help" => return Err(String::new()),
            _ if arg.starts_with('-') || !options.program.is_empty() => {
                return Err(format!("unexpected argument {arg}"))
            }
            _ => options.program = arg,
        }
    }
    if options.program.is_empty() {
        return Err("missing PROGRAM".into());
    }
    if options.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    Ok(options)
}

fn main() {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("svm-run: {message}");
            }
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(message) = run(&options) {
        eprintln!("svm-run: {message}");
        std::process::exit(1);
    }
}

fn run(options: &Options) -> Result<(), String> {
    let read = |path: &str| std::fs::read(path).map_err(|err| format!("{path}: {err}"));
    let elf = read(&options.program)?;
    let entrypoints: Vec<Option<&str>> = if options.entrypoints.is_empty() {
        vec![None]
    } else {
        options
            .entrypoints
            .iter()
            .map(|e| Some(e.as_str()))
            .collect()
    };
    let inputs: Vec<Option<&str>> = if options.inputs.is_empty() {
        vec![None]
    } else {
        options.inputs.iter().map(|i| Some(i.as_str())).collect()
    };
    let case_count = entrypoints.len() * inputs.len();
    if !options.expects.is_empty()
        && options.expects.len() != 1
        && options.expects.len() != case_count
    {
        return Err(format!(
            "{} --expect files for {case_count} cases",
            options.expects.len()
        ));
    }

    let mut failures = vec![];
    let mut cases = vec![];
    let mut outcomes = vec![];
    for entrypoint in &entrypoints {
        let (executable, compile_ns) = load(&elf, *entrypoint)?;
        for input_path in &inputs {
            let input = input_path.map(read).transpose()?.unwrap_or_default();
            let name = format!(
                "{} on {}",
                entrypoint.unwrap_or("entrypoint"),
                input_path.unwrap_or("<empty>")
            );
            let mut times = [vec![], vec![]];
            let mut first: Option<Outcome> = None;
            for (engine, jit) in [("interpreter", false), ("jit", true)] {
                for run in 0..options.runs {
                    let (outcome, elapsed) = execute(&executable, &input, options, jit);
                    times[jit as usize].push(elapsed);
                    match &first {
                        None => first = Some(outcome),
                        Some(expected) if *expected != outcome => {
                            let what = if run == 0 {
                                "interpreter and JIT disagree"
                            } else {
                                "nondeterministic"
                            };
                            report_difference(&name, engine, expected, &outcome);
                            return Err(format!("{name}: {what} ({engine} run {run})"));
                        }
                        Some(_) => {}
                    }
                }
            }
            let outcome = first.unwrap();
            let case_index = cases.len();
            let mut case = serde_json::json!({
                "entrypoint": entrypoint.unwrap_or("entrypoint"),
                "input": input_path,
                "r0": outcome.result.as_ref().ok(),
                "error": outcome.result.as_ref().err(),
                "cu": outcome.compute_units,
                "output_len": outcome.output.len(),
                "output_sha256": hex(&Sha256::digest(&outcome.output)),
                "logs": outcome.logs,
                "runs": options.runs,
                "repeat": options.repeat,
                "interpreter_ns": median(times[0].clone()),
                "jit_ns": median(times[1].clone()),
                "jit_compile_ns": compile_ns,
            });
            if options.hex {
                case["output_hex"] = hex(&outcome.output).into();
            }
            if let (Err(err), false) = (&outcome.result, options.allow_error) {
                failures.push(format!("{name}: program failed: {err}"));
            }
            if let Some(expect) = options.expects.get(if options.expects.len() == 1 {
                0
            } else {
                case_index
            }) {
                let expected = read(expect)?;
                let matches = expected == outcome.output;
                case["expect"] = expect.as_str().into();
                case["expect_matches"] = matches.into();
                if !matches {
                    let first_difference = expected
                        .iter()
                        .zip(&outcome.output)
                        .position(|(a, b)| a != b);
                    failures.push(format!(
                        "{name}: final input differs from {expect} (lengths {} vs {}, first difference at {:?})",
                        outcome.output.len(),
                        expected.len(),
                        first_difference
                    ));
                }
            }
            cases.push(case);
            outcomes.push(outcome);
        }
    }

    let mut report = serde_json::json!({ "program": options.program, "cases": cases });
    if let ([a, b], [ca, cb]) = (&outcomes[..], &cases[..]) {
        let ratio =
            |key: &str| ca[key].as_u64().unwrap() as f64 / cb[key].as_u64().unwrap().max(1) as f64;
        report["comparison"] = serde_json::json!({
            "r0_equal": a.result == b.result,
            "output_equal": a.output == b.output,
            "cu_ratio": ratio("cu"),
            "interpreter_time_ratio": ratio("interpreter_ns"),
            "jit_time_ratio": ratio("jit_ns"),
        });
    }
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

fn report_difference(name: &str, engine: &str, expected: &Outcome, actual: &Outcome) {
    eprintln!("svm-run: {name}: {engine} differs from the first interpreter run");
    if expected.result != actual.result {
        eprintln!("  result: {:?} vs {:?}", expected.result, actual.result);
    }
    if expected.compute_units != actual.compute_units {
        eprintln!(
            "  cu: {} vs {}",
            expected.compute_units, actual.compute_units
        );
    }
    if expected.output != actual.output {
        let position = expected
            .output
            .iter()
            .zip(&actual.output)
            .position(|(a, b)| a != b);
        eprintln!("  final input differs, first at byte {position:?}");
    }
    if expected.logs != actual.logs {
        eprintln!("  logs: {:?} vs {:?}", expected.logs, actual.logs);
    }
}
