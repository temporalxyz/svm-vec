# sBPF programs with 512-bit vector ops

Programs here are plain `no_std` Rust built for `bpfel-unknown-none` with a stock nightly
compiler. They call `core::arch`-style intrinsics from `svm-intrinsics`
(`_svm512_add_epi32(a, b)`, ...). The forked sbpf-linker in `../sbpf-linker` allocates the
vector registers and emits the vector ops, producing an sBPF v3 `.so`. See `../SPEC.md` for the
ops and their encoding.

## Prerequisites

- `rustup toolchain install nightly --component rust-src` (nightly ≥ 2026-08-05, LLVM 23).
- The forked linker, built once with
  `cargo build --release --manifest-path ../sbpf-linker/Cargo.toml`. `build.sh` does this for
  you. It loads LLVM at runtime from the rustc that invokes it, so it always matches
  `cargo +nightly`.

## Building

```sh
./build.sh                       # builds toolchain-tests/
./build.sh path/to/program       # any crate or workspace under programs/
```

or, inside a crate or workspace under `programs/`:

```sh
cargo +nightly build-sbpf
```

`build-sbpf` is an alias from `.cargo/config.toml` for
`build --release --target bpfel-unknown-none -Zbuild-std=core`. The same file sets the linker
and the target flags (`target-cpu=v4`, `--arch=v3`, 4 KiB stack, ...) for `bpfel-unknown-none`
only, so host crates under `programs/` (build scripts, host tools) are unaffected. The program
ends up at `target/bpfel-unknown-none/release/<crate>.so`. The linker also copies it to
`target/deploy/<crate>.so`, relative to the directory cargo runs in.

Cargo does not notice when the linker binary changes. After rebuilding sbpf-linker, run
`cargo clean` (or touch the program's sources) to relink.

Expect two harmless warnings: `allows-misaligned-mem-access` is an unstable target feature, and
`solana-compiler-builtins` is not linked. The test kernels need no libcalls. Add that crate if
yours do (for example, large `memcpy`s or 128-bit division).

## Writing a program

`Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
svm-intrinsics = { path = "../../svm-intrinsics" }   # adjust the paths
svm-rt = { path = "../svm-rt" }                       # panic handler

[profile.release]
opt-level = 3
codegen-units = 1
panic = "abort"
```

`src/lib.rs`:

```rust
#![no_std]
use svm_intrinsics::*;
use svm_rt as _; // panic handler (aborts through the `abort` syscall)

/// r1 = the input region (writable). Write outputs back into it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let a = _svm512_loadu_si512(input);
        let b = _svm512_loadu_si512(input.add(64));
        _svm512_storeu_si512(input.add(128), _svm512_add_epi32(a, b));
    }
    0
}
```

There is no `solana-program` dependency. The entrypoint gets the raw input region in `r1`, and
the return value is the program's exit code.

### Rules for vector values

A `__svm512i` is an opaque SSA token, not data. The linker's vector pass rejects, with an
error naming the function:

- storing a vector to memory, e.g. in an array or struct the optimizer cannot scalarize. Small
  arrays with constant indices are fine after unrolling.
- passing a vector to, or returning one from, a function that is not inlined. Mark helpers
  `#[inline(always)]`.
- an imm8 (`::<IMM>`) that is not a constant, or is out of range. Out-of-range immediates are
  also caught at compile time by the intrinsics.
- a vector live across a call to a non-inlined function that uses every remaining register, or
  across an indirect call.
- more than 30 vectors live at once when the excess cannot be spilled. Values whose remaining
  uses are all in the same basic block are spilled to 64-byte stack slots automatically, so
  straight-line overflow works. Values live across a loop or branch are not spilled.

`if`/`match` and loop-carried vectors are fine: phis become `mov`s only where the allocator
cannot give both sides the same register, and selects become a conditional `mov`.

## Tests

`toolchain-tests/` is a workspace of small kernels, one `.so` each:

| kernel | exercises |
|---|---|
| `vadd` | straight-line ops, `st`, `hsum` |
| `dot` | loop-carried `dpbusd` accumulator (no moves) |
| `nested` | nested loops with inner and outer accumulators |
| `branchy` | if/else merges inside a loop, a vector `select` |
| `misc` | an accumulator still live afterwards (`mov` inserted), imm8 ops, insert/extract, clmul |
| `pressure` | exactly 30 live vectors (`v0`–`v29`, no spills) |
| `spill` | 31 live vectors (one spill and reload through the stack) |
| `offsets` | constant displacements in `off` (negative, up to 32767, and one too far to fold) |
| `calls` | a vector live across a call to a non-inlined vector function |
| `all_ops` | every intrinsic once |

```sh
./build.sh && python3 toolchain-tests/check.py           # static checks
python3 toolchain-tests/check.py --list dot              # annotated disassembly (or a .so path)
python3 toolchain-tests/check.py --svm-run ../harness/target/release/svm-run   # end to end
```

`check.py` decodes each `.so` and checks that no `__svm_` symbol remains, that every `src=2` op
is well formed, and that each kernel has the expected ops. With `--svm-run`, it runs each kernel on generated
inputs and compares the input region afterwards with values computed in Python.

The pass's own unit and IR-level tests are in `../sbpf-linker`: `cargo test --lib vector`.
