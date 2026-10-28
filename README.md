# svm-vec

proof of concept: 512-bit vector ops for sbpf, written in stock rust, with no compiler changes.

```rust
let acc = _svm512_dpbusd_epi32(acc, x, w);   // no register numbers
```

becomes one 1-cu instruction, `dpbusd v0, v8, v9`, which the jit lowers to one `vpdpbusd zmm0, zmm8, zmm9`.

- ops are `CALL_IMM` with `src = 2`, register numbers in the imm (`SPEC.md`)
- 30 vector registers, hidden vm state, pinned to `zmm0`–`zmm29` by the jit
- sbpf-linker allocates the registers on whole-program llvm ir, after lto and before bpf codegen

## results

```sh
./bench.sh
```

zen 4 (epyc 9534):

| | scalar sbpf | vector ops | ratio |
|---|---|---|---|
| int8 mlp 256→128→16, cu | 490,048 | 2,108 | 232× |
| int8 mlp, jit | 361 µs | 1.6 µs | 225× |
| keccak-f[1600], cu / permutation | 11,824 | 988 | 12× |
| keccak-f[1600], jit / permutation | 8.9 µs | 0.32 µs | 28× |
| falcon-512 verify, cu | 182,764 | 20,770 | 8.8× |
| falcon-512 verify, jit | 294 µs | 11.7 µs | 25× |

- every output is checked against a host reference, on the interpreter and the jit. falcon: 64 signatures accepted, 128 tampered inputs rejected
- scalar falcon is blueshift's `solana-falcon512` program, timed by `compare/compare.py`. the other scalar baselines are our programs built without vector ops
- jit times are medians of 300 runs (mlp, keccak) and of 64 signatures (falcon)

against Dean's simd experiment (blueshift-gg/solana-sbpf@7941ce5, his avx-512 jit), timed the same way, `python3 compare/compare.py`:

| | Dean | ours | ours is |
|---|---:|---:|---:|
| falcon-512 verify, cu | 22,029 | 20,770 | 1.06× fewer |
| falcon-512 verify, jit | 12.8–12.9 µs | 11.6–12.3 µs | 1.04–1.11× faster |
| keccak-f, cu / permutation | 1,348 | 988 | 1.36× fewer |
| keccak-f, jit / permutation | 445 ns | 320–372 ns | 1.20–1.39× faster |

jit ranges are over separate invocations of the script.

the wider suite (`results/suite.md`), same method, Dean / ours:

| kernel | cu | jit | ours faster by |
|---|---:|---:|---:|
| sha-256, 16 blocks | 3,460 / 1,954 | 1.05 / 0.49 µs | 2.2× |
| blake3, 1 MiB | 1.88M / 1.07M | 756 / 218 µs | 3.5× (his kernel is 8-way, ours 16-way) |
| curve25519 field mul, 8 × 256 | 35,724 / 18,392 | 8.06 / 4.36 µs | 1.85× |
| sort 256 u32 | 1,761 / 1,031 | 501 / 273 ns | 1.84× |
| int8 mlp, unrolled | 1,696 / 1,257 | 1.84 / 1.46 µs | 1.26× |
| int8 matmul 32×256 · 256×128 | 50,479 / 25,482 | 16.6 / 15.4 µs | 1.08× |
| perps risk math, 256 positions | 821 / 869 | 0.57 / 0.79 µs | 0.72× (memory-bound, his bulk loads win) |

on the jit, every op of ours costs under 2.8 ns per cu, including loads and stores. details, method and caveats are in `results/`.

## layout

| dir | what |
|---|---|
| `SPEC.md` | ops, encoding, conventions, intrinsics abi |
| `sbpf/` | runtime: verifier, interpreter, avx-512 jit. fork of anza-xyz/sbpf, branch `vector-ops` |
| `sbpf-linker/` | register allocation and lowering pass. fork of blueshift-gg/sbpf-linker, branch `vector-pass` |
| `svm-intrinsics/` | `_svm512_*` intrinsics |
| `harness/` | `svm-run`: runs a `.so` on both engines, reports cu and time, fails if they disagree |
| `programs/` | build setup, `mlp`, `keccak`, `falcon`, toolchain test kernels |
| `compare/` | the comparison with Dean's simd experiment |
| `results/` | per-kernel results |

## size

- **sbpf:** +250 / −6 in 9 existing files. new `vector.rs` (883 lines), `vector_jit.rs` (1202), `x86_evex.rs` (180), `tests/vector.rs` (1226)
- **sbpf-linker:** +161 / −4 in 6 existing files. new `src/vector/` (~3,000 lines, ~700 of them tests)
- programs without vector ops take the old paths: the linker pass runs only if the inputs reference `__svm_*`, and every jit change is gated on the program using a vector register

## tests

```sh
(cd sbpf && cargo test)
(cd sbpf-linker && cargo test)
(cd programs && ./build.sh && python3 toolchain-tests/check.py --svm-run ../harness/target/release/svm-run)
```

## limitations

- interpreter and x86-64 jit only: no aarch64 jit, no firedancer
- gated by `Config::enable_vector_ops`. production needs a new sbpf version, or a feature gate plus proof that no deployed program contains a `src = 2` call (v3 accepts them at deploy today and faults when they run)
- vector values must stay in ssa. arrays of vectors indexed in non-unrolled loops are rejected. spilling only covers values whose remaining uses are in the same block
- cu is 1 per op, so multi-instruction ops (`hsum`, `insert`, compares) are underpriced
- the crates.io sbpf-assembler rejects `call` with `off`. the linker works around it; the fix belongs upstream
