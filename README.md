# svm-vec

proof of concept: 512-bit vector ops for sbpf, written in stock rust, with no compiler changes. 249 ops: the integer, bit, crypto (aes, gfni, clmul, sha-ni), gather/scatter and masked parts of avx-512 that exist on ice lake and zen 4.

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
| int8 mlp, jit | 361 µs | 0.91 µs | 396× |
| keccak-f[1600], cu / permutation | 11,824 | 988 | 12× |
| keccak-f[1600], jit / permutation | 8.9 µs | 0.26 µs | 34× |
| falcon-512 verify, cu | 182,764 | 20,770 | 8.8× |
| falcon-512 verify, jit | 294 µs | 10.8 µs | 27× |

- every output is checked against a host reference, on the interpreter and the jit. falcon: 64 signatures accepted, 128 tampered inputs rejected
- scalar falcon is blueshift's `solana-falcon512` program, timed by `compare/compare.py`. the other scalar baselines are our programs built without vector ops
- jit times are medians of 300 runs (mlp, keccak) and of 64 signatures (falcon)

against Dean's simd experiment (blueshift-gg/solana-sbpf@7941ce5, his avx-512 jit), timed the same way, `python3 compare/compare.py`:

| | Dean | ours | ours is |
|---|---:|---:|---:|
| falcon-512 verify, cu | 22,029 | 20,770 | 1.06× fewer |
| falcon-512 verify, jit | 12.8 µs | 10.6 µs | 1.21× faster |
| keccak-f, cu / permutation | 1,348 | 988 | 1.36× fewer |
| keccak-f, jit / permutation | 445 ns | 264–274 ns | 1.69× faster |

medians of 3 interleaved rounds.

the wider suite (`results/suite.md`), same method, Dean / ours:

| kernel | cu | jit | ours faster by |
|---|---:|---:|---:|
| sha-256, 16 blocks | 3,460 / 1,954 | 1.02 / 0.42 µs | 2.45× |
| blake3, 1 MiB | 1.88M / 1.07M | 751 / 179 µs | 4.2× (his kernel is 8-way, ours 16-way) |
| curve25519 field mul, 8 × 256 | 35,724 / 18,392 | 8.10 / 4.33 µs | 1.87× |
| sort 256 u32 | 1,761 / 1,031 | 503 / 235 ns | 2.14× |
| int8 mlp, unrolled | 1,696 / 1,257 | 1.61 / 0.48 µs | 3.33× |
| int8 matmul 32×256 · 256×128 | 50,479 / 25,482 | 15.9 / 5.14 µs | 3.10× |
| perps risk math, 256 positions | 821 / 869 | 563 / 324 ns | 1.74× |

on the jit every one of the 249 ops costs at most 1.7 ns per cu, loads, stores and gathers included (target: about 4). the jit checks bounds once per run of consecutive vector loads and stores from one base, which is most of the perps and matmul gains. details, method and caveats are in `results/`.

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

- **sbpf:** +254 / −6 in 9 existing files. new `vector.rs` (2,188 lines), `vector_jit.rs` (3,250), `x86_evex.rs` (319), `tests/vector.rs` (4,477)
- **sbpf-linker:** +173 / −4 in 6 existing files. new `src/vector/` (4,396 lines, 1,233 of them tests)
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
- cu is 1 per op except ops that do more work (gathers and scatters per lane, `ld.n`/`st.n` per register, a few multi-instruction ops). prices are a proposal checked against the per-op table, not tuned per vendor
- the hardware floor is ice lake / zen 4. there is no avx2 fallback, so validators without avx-512 cannot run the jit
- the crates.io sbpf-assembler rejects `call` with `off`. the linker works around it; the fix belongs upstream
