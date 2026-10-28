# Keccak-f[1600]: scalar sBPF vs Blueshift's AVX2 prototype vs svm vector ops

Measured on an AMD EPYC 9534 (Zen 4, AVX-512), native Linux, per permutation, using k = 100
permutations minus k = 0. CU are identical on both engines.

| | scalar sBPF (plain Rust u64) | Dean / Blueshift (AVX2 prototype) | ours (svm vector ops) |
|---|---:|---:|---:|
| CU per permutation | 11,824 | 3,316 (README: 3,312) | **988** |
| ops per round | ~493 scalar insns | 69 calls × 2 CU = 138 CU | **41.0** (34.0 compute + 6.0 table loads + 1.0 mov) |
| JIT ns per permutation | 8,900 | 1,210 | **312** |
| interpreter µs per permutation | 48 | 89 | 30 |

- **Scalar and ours:** `programs/keccak` on `harness/target/release/svm-run`, with `--expect` on
  both engines.
- **Dean, measured here:** his runtime and his generated `keccak_f`. The program is the
  simd-falcon one with one change: an entry that runs `keccak_init` and then k × `keccak_f`. The
  patch is 9 lines in `entrypoint`. Timing is his `Runner` with `RegisterCached`, the same setup as
  his `simd_falcon` bench.
- **Dean, from his docs:** the 69 ops per round and the bit-interleaved layout. The op count was
  cross-checked: his generated `keccak_f` has 1,656 = 69 × 24 calls.
- **Dean's full Falcon bench, run natively here:** 64 signatures pass. Median JIT verify is
  29.2 µs (his docs, Rosetta 2: 112 µs); Blueshift's scalar program is 295 µs; 53,274 CU.

Ours per permutation, from the disassembly of the k loop: 985 vector ops (ternlog 280,
permq 292, rolv.q 120, xor 84, roli.q 40, ld 144, mov 25) plus 3 scalar instructions. The table
base needs one `lddw` per permutation. Every load uses `ld [r1+off]`.

## Reproduce

```sh
cd programs/keccak && cargo +nightly build-sbpf   # build.rs self-checks the generated code
(cd host && cargo build --release && ./target/release/keccak-host /tmp/k 100 1)
../../harness/target/release/svm-run target/deploy/keccak.so --input /tmp/k/input-vector.bin \
    --expect /tmp/k/expected-vector.bin --runs 30 --budget 100000000   # input-scalar.bin: scalar
```
