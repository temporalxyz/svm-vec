# mlp

int8 mlp, 256 → 128 (relu, requantized to 0..127) → 16 logits. weights are s8 in .rodata, activations u8.
source: `programs/mlp`. host reference: `programs/mlp/host`.

zen 4 (epyc 9534), median of 300 runs. `./bench.sh` at load average 17, except the inline-translation row: a separate run at load average 135 (3 repetitions, median), where stock scalar JIT measured 410 µs.

| | cu | jit | interpreter |
|---|---|---|---|
| scalar sbpf, stock jit (`--features scalar-only`) | 490,048 | 360.7 µs | 1.69 ms |
| scalar sbpf, in a vector program (inline translation) | 490,051 | 198.1 µs | 2.04 ms |
| **vector ops** | **2,108** | **0.90 µs** | 67.8 µs |
| ratio, stock scalar / vector | **232×** | **401×** | 25× |

- output is byte-identical to the host reference on the interpreter and the jit, for seeds 1–5
- vector path: 72 vector ops in the binary. layer 1 hot loop per 4 inputs: 1 `ldbc.d` + 8 `ld` (constant `off`) + 8 `dpbusd` + 7 scalar, no `mov`s. accumulators stay in v0–v7 across the loop
- `off` displacement on ld/st took the vector path from 3,031 to 2,108 cu (−30%)
- side finding: the same scalar code is 2.1× faster in the jit (same run) when memory accesses go through the inline translation fast path instead of a rust call per access. that would help every program, not just vector ones

reproduce:

```sh
cd programs/mlp/host && cargo run --release -- ../data 1
cd programs && ./build.sh mlp
(cd mlp && cargo +nightly build-sbpf --features scalar-only --target-dir target-scalar)
D=programs/mlp/data
harness/target/release/svm-run programs/mlp/target/bpfel-unknown-none/release/libmlp.so \
  --input $D/input-scalar.bin --input $D/input-vector.bin \
  --expect $D/expected-scalar.bin --expect $D/expected-vector.bin --runs 300
```
