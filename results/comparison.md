# ours vs Dean

Dean = blueshift's sbpf simd experiment, blueshift-gg/solana-sbpf@7941ce5 (latest `simd-experiment`, with his avx-512 jit). each side runs its own programs on its own runtime: neither runtime runs the other's ops.

zen 4 (epyc 9534), 3 interleaved rounds, `python3 compare/compare.py`:

| | Dean | ours | ours is |
|---|---:|---:|---:|
| falcon-512 verify, cu (median of 64) | 22,029 | 20,770 | 1.06× fewer |
| falcon-512 verify, jit µs | 12.9 / 12.8 / 12.9 | 11.6 / 11.7 / 11.6 | 1.11× faster |
| keccak-f, cu / permutation | 1,348 | 988 | 1.36× fewer |
| keccak-f, jit ns / permutation | 446 / 445 / 445 | 319 / 320 / 321 | 1.39× faster |

Dean has no mlp, so there is nothing to compare there.

## method

- same timing for both: the mean of back-to-back runs on one memory mapping, a fresh vm each run, median over samples. his bench does this with his `Runner`, ours with `svm-run --repeat`
- falcon: 9 samples of ~2 ms per signature, median over the 64 signatures, for both
- the same 64 signatures for both (`programs/falcon/fixtures` is a copy of his)
- blueshift's scalar program, timed by both harnesses, agrees: 182,764 / 182,738 cu, 293–296 µs each

## not equal

- he builds with `cargo-build-sbf`, we build with stock nightly and our linker
- the falcon programs are separate ports. each has things the other lacks (`results/falcon.md`)
- shared machine, load average 20–30 during the runs. rounds within one invocation agree within 1%, but our times move between invocations (falcon 11.6–12.3 µs, keccak 320–372 ns over three) while his stay within 1%. both use the default jit config, including random no-op insertion, so this is likely code placement. not investigated
