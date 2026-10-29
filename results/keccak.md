# Keccak-f[1600]

zen 4 (epyc 9534), per permutation: k = 100 permutations minus k = 0. cu are identical on both engines.

| | scalar sbpf (plain rust u64) | Dean, blueshift-gg/solana-sbpf@7941ce5 | ours |
|---|---:|---:|---:|
| cu per permutation | 11,824 | 1,348 | **988** |
| ops per round | ~493 scalar insns | 28 calls × 2 cu | **41** × 1 cu |
| jit ns per permutation | 8,850 | 445 | **261–304** |
| interpreter µs per permutation | 48 | 54 (earlier run) | 43 |

- scalar and ours: `programs/keccak` on `svm-run`, output checked against the host reference on both engines
- Dean: his runtime and his generated `keccak_f`, through his `Runner` (`RegisterCached`). his falcon program gets a keccak-only entry (`compare/dean-keccak-only.patch`, 9 lines)
- the jit times for Dean and ours come from `compare/compare.py`, which times both the same way (`results/comparison.md`)
- each of his ops is a `mov64 r1, <slots>` plus a `call`, 2 cu. ours carry the registers in the call's imm, 1 cu. that is most of the cu gap: he needs fewer ops per round (28 against our 41), we pay half per op

ours per permutation, from the disassembly of the k loop: 985 vector ops (ternlog 280, permq 292, rolv.q 120, xor 84, roli.q 40, ld 144, mov 25) plus 3 scalar instructions.

## reproduce

```sh
./bench.sh                    # scalar and ours
python3 compare/compare.py    # Dean and ours, interleaved
```
