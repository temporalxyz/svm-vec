# Falcon-512 verify

`programs/falcon`: hash-to-point (SHAKE-256 on the vector keccak from `programs/keccak`), signature decompression (scalar, adapted from Dean's), NTT, product, inverse NTT and norms at 32 × 16-bit lanes. structure, input format and public key are the same as Dean's `simd-falcon` program, so both verify the same 64 PQClean signatures.

zen 4 (epyc 9534), 64 signatures:

| | blueshift scalar | Dean, blueshift-gg/solana-sbpf@7941ce5 | ours |
|---|---:|---:|---:|
| cu, median (min, max) | 182,764 (172,264, 254,225) | 22,029 (20,460, 31,872) | **20,770** (19,352, 28,134) |
| jit µs, median | 294 | 12.8 | **10.6** |

jit times are from `compare/compare.py`: same timing method for all three, 3 interleaved rounds (`results/comparison.md`).

## correctness

- all 64 signatures return 1 on the interpreter and the jit
- all 128 tampered inputs return 0 (a message bit flipped, `sig[100] ^= 0x10`, as in Dean's bench)
- for all 190 inputs that decode, the 512 hash-to-point coefficients, s2 and the final norm are byte-identical to `programs/falcon/host`: an independent port of PQClean's verify (bit-by-bit decoding, the `sha3` crate's SHAKE-256, schoolbook multiplication)
- `build.rs` checks the generated NTT against schoolbook arithmetic before the program is built

## where the cu go

fixture 5 (135-byte message, 21,730 cu). stage totals measured with feature builds that stop early, the split inside the hash from instruction counts:

| stage | ours | Dean (his README) |
|---|---:|---:|
| keccak-f, 10 permutations | 9,830 | 13,450 |
| absorb, squeeze, sampling | ~3,040 | ~1,300 |
| decompression | 6,928 | 6,400 |
| NTT, product, inverse NTT, norms | 1,932 | 2,200 |
| total | 21,730 | 23,442 |

the NTT part is 1,806 vector ops and 80 register moves, no spills.

he is ahead on sampling, with a compare-and-compress op we do not have, and on decompression, where his decoder has had two more rounds of tuning since the version ours is adapted from. a word compress op would save us about 2,500 cu (estimate).

## ops added for falcon

`mulhi.w`, `mulhiu.w`, `maxu.w`, `minu.w`, `perm2.w` (`vpermt2w`). `perm2.w` does the 2×2 transposes for the NTT levels inside a register. `mulhiu.w` and `maxu.w` are unused by the verifier.

## reproduce

```sh
./bench.sh                    # ours: correctness and cu
python3 compare/compare.py    # Dean, blueshift scalar and ours, timed the same way
```
