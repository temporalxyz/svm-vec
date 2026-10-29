# Strict pair vs src=2

svm-vec rodata-pair (df27ecb), sbpf 585f893, sbpf-linker 89a3999. zen 4 (epyc 9534). Same programs built twice from the same commits: `SVM_VECTOR_TRANSPORT=src2` (each op one `call src=2`) and the default `asm` (each op `lddw r1, <descriptor>; call sol_avx512`, descriptor in rodata). JIT and interpreter times are ns per program run, rounds R1 / R2 / R3, ratios from the medians.

| kernel | CU src=2 | CU pair | JIT ns src=2 R1/R2/R3 | JIT ns pair R1/R2/R3 | JIT pair/src=2 | interp pair/src=2 | JIT compile µs src=2 / pair | text+rodata bytes src=2 | text+rodata bytes pair | size pair/src=2 |
|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|
| sha256 | 1,954 | 1,954 | 426 / 526 / 471 | 448 / 476 / 482 | 1.01 | 1.60 | 1,014 / 1,224 | 16,752 | 61,664 | 3.68 |
| blake3 | 1,070,413 | 1,070,438 | 209,539 / 260,614 / 212,726 | 305,562 / 181,260 / 196,512 | 0.92 | 1.52 | 2,107 / 2,859 | 45,496 | 128,816 | 2.83 |
| fieldmul | 18,392 | 18,392 | 4,561 / 4,821 / 5,036 | 5,429 / 5,393 / 5,282 | 1.12 | 1.12 | 6,271 / 9,969 | 155,056 | 473,968 | 3.06 |
| matmul | 25,482 | 25,494 | 6,536 / 6,563 / 6,862 | 5,826 / 6,979 / 6,021 | 0.92 | 1.33 | 623 / 829 | 41,880 | 62,968 | 1.50 |
| sort | 1,031 | 1,032 | 344 / 281 / 321 | 259 / 377 / 247 | 0.81 | 1.37 | 567 / 656 | 9,792 | 38,584 | 3.94 |
| perps | 869 | 869 | 513 / 349 / 361 | 310 / 388 / 520 | 1.07 | 1.43 | 108 / 96 | 264 | 1,000 | 3.79 |
| mlp-unrolled | 1,257 | 1,257 | 572 / 639 / 719 | 947 / 646 / 706 | 1.10 | 1.51 | 725 / 944 | 45,608 | 78,168 | 1.71 |
| mlp | 2,108 | 2,109 | 1,030 / 1,009 / 1,120 | 1,280 / 1,160 / 1,170 | 1.14 | 1.53 | 147 / 172 | 75,064 | 76,800 | 1.02 |
| keccak (per permutation) | 988.01 | 988.01 | 273 / 293 / 348 | 284 / 302 / 276 | 0.97 | 1.57 | 572 / 804 | 23,784 | 47,360 | 1.99 |
| falcon (per verify, median of 64) | 20,770.50 | 20,771.50 | 11,375 / 12,555 / 11,380 | 12,075 / 12,436 / 18,995 | 1.09 | 1.60 | 1,472 / 1,927 | 70,552 | 141,760 | 2.01 |

- Outputs match the host references on both engines for every program, build and round.
- CU differ by +0 to +25 (scalar code generated around the inline asm block).
- JIT run time is within the noise of these runs (0.81–1.14×; single rounds vary by up to 1.7× on either side).
- Costs: text+rodata 1.0–3.9× (24 bytes of text per op instead of 8, plus 16 bytes of rodata per distinct descriptor), JIT compile 0.9–1.6×, interpreter 1.1–1.6×.

## Method

- Host: 256 cores, shared, nothing else running from this session. Load averages (1/5/15 min):
  - R1-start: 92.82, 62.45, 66.18
  - R1-end: 93.75, 70.90, 68.86
  - R2-start: 93.75, 70.90, 68.86
  - R2-end: 129.36, 88.12, 75.09
  - R3-start: 129.36, 88.12, 75.09
  - R3-end: 135.67, 101.74, 81.19
  - compile-start: 149.90, 111.46, 85.80
  - compile-end: 152.47, 112.63, 86.31
- One warm-up round discarded, then 3 rounds: src=2 then pair in R1 and R3, pair then src=2 in R2. Each round runs `compare/suite/<k>.sh ours` for sha256 … mlp-unrolled (method as in `suite.md`), `svm-run --runs 300` for mlp and keccak (k = 100 minus k = 0 permutations), and `programs/falcon/bench.py --runs 50` for falcon.
- JIT compile: median of 7 alternating compiles per program (`jit_compile_ns` of `svm-run`).
- Sizes: the file sizes of the two loadable segments (rodata, text) of each `.so`.
