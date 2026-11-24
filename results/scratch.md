# Scratch registers in SIMD-0714

Measured on AMD EPYC 9534 (Zen 4), pinned to core 77. Reproducible with `./scratch-bench/run.sh`.

## Simple example (`scratch-bench/example.c`)

In a pinned-register JIT (`vN` -> `zmmN`), every guest register is live across instructions. Any guest op that lowers to more than one x86 instruction needs scratch ZMMs so intermediates do not overwrite guest state.

Example: `hsum.d` (horizontal sum of 16 x u32):

```asm
; With 2 scratch ZMMs (zmm30, zmm31) — 0.894 ns/op:
vextracti64x4 $1, %zmm0, %ymm30
vpaddd        %ymm0, %ymm30, %ymm30
vextracti32x4 $1, %ymm30, %xmm31
vpaddd        %xmm31, %xmm30, %xmm30
vpshufd       $0x4e, %xmm30, %xmm31
vpaddd        %xmm31, %xmm30, %xmm30
vpshufd       $0xb1, %xmm30, %xmm31
vpaddd        %xmm31, %xmm30, %xmm30
vmovd         %xmm30, %eax

; With 0 scratch ZMMs — 3.411 ns/op (3.82x slower; 73.8% faster with scratch):
vmovdqa64     %zmm14, 0(%rsp)       ; spill 128 B
vmovdqa64     %zmm15, 64(%rsp)
...                                 ; same tree using zmm14/zmm15
vmovdqa64     0(%rsp), %zmm14       ; reload 128 B (store-to-load forwarding ~12 cyc)
vmovdqa64     64(%rsp), %zmm15
```

Why it costs ~12–13 cycles without scratch: on Zen 4, spilling and reloading a 512-bit ZMM through `[rsp]` hits the double-pumped 512-bit store-to-load forwarding path (~11–12 cycles) and serializes consecutive ops that spill the same victim register. Even when surrounded by 4 independent `vpaddd` ops (20% scratch-op density), that store-forwarding chain still caps loop throughput.

## Op-level speedups (`scratch-bench/bench_scratch.c`)

Zen 4 core 77 calibrated at 3.663 GHz. Median of 5 runs per row.

- `/dep`: dependent chain (latency-bound)
- `/ind`: 8 independent lanes (throughput-bound)
- `/mix`: 1 scratch op + 4 independent `vpaddd` ops (20% scratch-op density)

| Op / mode | With scratch | 0-scratch (or alt) | Cycles (scr -> alt) | Speedup | % faster |
|---|---:|---:|---:|---:|---:|
| `hsum.d` / ind (2 vs 0 scr) | 0.854 ns | 3.459 ns | 3.1 -> 12.7 | 4.05x | 75.3% |
| `hsum.d` / ind (2 vs 1 scr) | 0.970 ns | 3.611 ns | 3.6 -> 13.2 | 3.72x | 73.1% |
| `hsum.d` / mix (+4 `vpaddd`) | 1.736 ns | 3.589 ns | 6.4 -> 13.2 | 2.07x | 51.6% |
| `bcast.d[15]` / dep (1 vs 0 scr) | 1.638 ns | 3.542 ns | 6.0 -> 13.0 | 2.16x | 53.7% |
| `bcast.d[15]` / ind (1 vs 0 scr) | 0.553 ns | 3.629 ns | 2.0 -> 13.3 | 6.57x | 84.8% |
| `bcast.d[15]` / mix (+4 `vpaddd`) | 0.971 ns | 3.520 ns | 3.6 -> 12.9 | 3.63x | 72.4% |
| `insert.q[5]` / dep (1 vs 0 scr) | 1.932 ns | 3.464 ns | 7.1 -> 12.7 | 1.79x | 44.2% |
| `insert.q[5]` / ind (1 vs 0 scr) | 0.609 ns | 3.465 ns | 2.2 -> 12.7 | 5.69x | 82.4% |
| `extract.q[5]` / ind (1 vs 0 scr) | 0.559 ns | 3.600 ns | 2.1 -> 13.2 | 6.44x | 84.5% |
| `mul52lo` (alias) / dep (1 vs 0 scr) | 1.141 ns | 3.442 ns | 4.2 -> 12.6 | 3.02x | 66.9% |
| `mul52lo` (alias) / ind (1 vs 0 scr) | 0.278 ns | 3.556 ns | 1.0 -> 13.0 | 12.79x | 92.2% |
| `mul52lo` (alias) / mix (+4 `vpaddd`) | 0.702 ns | 3.360 ns | 2.6 -> 12.3 | 4.79x | 79.1% |
| `sha256rnds2` 16r / dep (1 vs 0 scr) | 1.261 ns | 2.962 ns | 4.6 -> 10.9 | 2.35x | 57.4% |
| `sha256rnds2` (16r vs 30r-hi) / dep | 1.231 ns | 4.403 ns | 4.5 -> 16.1 | 3.58x | 72.0% |
| ModRM `0xff` sanitizer / dep (1 vs 0 scr) | 0.583 ns | 3.704 ns | 2.1 -> 13.6 | 6.35x | 84.3% |
| ModRM `0xff` sanitizer / ind (1 vs 0 scr) | 0.285 ns | 3.601 ns | 1.0 -> 13.2 | 12.63x | 92.1% |

Summary across the 17 SIMD-0714 ops that need ZMM scratch (`hsum.{d,q}`, `extract.{b,w,d,q}` upper lanes, `insert.{b,w,d,q}`, `bcast.{b,w,d,q}` lane > 0, aliased `mul52{lo,hi}`, `sha256rnds2`, plus ModRM `0xff` sanitization):

- **44%–84% faster** (1.8x–6.4x) when latency-bound (`/dep`)
- **75%–92% faster** (4.0x–12.8x) when throughput-bound (`/ind`)
- **52%–79% faster** (2.1x–4.8x) in mixed loops with 20% scratch-op density (`/mix`)
- **2 scratch ZMMs is the exact minimum**: `hsum.d` with 1 scratch ZMM still spills and loses 73.1% vs 2 scratch ZMMs.
- **CU safety**: at 1 CU, 0-scratch `hsum.d` runs at ~3.5 ns/CU (near the 4 ns/CU ceiling), whereas 2 scratch ZMMs keep it at ~0.85 ns/CU.

## Address translation reuses the same 2 scratch ZMMs (`scratch-bench/bench_mmu.c`)

Address translation finishes before any load/store executes, and ALU ops do not translate addresses. The same 2 scratch ZMMs (`zmm30`/`zmm31`) therefore cover both vector-op lowering and L1-backed AVX-512 address translation (`vpbroadcastd` + `vpcmpud`).

Zen 4 core 77 calibrated at 3.468 GHz (latency ns / throughput ns per access, selected rows from `S1`–`S7`):

| Implementation | Res ZMM | Total ZMM | S2: 8/24 (AMM) | S3: 16/48 (DeFi) | S5: 32/64 (unif) | S6: 64/128 (stress) |
|---|---:|---:|---:|---:|---:|---:|
| `fd_bsearch` (scalar) | 0 | 0 | 16.5 / 16.1 | 21.2 / 20.4 | 28.2 / 27.2 | 32.4 / 31.3 |
| `agave_eytzinger` (scalar) | 0 | 0 | 17.7 / 5.4 | 20.8 / 6.2 | 21.4 / 6.5 | 23.8 / 7.3 |
| `l1_tlb16_u32` (L1-backed) | 0 | **2** | **9.5 / 1.8** | **9.8 / 1.8** | 25.6 / 22.4 | 31.1 / 29.7 |
| `l1_tlb32_u32` (L1-backed) | 0 | **2** | 12.0 / 2.2 | 11.7 / 2.1 | **12.5 / 2.2** | 29.5 / 26.7 |
| `l1_btree16_u32` (L1-backed, 0-miss) | 0 | **2** | 15.1 / 2.7 | 14.6 / 2.6 | 15.2 / 2.6 | **17.2 / 2.9** |
| `reg_tlb16_u32` (resident) | 2 | 4 | 9.3 / 1.7 | 10.0 / 1.7 | 25.0 / 21.0 | 29.8 / 28.1 |
| `reg_tlb32_u32` (resident) | 4 | 6 | 10.8 / 2.0 | 11.2 / 2.0 | 12.3 / 2.1 | 30.0 / 25.5 |
| `reg_tlb32_full` (resident) | 12 | **16** | 13.2 / 5.0 | 13.4 / 4.8 | 14.5 / 5.0 | 36.2 / 31.1 |

`l1_tlb32_u32` and `l1_btree16_u32` (0 resident, 2 scratch ZMMs) match or beat `reg_tlb32_full` (12 resident, 16 total ZMMs) across every working set. Reserving 16 ZMMs instead of 2–8 for the JIT gives no speedup on either vector-op lowering or address translation, while capping guest registers at 16 costs 1.7x–4.5x JIT time on register-heavy kernels (`blake3`, `fieldmul`, `sort`, `sha256`, `matmul`, `mlp-unrolled`).
