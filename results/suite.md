# sBPF SIMD suite: ours vs Dean (blueshift-gg/solana-sbpf@7941ce5)

Ratio = Dean / ours. JIT and interpreter times are ns per program run (median of each script's batches); rounds listed R1 / R2 / R3.

## Kernels

| kernel | CU Dean | CU ours | CU ratio | JIT ns Dean (R1/R2/R3) | JIT ns ours (R1/R2/R3) | JIT ratio (medians) | JIT ns/CU Dean | JIT ns/CU ours | interp ns/CU Dean | interp ns/CU ours |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| sha256 | 3,460 | 1,954 | 1.77 | 1,076 / 1,014 / 1,063 | 484 / 492 / 489 | 2.17 | 0.307 | 0.250 | 37.8 | 31.8 |
| blake3 | 1,879,811 | 1,070,413 | 1.76 | 702,081 / 756,188 / 762,648 | 217,653 / 253,140 / 217,410 | 3.47 | 0.402 | 0.203 | 25.5 | 27.7 |
| fieldmul | 35,724 | 18,392 | 1.94 | 8,076 / 8,055 / 8,058 | 4,365 / 4,362 / 4,357 | 1.85 | 0.226 | 0.237 | 35.4 | 32.5 |
| sort | 1,761 | 1,031 | 1.71 | 504 / 501 / 495 | 274 / 273 / 271 | 1.84 | 0.285 | 0.265 | 90.2 | 32.3 |
| perps | 821 | 869 | 0.94 | 565 / 574 / 516 | 787 / 792 / 786 | 0.72 | 0.688 | 0.906 | 52.8 | 23.6 |
| mlp-unrolled | 1,696 | 1,257 | 1.35 | 2,791 / 1,662 / 1,836 | 1,450 / 1,460 / 1,455 | 1.26 | 1.082 | 1.158 | 58.2 | 35.9 |
| falcon (per verify) | 22,029 | 20,770.5 | 1.06 | 12,800 / 12,900 / 20,300 | 11,576 / 11,632 / 11,630 | 1.11 | 0.586 | 0.560 | not measured | not measured |
| keccak (per permutation) | 1,348 | 988.01 | 1.36 | 445 / 445 / 445 | 316 / 314 / 315 | 1.41 | 0.330 | 0.319 | not measured | not measured |
| mlp (programs/mlp, not unrolled) | 1,696 | 2,108 | 0.80 | 1,652 / 1,622 / 1,620 | 1,577 / 1,571 / 1,591 | 1.03 | 0.957 | 0.748 | not measured | not measured |

## Kernels, briefly

- sha256: compression of 16 independent 64-byte blocks. both sides generated from one round generator (`programs/sha256/gen.rs`)
- blake3: Dean's own benchmark, `blake3::hash` of 1 MiB, run unchanged on his side. caveat: his kernel is 8-way on 256-bit slots, ours 16-way on 512-bit registers. a 16-way version on his ISA was not built; the porting agent estimates 40–50% fewer ops in the chunk phase (estimate). treat this row's ratio as an upper bound
- fieldmul: curve25519, 8 lanes × 256 sequential multiplications, radix 2^52 with madd52 on both sides. ours runs a register-only fast path and falls back to a general path when a 2^-38-probability carry condition fails, so its cu are input-dependent in theory
- sort: 256 u32 keys, both sides generated from one network generator (`programs/sort/net.rs`)
- perps: batch fixed-point risk math over 256 positions: pnl, funding, equity and a margin check per position, 5 i64 arrays in, 4 out. ours is a plain rust loop, his is generated straight-line code. memory-bound: ours does 288 single-register loads/stores, his 18 32-slot bulk transfers. Dean wins this one
- mlp-unrolled: `programs/mlp` fully unrolled (2,108 → 1,257 cu), against `compare/dean-mlp`
- falcon, keccak, mlp: from `compare/compare.py`

ops names: `ours=dean/variant` is the same function on both sides, `ours/variant` or `=dean/variant` exists on one side only, `(composite)` is built from several of that side's ops. `dep` is one dependent chain, `ind` is 8 independent chains.

## Ops

Per op instance, median over the 3 rounds of each script's own value. FLAG = JIT ns/CU above 4.

| op | CU Dean | CU ours | JIT ns/op Dean | JIT ns/op ours | JIT ns/CU Dean | JIT ns/CU ours | interp ns/CU Dean | interp ns/CU ours | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| =abs512.d/dep | 2 |  | 0.555 |  | 0.277 |  | 188.9 |  |  |
| =abs512.d/ind | 2 |  | 0.280 |  | 0.140 |  | 189.6 |  |  |
| =adds512.w/dep | 2 |  | 0.554 |  | 0.277 |  | 272.7 |  |  |
| =adds512.w/ind | 2 |  | 0.279 |  | 0.139 |  | 273.2 |  |  |
| =alignr512/dep | 2 |  | 0.557 |  | 0.278 |  | 70.1 |  |  |
| =alignr512/ind | 2 |  | 0.277 |  | 0.139 |  | 73.7 |  |  |
| =avg512.b/dep | 2 |  | 0.555 |  | 0.277 |  | 269.8 |  |  |
| =avg512.b/ind | 2 |  | 0.213 |  | 0.107 |  | 270.2 |  |  |
| =blendd512/dep | 2 |  | 0.545 |  | 0.273 |  | 33.3 |  |  |
| =blendd512/ind | 2 |  | 0.548 |  | 0.274 |  | 33.4 |  |  |
| =blendm512.d/dep | 2 |  | 0.608 |  | 0.304 |  | 113.6 |  |  |
| =blendm512.d/ind | 2 |  | 0.475 |  | 0.237 |  | 113.1 |  |  |
| =cmp512.gt.d/ind | 2 |  | 0.542 |  | 0.271 |  | 91.9 |  |  |
| =compress512.b/dep | 2 |  | 1.423 |  | 0.712 |  | 250.9 |  |  |
| =compress512.b/ind | 2 |  | 0.702 |  | 0.351 |  | 253.0 |  |  |
| =compress512.d/dep | 2 |  | 1.465 |  | 0.732 |  | 131.9 |  |  |
| =compress512.d/ind | 2 |  | 0.702 |  | 0.351 |  | 132.8 |  |  |
| =conflict512.d/dep | 2 |  | 1.644 |  | 0.822 |  | 181.0 |  |  |
| =conflict512.d/ind | 2 |  | 0.469 |  | 0.235 |  | 181.5 |  |  |
| =conflict512.q/dep | 2 |  | 1.643 |  | 0.821 |  | 139.2 |  |  |
| =conflict512.q/ind | 2 |  | 0.474 |  | 0.237 |  | 142.3 |  |  |
| =expand512.d/dep | 2 |  | 1.423 |  | 0.712 |  | 110.0 |  |  |
| =expand512.d/ind | 2 |  | 0.702 |  | 0.351 |  | 110.6 |  |  |
| =gf2p8affine512/dep | 2 |  | 0.827 |  | 0.413 |  | 250.3 |  |  |
| =gf2p8affine512/ind | 2 |  | 0.280 |  | 0.140 |  | 250.6 |  |  |
| =gf2p8mul512/dep | 2 |  | 0.827 |  | 0.413 |  | 233.2 |  |  |
| =gf2p8mul512/ind | 2 |  | 0.281 |  | 0.140 |  | 233.8 |  |  |
| =load256x2.rodata/ind | 2 |  | 1.522 |  | 0.761 |  | 20.7 |  |  |
| =loadm512.d/ind | 2 |  | 54.527 |  | 27.263 |  | 37.8 |  | FLAG dean |
| =lzcnt512.d/dep | 2 |  | 0.555 |  | 0.278 |  | 190.4 |  |  |
| =lzcnt512.d/ind | 2 |  | 0.281 |  | 0.140 |  | 193.8 |  |  |
| =mont16x32/dep | 2 |  | 1.927 |  | 0.964 |  | 88.3 |  |  |
| =mont16x32/ind | 2 |  | 0.992 |  | 0.496 |  | 87.6 |  |  |
| =mulhrs16x32/dep | 2 |  | 0.828 |  | 0.414 |  | 224.0 |  |  |
| =mulhrs16x32/ind | 2 |  | 0.282 |  | 0.141 |  | 214.4 |  |  |
| =multishift512/dep | 2 |  | 0.827 |  | 0.414 |  | 71.0 |  |  |
| =multishift512/ind | 2 |  | 0.281 |  | 0.140 |  | 70.9 |  |  |
| =permb512/dep | 2 |  | 1.372 |  | 0.686 |  | 69.0 |  |  |
| =permb512/ind | 2 |  | 0.285 |  | 0.142 |  | 70.1 |  |  |
| =permi2b512/dep | 2 |  | 1.372 |  | 0.686 |  | 72.6 |  |  |
| =permi2b512/ind | 2 |  | 0.298 |  | 0.149 |  | 75.1 |  |  |
| =permi2d512/dep | 2 |  | 1.371 |  | 0.686 |  | 34.0 |  |  |
| =permi2d512/ind | 2 |  | 0.300 |  | 0.150 |  | 34.1 |  |  |
| =permq512/dep | 2 |  | 1.098 |  | 0.549 |  | 138.4 |  |  |
| =permq512/ind | 2 |  | 0.281 |  | 0.141 |  | 138.4 |  |  |
| =permt2d512/dep | 2 |  | 1.371 |  | 0.686 |  | 32.7 |  |  |
| =permt2d512/ind | 2 |  | 0.304 |  | 0.152 |  | 32.6 |  |  |
| =permw512/dep | 2 |  | 1.372 |  | 0.686 |  | 206.5 |  |  |
| =permw512/ind | 2 |  | 0.284 |  | 0.142 |  | 205.5 |  |  |
| =sadbw512/dep | 2 |  | 0.828 |  | 0.414 |  | 69.0 |  |  |
| =sadbw512/ind | 2 |  | 0.282 |  | 0.141 |  | 69.5 |  |  |
| =shldv512.d/dep | 2 |  | 0.556 |  | 0.278 |  | 218.7 |  |  |
| =shldv512.d/ind | 2 |  | 0.279 |  | 0.140 |  | 219.7 |  |  |
| =shufd512/dep | 2 |  | 0.329 |  | 0.164 |  | 33.0 |  |  |
| =shufd512/ind | 2 |  | 0.229 |  | 0.115 |  | 33.0 |  |  |
| =sllv16x32/dep | 2 |  | 0.417 |  | 0.208 |  | 214.7 |  |  |
| =sllv16x32/ind | 2 |  | 0.277 |  | 0.138 |  | 217.2 |  |  |
| =srav16x32/dep | 2 |  | 0.419 |  | 0.209 |  | 222.8 |  |  |
| =srav16x32/ind | 2 |  | 0.278 |  | 0.139 |  | 222.9 |  |  |
| =storem512.d/ind | 2 |  | 33.705 |  | 16.853 |  | 30.6 |  | FLAG dean |
| =testm512.d/ind | 2 |  | 0.542 |  | 0.271 |  | 77.7 |  |  |
| =transpose8x8/dep | 2 |  | 6.463 |  | 3.231 |  | 31.5 |  |  |
| =transpose8x8q/dep | 2 |  | 10.980 |  | 5.490 |  | 132.3 |  | FLAG dean |
| =unpacklo512.d/dep | 2 |  | 0.326 |  | 0.163 |  | 69.4 |  |  |
| =unpacklo512.d/ind | 2 |  | 0.226 |  | 0.113 |  | 68.8 |  |  |
| add.b=add8x64/dep | 2 | 1 | 0.377 | 0.268 | 0.188 | 0.268 | 274.2 | 20.2 |  |
| add.b=add8x64/ind | 2 | 1 | 0.210 | 0.142 | 0.105 | 0.142 | 275.6 | 20.6 |  |
| add.d=add32x16/dep | 2 | 1 | 0.375 | 0.266 | 0.188 | 0.266 | 33.7 | 22.1 |  |
| add.d=add32x16/ind | 2 | 1 | 0.289 | 0.145 | 0.145 | 0.145 | 33.2 | 22.6 |  |
| add.q=add64x8/dep | 2 | 1 | 0.375 | 0.271 | 0.188 | 0.271 | 32.8 | 22.1 |  |
| add.q=add64x8/ind | 2 | 1 | 0.289 | 0.145 | 0.144 | 0.145 | 33.1 | 22.6 |  |
| add.w=add16x32/dep | 2 | 1 | 0.376 | 0.268 | 0.188 | 0.268 | 40.4 | 22.1 |  |
| add.w=add16x32/ind | 2 | 1 | 0.288 | 0.144 | 0.144 | 0.144 | 40.5 | 22.6 |  |
| and=and512/dep | 2 | 1 | 0.375 | 0.267 | 0.187 | 0.267 | 33.6 | 22.1 |  |
| and=and512/ind | 2 | 1 | 0.287 | 0.145 | 0.144 | 0.145 | 34.0 | 22.6 |  |
| andn=andn512/dep | 2 | 1 | 0.375 | 0.267 | 0.187 | 0.267 | 32.8 | 22.2 |  |
| andn=andn512/ind | 2 | 1 | 0.290 | 0.145 | 0.145 | 0.145 | 33.5 | 22.6 |  |
| clmul=clmul512/dep | 2 | 1 | 1.144 | 1.118 | 0.572 | 1.118 | 221.6 | 333.0 |  |
| clmul=clmul512/ind | 2 | 1 | 0.550 | 0.542 | 0.275 | 0.542 | 221.1 | 333.3 |  |
| cmpeq.b/dep |  | 1 |  | 1.658 |  | 1.658 |  | 20.3 |  |
| cmpeq.b/ind |  | 1 |  | 0.464 |  | 0.464 |  | 20.6 |  |
| cmpeq.d/dep |  | 1 |  | 1.661 |  | 1.661 |  | 22.3 |  |
| cmpeq.d/ind |  | 1 |  | 0.465 |  | 0.465 |  | 22.7 |  |
| cmpeq.q/dep |  | 1 |  | 1.660 |  | 1.660 |  | 22.6 |  |
| cmpeq.q/ind |  | 1 |  | 0.464 |  | 0.464 |  | 23.1 |  |
| cmpgt.d/dep |  | 1 |  | 1.662 |  | 1.662 |  | 22.2 |  |
| cmpgt.d/ind |  | 1 |  | 0.466 |  | 0.466 |  | 22.7 |  |
| cmpgt.q/dep |  | 1 |  | 1.659 |  | 1.659 |  | 22.7 |  |
| cmpgt.q/ind |  | 1 |  | 0.465 |  | 0.465 |  | 23.3 |  |
| cmpgtu.q/dep |  | 1 |  | 1.660 |  | 1.660 |  | 28.1 |  |
| cmpgtu.q/ind |  | 1 |  | 0.465 |  | 0.465 |  | 28.1 |  |
| dpbusd=dpbusd512/dep | 2 | 1 | 1.098 | 1.083 | 0.549 | 1.083 | 68.7 | 42.2 |  |
| dpbusd=dpbusd512/ind | 2 | 1 | 0.282 | 0.268 | 0.141 | 0.268 | 73.4 | 42.1 |  |
| dpwssd=dpwssd512/dep | 2 | 1 | 1.099 | 1.082 | 0.550 | 1.082 | 74.9 | 27.0 |  |
| dpwssd=dpwssd512/ind | 2 | 1 | 0.281 | 0.267 | 0.140 | 0.267 | 72.5 | 26.9 |  |
| extract.d/ind |  | 1 |  | 0.365 |  | 0.365 |  | 14.3 |  |
| extract.q=ext64/ind | 2 | 1 | 0.138 | 0.380 | 0.069 | 0.380 | 18.3 | 14.2 |  |
| hsum.d=(composite)/ind | 16 | 1 | 3.038 | 0.862 | 0.190 | 0.862 | 27.0 | 14.5 |  |
| hsum.q=(composite)/ind | 12 | 1 | 2.042 | 0.664 | 0.170 | 0.664 | 34.5 | 14.5 |  |
| insert.q+extract.q=ins64+ext64/dep | 5 | 3 | 1.401 | 3.528 | 0.280 | 1.176 | 15.5 | 12.7 |  |
| insert.q=ins64/dep | 2 | 1 | 0.337 | 1.671 | 0.168 | 1.671 | 18.9 | 21.3 |  |
| insert.q=ins64/ind | 2 | 1 | 0.295 | 0.587 | 0.148 | 0.587 | 18.3 | 21.3 |  |
| ld*16=load256x32/ind | 2 | 16 | 16.823 | 43.627 | 8.411 | 2.727 | 28.9 | 25.9 | FLAG dean |
| ld=load256x2/ind | 2 | 1 | 7.824 | 2.064 | 3.912 | 2.064 | 20.1 | 26.0 |  |
| ldbc.d/ind |  | 1 |  | 1.938 |  | 1.938 |  | 23.2 |  |
| ldbc.q/ind |  | 1 |  | 1.942 |  | 1.942 |  | 22.9 |  |
| madd52hi=madd52hi64x8/dep | 2 | 1 | 1.098 | 1.084 | 0.549 | 1.084 | 35.3 | 27.0 |  |
| madd52hi=madd52hi64x8/ind | 2 | 1 | 0.294 | 0.267 | 0.147 | 0.267 | 36.9 | 27.0 |  |
| madd52lo=madd52lo64x8/dep | 2 | 1 | 1.100 | 1.083 | 0.550 | 1.083 | 34.3 | 24.6 |  |
| madd52lo=madd52lo64x8/ind | 2 | 1 | 0.295 | 0.268 | 0.147 | 0.268 | 34.8 | 24.4 |  |
| maddubs=maddubs16x32/dep | 2 | 1 | 0.827 | 0.811 | 0.414 | 0.811 | 79.5 | 53.5 |  |
| maddubs=maddubs16x32/ind | 2 | 1 | 0.280 | 0.267 | 0.140 | 0.267 | 79.1 | 53.0 |  |
| maddwd=madd16x32/dep | 2 | 1 | 0.826 | 0.811 | 0.413 | 0.811 | 36.6 | 23.7 |  |
| maddwd=madd16x32/ind | 2 | 1 | 0.291 | 0.266 | 0.146 | 0.266 | 38.1 | 23.8 |  |
| max.d=max512.d/dep | 2 | 1 | 0.377 | 0.267 | 0.188 | 0.267 | 200.5 | 22.3 |  |
| max.d=max512.d/ind | 2 | 1 | 0.211 | 0.144 | 0.105 | 0.144 | 202.6 | 22.9 |  |
| max.w=max512.w/dep | 2 | 1 | 0.374 | 0.268 | 0.187 | 0.268 | 236.8 | 22.2 |  |
| max.w=max512.w/ind | 2 | 1 | 0.211 | 0.145 | 0.105 | 0.145 | 236.9 | 22.5 |  |
| minu.d=minu32x16/dep | 2 | 1 | 0.374 | 0.267 | 0.187 | 0.267 | 33.9 | 22.5 |  |
| minu.d=minu32x16/ind | 2 | 1 | 0.290 | 0.144 | 0.145 | 0.144 | 34.0 | 23.0 |  |
| minu.q=min512.uq/dep | 2 | 1 | 0.556 | 0.538 | 0.278 | 0.538 | 158.3 | 23.0 |  |
| minu.q=min512.uq/ind | 2 | 1 | 0.280 | 0.265 | 0.140 | 0.265 | 156.9 | 23.3 |  |
| minu.w=minu16x32/dep | 2 | 1 | 0.375 | 0.266 | 0.187 | 0.266 | 44.3 | 22.1 |  |
| minu.w=minu16x32/ind | 2 | 1 | 0.288 | 0.144 | 0.144 | 0.144 | 44.5 | 22.6 |  |
| mov=mov512/ind | 2 | 1 | 0.277 | 0.045 | 0.139 | 0.045 | 33.1 | 19.1 |  |
| movemask.b=movmsk8x64/ind | 2 | 1 | 0.543 | 0.540 | 0.271 | 0.540 | 32.6 | 36.5 |  |
| mulhi.w=mulhi16x32/dep | 2 | 1 | 0.828 | 0.810 | 0.414 | 0.810 | 40.9 | 22.3 |  |
| mulhi.w=mulhi16x32/ind | 2 | 1 | 0.289 | 0.267 | 0.145 | 0.267 | 41.0 | 22.8 |  |
| mulhiu.w=mulhu16x32/dep | 2 | 1 | 0.827 | 0.811 | 0.413 | 0.811 | 208.3 | 22.3 |  |
| mulhiu.w=mulhu16x32/ind | 2 | 1 | 0.281 | 0.267 | 0.140 | 0.267 | 210.9 | 22.7 |  |
| mullo.d=mullo32x16/dep | 2 | 1 | 0.827 | 0.810 | 0.414 | 0.810 | 33.0 | 22.6 |  |
| mullo.d=mullo32x16/ind | 2 | 1 | 0.291 | 0.266 | 0.145 | 0.266 | 33.2 | 22.9 |  |
| mullo.q=mullo64x8/dep | 2 | 1 | 0.829 | 0.812 | 0.415 | 0.812 | 33.2 | 25.7 |  |
| mullo.q=mullo64x8/ind | 2 | 1 | 0.290 | 0.266 | 0.145 | 0.266 | 33.3 | 26.6 |  |
| mullo.w=mullo16x32/dep | 2 | 1 | 0.828 | 0.811 | 0.414 | 0.811 | 41.5 | 22.2 |  |
| mullo.w=mullo16x32/ind | 2 | 1 | 0.293 | 0.269 | 0.147 | 0.269 | 41.3 | 22.8 |  |
| mulws=muls32x8/dep | 2 | 1 | 0.827 | 0.810 | 0.414 | 0.810 | 154.6 | 31.5 |  |
| mulws=muls32x8/ind | 2 | 1 | 0.280 | 0.267 | 0.140 | 0.267 | 153.7 | 31.4 |  |
| mulwu=mulu32x8/dep | 2 | 1 | 0.827 | 0.811 | 0.414 | 0.811 | 33.6 | 22.3 |  |
| mulwu=mulu32x8/ind | 2 | 1 | 0.291 | 0.267 | 0.146 | 0.267 | 34.4 | 22.7 |  |
| narrow.d.b=trunc512.db/dep | 2 | 1 | 1.371 | 1.355 | 0.686 | 1.355 | 207.4 | 25.8 |  |
| narrow.d.b=trunc512.db/ind | 2 | 1 | 0.283 | 0.267 | 0.141 | 0.267 | 206.2 | 25.9 |  |
| narrows.d.b/dep |  | 1 |  | 1.353 |  | 1.353 |  | 28.4 |  |
| narrows.d.b/ind |  | 1 |  | 0.266 |  | 0.266 |  | 28.4 |  |
| narrows.d.w/dep |  | 1 |  | 1.082 |  | 1.082 |  | 29.1 |  |
| narrows.d.w/ind |  | 1 |  | 0.266 |  | 0.266 |  | 29.1 |  |
| narrowus.d.b/dep |  | 1 |  | 1.354 |  | 1.354 |  | 26.4 |  |
| narrowus.d.b/ind |  | 1 |  | 0.266 |  | 0.266 |  | 26.3 |  |
| or=or512/dep | 2 | 1 | 0.375 | 0.267 | 0.187 | 0.267 | 33.4 | 22.2 |  |
| or=or512/ind | 2 | 1 | 0.289 | 0.145 | 0.145 | 0.145 | 33.8 | 22.6 |  |
| packs.d=packss512.d/dep | 2 | 1 | 0.555 | 0.538 | 0.278 | 0.538 | 469.3 | 29.4 |  |
| packs.d=packss512.d/ind | 2 | 1 | 0.277 | 0.266 | 0.139 | 0.266 | 470.0 | 29.7 |  |
| packus.w=packus512.w/dep | 2 | 1 | 0.555 | 0.538 | 0.277 | 0.538 | 778.1 | 48.4 |  |
| packus.w=packus512.w/ind | 2 | 1 | 0.278 | 0.266 | 0.139 | 0.266 | 780.9 | 47.3 |  |
| perm2.w/dep |  | 1 |  | 1.354 |  | 1.354 |  | 41.4 |  |
| perm2.w/ind |  | 1 |  | 0.266 |  | 0.266 |  | 41.7 |  |
| permd=permd512/dep | 2 | 1 | 1.371 | 1.354 | 0.686 | 1.354 | 32.9 | 22.4 |  |
| permd=permd512/ind | 2 | 1 | 0.292 | 0.267 | 0.146 | 0.267 | 34.3 | 22.5 |  |
| permq/dep |  | 1 |  | 1.353 |  | 1.353 |  | 24.5 |  |
| permq/ind |  | 1 |  | 0.265 |  | 0.265 |  | 24.3 |  |
| popcnt.b=popcnt512.b/dep | 2 | 1 | 0.554 | 0.539 | 0.277 | 0.539 | 294.2 | 23.3 |  |
| popcnt.b=popcnt512.b/ind | 2 | 1 | 0.278 | 0.268 | 0.139 | 0.268 | 294.3 | 23.2 |  |
| popcnt.q=popcnt512.q/dep | 2 | 1 | 0.555 | 0.537 | 0.278 | 0.537 | 156.7 | 20.9 |  |
| popcnt.q=popcnt512.q/ind | 2 | 1 | 0.278 | 0.265 | 0.139 | 0.265 | 155.8 | 21.0 |  |
| roli.d=ror32x16/dep | 2 | 1 | 0.556 | 0.539 | 0.278 | 0.539 | 33.9 | 25.5 |  |
| roli.d=ror32x16/ind | 2 | 1 | 0.280 | 0.265 | 0.140 | 0.265 | 34.0 | 26.3 |  |
| roli.q=ror64x8/dep | 2 | 1 | 0.556 | 0.539 | 0.278 | 0.539 | 153.4 | 20.2 |  |
| roli.q=ror64x8/ind | 2 | 1 | 0.279 | 0.264 | 0.140 | 0.264 | 155.0 | 20.5 |  |
| rolv.d=rolv32x16/dep | 2 | 1 | 0.555 | 0.539 | 0.278 | 0.539 | 34.3 | 31.9 |  |
| rolv.d=rolv32x16/ind | 2 | 1 | 0.291 | 0.266 | 0.146 | 0.266 | 35.1 | 31.5 |  |
| rolv.q=rolv64x8/dep | 2 | 1 | 0.556 | 0.538 | 0.278 | 0.538 | 33.0 | 29.1 |  |
| rolv.q=rolv64x8/ind | 2 | 1 | 0.292 | 0.266 | 0.146 | 0.266 | 33.3 | 29.0 |  |
| sari.q=sra64x8/dep | 2 | 1 | 0.414 | 0.268 | 0.207 | 0.268 | 33.5 | 20.4 |  |
| sari.q=sra64x8/ind | 2 | 1 | 0.279 | 0.267 | 0.139 | 0.267 | 34.0 | 21.0 |  |
| shli.d=sll32x16/dep | 2 | 1 | 0.414 | 0.268 | 0.207 | 0.268 | 33.2 | 22.3 |  |
| shli.d=sll32x16/ind | 2 | 1 | 0.278 | 0.267 | 0.139 | 0.267 | 33.5 | 22.8 |  |
| shli.w=sll16x32/dep | 2 | 1 | 0.415 | 0.266 | 0.207 | 0.266 | 213.9 | 29.5 |  |
| shli.w=sll16x32/ind | 2 | 1 | 0.277 | 0.265 | 0.139 | 0.265 | 213.9 | 29.4 |  |
| shlv.d=sllv32x16/dep | 2 | 1 | 0.415 | 0.266 | 0.207 | 0.266 | 33.3 | 23.3 |  |
| shlv.d=sllv32x16/ind | 2 | 1 | 0.291 | 0.266 | 0.146 | 0.266 | 34.3 | 23.4 |  |
| shri.q=srl64x8/dep | 2 | 1 | 0.416 | 0.268 | 0.208 | 0.268 | 33.7 | 19.7 |  |
| shri.q=srl64x8/ind | 2 | 1 | 0.280 | 0.267 | 0.140 | 0.267 | 33.8 | 20.0 |  |
| shrv.q=srlv64x8/dep | 2 | 1 | 0.415 | 0.268 | 0.207 | 0.268 | 33.9 | 26.7 |  |
| shrv.q=srlv64x8/ind | 2 | 1 | 0.291 | 0.268 | 0.146 | 0.268 | 34.5 | 26.6 |  |
| shufb=shufb512/dep | 2 | 1 | 0.556 | 0.538 | 0.278 | 0.538 | 49.8 | 40.0 |  |
| shufb=shufb512/ind | 2 | 1 | 0.291 | 0.266 | 0.146 | 0.266 | 51.2 | 40.0 |  |
| splat.d=bcast32x16/ind | 2 | 1 | 0.293 | 0.289 | 0.147 | 0.289 | 24.1 | 17.2 |  |
| splat.q=bcast64x8/ind | 2 | 1 | 0.293 | 0.289 | 0.146 | 0.289 | 23.6 | 17.0 |  |
| st*16=store256x32/ind | 2 | 16 | 18.515 | 44.307 | 9.258 | 2.769 | 29.1 | 18.7 | FLAG dean |
| st=store256x2/ind | 2 | 1 | 8.239 | 2.286 | 4.120 | 2.286 | 21.0 | 18.7 | FLAG dean |
| sub.d=sub32x16/dep | 2 | 1 | 0.375 | 0.268 | 0.187 | 0.268 | 33.0 | 22.1 |  |
| sub.d=sub32x16/ind | 2 | 1 | 0.289 | 0.143 | 0.145 | 0.143 | 32.8 | 22.6 |  |
| sub.q=sub64x8/dep | 2 | 1 | 0.378 | 0.267 | 0.189 | 0.267 | 34.9 | 22.2 |  |
| sub.q=sub64x8/ind | 2 | 1 | 0.288 | 0.145 | 0.144 | 0.145 | 33.9 | 22.6 |  |
| sx.w.d=sext512.wd/dep | 2 | 1 | 1.098 | 1.083 | 0.549 | 1.083 | 148.6 | 26.6 |  |
| sx.w.d=sext512.wd/ind | 2 | 1 | 0.281 | 0.267 | 0.140 | 0.267 | 150.6 | 26.1 |  |
| ternlog=ternlog512/dep | 2 | 1 | 0.285 | 0.267 | 0.143 | 0.267 | 39.5 | 32.3 |  |
| ternlog=ternlog512/ind | 2 | 1 | 0.303 | 0.157 | 0.151 | 0.157 | 41.4 | 32.5 |  |
| xor=xor512/dep | 2 | 1 | 0.375 | 0.267 | 0.187 | 0.267 | 33.6 | 22.1 |  |
| xor=xor512/ind | 2 | 1 | 0.290 | 0.145 | 0.145 | 0.145 | 34.4 | 22.6 |  |
| zero/ind |  | 1 |  | 0.046 |  | 0.046 |  | 17.9 |  |
| zx.b.d=zext512.bd/dep | 2 | 1 | 1.100 | 1.083 | 0.550 | 1.083 | 146.1 | 29.3 |  |
| zx.b.d=zext512.bd/ind | 2 | 1 | 0.283 | 0.266 | 0.141 | 0.266 | 148.0 | 29.6 |  |

Ops above 4 JIT ns/CU (median of 3 rounds):

- =loadm512.d/ind: dean 27.26
- =storem512.d/ind: dean 16.85
- =transpose8x8q/dep: dean 5.49
- ld*16=load256x32/ind: dean 8.41
- st*16=store256x32/ind: dean 9.26
- st=store256x2/ind: dean 4.12

## Method

- Machine: 256 logical CPUs, shared with other processes (load averages below).
- Suite kernels (sha256, blake3, fieldmul, sort, perps, mlp-unrolled, ops): each `compare/suite/<k>.sh ours|dean` run once to warm builds (discarded), then 3 rounds. Each round runs every kernel's two sides back to back: Dean then ours in rounds 1 and 3, ours then Dean in round 2. Every run printed ok=true. Each script checks outputs against its host reference on both engines and does its own timing (calibrated ~2 ms batches, median of 9-15 batches; see the porting notes).
- CU is identical in all 3 rounds for every suite kernel on both sides.
- ns/CU = median JIT (or interpreter) ns per run / CU per run.
- falcon, keccak, mlp: `./bench.sh` then `python3 compare/compare.py --rounds 3`, numbers taken from its per-round output (compare.py runs Dean then ours in every round; no alternation). compare.py does not time the interpreter, so interpreter columns for these rows are not measured. falcon CU is the median over 64 signatures (ours shows 20,770.5 as a median of an even count); keccak CU and time are per permutation from the (100 - 0) permutation difference. The mlp row is programs/mlp (2,108 CU), not mlp-unrolled.
- Ops: K=256 op instances per body, looped 16 times, baseline-subtracted (see ops notes). The ops table shows the median over 3 rounds of the per-op values each script printed.

## Load averages (uptime)

- before suite round 1 (order dean then ours): 34.33, 28.71, 28.13
- before suite round 2 (order ours then dean): 29.27, 28.40, 28.07
- before suite round 3 (order dean then ours): 26.44, 27.84, 27.90
- before bench.sh: 24.91, 27.04, 27.61
- before compare.py: 24.41, 26.84, 27.54
- after compare.py: 21.36, 25.67, 27.07

Raw runs: /home/ubuntu/svm-vec/target/suite/results.json

## reproduce

```sh
./bench.sh                              # builds and generates inputs
python3 compare/compare.py              # clones Dean's repo; falcon, keccak, mlp
compare/suite/<kernel>.sh ours|dean     # one json line: cu, jit_ns, interp_ns, ok
```
