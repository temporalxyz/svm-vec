# svm vector ops: poc spec

the contract between the runtime (`sbpf/`), the build pass (`sbpf-linker/`) and the intrinsics (`svm-intrinsics/`). if code and this file disagree, fix one of them.

## registers

- 30 vector registers `v0`–`v29`, 512 bits each, lanes little-endian
- lane widths: `b` = 64×u8, `w` = 32×u16, `d` = 16×u32, `q` = 8×u64
- vm state, not memory mapped. only reachable through vector ops
- zeroed at program entry. global to the invocation: preserved across internal calls and syscalls
- jit pins `vN` to `zmmN`. `zmm30`, `zmm31`, `k1`–`k7` are jit scratch

## encoding

a vector op is a `CALL_IMM` (`opc = 0x85`) with `src = 2`, `dst` = the mask mode (see masks), and `off` = 0 except for memory ops (see below):

```
imm[31:24] op
imm[23:16] imm8
imm[15]    0
imm[14:10] out
imm[9:5]   in0
imm[4:0]   in1
```

as one little-endian u64: `0x85 | (0x20 | mode) << 8 | off << 16 | imm << 32`.

- enabled by `Config::enable_vector_ops` (poc gate; production would bind it to a new sbpf version, because today v3 accepts `src=2` calls at deploy and faults at runtime). requires an sbpf version with static syscalls (v3+)
- verifier rejects: unknown op, any register field ≥ 30, a nonzero field the op doesn't use (see kind), imm8 out of range for the op, a mask mode the op does not allow, `off` ≠ 0 on a non-memory op, `imm[15]` set, `ldn`/`stn` registers past `v29`
- cu: 1 per op, except where the op table gives a cu. an op of c > 1 cu fails with `ExceededMaxInstructions` at its own pc, before any effect, unless c cu remain

## scalar conventions

- `r1` = scalar input (address or value). `r2` = mask. `r0` = scalar output
- ops without a scalar output leave `r0` unchanged. no op modifies any other scalar register
- memory ops (`ld`, `st`, `ldbc.*`, `ld.x`, `ld.y`, `st.x`, `st.y`, `ld.m`, `st.m`, `ldn`, `stn`, `gather.*`, `scatter.*`) access `r1 + sext(off)`, like scalar `ldx`/`stx`, so a constant displacement needs no extra scalar instruction. all other ops require `off = 0`
- memory ops: unaligned ok, the whole access must be inside one mapped region with the right permission, otherwise the same error a scalar access would raise. on a fault, vector registers and memory are unchanged
- `ld.m`/`st.m`: the access is the whole 64 bytes whatever the mask (a zero mask still needs the span mapped). `st.m` writes only the active lanes
- `ldn`/`stn`: one access of `64·n` bytes, registers `out..out+n-1` (`in0..in0+n-1`)
- gather/scatter: lane i accesses `E` bytes (the element size) at `A_i = r1 + sext(off) + (idx_i << imm8)`, wrapping, as a scalar access of its own. first every active lane is checked in ascending order, the first failure ends the op with that error and nothing written. then the active lanes are read, or written in ascending order (a later lane wins where two overlap). lanes may lie in different regions

## masks

`dst` is the mask mode. the mask is `r2`, bit i = lane i at the op's mask granularity, bits at or above the lane count are ignored.

| mode | meaning |
|---|---|
| 0 | unmasked |
| 1 | merge: lane i ← r2[i] ? result : old out |
| 2 | zero: lane i ← r2[i] ? result : 0 |

- mode 1 reads `out`. for the kinds that already read `out` (`a3`, `a3i`, `ai`) the old `out` is the accumulator
- results narrower than 512 bits (the narrowing ops, `extract128`, `extract256`, `gather.qd`): the mask applies to the result lanes, the bytes above stay 0 in every mode
- the op tables give the allowed modes: `m:G` = 0, 1, 2 at granularity `G` (`b`, `w`, `d`, `q`, or `W` = the width in imm8), `M:W` = 1 or 2, `S:G` = 0 or 1, `r2` = mode 0, `r2` is an operand. none = mode 0 only

mask modes of the ops in the first op table below (mode 0 only for `ld`, `st`, `zero`, `mov`, `cmpeq.*`, `cmpgt.*`, `cmpgtu.q`, `hsum.*`, `extract.*`, `insert.q`, `movemask.b`, `clmul`):

| ops | mask |
|---|---|
| `splat.*`, `ldbc.d`, `ldbc.q`, `add.*`, `sub.*`, `mullo.*`, `max*`/`min*`, `shli.*`, `shri.*`, `sari.*`, `roli.*`, `shlv.*`, `shrv.*`, `rolv.*` | m: own width |
| `mulwu`, `mulws`, `madd52lo/hi`, `and`, `or`, `xor`, `andn`, `ternlog`, `permq`, `popcnt.q`, `zx.d.q`, `sx.d.q` | m:q |
| `mulhi.w`, `mulhiu.w`, `maddubs`, `packs.d`, `perm2.w`, `narrows.d.w` | m:w |
| `maddwd`, `dpbusd`, `dpwssd`, `permd`, `zx.*.d`, `sx.*.d` | m:d |
| `packs.w`, `packus.w`, `shufb`, `popcnt.b`, `narrow*.d.b` | m:b |

## kinds (which fields are used)

| kind | reads | writes | imm8 |
|---|---|---|---|
| `r3` | in0, in1 | out | 0 |
| `a3` | out, in0, in1 | out | 0 (acc: out is also an input) |
| `a3i` | out, in0, in1 | out | per op |
| `r3i` | in0, in1 | out | per op |
| `r2` | in0 | out | 0 |
| `r2i` | in0 | out | per op |
| `r0` | — | out | 0 |
| `ls` | r1 | out | 0 (load/splat from scalar) |
| `st` | in0, r1 | memory | 0 |
| `sr` | in0 | r0 | 0 |
| `sri` | in0 | r0 | per op |
| `ai` | out, r1 | out | per op |
| `s3` | in0, in1 | r0 | 0 |
| `s3i` | in0, in1 | r0 | per op |
| `a2` | out, in0 | out | 0 |
| `a2i` | out, in0 | out | per op |
| `lsi` | r1, memory | out (`ldn`: out..out+n-1) | per op |
| `sti` | in0 (`stn`: in0..in0+n-1), r1 | memory | per op |
| `ga` | in0 (index), r1, memory | out | log2 scale |
| `sc` | in0 (index), in1 (data), r1 | memory | log2 scale |
| `s0` | r2 | r0 | 0 |

unused register fields must be 0.

## ops

intrinsic args map to `in0`, `in1` in order. for `a3`/`a3i`/`ai` the first arg is the accumulator (`out`). avx-512 column is the jit lowering (intel operand order: out, in0, in1).

| op | name | kind | semantics | avx-512 | intrinsic |
|---|---|---|---|---|---|
| 0x00 | ld | ls | out ← mem[r1..r1+64] | vmovdqu64 | `_svm512_loadu_si512(p)` |
| 0x01 | st | st | mem[r1..r1+64] ← in0 | vmovdqu64 | `_svm512_storeu_si512(p, a)` |
| 0x02 | ldbc.d | ls | out.d[i] ← mem32[r1] | vpbroadcastd m32 | `_svm512_loadbc_epi32(p)` |
| 0x03 | ldbc.q | ls | out.q[i] ← mem64[r1] | vpbroadcastq m64 | `_svm512_loadbc_epi64(p)` |
| 0x04 | splat.b | ls | out.b[i] ← r1[7:0] | vpbroadcastb r32 | `_svm512_set1_epi8(x)` |
| 0x05 | splat.w | ls | out.w[i] ← r1[15:0] | vpbroadcastw r32 | `_svm512_set1_epi16(x)` |
| 0x06 | splat.d | ls | out.d[i] ← r1[31:0] | vpbroadcastd r32 | `_svm512_set1_epi32(x)` |
| 0x07 | splat.q | ls | out.q[i] ← r1 | vpbroadcastq r64 | `_svm512_set1_epi64(x)` |
| 0x08 | zero | r0 | out ← 0 | vpxord | `_svm512_setzero_si512()` |
| 0x09 | mov | r2 | out ← in0 | vmovdqa64 | (pass-internal) |
| 0x10–0x13 | add.b/w/d/q | r3 | wrapping | vpaddb/w/d/q | `_svm512_add_epi8/16/32/64` |
| 0x14–0x17 | sub.b/w/d/q | r3 | wrapping, in0 − in1 | vpsubb/w/d/q | `_svm512_sub_epi8/16/32/64` |
| 0x18 | mullo.w | r3 | low 16 | vpmullw | `_svm512_mullo_epi16` |
| 0x19 | mullo.d | r3 | low 32 | vpmulld | `_svm512_mullo_epi32` |
| 0x1a | mullo.q | r3 | low 64 | vpmullq | `_svm512_mullo_epi64` |
| 0x1b | mulwu | r3 | out.q[i] ← u32(in0.d[2i]) · u32(in1.d[2i]) | vpmuludq | `_svm512_mul_epu32` |
| 0x1c | mulws | r3 | out.q[i] ← i32(in0.d[2i]) · i32(in1.d[2i]) | vpmuldq | `_svm512_mul_epi32` |
| 0x1d | mulhi.w | r3 | high 16 of the signed product | vpmulhw | `_svm512_mulhi_epi16` |
| 0x1e | mulhiu.w | r3 | high 16 of the unsigned product | vpmulhuw | `_svm512_mulhi_epu16` |
| 0x20 | max.d | r3 | signed | vpmaxsd | `_svm512_max_epi32` |
| 0x21 | min.d | r3 | signed | vpminsd | `_svm512_min_epi32` |
| 0x22 | maxu.d | r3 | unsigned | vpmaxud | `_svm512_max_epu32` |
| 0x23 | minu.d | r3 | unsigned | vpminud | `_svm512_min_epu32` |
| 0x24 | max.w | r3 | signed | vpmaxsw | `_svm512_max_epi16` |
| 0x25 | min.w | r3 | signed | vpminsw | `_svm512_min_epi16` |
| 0x26 | maxu.q | r3 | unsigned | vpmaxuq | `_svm512_max_epu64` |
| 0x27 | minu.q | r3 | unsigned | vpminuq | `_svm512_min_epu64` |
| 0x28 | and | r3 | | vpandq | `_svm512_and_si512` |
| 0x29 | or | r3 | | vporq | `_svm512_or_si512` |
| 0x2a | xor | r3 | | vpxorq | `_svm512_xor_si512` |
| 0x2b | andn | r3 | ~in0 & in1 | vpandnq | `_svm512_andnot_si512` |
| 0x2c | ternlog | a3i | out.bit ← imm8[(out.bit << 2) \| (in0.bit << 1) \| in1.bit], imm8 any | vpternlogq | `_svm512_ternarylogic_epi64::<IMM>(a, b, c)` |
| 0x2e | maxu.w | r3 | unsigned | vpmaxuw | `_svm512_max_epu16` |
| 0x2f | minu.w | r3 | unsigned | vpminuw | `_svm512_min_epu16` |
| 0x30–0x32 | shli.w/d/q | r2i | shl by imm8, imm8 < lane bits | vpsllw/d/q imm | `_svm512_slli_epi16/32/64::<IMM>` |
| 0x33–0x35 | shri.w/d/q | r2i | logical shr, imm8 < lane bits | vpsrlw/d/q imm | `_svm512_srli_epi16/32/64::<IMM>` |
| 0x36–0x38 | sari.w/d/q | r2i | arithmetic shr, imm8 < lane bits | vpsraw/d/q imm | `_svm512_srai_epi16/32/64::<IMM>` |
| 0x39 | roli.d | r2i | rotate left, imm8 < 32 | vprold | `_svm512_rol_epi32::<IMM>` |
| 0x3a | roli.q | r2i | rotate left, imm8 < 64 | vprolq | `_svm512_rol_epi64::<IMM>` |
| 0x3b | shlv.d | r3 | per lane in0 << in1, count ≥ 32 → 0 | vpsllvd | `_svm512_sllv_epi32` |
| 0x3c | shlv.q | r3 | count ≥ 64 → 0 | vpsllvq | `_svm512_sllv_epi64` |
| 0x3d | shrv.d | r3 | logical, count ≥ 32 → 0 | vpsrlvd | `_svm512_srlv_epi32` |
| 0x3e | shrv.q | r3 | logical, count ≥ 64 → 0 | vpsrlvq | `_svm512_srlv_epi64` |
| 0x3f | rolv.d | r3 | rotate left by in1 mod 32 | vprolvd | `_svm512_rolv_epi32` |
| 0x40 | rolv.q | r3 | rotate left by in1 mod 64 | vprolvq | `_svm512_rolv_epi64` |
| 0x48 | cmpeq.b | r3 | lane ← all ones if equal else 0 | vpcmpeqb k + vpmovm2b | `_svm512_cmpeq_epi8` |
| 0x49 | cmpeq.d | r3 | | vpcmpeqd k + vpmovm2d | `_svm512_cmpeq_epi32` |
| 0x4a | cmpeq.q | r3 | | vpcmpeqq k + vpmovm2q | `_svm512_cmpeq_epi64` |
| 0x4b | cmpgt.d | r3 | signed in0 > in1 | vpcmpgtd k + vpmovm2d | `_svm512_cmpgt_epi32` |
| 0x4c | cmpgt.q | r3 | signed | vpcmpgtq k + vpmovm2q | `_svm512_cmpgt_epi64` |
| 0x4d | cmpgtu.q | r3 | unsigned | vpcmpuq k (nle) + vpmovm2q | `_svm512_cmpgt_epu64` |
| 0x50 | dpbusd | a3 | out.d[i] += Σj<4 u8(in0.b[4i+j]) · s8(in1.b[4i+j]), wrapping | vpdpbusd | `_svm512_dpbusd_epi32(acc, a, b)` |
| 0x51 | dpwssd | a3 | out.d[i] += Σj<2 s16(in0.w[2i+j]) · s16(in1.w[2i+j]), wrapping | vpdpwssd | `_svm512_dpwssd_epi32(acc, a, b)` |
| 0x52 | maddubs | r3 | out.w[i] ← sat16(Σj<2 u8(in0.b[2i+j]) · s8(in1.b[2i+j])) | vpmaddubsw | `_svm512_maddubs_epi16` |
| 0x53 | maddwd | r3 | out.d[i] ← Σj<2 s16 · s16, wrapping | vpmaddwd | `_svm512_madd_epi16` |
| 0x54 | madd52lo | a3 | out.q[i] += low52(in0.q[i][51:0] · in1.q[i][51:0]) | vpmadd52luq | `_svm512_madd52lo_epu64(acc, a, b)` |
| 0x55 | madd52hi | a3 | out.q[i] += high52(same product) | vpmadd52huq | `_svm512_madd52hi_epu64(acc, a, b)` |
| 0x58 | packs.d | r3 | per 128-bit lane: [sat16(in0.d×4), sat16(in1.d×4)] | vpackssdw | `_svm512_packs_epi32` |
| 0x59 | packs.w | r3 | per 128-bit lane: [sat8(in0.w×8), sat8(in1.w×8)] | vpacksswb | `_svm512_packs_epi16` |
| 0x5a | packus.w | r3 | same, unsigned sat (neg → 0) | vpackuswb | `_svm512_packus_epi16` |
| 0x5b | permd | r3 | out.d[i] ← in1.d[in0.d[i] & 15] | vpermd (out, idx=in0, data=in1) | `_svm512_permutexvar_epi32(idx, a)` |
| 0x5c | permq | r3 | out.q[i] ← in1.q[in0.q[i] & 7] | vpermq (out, idx=in0, data=in1) | `_svm512_permutexvar_epi64(idx, a)` |
| 0x5d | shufb | r3 | per 128-bit lane L: out.b[i] ← in1.b[i] bit 7 ? 0 : in0.b[16L + (in1.b[i] & 15)] | vpshufb | `_svm512_shuffle_epi8(a, b)` |
| 0x5e | perm2.w | a3 | k = in0.w[i] & 63, out.w[i] ← k < 32 ? out.w[k] : in1.w[k − 32] (old out) | vpermt2w (out = table 1, idx = in0) | `_svm512_permutex2var_epi16(a, idx, b)` |
| 0x60 | narrowus.d.b | r2 | out.b[i] ← min(u32(in0.d[i]), 255) for i < 16, rest of out 0 | vpmovusdb | `_svm512_cvtusepi32_epi8` |
| 0x61 | narrows.d.b | r2 | out.b[i] ← sat8(i32) for i < 16, rest 0 | vpmovsdb | `_svm512_cvtsepi32_epi8` |
| 0x62 | narrows.d.w | r2 | out.w[i] ← sat16(i32) for i < 16, rest 0 | vpmovsdw | `_svm512_cvtsepi32_epi16` |
| 0x63 | narrow.d.b | r2 | out.b[i] ← truncate for i < 16, rest 0 | vpmovdb | `_svm512_cvtepi32_epi8` |
| 0x64 | zx.b.d | r2 | out.d[i] ← u8(in0.b[i]), i < 16 | vpmovzxbd | `_svm512_cvtepu8_epi32` |
| 0x65 | sx.b.d | r2 | out.d[i] ← i8(in0.b[i]) | vpmovsxbd | `_svm512_cvtepi8_epi32` |
| 0x66 | zx.w.d | r2 | out.d[i] ← u16(in0.w[i]) | vpmovzxwd | `_svm512_cvtepu16_epi32` |
| 0x67 | sx.w.d | r2 | out.d[i] ← i16(in0.w[i]) | vpmovsxwd | `_svm512_cvtepi16_epi32` |
| 0x68 | zx.d.q | r2 | out.q[i] ← u32(in0.d[i]), i < 8 | vpmovzxdq | `_svm512_cvtepu32_epi64` |
| 0x69 | sx.d.q | r2 | out.q[i] ← i32(in0.d[i]) | vpmovsxdq | `_svm512_cvtepi32_epi64` |
| 0x6a | popcnt.b | r2 | per byte | vpopcntb | `_svm512_popcnt_epi8` |
| 0x6b | popcnt.q | r2 | per u64 | vpopcntq | `_svm512_popcnt_epi64` |
| 0x70 | hsum.d | sr | r0 ← zext(Σ in0.d[i] mod 2^32) | extract/add tree on zmm30/31 | `_svm512_reduce_add_epi32` (→ i32) |
| 0x71 | hsum.q | sr | r0 ← Σ in0.q[i] mod 2^64 | extract/add tree | `_svm512_reduce_add_epi64` (→ i64) |
| 0x72 | extract.q | sri | r0 ← in0.q[imm8], imm8 < 8 | vextracti32x4 + vpextrq | `_svm512_extract_epi64::<IMM>` |
| 0x73 | extract.d | sri | r0 ← zext(in0.d[imm8]), imm8 < 16 | vextracti32x4 + vpextrd | `_svm512_extract_epi32::<IMM>` |
| 0x74 | insert.q | ai | out.q[imm8] ← r1, imm8 < 8 | (any correct sequence) | `_svm512_insert_epi64::<IMM>(a, x)` |
| 0x75 | movemask.b | sr | r0 ← bit i = sign bit of in0.b[i] | vpmovb2m k + kmovq | `_svm512_movepi8_mask` (→ u64) |
| 0x78 | clmul | r3i | per 128-bit lane L: out.lane ← clmul(in0.q[2L + (imm8 & 1)], in1.q[2L + (imm8 >> 4 & 1)]), imm8 ∈ {0x00, 0x01, 0x10, 0x11} | vpclmulqdq | `_svm512_clmulepi64_epi128::<IMM>` |

further ops, with the mask annotation in the kind column:

| op | name | kind | semantics | avx-512 | intrinsic | cu |
|---|---|---|---|---|---|---|
| 0xf6 | mpopcnt | s0 r2 | r0 ← popcount(r2) | popcnt rax, rdx | `_svm512_popcnt_mask(k)` | 1 |
| 0xf7 | mtzcnt | s0 r2 | r0 ← trailing zeros of r2, 64 for 0 | tzcnt | `_svm512_tzcnt_mask(k)` | 1 |
| 0xf8 | mlzcnt | s0 r2 | r0 ← leading zeros of r2, 64 for 0 | lzcnt | `_svm512_lzcnt_mask(k)` | 1 |
| 0x79 | blendm | r3i r2 | imm8 = W < 4. out.W[i] ← r2[i] ? in1.W[i] : in0.W[i] | vpblendm{b,w,d,q} | `_svm512_mask_blend_epi8/16/32/64(k, a, b)` | 1 |
| 0x7a | bcast.q | r2i m:q | imm8 < 8. out.q[i] ← in0.q[imm8] | imm8 = 0: vpbroadcastq out, xmm_in0. else valignq zmm30, in0, in0, imm8; vpbroadcastq out, xmm30 | `_svm512_broadcast_lane_epi64::<L>(a)`, `_svm512_broadcastq_epi64(a)` (L = 0) | 1 (imm8 = 0) / 2 |
| 0x7b | bcast.d | r2i m:d | imm8 < 16. out.d[i] ← in0.d[imm8] | as bcast.q with valignd / vpbroadcastd | `_svm512_broadcast_lane_epi32::<L>`, `_svm512_broadcastd_epi32` | 1 / 2 |
| 0x7c | bcast.w | r2i m:w | imm8 < 32. out.w[i] ← in0.w[imm8] | imm8 ≥ 8: vextracti32x4 xmm30, in0, imm8 >> 3. imm8 & 7 ≠ 0: vpsrldq xmm30, xmm, 2·(imm8 & 7). vpbroadcastw out, xmm | `_svm512_broadcast_lane_epi16::<L>`, `_svm512_broadcastw_epi16` | 1 + (imm8 ≥ 8) + (imm8 & 7 ≠ 0) |
| 0x7d | bcast.b | r2i m:b | imm8 < 64. out.b[i] ← in0.b[imm8] | as bcast.w: lane imm8 >> 4, byte shift imm8 & 15, vpbroadcastb | `_svm512_broadcast_lane_epi8::<L>`, `_svm512_broadcastb_epi8` | 1 + (imm8 ≥ 16) + (imm8 & 15 ≠ 0) |
| 0x7e | permqi | r2i m:q | imm8 any. out.q[4h + i] ← in0.q[4h + imm8[2i+1:2i]], h < 2, i < 4 | vpermq out, in0, imm8 | `_svm512_permutex_epi64::<IMM>(a)` | 1 |
| 0x7f | extract256 | r2i m:q | imm8 < 2. out.q[i] ← in0.q[4·imm8 + i] for i < 4, out.q[4..8] ← 0 | vextracti64x4 ymm_out, in0, imm8 | `_svm512_extracti64x4_epi64::<IMM>(a)` (result in the low half) | 1 |
| 0x5f | perm2i | a3i m:W | imm8 = W < 4. L = 64 >> W, k = out.W[i] & (2L − 1) (old out = index): out.W[i] ← k < L ? in0.W[k] : in1.W[k − L] | vpermi2{b,w,d,q} out, in0, in1 | `_svm512_mask2_permutex2var_epi8/16/32/64(a, idx, k, b)`, and `_svm512_permutex2var_*` / `_maskz_` where the pass picks it | 1 |
| 0x80 | align.q | r3i m:q | imm8 < 8. t[j] = j < 8 ? in1.q[j] : in0.q[j − 8], out.q[i] ← t[i + imm8] | valignq out, in0, in1, imm8 | `_svm512_alignr_epi64::<IMM>(a, b)` | 1 |
| 0x81 | align.d | r3i m:d | imm8 < 16. t[j] = j < 16 ? in1.d[j] : in0.d[j − 16], out.d[i] ← t[i + imm8] | valignd | `_svm512_alignr_epi32::<IMM>(a, b)` | 1 |
| 0x82 | extract128 | r2i m:d | imm8 < 4. out.x[0] ← in0.x[imm8], rest 0 | vextracti32x4 xmm_out, in0, imm8 | `_svm512_extracti32x4_epi32::<IMM>(a)` (mask `__mmask8`) | 1 |
| 0x83 | insert256 | r3i m:q | imm8 < 2. out ← in0, then out.y[imm8] ← in1.y[0] | vinserti64x4 out, in0, ymm_in1, imm8 | `_svm512_inserti64x4::<IMM>(a, b)` | 1 |
| 0x84 | insert128 | r3i m:d | imm8 < 4. out ← in0, then out.x[imm8] ← in1.x[0] | vinserti32x4 out, in0, xmm_in1, imm8 | `_svm512_inserti32x4::<IMM>(a, b)` | 1 |
| 0x85 | perm2t | a3i m:W | imm8 = W ∈ {0, 2, 3} (W = 1 is perm2.w). k = in0.W[i] & (2L − 1): out.W[i] ← k < L ? out.W[k] : in1.W[k − L] (old out = table 1) | vpermt2{b,d,q} out, in0, in1 | `_svm512_permutex2var_epi8/32/64(a, idx, b)`, `_svm512_mask_permutex2var_*(a, k, idx, b)`, `_svm512_maskz_permutex2var_*(k, a, idx, b)` | 1 |
| 0x86 | permb | r3 m:b | out.b[i] ← in1.b[in0.b[i] & 63] | vpermb out, in0 (idx), in1 | `_svm512_permutexvar_epi8(idx, a)` | 1 |
| 0x87 | permw | r3 m:w | out.w[i] ← in1.w[in0.w[i] & 31] | vpermw out, in0 (idx), in1 | `_svm512_permutexvar_epi16(idx, a)` | 1 |
| 0x88 | shufd | r2i m:d | imm8 any. out.d[4L + i] ← in0.d[4L + imm8[2i+1:2i]] | vpshufd | `_svm512_shuffle_epi32::<IMM>(a)` | 1 |
| 0x89 | shufhw | r2i m:w | imm8 any. per 128-bit lane L: out.w[8L + i] ← in0.w[8L + i], out.w[8L + 4 + i] ← in0.w[8L + 4 + imm8[2i+1:2i]], i < 4 | vpshufhw | `_svm512_shufflehi_epi16::<IMM>(a)` | 1 |
| 0x8a | shuflw | r2i m:w | imm8 any. per lane L: out.w[8L + i] ← in0.w[8L + imm8[2i+1:2i]], out.w[8L + 4 + i] ← in0.w[8L + 4 + i], i < 4 | vpshuflw | `_svm512_shufflelo_epi16::<IMM>(a)` | 1 |
| 0x8b | shuf128 | r3i m:q | imm8 any. out.x[0] ← in0.x[imm8[1:0]], out.x[1] ← in0.x[imm8[3:2]], out.x[2] ← in1.x[imm8[5:4]], out.x[3] ← in1.x[imm8[7:6]] | vshufi64x2 out, in0, in1, imm8 | `_svm512_shuffle_i64x2::<IMM>(a, b)` | 1 |
| 0x8c | shufps | r3i m:d | imm8 any. per lane L: out.d[4L] ← in0.d[4L + imm8[1:0]], out.d[4L+1] ← in0.d[4L + imm8[3:2]], out.d[4L+2] ← in1.d[4L + imm8[5:4]], out.d[4L+3] ← in1.d[4L + imm8[7:6]] | vshufps (pure bit movement, no MXCSR or NaN effect) | `_svm512_shuffle_ps_epi32::<IMM>(a, b)` | 1 |
| 0x8d | alignr | r3i m:b | imm8 < 32. per lane L: t = in0.x[L] · 2^128 + in1.x[L], out.x[L] ← (t >> 8·imm8) mod 2^128 | vpalignr | `_svm512_alignr_epi8::<IMM>(a, b)` | 1 |
| 0x8e | bshli | r2i | imm8 < 16. out.x[L] ← (in0.x[L] << 8·imm8) mod 2^128 | vpslldq | `_svm512_bslli_epi128::<IMM>(a)` | 1 |
| 0x8f | bshri | r2i | imm8 < 16. out.x[L] ← in0.x[L] >> 8·imm8 | vpsrldq | `_svm512_bsrli_epi128::<IMM>(a)` | 1 |
| 0x90 | unpacklo | r3i m:W | imm8 = W < 4. m = 16 >> W lanes per 128-bit lane L: out.W[mL + 2j] ← in0.W[mL + j], out.W[mL + 2j + 1] ← in1.W[mL + j], j < m/2 | vpunpckl{bw,wd,dq,qdq} | `_svm512_unpacklo_epi8/16/32/64(a, b)` | 1 |
| 0x91 | unpackhi | r3i m:W | same with source lane mL + m/2 + j | vpunpckh{bw,wd,dq,qdq} | `_svm512_unpackhi_epi8/16/32/64(a, b)` | 1 |
| 0x92 | compress | r2i M:W | imm8 = W < 4. j ← 0. for i < L ascending, if r2[i]: out.W[j] ← in0.W[i], j += 1. lanes ≥ j: old out (mode 1) or 0 (mode 2) | vpcompress{b,w,d,q} out{k2}[{z}], in0 | `_svm512_mask_compress_epi8/16/32/64(src, k, a)`, `_svm512_maskz_compress_*(k, a)` | 2 |
| 0x93 | expand | r2i M:W | imm8 = W < 4. j ← 0. for i < L ascending: if r2[i]: out.W[i] ← in0.W[j], j += 1, else old out (mode 1) or 0 (mode 2) | vpexpand{b,w,d,q} out{k2}[{z}], in0 | `_svm512_mask_expand_*(src, k, a)`, `_svm512_maskz_expand_*(k, a)` | 2 |
| 0x94 | packus.d | r3 m:w | per lane L: out.w[8L + k] ← clamp(i32(in0.d[4L + k]), 0, 65535), out.w[8L + 4 + k] ← same of in1, k < 4 | vpackusdw | `_svm512_packus_epi32(a, b)` | 1 |
| 0x95 | narrow.d.w | r2 m:w | out.w[i] ← in0.d[i] mod 2^16, i < 16, rest 0 | vpmovdw ymm_out, in0 | `_svm512_cvtepi32_epi16(a)` | 1 |
| 0x96 | narrowus.d.w | r2 m:w | out.w[i] ← min(u32(in0.d[i]), 2^16 − 1), rest 0 | vpmovusdw | `_svm512_cvtusepi32_epi16` | 1 |
| 0x97 | narrow.q.b | r2 m:b | out.b[i] ← in0.q[i] mod 2^8, i < 8, rest 0 | vpmovqb xmm_out, in0 | `_svm512_cvtepi64_epi8` | 1 |
| 0x98 | narrows.q.b | r2 m:b | out.b[i] ← clamp(i64(in0.q[i]), −2^7, 2^7 − 1), rest 0 | vpmovsqb | `_svm512_cvtsepi64_epi8` | 1 |
| 0x99 | narrowus.q.b | r2 m:b | out.b[i] ← min(u64(in0.q[i]), 2^8 − 1), rest 0 | vpmovusqb | `_svm512_cvtusepi64_epi8` | 1 |
| 0x9a | narrow.q.w | r2 m:w | out.w[i] ← in0.q[i] mod 2^16, i < 8, rest 0 | vpmovqw xmm_out, in0 | `_svm512_cvtepi64_epi16` | 1 |
| 0x9b | narrows.q.w | r2 m:w | signed saturation to i16, rest 0 | vpmovsqw | `_svm512_cvtsepi64_epi16` | 1 |
| 0x9c | narrowus.q.w | r2 m:w | min(u64, 2^16 − 1), rest 0 | vpmovusqw | `_svm512_cvtusepi64_epi16` | 1 |
| 0x9d | narrow.q.d | r2 m:d | out.d[i] ← in0.q[i] mod 2^32, i < 8, rest 0 | vpmovqd ymm_out, in0 | `_svm512_cvtepi64_epi32` | 1 |
| 0x9e | narrows.q.d | r2 m:d | signed saturation to i32, rest 0 | vpmovsqd | `_svm512_cvtsepi64_epi32` | 1 |
| 0x9f | narrowus.q.d | r2 m:d | min(u64, 2^32 − 1), rest 0 | vpmovusqd | `_svm512_cvtusepi64_epi32` | 1 |
| 0xa0 | narrow.w.b | r2 m:b | out.b[i] ← in0.w[i] mod 2^8, i < 32, rest 0 | vpmovwb ymm_out, in0 | `_svm512_cvtepi16_epi8` | 1 |
| 0xa1 | narrows.w.b | r2 m:b | clamp(i16, −2^7, 2^7 − 1), rest 0 | vpmovswb | `_svm512_cvtsepi16_epi8` | 1 |
| 0xa2 | narrowus.w.b | r2 m:b | min(u16, 2^8 − 1), rest 0 | vpmovuswb | `_svm512_cvtusepi16_epi8` | 1 |
| 0xa3 | zx.b.w | r2 m:w | out.w[i] ← u8(in0.b[i]), i < 32 | vpmovzxbw out, ymm_in0 | `_svm512_cvtepu8_epi16` | 1 |
| 0xa4 | sx.b.w | r2 m:w | out.w[i] ← i8(in0.b[i]) | vpmovsxbw | `_svm512_cvtepi8_epi16` | 1 |
| 0xa5 | zx.b.q | r2 m:q | out.q[i] ← u8(in0.b[i]), i < 8 | vpmovzxbq out, xmm_in0 | `_svm512_cvtepu8_epi64` | 1 |
| 0xa6 | sx.b.q | r2 m:q | out.q[i] ← i8(in0.b[i]) | vpmovsxbq | `_svm512_cvtepi8_epi64` | 1 |
| 0xa7 | zx.w.q | r2 m:q | out.q[i] ← u16(in0.w[i]), i < 8 | vpmovzxwq | `_svm512_cvtepu16_epi64` | 1 |
| 0xa8 | sx.w.q | r2 m:q | out.q[i] ← i16(in0.w[i]) | vpmovsxwq | `_svm512_cvtepi16_epi64` | 1 |
| 0xa9 | extract.b | sri | imm8 < 64. r0 ← zext(in0.b[imm8]) | vextracti32x4 xmm30, in0, imm8 >> 4 (if ≠ 0); vpextrb eax, xmm, imm8 & 15 | `_svm512_extract_epi8::<IMM>(a)` (→ i32) | 2 |
| 0xaa | extract.w | sri | imm8 < 32. r0 ← zext(in0.w[imm8]) | vextracti32x4 (if imm8 ≥ 8); vpextrw eax, xmm, imm8 & 7 | `_svm512_extract_epi16::<IMM>(a)` (→ i32) | 2 |
| 0xab | insert.b | ai | imm8 < 64. out.b[imm8] ← r1[7:0] | vextracti32x4 xmm30, out, imm8 >> 4; vpinsrb xmm30, xmm30, esi, imm8 & 15; vinserti32x4 out, out, xmm30, imm8 >> 4 | `_svm512_insert_epi8::<IMM>(a, x)` | 3 |
| 0xac | insert.w | ai | imm8 < 32. out.w[imm8] ← r1[15:0] | same with vpinsrw | `_svm512_insert_epi16::<IMM>(a, x)` | 3 |
| 0xad | insert.d | ai | imm8 < 16. out.d[imm8] ← r1[31:0] | same with vpinsrd | `_svm512_insert_epi32::<IMM>(a, x)` | 3 |
| 0xae | movr.q | ls | out.q[0] ← r1, out.q[1..8] ← 0 | vmovq xmm_out, rsi | `_svm512_cvtsi64_si512(x)` | 1 |
| 0xaf | mul52lo | r3 m:q | out.q[i] ← (in0.q[i][51:0] · in1.q[i][51:0]) mod 2^52 | vpmadd52luq into a zeroed register | `_svm512_mul52lo_epu64(a, b)` | 2 |
| 0xb0 | mul52hi | r3 m:q | out.q[i] ← bits 103:52 of the same product | vpmadd52huq into a zeroed register | `_svm512_mul52hi_epu64(a, b)` | 2 |
| 0xb1 | abs.q | r2 m:q | out.q[i] ← \|i64\| mod 2^64 (abs(MIN) = MIN) | vpabsq | `_svm512_abs_epi64(a)` | 1 |
| 0xb2 | max.q | r3 m:q | signed | vpmaxsq | `_svm512_max_epi64` | 1 |
| 0xb3 | min.q | r3 m:q | signed | vpminsq | `_svm512_min_epi64` | 1 |
| 0xb4 | abs.b | r2 m:b | as abs.q, 8 bit | vpabsb | `_svm512_abs_epi8(a)` | 1 |
| 0xb5 | abs.w | r2 m:w | 16 bit | vpabsw | `_svm512_abs_epi16(a)` | 1 |
| 0xb6 | abs.d | r2 m:d | 32 bit | vpabsd | `_svm512_abs_epi32(a)` | 1 |
| 0xb7 | max.b | r3 m:b | signed | vpmaxsb | `_svm512_max_epi8` | 1 |
| 0xb8 | min.b | r3 m:b | signed | vpminsb | `_svm512_min_epi8` | 1 |
| 0xb9 | maxu.b | r3 m:b | unsigned | vpmaxub | `_svm512_max_epu8` | 1 |
| 0xba | minu.b | r3 m:b | unsigned | vpminub | `_svm512_min_epu8` | 1 |
| 0xbb | adds.b | r3 m:b | sat8(i8 + i8) | vpaddsb | `_svm512_adds_epi8` | 1 |
| 0xbc | adds.w | r3 m:w | sat16(i16 + i16) | vpaddsw | `_svm512_adds_epi16` | 1 |
| 0xbd | addus.b | r3 m:b | min(u8 + u8, 255) | vpaddusb | `_svm512_adds_epu8` | 1 |
| 0xbe | addus.w | r3 m:w | min(u16 + u16, 65535) | vpaddusw | `_svm512_adds_epu16` | 1 |
| 0xbf | subs.b | r3 m:b | sat8(in0 − in1), signed | vpsubsb | `_svm512_subs_epi8` | 1 |
| 0xc0 | subs.w | r3 m:w | sat16(in0 − in1), signed | vpsubsw | `_svm512_subs_epi16` | 1 |
| 0xc1 | subus.b | r3 m:b | max(in0 − in1, 0), unsigned | vpsubusb | `_svm512_subs_epu8` | 1 |
| 0xc2 | subus.w | r3 m:w | max(in0 − in1, 0), unsigned | vpsubusw | `_svm512_subs_epu16` | 1 |
| 0xc3 | avg.b | r3 m:b | (u8 + u8 + 1) >> 1, exact | vpavgb | `_svm512_avg_epu8` | 1 |
| 0xc4 | avg.w | r3 m:w | (u16 + u16 + 1) >> 1, exact | vpavgw | `_svm512_avg_epu16` | 1 |
| 0xc5 | sad | r3 | out.q[j] ← Σk<8 \|u8(in0.b[8j+k]) − u8(in1.b[8j+k])\| | vpsadbw | `_svm512_sad_epu8(a, b)` | 1 |
| 0x1f | mulhrs.w | r3 m:w | out.w[i] ← (((i16 · i16) >> 14) + 1) >> 1, low 16 bits | vpmulhrsw | `_svm512_mulhrs_epi16` | 1 |
| 0x56 | dpbusds | a3 m:d | as dpbusd, the exact sum saturated to i32 | vpdpbusds | `_svm512_dpbusds_epi32(acc, a, b)` | 1 |
| 0x57 | dpwssds | a3 m:d | as dpwssd, the exact sum saturated to i32 | vpdpwssds | `_svm512_dpwssds_epi32(acc, a, b)` | 1 |
| 0x46 | cmp | s3i | imm8[2:0] = P ∈ {0 eq, 1 lt, 2 le, 4 ne, 5 ge, 6 gt}, imm8[3] = unsigned, imm8[5:4] = W, imm8[7:6] = 0. r0 bit i ← P(in0.W[i], in1.W[i]), bits ≥ lane count 0 | vpcmp[u]{b,w,d,q} k1 + kmovq | `_svm512_cmp_epi8/16/32/64_mask::<P>(a, b)`, `_svm512_cmp_epu*_mask::<P>`, `_svm512_cmp{eq,lt,le,neq,ge,gt}_ep{i,u}*_mask(a, b)` | 2 |
| 0x4f | bitshuf | s3 | r0 bit (8j + k) ← bit (in1.b[8j+k] & 63) of in0.q[j] | vpshufbitqmb k1 + kmovq | `_svm512_bitshuffle_epi64_mask(b, c)` | 2 |
| 0x47 | test | s3i | imm8 = W < 4. r0 bit i ← (in0.W[i] & in1.W[i]) ≠ 0, bits ≥ lane count 0 | vptestm{b,w,d,q} k1 + kmovq | `_svm512_test_epi8/16/32/64_mask(a, b)` | 2 |
| 0x4e | testn | s3i | imm8 = W < 4. r0 bit i ← (in0.W[i] & in1.W[i]) = 0 | vptestnm{b,w,d,q} k1 + kmovq | `_svm512_testn_epi8/16/32/64_mask(a, b)` | 2 |
| 0x76 | movemask.w | sr | r0 bit i ← bit 15 of in0.w[i] | vpmovw2m k1 + kmovq | `_svm512_movepi16_mask(a)` | 2 |
| 0x77 | movemask.d | sr | r0 bit i ← bit 31 of in0.d[i] | vpmovd2m k1 + kmovq | `_svm512_movepi32_mask(a)` | 2 |
| 0xc6 | movemask.q | sr | r0 bit i ← bit 63 of in0.q[i] | vpmovq2m k1 + kmovq | `_svm512_movepi64_mask(a)` | 2 |
| 0xc7 | movm.b | r0 r2 | out.b[i] ← r2[i] ? 0xff : 0 | kmovq k2 + vpmovm2b | `_svm512_movm_epi8(k)` | 1 |
| 0xc8 | movm.w | r0 r2 | out.w[i] ← r2[i] ? 0xffff : 0 | vpmovm2w | `_svm512_movm_epi16(k)` | 1 |
| 0xc9 | movm.d | r0 r2 | out.d[i] ← r2[i] ? all ones : 0 | vpmovm2d | `_svm512_movm_epi32(k)` | 1 |
| 0xca | movm.q | r0 r2 | out.q[i] ← r2[i] ? all ones : 0 | vpmovm2q | `_svm512_movm_epi64(k)` | 1 |
| 0x2d | ternlog.d | a3i m:d | as ternlog, mask granularity d | vpternlogd | `_svm512_ternarylogic_epi32::<IMM>(a, b, c)` | 1 |
| 0x41 | shlv.w | r3 m:w | c = u(in1.w[i]): c < 16 ? in0.w[i] << c : 0 | vpsllvw | `_svm512_sllv_epi16(a, b)` | 1 |
| 0x42 | shrv.w | r3 m:w | c < 16 ? in0.w[i] >> c : 0 | vpsrlvw | `_svm512_srlv_epi16(a, b)` | 1 |
| 0x43 | sarv.w | r3 m:w | s(in0.w[i]) >> min(u(in1.w[i]), 15) | vpsravw | `_svm512_srav_epi16(a, b)` | 1 |
| 0x44 | sarv.d | r3 m:d | s(in0.d[i]) >> min(u(in1.d[i]), 31) | vpsravd | `_svm512_srav_epi32(a, b)` | 1 |
| 0x45 | sarv.q | r3 m:q | s(in0.q[i]) >> min(u(in1.q[i]), 63) | vpsravq | `_svm512_srav_epi64(a, b)` | 1 |
| 0xce | shldi.w | r3i m:w | imm8 = n < 16. out.w[i] ← high 16 of (in0.w[i] · 2^16 + in1.w[i]) << n | vpshldw | `_svm512_shldi_epi16::<IMM>(a, b)` | 1 |
| 0xcf | shldi.d | r3i m:d | n < 32, 32-bit lanes | vpshldd | `_svm512_shldi_epi32::<IMM>(a, b)` | 1 |
| 0xd0 | shldi.q | r3i m:q | n < 64, 64-bit lanes | vpshldq | `_svm512_shldi_epi64::<IMM>(a, b)` | 1 |
| 0xd1 | shrdi.w | r3i m:w | imm8 = n < 16. out.w[i] ← low 16 of (in1.w[i] · 2^16 + in0.w[i]) >> n | vpshrdw | `_svm512_shrdi_epi16::<IMM>(a, b)` | 1 |
| 0xd2 | shrdi.d | r3i m:d | n < 32 | vpshrdd | `_svm512_shrdi_epi32::<IMM>(a, b)` | 1 |
| 0xd3 | shrdi.q | r3i m:q | n < 64 | vpshrdq | `_svm512_shrdi_epi64::<IMM>(a, b)` | 1 |
| 0xd4 | shldv.w | a3 m:w | c = in1.w[i] mod 16: out.w[i] ← high 16 of (out.w[i] · 2^16 + in0.w[i]) << c | vpshldvw | `_svm512_shldv_epi16(a, b, c)` (a = out) | 1 |
| 0xd5 | shldv.d | a3 m:d | c = in1.d[i] mod 32 | vpshldvd | `_svm512_shldv_epi32(a, b, c)` | 1 |
| 0xd6 | shldv.q | a3 m:q | c = in1.q[i] mod 64 | vpshldvq | `_svm512_shldv_epi64(a, b, c)` | 1 |
| 0xd7 | shrdv.w | a3 m:w | c = in1.w[i] mod 16: out.w[i] ← low 16 of (in0.w[i] · 2^16 + out.w[i]) >> c | vpshrdvw | `_svm512_shrdv_epi16(a, b, c)` (a = out) | 1 |
| 0xd8 | shrdv.d | a3 m:d | c = in1.d[i] mod 32 | vpshrdvd | `_svm512_shrdv_epi32(a, b, c)` | 1 |
| 0xd9 | shrdv.q | a3 m:q | c = in1.q[i] mod 64 | vpshrdvq | `_svm512_shrdv_epi64(a, b, c)` | 1 |
| 0xcb | conflict.d | r2 m:d | out.d[i] ← Σj<i [in0.d[j] = in0.d[i]] << j | vpconflictd | `_svm512_conflict_epi32(a)` | 4 |
| 0xcc | conflict.q | r2 m:q | out.q[i] ← Σj<i [in0.q[j] = in0.q[i]] << j | vpconflictq | `_svm512_conflict_epi64(a)` | 3 |
| 0x6e | lzcnt.d | r2 m:d | leading zeros of in0.d[i], 32 for 0 | vplzcntd | `_svm512_lzcnt_epi32(a)` | 1 |
| 0x6f | lzcnt.q | r2 m:q | leading zeros of in0.q[i], 64 for 0 | vplzcntq | `_svm512_lzcnt_epi64(a)` | 1 |
| 0x6c | popcnt.w | r2 m:w | per u16 | vpopcntw | `_svm512_popcnt_epi16(a)` | 1 |
| 0x6d | popcnt.d | r2 m:d | per u32 | vpopcntd | `_svm512_popcnt_epi32(a)` | 1 |
| 0xcd | multishift | r3 m:b | c = in0.b[8j+k] & 63: out.b[8j+k] ← low 8 of in1.q[j] rotated right by c | vpmultishiftqb (control in0, data in1) | `_svm512_multishift_epi64_epi8(a, b)` | 1 |
| 0xe1 | sha256rnds2 | a3 | out.x[0] ← two SHA-256 rounds (Intel SDM SHA256RNDS2) with CDGH = out.x[0], ABEF = in0.x[0], WK = in1.q[0]. bits 511:128 of out unchanged | sha256rnds2 (WK in xmm0) | `_svm512_sha256rnds2_epu32(cdgh, abef, wk)` | 2 |
| 0xe2 | sha256msg1 | a2 | out.x[0] ← SHA256MSG1(out.x[0], in0.x[0]), upper unchanged | sha256msg1 | `_svm512_sha256msg1_epu32(a, b)` | 1 |
| 0xe4 | sha1rnds4 | a2i | imm8 < 4. out.x[0] ← SHA1RNDS4(out.x[0], in0.x[0], imm8), upper unchanged | sha1rnds4 | `_svm512_sha1rnds4_epu32::<F>(a, b)` | 1 |
| 0xe3 | sha256msg2 | a2 | out.x[0] ← SHA256MSG2(out.x[0], in0.x[0]), upper unchanged | sha256msg2 | `_svm512_sha256msg2_epu32(a, b)` | 1 |
| 0xe5 | sha1nexte | a2 | out.x[0] ← SHA1NEXTE(out.x[0], in0.x[0]), upper unchanged | sha1nexte | `_svm512_sha1nexte_epu32(a, b)` | 1 |
| 0xe6 | sha1msg1 | a2 | out.x[0] ← SHA1MSG1(out.x[0], in0.x[0]), upper unchanged | sha1msg1 | `_svm512_sha1msg1_epu32(a, b)` | 1 |
| 0xe7 | sha1msg2 | a2 | out.x[0] ← SHA1MSG2(out.x[0], in0.x[0]), upper unchanged | sha1msg2 | `_svm512_sha1msg2_epu32(a, b)` | 1 |
| 0xda | gf2p8mul | r3 m:b | out.b[i] ← in0.b[i] · in1.b[i] in GF(2^8) mod x^8 + x^4 + x^3 + x + 1 | vgf2p8mulb | `_svm512_gf2p8mul_epi8(a, b)` | 1 |
| 0xdb | gf2p8affine | r3i m:b | imm8 any. x = in0.b[8j+i], A = in1.q[j]: bit t of out.b[8j+i] ← parity(A.b[7 − t] & x) ⊕ imm8 bit t | vgf2p8affineqb | `_svm512_gf2p8affine_epi64_epi8::<B>(x, a)` | 1 |
| 0xdc | gf2p8affineinv | r3i m:b | as gf2p8affine with x replaced by its inverse in GF(2^8) (0 for 0) | vgf2p8affineinvqb | `_svm512_gf2p8affineinv_epi64_epi8::<B>(x, a)` | 1 |
| 0xdd | aesenc | r3 | per 128-bit lane: out ← MixColumns(ShiftRows(SubBytes(in0))) ⊕ in1 (FIPS-197, byte k of a lane is row k mod 4, column k / 4 of the state) | vaesenc | `_svm512_aesenc_epi128(a, round_key)` | 1 |
| 0xde | aesenclast | r3 | per lane: ShiftRows(SubBytes(in0)) ⊕ in1 | vaesenclast | `_svm512_aesenclast_epi128(a, round_key)` | 1 |
| 0xdf | aesdec | r3 | per lane: InvMixColumns(InvShiftRows(InvSubBytes(in0))) ⊕ in1 (equivalent inverse cipher) | vaesdec | `_svm512_aesdec_epi128(a, round_key)` | 1 |
| 0xe0 | aesdeclast | r3 | per lane: InvShiftRows(InvSubBytes(in0)) ⊕ in1 | vaesdeclast | `_svm512_aesdeclast_epi128(a, round_key)` | 1 |
| 0xe8 | gather.qq | ga m:q | imm8 < 4. idx = in0.q[i], out.q[i] ← mem64[A_i] for the active lanes. inactive: old out (mode 1), 0 (mode 2) | vpgatherqq | `_svm512_i64gather_epi64::<SCALE>(idx, p)`, `_svm512_mask_i64gather_epi64::<SCALE>(src, k, idx, p)`, `_svm512_maskz_i64gather_epi64::<SCALE>(k, idx, p)` | 8 |
| 0xec | scatter.qq | sc S:q | imm8 < 4. mem64[A_i] ← in1.q[i] for the active lanes, ascending | vpscatterqq | `_svm512_i64scatter_epi64::<SCALE>(p, idx, a)`, `_svm512_mask_i64scatter_epi64::<SCALE>(p, k, idx, a)` | 8 |
| 0x0a | ld.m | lsi M:W | imm8 = W < 4. out.W[i] ← mem.W[i] where r2[i], else old out (mode 1) or 0 (mode 2) | vmovdqu{8,16,32,64} {k2} | `_svm512_mask_loadu_epi8/16/32/64(src, k, p)`, `_svm512_maskz_loadu_*(k, p)` | 2 |
| 0x0b | st.m | sti M:W | imm8 = W < 4, mode 1 only. mem.W[i] ← in0.W[i] where r2[i], the other bytes are not written | vmovdqu{8,16,32,64} {k2} | `_svm512_mask_storeu_epi8/16/32/64(p, k, a)` | 2 |
| 0x0c | ldn | lsi | imm8 = n, 2 ≤ n, out + n ≤ 30. out+k ← mem[r1 + 64k, 64], k < n | n × vmovdqu64 | `_svm512_loadu_n::<N>(p)`, or fused from `ld`s | n |
| 0x0d | stn | sti | imm8 = n, 2 ≤ n, in0 + n ≤ 30. mem[r1 + 64k, 64] ← in0+k | n × vmovdqu64 | `_svm512_storeu_n::<N>(p, [a; N])`, or fused from `st`s | n |
| 0xe9 | gather.dd | ga m:d | imm8 < 4. idx = sext(in0.d[i]), out.d[i] ← mem32[A_i], 16 lanes | vpgatherdd | `_svm512_i32gather_epi32::<SCALE>(idx, p)`, `_svm512_mask_i32gather_epi32::<SCALE>(src, k, idx, p)`, `_svm512_maskz_i32gather_epi32::<SCALE>(k, idx, p)` | 16 |
| 0xea | gather.dq | ga m:q | imm8 < 4. idx = sext(in0.d[i]) for i < 8 (the low 256 bits of in0), out.q[i] ← mem64[A_i] | vpgatherdq | `_svm512_i32gather_epi64::<SCALE>(idx, p)`, `_svm512_mask_i32gather_epi64::<SCALE>(src, k, idx, p)`, `_svm512_maskz_i32gather_epi64::<SCALE>(k, idx, p)` | 8 |
| 0xeb | gather.qd | ga m:d | imm8 < 4. idx = in0.q[i], out.d[i] ← mem32[A_i] for i < 8. out.d[8..16] ← 0 in every mode | vpgatherqd | `_svm512_i64gather_epi32::<SCALE>(idx, p)`, `_svm512_mask_i64gather_epi32::<SCALE>(src, k, idx, p)`, `_svm512_maskz_i64gather_epi32::<SCALE>(k, idx, p)` | 8 |
| 0xed | scatter.dd | sc S:d | imm8 < 4. idx = sext(in0.d[i]), mem32[A_i] ← in1.d[i], 16 lanes | vpscatterdd | `_svm512_i32scatter_epi32::<SCALE>(p, idx, a)`, `_svm512_mask_i32scatter_epi32::<SCALE>(p, k, idx, a)` | 16 |
| 0xee | scatter.dq | sc S:q | imm8 < 4. idx = sext(in0.d[i]) for i < 8, mem64[A_i] ← in1.q[i] | vpscatterdq | `_svm512_i32scatter_epi64::<SCALE>(p, idx, a)`, `_svm512_mask_i32scatter_epi64::<SCALE>(p, k, idx, a)` | 8 |
| 0xef | scatter.qd | sc S:d | imm8 < 4. idx = in0.q[i], mem32[A_i] ← in1.d[i] for i < 8 | vpscatterqd | `_svm512_i64scatter_epi32::<SCALE>(p, idx, a)`, `_svm512_mask_i64scatter_epi32::<SCALE>(p, k, idx, a)` | 8 |
| 0x0e | ld.y | ls | out bytes 0..32 ← mem[r1, 32], bytes 32..64 ← 0 | vmovdqu64 ymm | `_svm512_loadu_si256(p)` | 1 |
| 0x0f | st.y | st | mem[r1, 32] ← in0 bytes 0..32 | vmovdqu64 ymm | `_svm512_storeu_si256(p, a)` | 1 |
| 0xf0 | ld.x | ls | out bytes 0..16 ← mem[r1, 16], bytes 16..64 ← 0 | vmovdqu32 xmm | `_svm512_loadu_si128(p)` | 1 |
| 0xf1 | st.x | st | mem[r1, 16] ← in0 bytes 0..16 | vmovdqu32 xmm | `_svm512_storeu_si128(p, a)` | 1 |
| 0xf2 | ldbc.b | ls m:b | out.b[i] ← mem8[r1] | vpbroadcastb m8 | `_svm512_loadbc_epi8(p)`, `_svm512_mask_loadbc_epi8(src, k, p)`, `_svm512_maskz_loadbc_epi8(k, p)` | 1 |
| 0xf3 | ldbc.w | ls m:w | out.w[i] ← mem16[r1] | vpbroadcastw m16 | `_svm512_loadbc_epi16(p)` (+ `_mask`, `_maskz`) | 1 |
| 0xf4 | ldbc.x | ls m:d | each 128-bit lane of out ← mem[r1, 16] | vbroadcasti32x4 m128 | `_svm512_loadbc_i32x4(p)` (+ `_mask`, `_maskz`) | 1 |
| 0xf5 | ldbc.y | ls m:q | each 256-bit half of out ← mem[r1, 32] | vbroadcasti64x4 m256 | `_svm512_loadbc_i64x4(p)` (+ `_mask`, `_maskz`) | 1 |

SHA ops use only the low 128 bits (lane 0), with the dword order and pseudo-code of the Intel SDM, and keep bits 511:128 of `out`. `SCALE` of the gather and scatter intrinsics is 1, 2, 4 or 8 (imm8 = log2).

all other op values are reserved and rejected.

## intrinsics abi (what the build pass sees)

- `__svm512i` is `#[repr(transparent)] struct(u32)`, `Copy`. the u32 is an opaque ssa token, never a register number
- every intrinsic is `#[inline(always)]` and calls one marker: `extern "C" fn __svm_<name with '.' → '_'>(...)`, e.g. `__svm_add_d(a: u32, b: u32) -> u32`, `__svm_ld(p: *const u8) -> u32`, `__svm_st(a: u32, p: *mut u8)`, `__svm_splat_d(x: u64) -> u32`, `__svm_hsum_d(a: u32) -> u64`, `__svm_shli_d(a: u32, imm: u32) -> u32`
- argument order: vector args in intrinsic order, then scalar (`r1`) arg, then the mask (`r2`, u64), then imm8 last. imm8 must be an ir constant
- masked forms: `__svm_<name>_mask(src, ..., k)` (mode 1) and `__svm_<name>_maskz(..., k)` (mode 2). `src` is an extra first vector arg (tied to `out`), except for the kinds that already read `out`, whose accumulator is the source. intrinsics: `_svm512_mask_<op>(src, k, ...)` and `_svm512_maskz_<op>(k, ...)` as in `core::arch`, `k` of type `__mmask8/16/32/64` (u8/u16/u32/u64), zero-extended. ops that read `r2` as an operand (`blendm`, `mpopcnt`, ...) take it as a u64 argument in the same position
- ops with an `r0` result (`sr`, `sri`, `s3`, `s3i`, `s0`) return u64
- `ldn`/`stn`: `__svm_ldn(p, k, n) -> u32` is register k of an `ldn` of n registers at p, `__svm_stn(a, p, k, n)` likewise. the pass emits one `ldn`/`stn` for registers k = 0, 1, … in consecutive calls (nothing that accesses memory between them) if it can give them consecutive registers, otherwise single `ld`/`st` at `p + 64k`. it also fuses runs of `ld` (`st`) in the same way: next to each other, from one base through inbounds GEPs (one allocated object, hence one region) at offsets 64 apart
- the pass replaces every marker call with inline asm emitting the encoded op, assigns `v0`–`v29`, and rejects programs where a vector value escapes ssa (stored to memory, passed to or returned from a non-marker call, etc.) with a clear error. phis and `select`s of vectors are allowed (lowered to `mov`s)
- `_svm512_permutex2var_*` (and `_maskz_`) is `perm2t` (`perm2.w` for 16 bits), which ties the first table to `out`. If the index has no other use and the first table does, the pass emits `perm2i` instead, which ties the index
- a program built without the pass fails to link (undefined `__svm_*`)
