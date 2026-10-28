# svm vector ops: poc spec

the contract between the runtime (`sbpf/`), the build pass (`sbpf-linker/`) and the intrinsics (`svm-intrinsics/`). if code and this file disagree, fix one of them.

## registers

- 30 vector registers `v0`–`v29`, 512 bits each, lanes little-endian
- lane widths: `b` = 64×u8, `w` = 32×u16, `d` = 16×u32, `q` = 8×u64
- vm state, not memory mapped. only reachable through vector ops
- zeroed at program entry. global to the invocation: preserved across internal calls and syscalls
- jit pins `vN` to `zmmN`. `zmm30`, `zmm31`, `k1`–`k7` are jit scratch

## encoding

a vector op is a `CALL_IMM` (`opc = 0x85`) with `src = 2`, `dst = 0`, and `off` = 0 except for memory ops (see below):

```
imm[31:24] op
imm[23:16] imm8
imm[15]    0
imm[14:10] out
imm[9:5]   in0
imm[4:0]   in1
```

as one little-endian u64: `0x85 | 0x20 << 8 | imm << 32`.

- enabled by `Config::enable_vector_ops` (poc gate; production would bind it to a new sbpf version, because today v3 accepts `src=2` calls at deploy and faults at runtime). requires an sbpf version with static syscalls (v3+)
- verifier rejects: unknown op, any register field ≥ 30, a nonzero field the op doesn't use (see kind), imm8 out of range for the op, `dst` ≠ 0, `off` ≠ 0 on a non-memory op
- cu: 1 per op (it is one instruction)

## scalar conventions

- `r1` = scalar input (address or value). `r0` = scalar output
- ops without a scalar output leave `r0` unchanged. no op modifies any other scalar register
- memory ops (`ld`, `st`, `ldbc.d`, `ldbc.q`) access `r1 + sext(off)`, like scalar `ldx`/`stx`, so a constant displacement needs no extra scalar instruction. all other ops require `off = 0`
- memory ops: unaligned ok, the whole access must be inside one mapped region with the right permission, otherwise the same error a scalar access would raise. on a fault, vector registers are unchanged

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

all other op values are reserved and rejected.

## intrinsics abi (what the build pass sees)

- `__svm512i` is `#[repr(transparent)] struct(u32)`, `Copy`. the u32 is an opaque ssa token, never a register number
- every intrinsic is `#[inline(always)]` and calls one marker: `extern "C" fn __svm_<name with '.' → '_'>(...)`, e.g. `__svm_add_d(a: u32, b: u32) -> u32`, `__svm_ld(p: *const u8) -> u32`, `__svm_st(a: u32, p: *mut u8)`, `__svm_splat_d(x: u64) -> u32`, `__svm_hsum_d(a: u32) -> u64`, `__svm_shli_d(a: u32, imm: u32) -> u32`
- argument order: vector args in intrinsic order, then scalar (`r1`) arg, then imm8 last. imm8 must be an ir constant
- the pass replaces every marker call with inline asm emitting the encoded op, assigns `v0`–`v29`, and rejects programs where a vector value escapes ssa (stored to memory, passed to or returned from a non-marker call, etc.) with a clear error. phis and `select`s of vectors are allowed (lowered to `mov`s)
- a program built without the pass fails to link (undefined `__svm_*`)
