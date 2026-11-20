# svm vector ops: poc spec

the contract between the runtime (`sbpf/`), the build pass (`sbpf-linker/`) and the intrinsics (`svm-intrinsics/`). if code and this file disagree, fix one of them.

## registers

- 16 vector registers `v0`–`v15`, 512 bits each, lanes little-endian
- lane widths: `b` = 64×u8, `w` = 32×u16, `d` = 16×u32, `q` = 8×u64
- vm state, not memory mapped. only reachable through vector ops
- zeroed at program entry. global to the invocation: preserved across internal calls and syscalls
- jit pins `vN` to `zmmN` and uses no other vector register and no mask register
- every op is one host instruction. no op moves a lane to or from a scalar register and there is no horizontal sum: programs `st` the vector to memory and use scalar code

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
- verifier rejects: unknown op, any register field ≥ 16, a nonzero field the op doesn't use (see kind), imm8 out of range for the op, `dst` ≠ 0, `off` ≠ 0 on a non-memory op
- cu: 1 per op (it is one instruction)

## scalar conventions

- `r1` = scalar input (address or value)
- no op modifies a scalar register
- memory ops (`ld`, `st`, `ldbc.d`, `ldbc.q`) access `r1 + sext(off)`, like scalar `ldx`/`stx`, so a constant displacement needs no extra scalar instruction. all other ops require `off = 0`
- memory ops: unaligned ok, the whole access must be inside one mapped region with the right permission, otherwise the same error a scalar access would raise. on a fault, vector registers are unchanged

## kinds (which fields are used)

| kind | reads | writes | imm8 |
|---|---|---|---|
| `r3` | in0, in1 | out | 0 |
| `a3` | out, in0, in1 | out | 0 (acc: out is also an input) |
| `a3i` | out, in0, in1 | out | per op |
| `r2` | in0 | out | 0 |
| `r2i` | in0 | out | per op |
| `r0` | — | out | 0 |
| `ls` | r1 | out | 0 (load/splat from scalar) |
| `st` | in0, r1 | memory | 0 |

unused register fields must be 0.

## ops

intrinsic args map to `in0`, `in1` in order. for `a3`/`a3i` the first arg is the accumulator (`out`). avx-512 column is the jit lowering (intel operand order: out, in0, in1).

| op | name | kind | semantics | avx-512 | intrinsic |
|---|---|---|---|---|---|
| 0x00 | ld | ls | out ← mem[r1..r1+64] | vmovdqu64 | `_svm512_loadu_si512(p)` |
| 0x01 | st | st | mem[r1..r1+64] ← in0 | vmovdqu64 | `_svm512_storeu_si512(p, a)` |
| 0x02 | ldbc.d | ls | out.d[i] ← mem32[r1] | vpbroadcastd m32 | `_svm512_loadbc_epi32(p)` |
| 0x03 | ldbc.q | ls | out.q[i] ← mem64[r1] | vpbroadcastq m64 | `_svm512_loadbc_epi64(p)` |
| 0x06 | splat.d | ls | out.d[i] ← r1[31:0] | vpbroadcastd r32 | `_svm512_set1_epi32(x)` |
| 0x08 | zero | r0 | out ← 0 | vpxord | `_svm512_setzero_si512()` |
| 0x09 | mov | r2 | out ← in0 | vmovdqa64 | (pass-internal) |
| 0x11 | add.w | r3 | wrapping | vpaddw | `_svm512_add_epi16` |
| 0x13 | add.q | r3 | wrapping | vpaddq | `_svm512_add_epi64` |
| 0x15 | sub.w | r3 | wrapping, in0 − in1 | vpsubw | `_svm512_sub_epi16` |
| 0x17 | sub.q | r3 | wrapping, in0 − in1 | vpsubq | `_svm512_sub_epi64` |
| 0x18 | mullo.w | r3 | low 16 | vpmullw | `_svm512_mullo_epi16` |
| 0x1a | mullo.q | r3 | low 64 | vpmullq | `_svm512_mullo_epi64` |
| 0x1d | mulhi.w | r3 | high 16 of the signed product | vpmulhw | `_svm512_mulhi_epi16` |
| 0x20 | max.d | r3 | signed | vpmaxsd | `_svm512_max_epi32` |
| 0x21 | min.d | r3 | signed | vpminsd | `_svm512_min_epi32` |
| 0x27 | minu.q | r3 | unsigned | vpminuq | `_svm512_min_epu64` |
| 0x28 | and | r3 | | vpandq | `_svm512_and_si512` |
| 0x2a | xor | r3 | | vpxorq | `_svm512_xor_si512` |
| 0x2c | ternlog | a3i | out.bit ← imm8[(out.bit << 2) \| (in0.bit << 1) \| in1.bit], imm8 any | vpternlogq | `_svm512_ternarylogic_epi64::<IMM>(a, b, c)` |
| 0x2f | minu.w | r3 | unsigned | vpminuw | `_svm512_min_epu16` |
| 0x32 | shli.q | r2i | shl by imm8, imm8 < 64 | vpsllq imm | `_svm512_slli_epi64::<IMM>` |
| 0x35 | shri.q | r2i | logical shr, imm8 < 64 | vpsrlq imm | `_svm512_srli_epi64::<IMM>` |
| 0x37 | sari.d | r2i | arithmetic shr, imm8 < 32 | vpsrad imm | `_svm512_srai_epi32::<IMM>` |
| 0x38 | sari.q | r2i | arithmetic shr, imm8 < 64 | vpsraq imm | `_svm512_srai_epi64::<IMM>` |
| 0x3a | roli.q | r2i | rotate left, imm8 < 64 | vprolq | `_svm512_rol_epi64::<IMM>` |
| 0x40 | rolv.q | r3 | rotate left by in1 mod 64 | vprolvq | `_svm512_rolv_epi64` |
| 0x50 | dpbusd | a3 | out.d[i] += Σj<4 u8(in0.b[4i+j]) · s8(in1.b[4i+j]), wrapping | vpdpbusd | `_svm512_dpbusd_epi32(acc, a, b)` |
| 0x51 | dpwssd | a3 | out.d[i] += Σj<2 s16(in0.w[2i+j]) · s16(in1.w[2i+j]), wrapping | vpdpwssd | `_svm512_dpwssd_epi32(acc, a, b)` |
| 0x53 | maddwd | r3 | out.d[i] ← Σj<2 s16 · s16, wrapping | vpmaddwd | `_svm512_madd_epi16` |
| 0x5c | permq | r3 | out.q[i] ← in1.q[in0.q[i] & 7] | vpermq (out, idx=in0, data=in1) | `_svm512_permutexvar_epi64(idx, a)` |
| 0x5d | shufb | r3 | per 128-bit lane L: out.b[i] ← in1.b[i] bit 7 ? 0 : in0.b[16L + (in1.b[i] & 15)] | vpshufb | `_svm512_shuffle_epi8(a, b)` |
| 0x5e | perm2.w | a3 | k = in0.w[i] & 63, out.w[i] ← k < 32 ? out.w[k] : in1.w[k − 32] (old out) | vpermt2w (out = table 1, idx = in0) | `_svm512_permutex2var_epi16(a, idx, b)` |
| 0x5f | perm2.q | a3 | k = in0.q[i] & 15, out.q[i] ← k < 8 ? out.q[k] : in1.q[k − 8] (old out) | vpermt2q (out = table 1, idx = in0) | `_svm512_permutex2var_epi64(a, idx, b)` |
| 0x60 | narrowus.d.b | r2 | out.b[i] ← min(u32(in0.d[i]), 255) for i < 16, rest of out 0 | vpmovusdb | `_svm512_cvtusepi32_epi8` |
| 0xb1 | abs.q | r2 | out.q[i] ← \|i64(in0.q[i])\|, i64 min → i64 min | vpabsq | `_svm512_abs_epi64` |

all other op values are reserved and rejected. that includes the codes of ops removed from earlier versions of this list: 0x04, 0x05, 0x07, 0x10, 0x12, 0x14, 0x16, 0x19, 0x1b, 0x1c, 0x1e, 0x22–0x26, 0x29, 0x2b, 0x2e, 0x30, 0x31, 0x33, 0x34, 0x36, 0x39, 0x3b–0x3f, 0x48–0x4d, 0x52, 0x54, 0x55, 0x58–0x5b, 0x61–0x6b, 0x6f, 0x75, 0x78, 0xb2, 0xb3.

## intrinsics abi (what the build pass sees)

- `__svm512i` is `#[repr(transparent)] struct(u32)`, `Copy`. the u32 is an opaque ssa token, never a register number
- every intrinsic is `#[inline(always)]` and calls one marker: `extern "C" fn __svm_<name with '.' → '_'>(...)`, e.g. `__svm_add_q(a: u32, b: u32) -> u32`, `__svm_ld(p: *const u8) -> u32`, `__svm_st(a: u32, p: *mut u8)`, `__svm_splat_d(x: u64) -> u32`, `__svm_shli_q(a: u32, imm: u32) -> u32`
- argument order: vector args in intrinsic order, then scalar (`r1`) arg, then imm8 last. imm8 must be an ir constant
- the pass replaces every marker call with inline asm emitting the encoded op, assigns `v0`–`v15`, and rejects programs where a vector value escapes ssa (stored to memory, passed to or returned from a non-marker call, etc.) with a clear error. phis and `select`s of vectors are allowed (lowered to `mov`s)
- a program built without the pass fails to link (undefined `__svm_*`)
