/*
 * Zen 4 microbenchmark for SIMD-0714 scratch-register lowering:
 *   - 2 scratch ZMMs (zmm30/31) vs 0 scratch ZMMs (stack spill/reload)
 *   - 1 vs 2 scratch ZMMs for hsum.d (which needs 2)
 *   - /dep (latency chain), /ind (throughput), and /mix (1 scratch op + 4 vpaddd)
 *   - sha256rnds2 in 16-reg layout (xmm0-15) vs 30-reg high regs (zmm16-18)
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <time.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void pin_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static double calibrate_ghz(void) {
    uint64_t n = 50000000ULL;
    uint64_t x = 1;
    uint64_t t0 = now_ns();
    __asm__ volatile(
        ".p2align 6\n\t"
        "1:\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "add $1, %0\n\t"
        "dec %1\n\t"
        "jnz 1b\n\t"
        : "+r"(x), "+r"(n)
        :
        : "cc"
    );
    uint64_t dt = now_ns() - t0;
    volatile uint64_t sink = x; (void)sink;
    return (double)(50000000ULL * 8ULL) / (double)dt;
}

#define N_ITERS 4000000ULL
#define UNROLL 8

#define BENCH_ASM(name, init_asm, body1, body2, body3, body4, body5, body6, body7, body8, clobbers) \
static double bench_##name(void) { \
    __attribute__((aligned(64))) uint8_t spill_area[256] = {0}; \
    uint64_t n = N_ITERS; \
    uint64_t out_r = 0; \
    uint64_t in_r = 0x123456789abcdef0ULL; \
    uint64_t t0 = now_ns(); \
    __asm__ volatile( \
        init_asm \
        ".p2align 6\n\t" \
        "1:\n\t" \
        body1 \
        body2 \
        body3 \
        body4 \
        body5 \
        body6 \
        body7 \
        body8 \
        "dec %[n]\n\t" \
        "jnz 1b\n\t" \
        : [n] "+r"(n), [out] "+r"(out_r) \
        : [sp] "r"(spill_area), [inr] "r"(in_r) \
        : "cc", "memory", clobbers \
    ); \
    uint64_t dt = now_ns() - t0; \
    volatile uint64_t sink = out_r; (void)sink; \
    return (double)dt / (double)(N_ITERS * UNROLL); \
}

#define ALL_ZMMS \
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7", \
    "zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15", \
    "zmm16","zmm17","zmm18","zmm30","zmm31"

#define INIT_ZMMS \
    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t" \
    "vpxord %%zmm1, %%zmm1, %%zmm1\n\t" \
    "vpxord %%zmm2, %%zmm2, %%zmm2\n\t" \
    "vpxord %%zmm3, %%zmm3, %%zmm3\n\t" \
    "vpxord %%zmm4, %%zmm4, %%zmm4\n\t" \
    "vpxord %%zmm5, %%zmm5, %%zmm5\n\t" \
    "vpxord %%zmm6, %%zmm6, %%zmm6\n\t" \
    "vpxord %%zmm7, %%zmm7, %%zmm7\n\t" \
    "vpxord %%zmm8, %%zmm8, %%zmm8\n\t" \
    "vpxord %%zmm9, %%zmm9, %%zmm9\n\t" \
    "vpxord %%zmm10, %%zmm10, %%zmm10\n\t" \
    "vpxord %%zmm11, %%zmm11, %%zmm11\n\t" \
    "vpxord %%zmm12, %%zmm12, %%zmm12\n\t" \
    "vpxord %%zmm13, %%zmm13, %%zmm13\n\t" \
    "vpxord %%zmm14, %%zmm14, %%zmm14\n\t" \
    "vpxord %%zmm15, %%zmm15, %%zmm15\n\t" \
    "vpxord %%zmm16, %%zmm16, %%zmm16\n\t" \
    "vpxord %%zmm17, %%zmm17, %%zmm17\n\t" \
    "vpxord %%zmm18, %%zmm18, %%zmm18\n\t"

/* 4 independent ALU ops interleaved after a scratch op (20% scratch-op density) */
#define ALU4 \
    "vpaddd %%zmm8, %%zmm9, %%zmm10\n\t" \
    "vpaddd %%zmm9, %%zmm10, %%zmm11\n\t" \
    "vpaddd %%zmm10, %%zmm11, %%zmm12\n\t" \
    "vpaddd %%zmm11, %%zmm12, %%zmm13\n\t"

/* 1. hsum.d (exact vector_jit.rs sequence): needs 2 scratch ZMMs */
#define HSUMD_2SCR(idx) \
    "vextracti64x4 $1, %%zmm" #idx ", %%ymm30\n\t" \
    "vpaddd %%ymm" #idx ", %%ymm30, %%ymm30\n\t" \
    "vextracti32x4 $1, %%ymm30, %%xmm31\n\t" \
    "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t" \
    "vpshufd $0x4e, %%xmm30, %%xmm31\n\t" \
    "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t" \
    "vpshufd $0xb1, %%xmm30, %%xmm31\n\t" \
    "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t" \
    "vmovd %%xmm30, %k[out]\n\t"

#define HSUMD_1SCR(idx) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "vextracti64x4 $1, %%zmm" #idx ", %%ymm30\n\t" \
    "vpaddd %%ymm" #idx ", %%ymm30, %%ymm30\n\t" \
    "vextracti32x4 $1, %%ymm30, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm30, %%xmm30\n\t" \
    "vpshufd $0x4e, %%xmm30, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm30, %%xmm30\n\t" \
    "vpshufd $0xb1, %%xmm30, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm30, %%xmm30\n\t" \
    "vmovd %%xmm30, %k[out]\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

#define HSUMD_0SCR(idx) \
    "vmovdqa64 %%zmm14, 0(%[sp])\n\t" \
    "vmovdqa64 %%zmm15, 64(%[sp])\n\t" \
    "vextracti64x4 $1, %%zmm" #idx ", %%ymm14\n\t" \
    "vpaddd %%ymm" #idx ", %%ymm14, %%ymm14\n\t" \
    "vextracti32x4 $1, %%ymm14, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t" \
    "vpshufd $0x4e, %%xmm14, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t" \
    "vpshufd $0xb1, %%xmm14, %%xmm15\n\t" \
    "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t" \
    "vmovd %%xmm14, %k[out]\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm14\n\t" \
    "vmovdqa64 64(%[sp]), %%zmm15\n\t"

BENCH_ASM(hsumd_2scr, INIT_ZMMS,
    HSUMD_2SCR(0), HSUMD_2SCR(1), HSUMD_2SCR(2), HSUMD_2SCR(3),
    HSUMD_2SCR(4), HSUMD_2SCR(5), HSUMD_2SCR(6), HSUMD_2SCR(7), ALL_ZMMS)
BENCH_ASM(hsumd_1scr, INIT_ZMMS,
    HSUMD_1SCR(0), HSUMD_1SCR(1), HSUMD_1SCR(2), HSUMD_1SCR(3),
    HSUMD_1SCR(4), HSUMD_1SCR(5), HSUMD_1SCR(6), HSUMD_1SCR(7), ALL_ZMMS)
BENCH_ASM(hsumd_0scr, INIT_ZMMS,
    HSUMD_0SCR(0), HSUMD_0SCR(1), HSUMD_0SCR(2), HSUMD_0SCR(3),
    HSUMD_0SCR(4), HSUMD_0SCR(5), HSUMD_0SCR(6), HSUMD_0SCR(7), ALL_ZMMS)
BENCH_ASM(hsumd_2scr_mix, INIT_ZMMS,
    HSUMD_2SCR(0) ALU4, HSUMD_2SCR(1) ALU4, HSUMD_2SCR(2) ALU4, HSUMD_2SCR(3) ALU4,
    HSUMD_2SCR(4) ALU4, HSUMD_2SCR(5) ALU4, HSUMD_2SCR(6) ALU4, HSUMD_2SCR(7) ALU4, ALL_ZMMS)
BENCH_ASM(hsumd_0scr_mix, INIT_ZMMS,
    HSUMD_0SCR(0) ALU4, HSUMD_0SCR(1) ALU4, HSUMD_0SCR(2) ALU4, HSUMD_0SCR(3) ALU4,
    HSUMD_0SCR(4) ALU4, HSUMD_0SCR(5) ALU4, HSUMD_0SCR(6) ALU4, HSUMD_0SCR(7) ALU4, ALL_ZMMS)

/* 2. bcast.d[15] (valignd + vpbroadcastd): needs 1 scratch ZMM */
#define BCASTD_SCR(dst, src) \
    "valignd $15, %%" #src ", %%" #src ", %%zmm30\n\t" \
    "vpbroadcastd %%xmm30, %%" #dst "\n\t"

#define BCASTD_0SCR(dst, src) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "valignd $15, %%" #src ", %%" #src ", %%zmm15\n\t" \
    "vpbroadcastd %%xmm15, %%" #dst "\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

BENCH_ASM(bcastd_scr_dep, INIT_ZMMS,
    BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0),
    BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0), BCASTD_SCR(zmm0, zmm0), ALL_ZMMS)
BENCH_ASM(bcastd_0scr_dep, INIT_ZMMS,
    BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0),
    BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0), BCASTD_0SCR(zmm0, zmm0), ALL_ZMMS)
BENCH_ASM(bcastd_scr_ind, INIT_ZMMS,
    BCASTD_SCR(zmm0, zmm8), BCASTD_SCR(zmm1, zmm9), BCASTD_SCR(zmm2, zmm10), BCASTD_SCR(zmm3, zmm11),
    BCASTD_SCR(zmm4, zmm12), BCASTD_SCR(zmm5, zmm13), BCASTD_SCR(zmm6, zmm14), BCASTD_SCR(zmm7, zmm8), ALL_ZMMS)
BENCH_ASM(bcastd_0scr_ind, INIT_ZMMS,
    BCASTD_0SCR(zmm0, zmm8), BCASTD_0SCR(zmm1, zmm9), BCASTD_0SCR(zmm2, zmm10), BCASTD_0SCR(zmm3, zmm11),
    BCASTD_0SCR(zmm4, zmm12), BCASTD_0SCR(zmm5, zmm13), BCASTD_0SCR(zmm6, zmm14), BCASTD_0SCR(zmm7, zmm8), ALL_ZMMS)
BENCH_ASM(bcastd_scr_mix, INIT_ZMMS,
    BCASTD_SCR(zmm0, zmm8) ALU4, BCASTD_SCR(zmm1, zmm9) ALU4, BCASTD_SCR(zmm2, zmm10) ALU4, BCASTD_SCR(zmm3, zmm11) ALU4,
    BCASTD_SCR(zmm4, zmm12) ALU4, BCASTD_SCR(zmm5, zmm13) ALU4, BCASTD_SCR(zmm6, zmm14) ALU4, BCASTD_SCR(zmm7, zmm8) ALU4, ALL_ZMMS)
BENCH_ASM(bcastd_0scr_mix, INIT_ZMMS,
    BCASTD_0SCR(zmm0, zmm8) ALU4, BCASTD_0SCR(zmm1, zmm9) ALU4, BCASTD_0SCR(zmm2, zmm10) ALU4, BCASTD_0SCR(zmm3, zmm11) ALU4,
    BCASTD_0SCR(zmm4, zmm12) ALU4, BCASTD_0SCR(zmm5, zmm13) ALU4, BCASTD_0SCR(zmm6, zmm14) ALU4, BCASTD_0SCR(zmm7, zmm8) ALU4, ALL_ZMMS)

/* 3. insert.q[5] */
#define INSQ_SCR(dst) \
    "vextracti32x4 $2, %%" #dst ", %%xmm30\n\t" \
    "vpinsrq $1, %[inr], %%xmm30, %%xmm30\n\t" \
    "vinserti32x4 $2, %%xmm30, %%" #dst ", %%" #dst "\n\t"

#define INSQ_0SCR(dst) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "vextracti32x4 $2, %%" #dst ", %%xmm15\n\t" \
    "vpinsrq $1, %[inr], %%xmm15, %%xmm15\n\t" \
    "vinserti32x4 $2, %%xmm15, %%" #dst ", %%" #dst "\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

BENCH_ASM(insq_scr_dep, INIT_ZMMS,
    INSQ_SCR(zmm0), INSQ_SCR(zmm0), INSQ_SCR(zmm0), INSQ_SCR(zmm0),
    INSQ_SCR(zmm0), INSQ_SCR(zmm0), INSQ_SCR(zmm0), INSQ_SCR(zmm0), ALL_ZMMS)
BENCH_ASM(insq_0scr_dep, INIT_ZMMS,
    INSQ_0SCR(zmm0), INSQ_0SCR(zmm0), INSQ_0SCR(zmm0), INSQ_0SCR(zmm0),
    INSQ_0SCR(zmm0), INSQ_0SCR(zmm0), INSQ_0SCR(zmm0), INSQ_0SCR(zmm0), ALL_ZMMS)
BENCH_ASM(insq_scr_ind, INIT_ZMMS,
    INSQ_SCR(zmm0), INSQ_SCR(zmm1), INSQ_SCR(zmm2), INSQ_SCR(zmm3),
    INSQ_SCR(zmm4), INSQ_SCR(zmm5), INSQ_SCR(zmm6), INSQ_SCR(zmm7), ALL_ZMMS)
BENCH_ASM(insq_0scr_ind, INIT_ZMMS,
    INSQ_0SCR(zmm0), INSQ_0SCR(zmm1), INSQ_0SCR(zmm2), INSQ_0SCR(zmm3),
    INSQ_0SCR(zmm4), INSQ_0SCR(zmm5), INSQ_0SCR(zmm6), INSQ_0SCR(zmm7), ALL_ZMMS)

/* 4. extract.q[5] */
#define EXTQ_SCR(src) \
    "vextracti32x4 $2, %%" #src ", %%xmm30\n\t" \
    "vpextrq $1, %%xmm30, %[out]\n\t"

#define EXTQ_0SCR(src) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "vextracti32x4 $2, %%" #src ", %%xmm15\n\t" \
    "vpextrq $1, %%xmm15, %[out]\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

BENCH_ASM(extq_scr_ind, INIT_ZMMS,
    EXTQ_SCR(zmm0), EXTQ_SCR(zmm1), EXTQ_SCR(zmm2), EXTQ_SCR(zmm3),
    EXTQ_SCR(zmm4), EXTQ_SCR(zmm5), EXTQ_SCR(zmm6), EXTQ_SCR(zmm7), ALL_ZMMS)
BENCH_ASM(extq_0scr_ind, INIT_ZMMS,
    EXTQ_0SCR(zmm0), EXTQ_0SCR(zmm1), EXTQ_0SCR(zmm2), EXTQ_0SCR(zmm3),
    EXTQ_0SCR(zmm4), EXTQ_0SCR(zmm5), EXTQ_0SCR(zmm6), EXTQ_0SCR(zmm7), ALL_ZMMS)

/* 5. mul52lo aliased (out == in0) */
#define MUL52_SCR(dst, b) \
    "vpxorq %%zmm30, %%zmm30, %%zmm30\n\t" \
    "vpmadd52luq %%" #b ", %%" #dst ", %%zmm30\n\t" \
    "vmovdqa64 %%zmm30, %%" #dst "\n\t"

#define MUL52_0SCR(dst, b) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "vpxorq %%zmm15, %%zmm15, %%zmm15\n\t" \
    "vpmadd52luq %%" #b ", %%" #dst ", %%zmm15\n\t" \
    "vmovdqa64 %%zmm15, %%" #dst "\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

BENCH_ASM(mul52_scr_dep, INIT_ZMMS,
    MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8),
    MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm0, zmm8), ALL_ZMMS)
BENCH_ASM(mul52_0scr_dep, INIT_ZMMS,
    MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8),
    MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm0, zmm8), ALL_ZMMS)
BENCH_ASM(mul52_scr_ind, INIT_ZMMS,
    MUL52_SCR(zmm0, zmm8), MUL52_SCR(zmm1, zmm9), MUL52_SCR(zmm2, zmm10), MUL52_SCR(zmm3, zmm11),
    MUL52_SCR(zmm4, zmm12), MUL52_SCR(zmm5, zmm13), MUL52_SCR(zmm6, zmm14), MUL52_SCR(zmm7, zmm8), ALL_ZMMS)
BENCH_ASM(mul52_0scr_ind, INIT_ZMMS,
    MUL52_0SCR(zmm0, zmm8), MUL52_0SCR(zmm1, zmm9), MUL52_0SCR(zmm2, zmm10), MUL52_0SCR(zmm3, zmm11),
    MUL52_0SCR(zmm4, zmm12), MUL52_0SCR(zmm5, zmm13), MUL52_0SCR(zmm6, zmm14), MUL52_0SCR(zmm7, zmm8), ALL_ZMMS)
BENCH_ASM(mul52_scr_mix, INIT_ZMMS,
    MUL52_SCR(zmm0, zmm8) ALU4, MUL52_SCR(zmm1, zmm9) ALU4, MUL52_SCR(zmm2, zmm10) ALU4, MUL52_SCR(zmm3, zmm11) ALU4,
    MUL52_SCR(zmm4, zmm12) ALU4, MUL52_SCR(zmm5, zmm13) ALU4, MUL52_SCR(zmm6, zmm14) ALU4, MUL52_SCR(zmm7, zmm8) ALU4, ALL_ZMMS)
BENCH_ASM(mul52_0scr_mix, INIT_ZMMS,
    MUL52_0SCR(zmm0, zmm8) ALU4, MUL52_0SCR(zmm1, zmm9) ALU4, MUL52_0SCR(zmm2, zmm10) ALU4, MUL52_0SCR(zmm3, zmm11) ALU4,
    MUL52_0SCR(zmm4, zmm12) ALU4, MUL52_0SCR(zmm5, zmm13) ALU4, MUL52_0SCR(zmm6, zmm14) ALU4, MUL52_0SCR(zmm7, zmm8) ALU4, ALL_ZMMS)

/* 6. sha256rnds2: 16-reg w/ scratch vs 16-reg 0-scratch vs 30-reg high regs (zmm16/17/18) */
#define SHA_16_SCR(o, a, wk) \
    "vinserti32x4 $0, %%xmm0, %%zmm30, %%zmm30\n\t" \
    "vinserti32x4 $0, %%" #wk ", %%zmm0, %%zmm0\n\t" \
    "sha256rnds2 %%xmm0, %%" #a ", %%" #o "\n\t" \
    "vinserti32x4 $0, %%xmm30, %%zmm0, %%zmm0\n\t"

#define SHA_16_0SCR(o, a, wk) \
    "vmovdqa %%xmm0, 0(%[sp])\n\t" \
    "vinserti32x4 $0, %%" #wk ", %%zmm0, %%zmm0\n\t" \
    "sha256rnds2 %%xmm0, %%" #a ", %%" #o "\n\t" \
    "vinserti32x4 $0, 0(%[sp]), %%zmm0, %%zmm0\n\t"

#define SHA_30_HIGH(o, a, wk) \
    "vinserti32x4 $0, %%xmm1, %%zmm30, %%zmm30\n\t" \
    "vinserti32x4 $1, %%xmm2, %%zmm30, %%zmm30\n\t" \
    "vinserti32x4 $2, %%xmm0, %%zmm30, %%zmm30\n\t" \
    "vinserti32x4 $0, %%xmm" #o ", %%zmm1, %%zmm1\n\t" \
    "vinserti32x4 $0, %%xmm" #a ", %%zmm2, %%zmm2\n\t" \
    "vinserti32x4 $0, %%xmm" #wk ", %%zmm0, %%zmm0\n\t" \
    "sha256rnds2 %%xmm0, %%xmm2, %%xmm1\n\t" \
    "vextracti32x4 $2, %%zmm30, %%xmm31\n\t" \
    "vinserti32x4 $0, %%xmm31, %%zmm0, %%zmm0\n\t" \
    "vinserti32x4 $0, %%xmm1, %%zmm" #o ", %%zmm" #o "\n\t" \
    "vextracti32x4 $1, %%zmm30, %%xmm31\n\t" \
    "vinserti32x4 $0, %%xmm31, %%zmm2, %%zmm2\n\t" \
    "vinserti32x4 $0, %%xmm30, %%zmm1, %%zmm1\n\t"

BENCH_ASM(sha_16_scr_dep, INIT_ZMMS,
    SHA_16_SCR(xmm1, xmm2, xmm3), SHA_16_SCR(xmm2, xmm1, xmm3),
    SHA_16_SCR(xmm1, xmm2, xmm3), SHA_16_SCR(xmm2, xmm1, xmm3),
    SHA_16_SCR(xmm1, xmm2, xmm3), SHA_16_SCR(xmm2, xmm1, xmm3),
    SHA_16_SCR(xmm1, xmm2, xmm3), SHA_16_SCR(xmm2, xmm1, xmm3), ALL_ZMMS)
BENCH_ASM(sha_16_0scr_dep, INIT_ZMMS,
    SHA_16_0SCR(xmm1, xmm2, xmm3), SHA_16_0SCR(xmm2, xmm1, xmm3),
    SHA_16_0SCR(xmm1, xmm2, xmm3), SHA_16_0SCR(xmm2, xmm1, xmm3),
    SHA_16_0SCR(xmm1, xmm2, xmm3), SHA_16_0SCR(xmm2, xmm1, xmm3),
    SHA_16_0SCR(xmm1, xmm2, xmm3), SHA_16_0SCR(xmm2, xmm1, xmm3), ALL_ZMMS)
BENCH_ASM(sha_30_high_dep, INIT_ZMMS,
    SHA_30_HIGH(16, 17, 18), SHA_30_HIGH(17, 16, 18),
    SHA_30_HIGH(16, 17, 18), SHA_30_HIGH(17, 16, 18),
    SHA_30_HIGH(16, 17, 18), SHA_30_HIGH(17, 16, 18),
    SHA_30_HIGH(16, 17, 18), SHA_30_HIGH(17, 16, 18), ALL_ZMMS)

/* 7. ModRM 0xff sanitizer break */
#define SANIT_SCR(dst, src) \
    "vmovdqa64 %%" #src ", %%zmm30\n\t" \
    "vprold $3, %%zmm30, %%" #dst "\n\t"

#define SANIT_0SCR(dst, src) \
    "vmovdqa64 %%zmm15, 0(%[sp])\n\t" \
    "vmovdqa64 %%" #src ", %%zmm15\n\t" \
    "vprold $3, %%zmm15, %%" #dst "\n\t" \
    "vmovdqa64 0(%[sp]), %%zmm15\n\t"

BENCH_ASM(sanit_scr_dep, INIT_ZMMS,
    SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7),
    SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7), SANIT_SCR(zmm7, zmm7), ALL_ZMMS)
BENCH_ASM(sanit_0scr_dep, INIT_ZMMS,
    SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7),
    SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7), SANIT_0SCR(zmm7, zmm7), ALL_ZMMS)
BENCH_ASM(sanit_scr_ind, INIT_ZMMS,
    SANIT_SCR(zmm0, zmm7), SANIT_SCR(zmm1, zmm7), SANIT_SCR(zmm2, zmm7), SANIT_SCR(zmm3, zmm7),
    SANIT_SCR(zmm4, zmm7), SANIT_SCR(zmm5, zmm7), SANIT_SCR(zmm6, zmm7), SANIT_SCR(zmm8, zmm7), ALL_ZMMS)
BENCH_ASM(sanit_0scr_ind, INIT_ZMMS,
    SANIT_0SCR(zmm0, zmm7), SANIT_0SCR(zmm1, zmm7), SANIT_0SCR(zmm2, zmm7), SANIT_0SCR(zmm3, zmm7),
    SANIT_0SCR(zmm4, zmm7), SANIT_0SCR(zmm5, zmm7), SANIT_0SCR(zmm6, zmm7), SANIT_0SCR(zmm8, zmm7), ALL_ZMMS)

static double med5(double (*fn)(void)) {
    double a[5];
    fn();
    for (int i = 0; i < 5; i++) a[i] = fn();
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (a[j] < a[i]) { double t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[2];
}

int main(int argc, char **argv) {
    int cpu = (argc > 1) ? atoi(argv[1]) : 77;
    pin_cpu(cpu);
    double ghz = calibrate_ghz();
    printf("Zen 4 core %d calibrated: %.3f GHz\n\n", cpu, ghz);
    printf("%-32s %8s %8s %7s %7s %7s %8s\n",
           "Op / mode", "with scr", "0-scr", "cyc(s)", "cyc(0)", "ratio", "faster");
    printf("---------------------------------------------------------------------------------------\n");

    struct { const char *name; double (*a)(void); double (*b)(void); } rows[] = {
        {"hsum.d / ind (2 vs 0 scr)",      bench_hsumd_2scr,      bench_hsumd_0scr},
        {"hsum.d / ind (2 vs 1 scr)",      bench_hsumd_2scr,      bench_hsumd_1scr},
        {"hsum.d / mix (+4 vpaddd)",       bench_hsumd_2scr_mix,  bench_hsumd_0scr_mix},
        {"bcast.d[15] / dep (1 vs 0 scr)", bench_bcastd_scr_dep,  bench_bcastd_0scr_dep},
        {"bcast.d[15] / ind (1 vs 0 scr)", bench_bcastd_scr_ind,  bench_bcastd_0scr_ind},
        {"bcast.d[15] / mix (+4 vpaddd)",  bench_bcastd_scr_mix,  bench_bcastd_0scr_mix},
        {"insert.q[5] / dep (1 vs 0 scr)", bench_insq_scr_dep,    bench_insq_0scr_dep},
        {"insert.q[5] / ind (1 vs 0 scr)", bench_insq_scr_ind,    bench_insq_0scr_ind},
        {"extract.q[5] / ind (1 vs 0 scr)",bench_extq_scr_ind,    bench_extq_0scr_ind},
        {"mul52lo(alias)/dep (1 vs 0 scr)",bench_mul52_scr_dep,   bench_mul52_0scr_dep},
        {"mul52lo(alias)/ind (1 vs 0 scr)",bench_mul52_scr_ind,   bench_mul52_0scr_ind},
        {"mul52lo(alias)/mix (+4 vpaddd)", bench_mul52_scr_mix,   bench_mul52_0scr_mix},
        {"sha256rnds2 16r(1 vs 0 scr)/dep",bench_sha_16_scr_dep,  bench_sha_16_0scr_dep},
        {"sha256rnds2 (16r vs 30r-hi)/dep",bench_sha_16_scr_dep,  bench_sha_30_high_dep},
        {"sanitizer(vprold)/dep (1v0 scr)",bench_sanit_scr_dep,   bench_sanit_0scr_dep},
        {"sanitizer(vprold)/ind (1v0 scr)",bench_sanit_scr_ind,   bench_sanit_0scr_ind},
    };
    int n = (int)(sizeof(rows) / sizeof(rows[0]));
    for (int i = 0; i < n; i++) {
        double s = med5(rows[i].a);
        double ns = med5(rows[i].b);
        double pct = (1.0 - s / ns) * 100.0;
        printf("%-32s %6.3f ns %6.3f ns %6.2f  %6.2f  %5.2fx  %6.1f%%\n",
               rows[i].name, s, ns, s * ghz, ns * ghz, ns / s, pct);
    }
    return 0;
}
