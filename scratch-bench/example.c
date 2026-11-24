/*
 * Minimal example: why SIMD-0714 JIT lowering needs 2 scratch ZMM registers.
 *
 * Guest registers v0..v15 are pinned to zmm0..zmm15. Multi-instruction ops
 * like hsum.d (horizontal sum of 16 x u32) and bcast.d (broadcast upper lane)
 * cannot clobber their inputs or any other guest register.
 *
 * - With 2 scratch ZMMs (zmm30, zmm31): intermediates stay in vector regs.
 * - With 0 scratch ZMMs: the JIT must spill victim guest regs to [rsp] and
 *   reload them around every op, paying 512-bit store-to-load forwarding
 *   latency (~11-12 cycles on Zen 4) plus 128-256 B of L1 stack traffic.
 *
 * Build & run: clang -O2 -march=native example.c -o example && ./example
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* hsum.d using zmm30 and zmm31 as scratch; zmm0..zmm15 untouched */
static uint32_t hsum_d_with_scratch(__m512i v, uint64_t iters) {
    uint32_t out = 0;
    __asm__ volatile(
        ".p2align 6\n\t"
        "1:\n\t"
        "vextracti64x4 $1, %[v], %%ymm30\n\t"
        "vpaddd %t[v], %%ymm30, %%ymm30\n\t"
        "vextracti32x4 $1, %%ymm30, %%xmm31\n\t"
        "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t"
        "vpshufd $0x4e, %%xmm30, %%xmm31\n\t"
        "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t"
        "vpshufd $0xb1, %%xmm30, %%xmm31\n\t"
        "vpaddd %%xmm31, %%xmm30, %%xmm30\n\t"
        "vmovd %%xmm30, %[out]\n\t"
        "dec %[n]\n\t"
        "jnz 1b\n\t"
        : [out] "=&r"(out), [n] "+r"(iters)
        : [v] "v"(v)
        : "cc", "zmm30", "zmm31"
    );
    return out;
}

/* Same hsum.d with 0 scratch ZMMs: spills zmm14 and zmm15 to [rsp] every op */
static uint32_t hsum_d_no_scratch(__m512i v, uint64_t iters) {
    __attribute__((aligned(64))) uint8_t spill[128] = {0};
    uint32_t out = 0;
    __asm__ volatile(
        ".p2align 6\n\t"
        "1:\n\t"
        "vmovdqa64 %%zmm14, 0(%[sp])\n\t"
        "vmovdqa64 %%zmm15, 64(%[sp])\n\t"
        "vextracti64x4 $1, %[v], %%ymm14\n\t"
        "vpaddd %t[v], %%ymm14, %%ymm14\n\t"
        "vextracti32x4 $1, %%ymm14, %%xmm15\n\t"
        "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t"
        "vpshufd $0x4e, %%xmm14, %%xmm15\n\t"
        "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t"
        "vpshufd $0xb1, %%xmm14, %%xmm15\n\t"
        "vpaddd %%xmm15, %%xmm14, %%xmm14\n\t"
        "vmovd %%xmm14, %[out]\n\t"
        "vmovdqa64 0(%[sp]), %%zmm14\n\t"
        "vmovdqa64 64(%[sp]), %%zmm15\n\t"
        "dec %[n]\n\t"
        "jnz 1b\n\t"
        : [out] "=&r"(out), [n] "+r"(iters)
        : [v] "v"(v), [sp] "r"(spill)
        : "cc", "memory", "zmm14", "zmm15"
    );
    return out;
}

int main(void) {
    uint32_t lanes[16];
    uint32_t expected = 0;
    for (int i = 0; i < 16; i++) {
        lanes[i] = (uint32_t)(i + 1) * 7U;
        expected += lanes[i];
    }
    __m512i v = _mm512_loadu_si512((const void *)lanes);

    const uint64_t N = 20000000ULL;
    uint32_t a = hsum_d_with_scratch(v, 1000);
    uint32_t b = hsum_d_no_scratch(v, 1000);
    if (a != expected || b != expected) {
        fprintf(stderr, "mismatch: got %u / %u, expected %u\n", a, b, expected);
        return 1;
    }

    uint64_t t0 = now_ns();
    a = hsum_d_with_scratch(v, N);
    uint64_t t1 = now_ns();
    b = hsum_d_no_scratch(v, N);
    uint64_t t2 = now_ns();

    double ns_scr = (double)(t1 - t0) / (double)N;
    double ns_noscr = (double)(t2 - t1) / (double)N;
    double pct = (1.0 - ns_scr / ns_noscr) * 100.0;
    printf("hsum.d result = %u (matches expected %u)\n", a, expected);
    printf("  with 2 scratch ZMMs (zmm30/31): %.3f ns/op\n", ns_scr);
    printf("  with 0 scratch ZMMs (spilled):  %.3f ns/op\n", ns_noscr);
    printf("  speedup: %.2fx (%.1f%% faster)\n", ns_noscr / ns_scr, pct);
    return 0;
}
