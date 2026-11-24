#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;

#define VM_INPUT_START    0x400000000ULL
#define VM_OFFSET_MASK    0x0FFFFFFFFULL
#define MAX_INPUT_REGIONS 256
#define REGION_BUF_SZ     4096

typedef struct {
  u64 vaddr_offset;
  u64 haddr;
  u32 region_sz;
  u32 address_space_reserved;
  u8  is_writable;
} input_region_t;

typedef struct {
  u64            region_haddr[6];
  u32            region_ld_sz[6];
  u32            region_st_sz[6];
  input_region_t input_regions[MAX_INPUT_REGIONS];
  u32            input_cnt;
  u32            in_lo32[MAX_INPUT_REGIONS + 16]  __attribute__((aligned(64)));
  u32            in_hi32[MAX_INPUT_REGIONS + 16]  __attribute__((aligned(64)));
  u64            in_delta[MAX_INPUT_REGIONS + 16] __attribute__((aligned(64)));
  u32            btree_root_hi32[16]              __attribute__((aligned(64)));
  u64            eytz_addr[MAX_INPUT_REGIONS + 1];
  u32            eytz_idx[MAX_INPUT_REGIONS + 1];
} vm_mem_t;

static inline u32
fd_bsearch_input( vm_mem_t const * vm, u64 offset ) {
  u32 left  = 0U;
  u32 right = vm->input_cnt - 1U;
  while( left < right ) {
    u32 mid = (left + right) >> 1;
    if( offset >= vm->input_regions[mid].vaddr_offset + vm->input_regions[mid].address_space_reserved )
      left = mid + 1U;
    else
      right = mid;
  }
  return left;
}

static inline u64
fd_translate_ld( vm_mem_t const * vm, u64 vaddr ) {
  u64 region = vaddr >> 32;
  if( region > 5ULL ) region = 5ULL;
  u64 offset = vaddr & VM_OFFSET_MASK;
  if( region == 4ULL ) {
    u32 idx = fd_bsearch_input( vm, offset );
    input_region_t const * r = &vm->input_regions[idx];
    u64 rel = offset - r->vaddr_offset;
    if( __builtin_expect( offset < r->vaddr_offset || rel > r->region_sz || 8ULL > r->region_sz - rel, 0 ) )
      return 0ULL;
    return r->haddr + rel;
  }
  u64 rsz = (u64)vm->region_ld_sz[region];
  return (offset <= rsz && 8ULL <= rsz - offset) ? (vm->region_haddr[region] + offset) : 0ULL;
}

typedef struct { u64 lo, hi, delta; } tlb_entry_t;

__attribute__((noinline, preserve_all))
static u64
resolve_miss_ld( vm_mem_t const * vm, u64 vaddr, tlb_entry_t * out ) {
  u64 region = vaddr >> 32;
  u64 offset = vaddr & VM_OFFSET_MASK;
  if( region == 4ULL ) {
    u32 idx = fd_bsearch_input( vm, offset );
    input_region_t const * r = &vm->input_regions[idx];
    u64 rel = offset - r->vaddr_offset;
    if( offset < r->vaddr_offset || rel > r->region_sz || 8ULL > r->region_sz - rel ) return 0ULL;
    out->lo    = VM_INPUT_START + r->vaddr_offset;
    out->hi    = out->lo + r->region_sz;
    out->delta = r->haddr - out->lo;
    return r->haddr + rel;
  }
  u64 rsz = (u64)vm->region_ld_sz[region];
  if( offset > rsz || 8ULL > rsz - offset ) return 0ULL;
  out->lo    = region << 32;
  out->hi    = out->lo + rsz;
  out->delta = vm->region_haddr[region] - out->lo;
  return vm->region_haddr[region] + offset;
}

static u32 eytz_build( vm_mem_t * vm, u32 in_i, u32 out_i ) {
  if( out_i > vm->input_cnt ) return in_i;
  in_i = eytz_build( vm, in_i, out_i * 2U );
  vm->eytz_addr[out_i] = VM_INPUT_START + vm->input_regions[in_i].vaddr_offset;
  vm->eytz_idx[out_i]  = in_i;
  return eytz_build( vm, in_i + 1U, out_i * 2U + 1U );
}

static inline u64
eytz_translate_ld( vm_mem_t const * vm, u64 vaddr ) {
  u32 n = vm->input_cnt;
  u32 k = 1U;
  while( k <= n ) {
    k = (k << 1) + (u32)(vm->eytz_addr[k] <= vaddr);
  }
  k >>= __builtin_ctz( k ) + 1;
  if( __builtin_expect( k == 0U, 0 ) ) return 0ULL;
  u32 idx = vm->eytz_idx[k];
  input_region_t const * r = &vm->input_regions[idx];
  u64 off = vaddr - VM_INPUT_START;
  u64 rel = off - r->vaddr_offset;
  return (rel + 8ULL <= r->region_sz) ? (r->haddr + rel) : 0ULL;
}

typedef struct __attribute__((aligned(64))) { u64 lo[32], hi[32], delta[32]; u32 rr; } tlb64_mem_t;
typedef struct __attribute__((aligned(64))) { u32 lo32[32], hi32[32]; u64 delta[32]; u32 rr; } tlb32_mem_t;

__attribute__((noinline))
static u64 calibrate_add( u64 n ) {
  u64 x = 1;
  for( u64 i = 0; i < n; i++ ) {
    __asm__ volatile(
      "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
      "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
      : "+r"(x) );
  }
  return x;
}

/* Instantiate EVERY benchmark cleanly for BOTH DEP=1 (serial latency) and DEP=0 (pure ILP throughput)
 * so clang NEVER emits a spurious `cmovne` dependency in throughput mode! */
#define DEFINE_BENCH_PAIR(NAME, INIT_CODE, ...) \
__attribute__((noinline)) \
static u64 bench_##NAME##_lat( vm_mem_t const * vm, u64 const * addrs, size_t n ) { \
  INIT_CODE; \
  u64 acc = 0; \
  for( size_t i = 0; i < n; i++ ) { \
    u64 v = addrs[i] + (acc & 8ULL); \
    u64 h; \
    __VA_ARGS__; \
    acc ^= *(u64 const *)h; \
  } \
  return acc; \
} \
__attribute__((noinline)) \
static u64 bench_##NAME##_tp( vm_mem_t const * vm, u64 const * addrs, size_t n ) { \
  INIT_CODE; \
  u64 acc = 0; \
  for( size_t i = 0; i < n; i++ ) { \
    u64 v = addrs[i]; \
    u64 h; \
    __VA_ARGS__; \
    acc ^= *(u64 const *)h; \
  } \
  return acc; \
}

/* 0. Roofline flat base+vaddr */
DEFINE_BENCH_PAIR( roofline_flat,
  u64 base = vm->region_haddr[2] - 0x200000000ULL,
  {
    u64 vv = 0x200000000ULL + (v & 0xFE0ULL);
    u32 off = (u32)vv;
    h = (off <= REGION_BUF_SZ - 8U) ? (base + vv) : 0ULL;
    __asm__ volatile("" : "+r"(h));
  }
)

/* 1. Firedancer main scalar bsearch */
DEFINE_BENCH_PAIR( fd_bsearch, (void)0, { h = fd_translate_ld( vm, v ); } )

/* 2. Agave branchless Eytzinger */
DEFINE_BENCH_PAIR( agave_eytz, (void)0, { h = ((v >> 32) == 4ULL) ? eytz_translate_ld( vm, v ) : fd_translate_ld( vm, v ); } )

/* 3. Firedancer ripatel/jit 1-slot scalar TLB */
DEFINE_BENCH_PAIR( ripatel_tlb1,
  u64 tlb_lo = ~0ULL; u64 tlb_hi = 0ULL; u64 tlb_delta = 0ULL,
  {
    if( __builtin_expect( v >= tlb_lo && v + 8ULL <= tlb_hi, 1 ) ) {
      h = v + tlb_delta;
    } else {
      tlb_entry_t e; h = resolve_miss_ld( vm, v, &e );
      tlb_lo = e.lo; tlb_hi = e.hi; tlb_delta = e.delta;
    }
  }
)

/* 4. L1-backed 8-slot u64 (0 res ZMM, 2 scratch) */
DEFINE_BENCH_PAIR( l1_tlb8_u64,
  tlb64_mem_t tlb;
  for( int j = 0; j < 8; j++ ) { tlb.lo[j] = ~0ULL; tlb.hi[j] = 0; tlb.delta[j] = 0; }
  tlb.rr = 0; tlb64_mem_t * p = &tlb,
  {
    __asm__ volatile("" : "+m"(*p));
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo), zv) & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi));
    if( __builtin_expect( k != 0, 1 ) ) h = v + p->delta[__builtin_ctz((u32)k)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = p->rr++ & 7U; p->lo[s]=e.lo; p->hi[s]=e.hi; p->delta[s]=e.delta; }
  }
)

/* 5. L1-backed 16-slot u64 (0 res ZMM, 2 scratch) */
DEFINE_BENCH_PAIR( l1_tlb16_u64,
  tlb64_mem_t tlb;
  for( int j = 0; j < 16; j++ ) { tlb.lo[j] = ~0ULL; tlb.hi[j] = 0; tlb.delta[j] = 0; }
  tlb.rr = 0; tlb64_mem_t * p = &tlb,
  {
    __asm__ volatile("" : "+m"(*p));
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo), zv)     & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi));
    __mmask8 k1 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo + 8), zv) & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi + 8));
    u32 m = (u32)k0 | ((u32)k1 << 8);
    if( __builtin_expect( m != 0, 1 ) ) h = v + p->delta[__builtin_ctz(m)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = p->rr++ & 15U; p->lo[s]=e.lo; p->hi[s]=e.hi; p->delta[s]=e.delta; }
  }
)

/* 6. L1-backed 32-slot u64 (0 res ZMM, 2 scratch) */
DEFINE_BENCH_PAIR( l1_tlb32_u64,
  tlb64_mem_t tlb;
  for( int j = 0; j < 32; j++ ) { tlb.lo[j] = ~0ULL; tlb.hi[j] = 0; tlb.delta[j] = 0; }
  tlb.rr = 0; tlb64_mem_t * p = &tlb,
  {
    __asm__ volatile("" : "+m"(*p));
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo), zv)      & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi));
    __mmask8 k1 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo + 8), zv)  & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi + 8));
    __mmask8 k2 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo + 16), zv) & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi + 16));
    __mmask8 k3 = _mm512_cmple_epu64_mask(_mm512_load_si512(p->lo + 24), zv) & _mm512_cmple_epu64_mask(zend, _mm512_load_si512(p->hi + 24));
    u32 m = (u32)k0 | ((u32)k1 << 8) | ((u32)k2 << 16) | ((u32)k3 << 24);
    if( __builtin_expect( m != 0, 1 ) ) h = v + p->delta[__builtin_ctz(m)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = p->rr++ & 31U; p->lo[s]=e.lo; p->hi[s]=e.hi; p->delta[s]=e.delta; }
  }
)

/* 7. L1-backed 16-slot u32 input TLB (0 res ZMM, 2 scratch) */
DEFINE_BENCH_PAIR( l1_tlb16_u32,
  tlb32_mem_t tlb;
  for( int j = 0; j < 16; j++ ) { tlb.lo32[j] = ~0U; tlb.hi32[j] = 0; tlb.delta[j] = 0; }
  tlb.rr = 0; tlb32_mem_t * p = &tlb,
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __asm__ volatile("" : "+m"(*p));
      __m512i zv = _mm512_set1_epi32((int)off), zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k = _mm512_cmple_epu32_mask(_mm512_load_si512(p->lo32), zv) & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(p->hi32));
      if( __builtin_expect( k != 0, 1 ) ) h = v + p->delta[__builtin_ctz((u32)k)];
      else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = p->rr++ & 15U; p->lo32[s]=(u32)(e.lo-VM_INPUT_START); p->hi32[s]=(u32)(e.hi-VM_INPUT_START); p->delta[s]=e.delta; }
    } else h = fd_translate_ld(vm, v);
  }
)

/* 8. L1-backed 32-slot u32 input TLB (0 res ZMM, 2 scratch) */
DEFINE_BENCH_PAIR( l1_tlb32_u32,
  tlb32_mem_t tlb;
  for( int j = 0; j < 32; j++ ) { tlb.lo32[j] = ~0U; tlb.hi32[j] = 0; tlb.delta[j] = 0; }
  tlb.rr = 0; tlb32_mem_t * p = &tlb,
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __asm__ volatile("" : "+m"(*p));
      __m512i zv = _mm512_set1_epi32((int)off), zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k0 = _mm512_cmple_epu32_mask(_mm512_load_si512(p->lo32), zv)      & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(p->hi32));
      __mmask16 k1 = _mm512_cmple_epu32_mask(_mm512_load_si512(p->lo32 + 16), zv) & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(p->hi32 + 16));
      u32 m = (u32)k0 | ((u32)k1 << 16);
      if( __builtin_expect( m != 0, 1 ) ) h = v + p->delta[__builtin_ctz(m)];
      else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = p->rr++ & 31U; p->lo32[s]=(u32)(e.lo-VM_INPUT_START); p->hi32[s]=(u32)(e.hi-VM_INPUT_START); p->delta[s]=e.delta; }
    } else h = fd_translate_ld(vm, v);
  }
)

/* 9. L1-backed 16-way SIMD B-tree (0 res ZMM, 2 scratch, ZERO misses up to 256!) */
DEFINE_BENCH_PAIR( l1_btree16_u32, (void)0,
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __asm__ volatile("" : "+r"(vm));
      __m512i zv   = _mm512_set1_epi32((int)off);
      __mmask16 rk = _mm512_cmplt_epu32_mask(zv, _mm512_load_si512(vm->btree_root_hi32));
      u32 base = (u32)__builtin_ctz((u32)rk | (1U << 15)) << 4;
      __m512i zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k = _mm512_cmple_epu32_mask(_mm512_load_si512(vm->in_lo32 + base), zv) & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(vm->in_hi32 + base));
      h = __builtin_expect( k != 0, 1 ) ? (v + vm->in_delta[base + (u32)__builtin_ctz((u32)k)]) : 0ULL;
    } else h = fd_translate_ld(vm, v);
  }
)

/* 10. Hybrid: 1-slot scalar TLB + L1-backed 16-way SIMD B-tree on miss (0 res ZMM, 2 scratch!) */
DEFINE_BENCH_PAIR( hybrid_tlb1_btree16,
  u64 tlb_lo = ~0ULL; u64 tlb_hi = 0ULL; u64 tlb_delta = 0ULL,
  {
    if( __builtin_expect( v >= tlb_lo && v + 8ULL <= tlb_hi, 1 ) ) {
      h = v + tlb_delta;
    } else {
      u64 reg = v >> 32; u32 off = (u32)v;
      if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
        __asm__ volatile("" : "+r"(vm));
        __m512i zv   = _mm512_set1_epi32((int)off);
        __mmask16 rk = _mm512_cmplt_epu32_mask(zv, _mm512_load_si512(vm->btree_root_hi32));
        u32 base = (u32)__builtin_ctz((u32)rk | (1U << 15)) << 4;
        __m512i zend = _mm512_set1_epi32((int)(off + 8U));
        __mmask16 k = _mm512_cmple_epu32_mask(_mm512_load_si512(vm->in_lo32 + base), zv) & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(vm->in_hi32 + base));
        if( __builtin_expect( k != 0, 1 ) ) {
          u32 idx = base + (u32)__builtin_ctz((u32)k);
          tlb_lo    = VM_INPUT_START + vm->in_lo32[idx];
          tlb_hi    = VM_INPUT_START + vm->in_hi32[idx];
          tlb_delta = vm->in_delta[idx];
          h = v + tlb_delta;
        } else h = 0ULL;
      } else h = fd_translate_ld(vm, v);
    }
  }
)

/* 11. reg1_btree16_u32 (1 res ZMM for root + 2 scratch = 3 ZMM, ZERO misses up to 256!) */
DEFINE_BENCH_PAIR( reg1_btree16_u32,
  __m512i z_root = _mm512_load_si512(vm->btree_root_hi32),
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __asm__ volatile("" : "+r"(vm));
      __m512i zv   = _mm512_set1_epi32((int)off);
      __mmask16 rk = _mm512_cmplt_epu32_mask(zv, z_root);
      u32 base = (u32)__builtin_ctz((u32)rk | (1U << 15)) << 4;
      __m512i zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k = _mm512_cmple_epu32_mask(_mm512_load_si512(vm->in_lo32 + base), zv) & _mm512_cmple_epu32_mask(zend, _mm512_load_si512(vm->in_hi32 + base));
      h = __builtin_expect( k != 0, 1 ) ? (v + vm->in_delta[base + (u32)__builtin_ctz((u32)k)]) : 0ULL;
    } else h = fd_translate_ld(vm, v);
  }
)

/* 12. reg_tlb8_memd (2 res + 2 scratch = 4 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb8_memd,
  __m512i z_lo = _mm512_set1_epi64(-1LL); __m512i z_hi = _mm512_setzero_si512();
  u64 delta[8] __attribute__((aligned(64))) = {0}; u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k = _mm512_cmple_epu64_mask(z_lo, zv) & _mm512_cmple_epu64_mask(zend, z_hi);
    if( __builtin_expect( k != 0, 1 ) ) h = v + delta[__builtin_ctz((u32)k)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 7U; __mmask8 sm = (__mmask8)(1U << s); z_lo = _mm512_mask_set1_epi64(z_lo, sm, (long long)e.lo); z_hi = _mm512_mask_set1_epi64(z_hi, sm, (long long)e.hi); delta[s] = e.delta; }
  }
)

/* 13. reg_tlb8_full (3 res + 2 scratch = 5 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb8_full,
  __m512i z_lo = _mm512_set1_epi64(-1LL); __m512i z_hi = _mm512_setzero_si512(); __m512i z_d = _mm512_setzero_si512();
  u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k = _mm512_cmple_epu64_mask(z_lo, zv) & _mm512_cmple_epu64_mask(zend, z_hi);
    if( __builtin_expect( k != 0, 1 ) ) { __m512i sel = _mm512_maskz_compress_epi64(k, z_d); h = v + (u64)_mm_cvtsi128_si64(_mm512_castsi512_si128(sel)); }
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 7U; __mmask8 sm = (__mmask8)(1U << s); z_lo = _mm512_mask_set1_epi64(z_lo, sm, (long long)e.lo); z_hi = _mm512_mask_set1_epi64(z_hi, sm, (long long)e.hi); z_d = _mm512_mask_set1_epi64(z_d, sm, (long long)e.delta); }
  }
)

/* 14. reg_tlb16_u32 (2 res + 2 scratch = 4 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb16_u32,
  __m512i z_lo = _mm512_set1_epi32(-1); __m512i z_hi = _mm512_setzero_si512();
  u64 delta[16] __attribute__((aligned(64))) = {0}; u32 rr = 0,
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __m512i zv = _mm512_set1_epi32((int)off), zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k = _mm512_cmple_epu32_mask(z_lo, zv) & _mm512_cmple_epu32_mask(zend, z_hi);
      if( __builtin_expect( k != 0, 1 ) ) h = v + delta[__builtin_ctz((u32)k)];
      else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 15U; __mmask16 sm = (__mmask16)(1U << s); z_lo = _mm512_mask_set1_epi32(z_lo, sm, (int)(e.lo-VM_INPUT_START)); z_hi = _mm512_mask_set1_epi32(z_hi, sm, (int)(e.hi-VM_INPUT_START)); delta[s] = e.delta; }
    } else h = fd_translate_ld(vm, v);
  }
)

/* 15. reg_tlb16_memd (4 res + 2 scratch = 6 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb16_memd,
  __m512i z_lo0 = _mm512_set1_epi64(-1LL); __m512i z_lo1 = _mm512_set1_epi64(-1LL);
  __m512i z_hi0 = _mm512_setzero_si512();  __m512i z_hi1 = _mm512_setzero_si512();
  u64 delta[16] __attribute__((aligned(64))) = {0}; u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(z_lo0, zv) & _mm512_cmple_epu64_mask(zend, z_hi0);
    __mmask8 k1 = _mm512_cmple_epu64_mask(z_lo1, zv) & _mm512_cmple_epu64_mask(zend, z_hi1);
    u32 m = (u32)k0 | ((u32)k1 << 8);
    if( __builtin_expect( m != 0, 1 ) ) h = v + delta[__builtin_ctz(m)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 15U; __mmask8 sm = (__mmask8)(1U << (s & 7U)); if(s<8U){z_lo0=_mm512_mask_set1_epi64(z_lo0,sm,(long long)e.lo);z_hi0=_mm512_mask_set1_epi64(z_hi0,sm,(long long)e.hi);}else{z_lo1=_mm512_mask_set1_epi64(z_lo1,sm,(long long)e.lo);z_hi1=_mm512_mask_set1_epi64(z_hi1,sm,(long long)e.hi);} delta[s] = e.delta; }
  }
)

/* 16. reg_tlb16_full (6 res + 2 scratch = 8 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb16_full,
  __m512i z_lo0 = _mm512_set1_epi64(-1LL); __m512i z_lo1 = _mm512_set1_epi64(-1LL);
  __m512i z_hi0 = _mm512_setzero_si512();  __m512i z_hi1 = _mm512_setzero_si512();
  __m512i z_d0  = _mm512_setzero_si512();  __m512i z_d1  = _mm512_setzero_si512();
  u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(z_lo0, zv) & _mm512_cmple_epu64_mask(zend, z_hi0);
    __mmask8 k1 = _mm512_cmple_epu64_mask(z_lo1, zv) & _mm512_cmple_epu64_mask(zend, z_hi1);
    if( __builtin_expect( (k0 | k1) != 0, 1 ) ) {
      __m512i sel = _mm512_or_si512(_mm512_maskz_compress_epi64(k0, z_d0), _mm512_maskz_compress_epi64(k1, z_d1));
      h = v + (u64)_mm_cvtsi128_si64(_mm512_castsi512_si128(sel));
    } else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 15U; __mmask8 sm = (__mmask8)(1U << (s & 7U)); if(s<8U){z_lo0=_mm512_mask_set1_epi64(z_lo0,sm,(long long)e.lo);z_hi0=_mm512_mask_set1_epi64(z_hi0,sm,(long long)e.hi);z_d0=_mm512_mask_set1_epi64(z_d0,sm,(long long)e.delta);}else{z_lo1=_mm512_mask_set1_epi64(z_lo1,sm,(long long)e.lo);z_hi1=_mm512_mask_set1_epi64(z_hi1,sm,(long long)e.hi);z_d1=_mm512_mask_set1_epi64(z_d1,sm,(long long)e.delta);} }
  }
)

/* 17. reg_tlb32_u32 (4 res + 2 scratch = 6 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb32_u32,
  __m512i z_lo0 = _mm512_set1_epi32(-1);   __m512i z_lo1 = _mm512_set1_epi32(-1);
  __m512i z_hi0 = _mm512_setzero_si512();  __m512i z_hi1 = _mm512_setzero_si512();
  u64 delta[32] __attribute__((aligned(64))) = {0}; u32 rr = 0,
  {
    u64 reg = v >> 32; u32 off = (u32)v;
    if( __builtin_expect( reg == 4ULL && off <= 0xFFFFFFF7U, 1 ) ) {
      __m512i zv = _mm512_set1_epi32((int)off), zend = _mm512_set1_epi32((int)(off + 8U));
      __mmask16 k0 = _mm512_cmple_epu32_mask(z_lo0, zv) & _mm512_cmple_epu32_mask(zend, z_hi0);
      __mmask16 k1 = _mm512_cmple_epu32_mask(z_lo1, zv) & _mm512_cmple_epu32_mask(zend, z_hi1);
      u32 m = (u32)k0 | ((u32)k1 << 16);
      if( __builtin_expect( m != 0, 1 ) ) h = v + delta[__builtin_ctz(m)];
      else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 31U; __mmask16 sm = (__mmask16)(1U << (s & 15U)); if(s<16U){z_lo0=_mm512_mask_set1_epi32(z_lo0,sm,(int)(e.lo-VM_INPUT_START));z_hi0=_mm512_mask_set1_epi32(z_hi0,sm,(int)(e.hi-VM_INPUT_START));}else{z_lo1=_mm512_mask_set1_epi32(z_lo1,sm,(int)(e.lo-VM_INPUT_START));z_hi1=_mm512_mask_set1_epi32(z_hi1,sm,(int)(e.hi-VM_INPUT_START));} delta[s] = e.delta; }
    } else h = fd_translate_ld(vm, v);
  }
)

/* 18. reg_tlb32_memd (8 res + 2 scratch = 10 ZMM) */
DEFINE_BENCH_PAIR( reg_tlb32_memd,
  __m512i z_lo0 = _mm512_set1_epi64(-1LL); __m512i z_lo1 = _mm512_set1_epi64(-1LL); __m512i z_lo2 = _mm512_set1_epi64(-1LL); __m512i z_lo3 = _mm512_set1_epi64(-1LL);
  __m512i z_hi0 = _mm512_setzero_si512();  __m512i z_hi1 = _mm512_setzero_si512();  __m512i z_hi2 = _mm512_setzero_si512();  __m512i z_hi3 = _mm512_setzero_si512();
  u64 delta[32] __attribute__((aligned(64))) = {0}; u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(z_lo0, zv) & _mm512_cmple_epu64_mask(zend, z_hi0);
    __mmask8 k1 = _mm512_cmple_epu64_mask(z_lo1, zv) & _mm512_cmple_epu64_mask(zend, z_hi1);
    __mmask8 k2 = _mm512_cmple_epu64_mask(z_lo2, zv) & _mm512_cmple_epu64_mask(zend, z_hi2);
    __mmask8 k3 = _mm512_cmple_epu64_mask(z_lo3, zv) & _mm512_cmple_epu64_mask(zend, z_hi3);
    u32 m = (u32)k0 | ((u32)k1 << 8) | ((u32)k2 << 16) | ((u32)k3 << 24);
    if( __builtin_expect( m != 0, 1 ) ) h = v + delta[__builtin_ctz(m)];
    else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 31U; __mmask8 sm = (__mmask8)(1U << (s & 7U));
      if(s<8U){z_lo0=_mm512_mask_set1_epi64(z_lo0,sm,(long long)e.lo);z_hi0=_mm512_mask_set1_epi64(z_hi0,sm,(long long)e.hi);}
      else if(s<16U){z_lo1=_mm512_mask_set1_epi64(z_lo1,sm,(long long)e.lo);z_hi1=_mm512_mask_set1_epi64(z_hi1,sm,(long long)e.hi);}
      else if(s<24U){z_lo2=_mm512_mask_set1_epi64(z_lo2,sm,(long long)e.lo);z_hi2=_mm512_mask_set1_epi64(z_hi2,sm,(long long)e.hi);}
      else{z_lo3=_mm512_mask_set1_epi64(z_lo3,sm,(long long)e.lo);z_hi3=_mm512_mask_set1_epi64(z_hi3,sm,(long long)e.hi);}
      delta[s] = e.delta;
    }
  }
)

/* 19. reg_tlb32_full (12 res + 4 scratch = 16 ZMM total) */
DEFINE_BENCH_PAIR( reg_tlb32_full,
  __m512i z_lo0 = _mm512_set1_epi64(-1LL); __m512i z_lo1 = _mm512_set1_epi64(-1LL); __m512i z_lo2 = _mm512_set1_epi64(-1LL); __m512i z_lo3 = _mm512_set1_epi64(-1LL);
  __m512i z_hi0 = _mm512_setzero_si512();  __m512i z_hi1 = _mm512_setzero_si512();  __m512i z_hi2 = _mm512_setzero_si512();  __m512i z_hi3 = _mm512_setzero_si512();
  __m512i z_d0  = _mm512_setzero_si512();  __m512i z_d1  = _mm512_setzero_si512();  __m512i z_d2  = _mm512_setzero_si512();  __m512i z_d3  = _mm512_setzero_si512();
  u32 rr = 0,
  {
    __m512i zv = _mm512_set1_epi64((long long)v), zend = _mm512_set1_epi64((long long)(v + 8ULL));
    __mmask8 k0 = _mm512_cmple_epu64_mask(z_lo0, zv) & _mm512_cmple_epu64_mask(zend, z_hi0);
    __mmask8 k1 = _mm512_cmple_epu64_mask(z_lo1, zv) & _mm512_cmple_epu64_mask(zend, z_hi1);
    __mmask8 k2 = _mm512_cmple_epu64_mask(z_lo2, zv) & _mm512_cmple_epu64_mask(zend, z_hi2);
    __mmask8 k3 = _mm512_cmple_epu64_mask(z_lo3, zv) & _mm512_cmple_epu64_mask(zend, z_hi3);
    if( __builtin_expect( (k0 | k1 | k2 | k3) != 0, 1 ) ) {
      __m512i sel = _mm512_or_si512(
        _mm512_or_si512(_mm512_maskz_compress_epi64(k0, z_d0), _mm512_maskz_compress_epi64(k1, z_d1)),
        _mm512_or_si512(_mm512_maskz_compress_epi64(k2, z_d2), _mm512_maskz_compress_epi64(k3, z_d3)));
      h = v + (u64)_mm_cvtsi128_si64(_mm512_castsi512_si128(sel));
    } else { tlb_entry_t e; h = resolve_miss_ld(vm, v, &e); u32 s = rr++ & 31U; __mmask8 sm = (__mmask8)(1U << (s & 7U));
      if(s<8U){z_lo0=_mm512_mask_set1_epi64(z_lo0,sm,(long long)e.lo);z_hi0=_mm512_mask_set1_epi64(z_hi0,sm,(long long)e.hi);z_d0=_mm512_mask_set1_epi64(z_d0,sm,(long long)e.delta);}
      else if(s<16U){z_lo1=_mm512_mask_set1_epi64(z_lo1,sm,(long long)e.lo);z_hi1=_mm512_mask_set1_epi64(z_hi1,sm,(long long)e.hi);z_d1=_mm512_mask_set1_epi64(z_d1,sm,(long long)e.delta);}
      else if(s<24U){z_lo2=_mm512_mask_set1_epi64(z_lo2,sm,(long long)e.lo);z_hi2=_mm512_mask_set1_epi64(z_hi2,sm,(long long)e.hi);z_d2=_mm512_mask_set1_epi64(z_d2,sm,(long long)e.delta);}
      else{z_lo3=_mm512_mask_set1_epi64(z_lo3,sm,(long long)e.lo);z_hi3=_mm512_mask_set1_epi64(z_hi3,sm,(long long)e.hi);z_d3=_mm512_mask_set1_epi64(z_d3,sm,(long long)e.delta);}
    }
  }
)

/* ============================================================================
 * Setup + Driver
 * ============================================================================ */
static u64 rng_s = 0xfeedfacecafebeefULL;
static inline u64 xorshift64( void ) {
  u64 x = rng_s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return rng_s = x;
}

static vm_mem_t * init_vm( u32 n_in, u8 ** backing_out ) {
  vm_mem_t * vm = aligned_alloc( 64, sizeof(vm_mem_t) );
  memset( vm, 0, sizeof(vm_mem_t) );
  size_t total = 3 + n_in;
  u8 * backing = aligned_alloc( 64, total * REGION_BUF_SZ );
  for( size_t i = 0; i < total * REGION_BUF_SZ; i += 8 ) *(u64 *)(backing + i) = xorshift64();
  *backing_out = backing;

  vm->region_haddr[0] = (u64)(backing + 0 * REGION_BUF_SZ); vm->region_ld_sz[0] = REGION_BUF_SZ;
  vm->region_haddr[2] = (u64)(backing + 1 * REGION_BUF_SZ); vm->region_ld_sz[2] = REGION_BUF_SZ; vm->region_st_sz[2] = REGION_BUF_SZ;
  vm->region_haddr[3] = (u64)(backing + 2 * REGION_BUF_SZ); vm->region_ld_sz[3] = REGION_BUF_SZ; vm->region_st_sz[3] = REGION_BUF_SZ;

  u32 perm[MAX_INPUT_REGIONS];
  for( u32 i = 0; i < n_in; i++ ) perm[i] = i;
  for( u32 i = n_in - 1; i > 0; i-- ) { u32 j = (u32)(xorshift64() % (i + 1)); u32 t = perm[i]; perm[i] = perm[j]; perm[j] = t; }

  vm->input_cnt = n_in;
  for( u32 i = 0; i < MAX_INPUT_REGIONS + 16; i++ ) { vm->in_lo32[i] = ~0U; vm->in_hi32[i] = 0; vm->in_delta[i] = 0; }
  u64 cur = 0;
  for( u32 i = 0; i < n_in; i++ ) {
    u32 rsz = 1024U + (u32)((xorshift64() % 3) * 1024U);
    u32 res = rsz + 10240U;
    u64 ha  = (u64)(backing + (3 + perm[i]) * REGION_BUF_SZ);
    vm->input_regions[i].vaddr_offset = cur;
    vm->input_regions[i].haddr = ha;
    vm->input_regions[i].region_sz = rsz;
    vm->input_regions[i].address_space_reserved = res;
    vm->input_regions[i].is_writable = 1;
    vm->in_lo32[i] = (u32)cur;
    vm->in_hi32[i] = (u32)(cur + rsz);
    vm->in_delta[i] = ha - (VM_INPUT_START + cur);
    cur += res;
  }
  u32 nblk = (n_in + 15U) >> 4;
  for( u32 b = 0; b < 16; b++ ) {
    if( b < nblk ) {
      u32 last = ((b + 1U) << 4) - 1U;
      if( last >= n_in ) last = n_in - 1U;
      vm->btree_root_hi32[b] = (u32)(vm->input_regions[last].vaddr_offset + vm->input_regions[last].address_space_reserved);
    } else vm->btree_root_hi32[b] = ~0U;
  }
  eytz_build( vm, 0, 1 );
  return vm;
}

static u64 * gen_input_trace( vm_mem_t const * vm, size_t n, u32 active, int zipf ) {
  u64 * a = malloc( n * sizeof(u64) );
  for( size_t i = 0; i < n; i++ ) {
    u64 r = xorshift64();
    u32 idx;
    if( zipf ) {
      u32 hot = active > 4 ? (active / 3) : active;
      idx = (((r >> 8) % 100) < 75) ? (u32)((r >> 16) % hot) : (u32)((r >> 16) % active);
    } else {
      idx = (u32)((r >> 16) % active);
    }
    input_region_t const * ir = &vm->input_regions[idx];
    u64 off = ((r >> 32) % ((ir->region_sz - 32U) / 16U)) * 16ULL;
    a[i] = VM_INPUT_START + ir->vaddr_offset + off;
  }
  return a;
}

typedef u64 (*fn_t)( vm_mem_t const *, u64 const *, size_t );
typedef struct { char const * name; int res; int tot; fn_t lat; fn_t tp; int is_roof; } impl_t;

static double tns( struct timespec a, struct timespec b ) {
  return (double)(b.tv_sec - a.tv_sec) * 1e9 + (double)(b.tv_nsec - a.tv_nsec);
}
static int cmpd( void const * a, void const * b ) {
  double x = *(double const *)a, y = *(double const *)b;
  return (x > y) - (x < y);
}

static double core_ghz = 3.7;

static void run_suite( char const * label, u32 total_in, u32 active_in, int zipf ) {
  const size_t N = 1 << 18;
  const int REPS = 7;
  u8 * backing = NULL;
  vm_mem_t * vm = init_vm( total_in, &backing );
  u64 * addrs = gen_input_trace( vm, N, active_in, zipf );

  impl_t impls[] = {
    { "roofline: flat base+vaddr",     0,  0, bench_roofline_flat_lat,     bench_roofline_flat_tp,     1 },
    { "fd_bsearch (main)",             0,  0, bench_fd_bsearch_lat,        bench_fd_bsearch_tp,        0 },
    { "agave_eytzinger",               0,  0, bench_agave_eytz_lat,        bench_agave_eytz_tp,        0 },
    { "ripatel_tlb1 (jit)",            0,  0, bench_ripatel_tlb1_lat,      bench_ripatel_tlb1_tp,      0 },
    { "l1_tlb8_u64  (0 res)",          0,  2, bench_l1_tlb8_u64_lat,       bench_l1_tlb8_u64_tp,       0 },
    { "l1_tlb16_u64 (0 res)",          0,  2, bench_l1_tlb16_u64_lat,      bench_l1_tlb16_u64_tp,      0 },
    { "l1_tlb32_u64 (0 res)",          0,  2, bench_l1_tlb32_u64_lat,      bench_l1_tlb32_u64_tp,      0 },
    { "l1_tlb16_u32 (0 res)",          0,  2, bench_l1_tlb16_u32_lat,      bench_l1_tlb16_u32_tp,      0 },
    { "l1_tlb32_u32 (0 res)",          0,  2, bench_l1_tlb32_u32_lat,      bench_l1_tlb32_u32_tp,      0 },
    { "l1_btree16_u32 (0 res,0ms)",    0,  2, bench_l1_btree16_u32_lat,    bench_l1_btree16_u32_tp,    0 },
    { "hybrid_tlb1+btree16 (0 res)",   0,  2, bench_hybrid_tlb1_btree16_lat, bench_hybrid_tlb1_btree16_tp, 0 },
    { "reg1_btree16_u32 (1 res,0ms)",  1,  3, bench_reg1_btree16_u32_lat,  bench_reg1_btree16_u32_tp,  0 },
    { "reg_tlb8_memd  (2 res)",        2,  4, bench_reg_tlb8_memd_lat,     bench_reg_tlb8_memd_tp,     0 },
    { "reg_tlb8_full  (3 res)",        3,  5, bench_reg_tlb8_full_lat,     bench_reg_tlb8_full_tp,     0 },
    { "reg_tlb16_u32  (2 res)",        2,  4, bench_reg_tlb16_u32_lat,     bench_reg_tlb16_u32_tp,     0 },
    { "reg_tlb16_memd (4 res)",        4,  6, bench_reg_tlb16_memd_lat,    bench_reg_tlb16_memd_tp,    0 },
    { "reg_tlb16_full (6 res)",        6,  8, bench_reg_tlb16_full_lat,    bench_reg_tlb16_full_tp,    0 },
    { "reg_tlb32_u32  (4 res)",        4,  6, bench_reg_tlb32_u32_lat,     bench_reg_tlb32_u32_tp,     0 },
    { "reg_tlb32_memd (8 res)",        8, 10, bench_reg_tlb32_memd_lat,    bench_reg_tlb32_memd_tp,    0 },
    { "reg_tlb32_full (12 res)",      12, 16, bench_reg_tlb32_full_lat,    bench_reg_tlb32_full_tp,    0 },
  };
  int nim = (int)(sizeof(impls) / sizeof(impls[0]));

  printf( "\n=== %s (total_in=%u, active_in=%u, zipf=%d) ===\n", label, total_in, active_in, zipf );
  printf( "%-30s | %3s | %3s | %8s (%5s) | %8s (%5s)\n", "implementation", "res", "tot", "lat ns", "~cyc", "tp ns", "~cyc" );
  printf( "-------------------------------+-----+-----+-----------------+-----------------\n" );

  u64 ref_lat = 0, ref_tp = 0;
  for( int k = 0; k < nim; k++ ) {
    double tl[REPS], tt[REPS];
    u64 cl = 0, ct = 0;
    for( int r = 0; r < REPS + 1; r++ ) {
      struct timespec t0, t1, t2;
      clock_gettime( CLOCK_MONOTONIC, &t0 );
      cl = impls[k].lat( vm, addrs, N );
      clock_gettime( CLOCK_MONOTONIC, &t1 );
      ct = impls[k].tp( vm, addrs, N );
      clock_gettime( CLOCK_MONOTONIC, &t2 );
      if( r > 0 ) {
        tl[r - 1] = tns( t0, t1 ) / (double)N;
        tt[r - 1] = tns( t1, t2 ) / (double)N;
      }
    }
    if( !impls[k].is_roof ) {
      if( ref_lat == 0 ) { ref_lat = cl; ref_tp = ct; }
      else if( cl != ref_lat || ct != ref_tp ) { fprintf( stderr, "MISMATCH %s %s\n", label, impls[k].name ); exit( 1 ); }
    }
    qsort( tl, REPS, sizeof(double), cmpd );
    qsort( tt, REPS, sizeof(double), cmpd );
    double ml = tl[REPS / 2], mt = tt[REPS / 2];
    printf( "%-30s | %3d | %3d | %8.2f (%5.1f) | %8.2f (%5.1f)\n",
            impls[k].name, impls[k].res, impls[k].tot, ml, ml * core_ghz, mt, mt * core_ghz );
  }
  free( addrs ); free( backing ); free( vm );
}

int main( void ) {
  struct timespec t0, t1;
  volatile u64 sink = calibrate_add( 5000000ULL );
  clock_gettime( CLOCK_MONOTONIC, &t0 );
  sink ^= calibrate_add( 20000000ULL );
  clock_gettime( CLOCK_MONOTONIC, &t1 );
  (void)sink;
  core_ghz = 160e6 / tns( t0, t1 );
  printf( "Calibrated core clock on pinned Zen 4 core: %.3f GHz\n", core_ghz );

  run_suite( "S1: 1 active / 24 (single hot acct)",   24,   1, 0 );
  run_suite( "S2: 8 active / 24 (typical AMM)",       24,   8, 0 );
  run_suite( "S3: 16 active / 48 (multi-hop DeFi)",   48,  16, 0 );
  run_suite( "S4: 32 active / 64 (heavy zipf)",       64,  32, 1 );
  run_suite( "S5: 32 active / 64 (heavy unif)",       64,  32, 0 );
  run_suite( "S6: 64 active / 128 (stress unif)",    128,  64, 0 );
  run_suite( "S7: 128 active / 128 (extreme unif)",  128, 128, 0 );
  return 0;
}
