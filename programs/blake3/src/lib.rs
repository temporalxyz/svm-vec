//! BLAKE3 hash of up to 1 MiB, 16 chunks or parents at a time (one per u32 lane).
//!
//! input region: [0..8) len, [8..40) digest (output), [64..64 + len) message.
//! heap: chaining values word-sliced, word w of node j at HEAP + w * STRIDE + 4 j, so a group of
//! 16 nodes is one vector per word and a tree level reads 32 nodes and writes 16 in place.
//! the tree pairs adjacent nodes left to right and carries an odd last node up, which is BLAKE3's
//! left-complete tree.

#![no_std]

use svm_intrinsics::*;
use svm_rt as _;

#[repr(C, align(64))]
pub struct Tab<const N: usize>([[u16; 32]; N]);

include!(concat!(env!("OUT_DIR"), "/gen.rs"));

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;

const DATA: usize = 64;
const MAX_LEN: u64 = 1 << 20;
const HEAP: usize = 0x3_0000_0000;
const STRIDE: usize = 4 * (1024 + 32);
const CARRY: usize = HEAP + 8 * STRIDE;
const PAD: usize = CARRY + 64;
/// compression scratch: message words, the counter (at SCR + CTR)
const SCR: usize = PAD + 64;

static K: [u32; 9] = [IV[0], IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7], 64];

#[inline(always)]
unsafe fn word(w: usize, node: usize) -> *mut u32 {
    (HEAP + w * STRIDE + 4 * node) as *mut u32
}

#[inline(always)]
unsafe fn iv() -> [__svm512i; 8] {
    core::array::from_fn(|i| _svm512_loadbc_epi32(K.as_ptr().add(i) as *const i32))
}

/// 16 chunks, lane l at `base + off(l)`, counters `c0 + l`; returns the word-sliced CVs
#[inline(always)]
unsafe fn chunks16(base: *const u8, c0: usize, off: impl Fn(usize) -> usize) -> [__svm512i; 8] {
    let iota = _svm512_loadu_si512((TAB.0.as_ptr() as *const u8).add(T_IOTA));
    _svm512_storeu_si512((SCR + CTR) as *mut u8, _svm512_add_epi32(_svm512_set1_epi32(c0 as i32), iota));
    let mut cv = iv();
    for b in 0..16 {
        let p = base.add(64 * b);
        let m = transpose(core::array::from_fn(|l| _svm512_loadu_si512(p.add(off(l)))));
        let flags = if b == 0 { CHUNK_START } else if b == 15 { CHUNK_END } else { 0 };
        cv = compress_chunk(cv, m, SCR as *mut u8, bl64, || _svm512_set1_epi32(flags as i32));
    }
    cv
}

#[inline(always)]
fn bl64() -> __svm512i {
    unsafe { _svm512_loadbc_epi32(K.as_ptr().add(8) as *const i32) }
}

#[inline(always)]
unsafe fn store_group(g: usize, cv: [__svm512i; 8]) {
    for w in 0..8 {
        _svm512_storeu_si512(word(w, 16 * g) as *mut u8, cv[w]);
    }
}

/// one chunk of `len` <= 1024 bytes, every lane the same; returns word w of its CV
#[inline(never)]
unsafe fn chunk1(p: *const u8, len: usize, counter: usize, extra: u32, out: *mut u32, stride: usize) {
    let nb = if len == 0 { 1 } else { len.div_ceil(64) };
    _svm512_storeu_si512((SCR + CTR) as *mut u8, _svm512_set1_epi32(counter as i32));
    let mut cv = iv();
    for b in 0..nb {
        let mut q = p.add(64 * b);
        let blen = (len - 64 * b).min(64);
        if blen < 64 {
            let pad = PAD as *mut u8;
            for i in 0..8 {
                (pad as *mut u64).add(i).write_volatile(0);
            }
            for i in 0..blen {
                pad.add(i).write_volatile(q.add(i).read_volatile());
            }
            q = pad;
        }
        let m = core::array::from_fn(|w| _svm512_loadbc_epi32(q.add(4 * w) as *const i32));
        let mut flags = if b == 0 { CHUNK_START } else { 0 };
        if b == nb - 1 {
            flags |= CHUNK_END | extra;
        }
        cv = compress_chunk(cv, m, SCR as *mut u8, || _svm512_set1_epi32(blen as i32), || _svm512_set1_epi32(flags as i32));
    }
    for w in 0..8 {
        out.byte_add(w * stride).write_volatile(_svm512_extract_epi32::<0>(cv[w]) as u32);
    }
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let len = (input as *const u64).read_unaligned();
    if len > MAX_LEN {
        return 1;
    }
    let len = len as usize;
    let data = input.add(DATA) as *const u8;
    let digest = input.add(8) as *mut u32;
    if len <= 1024 {
        chunk1(data, len, 0, ROOT, digest, 4);
        return 0;
    }
    let full = len / 1024;
    let groups = full / 16;
    for g in 0..groups {
        store_group(g, chunks16(data.add(16384 * g), 16 * g, |l| 1024 * l));
    }
    let rest = full - 16 * groups;
    if rest > 0 {
        store_group(groups, chunks16(data.add(16384 * groups), 16 * groups, |l| 1024 * l.min(rest - 1)));
    }
    let mut n = full;
    if len % 1024 != 0 {
        chunk1(data.add(1024 * full), len % 1024, full, 0, word(0, full), STRIDE);
        n += 1;
    }
    let t = TAB.0.as_ptr() as *const u8;
    while n > 2 {
        let pairs = n / 2;
        let odd = n % 2 == 1;
        if odd {
            for w in 0..8 {
                (CARRY as *mut u32).add(w).write_volatile(word(w, n - 1).read_volatile());
            }
        }
        for g in 0..pairs.div_ceil(16) {
            let even = _svm512_loadu_si512(t.add(T_EVEN));
            let oddi = _svm512_loadu_si512(t.add(T_ODD));
            let a: [__svm512i; 8] = core::array::from_fn(|w| _svm512_loadu_si512(word(w, 32 * g) as *const u8));
            let b: [__svm512i; 8] = core::array::from_fn(|w| _svm512_loadu_si512(word(w, 32 * g + 16) as *const u8));
            let m = core::array::from_fn(|i| {
                let w = i % 8;
                _svm512_permutex2var_epi16(a[w], if i < 8 { even } else { oddi }, b[w])
            });
            store_group(g, compress_parent(m, SCR as *mut u8, bl64, || _svm512_set1_epi32(PARENT as i32)));
        }
        if odd {
            for w in 0..8 {
                word(w, pairs).write_volatile((CARRY as *const u32).add(w).read_volatile());
            }
        }
        n = pairs + odd as usize;
    }
    let m = core::array::from_fn(|i| _svm512_loadbc_epi32(word(i % 8, i / 8) as *const i32));
    let cv = compress_parent(m, SCR as *mut u8, bl64, || _svm512_set1_epi32((PARENT | ROOT) as i32));
    for w in 0..8 {
        digest.add(w).write_unaligned(_svm512_extract_epi32::<0>(cv[w]) as u32);
    }
    0
}
