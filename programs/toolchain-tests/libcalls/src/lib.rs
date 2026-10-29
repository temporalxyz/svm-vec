//! Vectors live across memory copies that codegen lowers to calls to the
//! program's own vector `memcpy` and `memset`.
//!
//! input: [0, 8) n1, [8, 16) n2, [64, 128) a, [128, 192) b, [512, 512 + n1)
//! src. output: [192, 256) a, [256, 320) b, [1024, 1024 + n1) src,
//! [1536, 1536 + n2) zero.
#![no_std]

use core::ffi::c_void;

use svm_intrinsics::*;
use svm_rt as _;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void {
    unsafe {
        let (d, s) = (d.cast::<u8>(), s.cast::<u8>());
        let mut i = 0;
        while i + 64 <= n {
            _svm512_storeu_si512(d.add(i), _svm512_loadu_si512(s.add(i)));
            i += 64;
        }
        while i < n {
            core::ptr::write_volatile(d.add(i), core::ptr::read_volatile(s.add(i)));
            i += 1;
        }
        d.cast()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(d: *mut c_void, c: i32, n: usize) -> *mut c_void {
    unsafe {
        let d = d.cast::<u8>();
        let v = _svm512_set1_epi8(c as i8);
        let mut i = 0;
        while i + 64 <= n {
            _svm512_storeu_si512(d.add(i), v);
            i += 64;
        }
        while i < n {
            core::ptr::write_volatile(d.add(i), c as u8);
            i += 1;
        }
        d.cast()
    }
}

/// No vector code of its own: the clobbers come from `memset`.
#[inline(never)]
unsafe fn clear(p: *mut u8, n: usize) {
    unsafe { core::ptr::write_bytes(p, 0, n) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    unsafe {
        let n1 = *(input as *const u64) as usize;
        let n2 = *(input.add(8) as *const u64) as usize;
        let a = _svm512_loadu_si512(input.add(64));
        core::ptr::copy_nonoverlapping(input.add(512), input.add(1024), n1);
        _svm512_storeu_si512(input.add(192), a);
        let b = _svm512_loadu_si512(input.add(128));
        clear(input.add(1536), n2);
        _svm512_storeu_si512(input.add(256), b);
    }
    0
}
