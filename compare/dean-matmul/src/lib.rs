//! programs/matmul (int8 GEMM, C = A x B, 32 x 256 x 128) on Dean's vector intrinsics.
//!
//! Instruction data (no accounts) is programs/matmul's 24 KiB input region unchanged: A as u8 at
//! [0..8192), C written as i32 at [8192..24576).

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_permd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_dpbusd512(ops: u64) -> u64);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 24576;
static Z: usize = 0;

#[cfg(target_os = "solana")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    if (input.add(8) as *const u64).read_unaligned() < LEN as u64 {
        return 0x0000_0003_0000_0000;
    }
    // an opaque 0: keeps the .rodata addresses of B away from the JIT's constant tracking (build.rs)
    vector(input.add(16), core::ptr::read_volatile(&Z));
    0
}
