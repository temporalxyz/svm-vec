//! programs/perps on Dean's vector intrinsics.
//!
//! Instruction data (no accounts) is perps-host's 18432 byte input region unchanged: the five
//! input arrays at [0..10240), pnl, funding, equity, healthy written to [10240..18432).

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_bcast64x8(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_add64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sub64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mullo64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sra64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_abs512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_cmp512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_blendm512(ops: u64, mask: u64) -> u64);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 18432;

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
    vector(input.add(16));
    0
}
