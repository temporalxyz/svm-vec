//! programs/sha256 (SHA-256 compression of 16 independent 64-byte blocks) on Dean's vector
//! intrinsics.
//!
//! Instruction data (no accounts) is programs/sha256's 1536 byte input region unchanged: 16 blocks
//! at [0..1024), the 16 x 32-byte outputs written to [1024..1536). Each output is the raw
//! compression function from the standard IV over that one block (no padding or length block),
//! the 8 state words big-endian, as `sha2::compress256` gives them.

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_transpose8x8q(ops: u64) -> u64);
define_syscall!(fn sol_avx2_transpose8x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permi2b512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx2_add32x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ror32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_srl32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_ternlog512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mov512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_shufb512(ops: u64) -> u64);

#[repr(C, align(64))]
pub struct Tab32<const N: usize>([u32; N]);
#[repr(C, align(64))]
pub struct Tab8<const N: usize>([u8; N]);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 1536;

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
    let data = input.add(16);
    sha256x16(data, data.add(1024));
    0
}
