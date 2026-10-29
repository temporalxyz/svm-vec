//! programs/fieldmul on Dean's vector intrinsics: 8 lanes of GF(2^255 - 19), r = a, then 256 times
//! r = r * b, output canonical. Instruction data (no accounts) is programs/fieldmul's 1024 byte
//! region unchanged: lane l has a at 64 l and b at 64 l + 32; r is written at 512 + 64 l and b
//! again at 512 + 64 l + 32.

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_transpose8x8q(ops: u64) -> u64);
define_syscall!(fn sol_avx512_bcast64x8(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_mov512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_shrd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_srl64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sll64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_and512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_or512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_add64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sub64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mullo64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_madd52lo64x8(ops: u64) -> u64);
define_syscall!(fn sol_avx512_madd52hi64x8(ops: u64) -> u64);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 1024;

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
    vector(data, data.add(512));
    0
}
