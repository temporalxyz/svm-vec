//! programs/sort (256 u32 keys ascending) on Dean's vector intrinsics.
//!
//! Instruction data (no accounts) is programs/sort's 2048 byte input region unchanged: the keys
//! at [0..1024), the sorted keys written to [1024..2048). build.rs generates the same sorting
//! network (programs/sort/net.rs) with slots picked at build time.

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_minu32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_max512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_unpacklo512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_unpackhi512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permt2d512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_mov512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_shufd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permq512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_permd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_blendd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_blendm512(ops: u64, mask: u64) -> u64);

#[repr(C, align(32))]
struct Tab<const N: usize>([[u32; 16]; N]);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 2048;

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
    vector(data, data.add(1024));
    0
}
