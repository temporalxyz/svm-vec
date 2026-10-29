//! programs/mlp's vector path (int8 mlp 256 -> 128 relu -> 16) on Dean's vector intrinsics.
//!
//! Instruction data (no accounts) is programs/mlp's 384 byte input region unchanged: x as u8 at
//! [64..320), logits written to [320..384) as i32; the mode word is ignored. Returns argmax.

#![no_std]

use solana_define_syscall::define_syscall;

define_syscall!(fn sol_avx_load256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx_store256(ops: u64, addr: u64) -> u64);
define_syscall!(fn sol_avx512_bcast32x16(ops: u64, value: u64) -> u64);
define_syscall!(fn sol_avx512_permd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_dpbusd512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_sra32x16(ops: u64) -> u64);
define_syscall!(fn sol_avx512_max512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_min512(ops: u64) -> u64);
define_syscall!(fn sol_avx512_trunc512(ops: u64) -> u64);

include!(concat!(env!("OUT_DIR"), "/vector.rs"));

const LEN: usize = 384;
const OUT: usize = 16;

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
    let out = data.add(320);
    vector(data.add(64), out);
    let out = out as *const i32;
    let mut best = 0;
    for o in 1..OUT {
        if out.add(o).read_unaligned() > out.add(best).read_unaligned() {
            best = o;
        }
    }
    best as u64
}
