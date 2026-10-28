//! Minimal runtime for `no_std` sBPF programs: a panic handler that aborts
//! through the `abort` syscall. Link it with `use svm_rt as _;`.
#![no_std]

#[cfg(target_arch = "bpf")]
unsafe extern "C" {
    /// The sBPF `abort` syscall (resolved statically by sbpf-linker).
    fn abort() -> !;
}

#[cfg(target_arch = "bpf")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
