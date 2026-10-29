// int8 GEMM shared by both programs' build.rs and the host reference:
// C (M x N, i32) = A (M x K, u8, the input) x B (K x N, s8, constant weights).

pub const M: usize = 32;
pub const K: usize = 256;
pub const N: usize = 128;
/// input region: A row-major at 0, C row-major (i32 little-endian) at C_OFF
pub const C_OFF: usize = M * K;
pub const LEN: usize = C_OFF + 4 * M * N;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// B row-major, B[k][n]
pub fn weights() -> Vec<[i8; N]> {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    (0..K).map(|_| core::array::from_fn(|_| r.next() as u8 as i8)).collect()
}

/// A row-major
pub fn input(seed: u64) -> Vec<u8> {
    let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    (0..M * K).map(|_| r.next() as u8).collect()
}

pub fn matmul(a: &[u8], b: &[[i8; N]]) -> Vec<i32> {
    let mut c = vec![0i32; M * N];
    for m in 0..M {
        for n in 0..N {
            for k in 0..K {
                c[m * N + n] = c[m * N + n].wrapping_add(a[m * K + k] as i32 * b[k][n] as i32);
            }
        }
    }
    c
}
