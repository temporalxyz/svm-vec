// int8 mlp shared by the program's build.rs and the host reference.
// 256 u8 inputs -> 128 relu (requantized to 0..127) -> 16 i32 logits.

pub const IN: usize = 256;
pub const HID: usize = 128;
pub const OUT: usize = 16;
pub const SHIFT: u32 = 10;

pub struct Model {
    pub w1: Vec<[i8; IN]>,
    pub b1: Vec<i32>,
    pub w2: Vec<[i8; HID]>,
    pub b2: Vec<i32>,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn i8(&mut self) -> i8 {
        (self.next() % 128) as i8 - 64
    }
    fn bias(&mut self) -> i32 {
        (self.next() % 2048) as i32 - 1024
    }
}

pub fn model() -> Model {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let w1 = (0..HID).map(|_| core::array::from_fn(|_| r.i8())).collect();
    let b1 = (0..HID).map(|_| r.bias()).collect();
    let w2 = (0..OUT).map(|_| core::array::from_fn(|_| r.i8())).collect();
    let b2 = (0..OUT).map(|_| r.bias()).collect();
    Model { w1, b1, w2, b2 }
}

pub fn input(seed: u64) -> [u8; IN] {
    let mut r = Rng(seed | 1);
    core::array::from_fn(|_| (r.next() % 128) as u8)
}

pub fn forward(m: &Model, x: &[u8; IN]) -> [i32; OUT] {
    let mut h = [0u8; HID];
    for o in 0..HID {
        let acc = (0..IN).fold(m.b1[o], |a, i| a.wrapping_add(x[i] as i32 * m.w1[o][i] as i32));
        h[o] = (acc >> SHIFT).clamp(0, 127) as u8;
    }
    core::array::from_fn(|o| {
        (0..HID).fold(m.b2[o], |a, i| a.wrapping_add(h[i] as i32 * m.w2[o][i] as i32))
    })
}

// vector layout for dpbusd: [group of 4 inputs][output][4 bytes], so one 64-byte load
// holds 16 outputs x 4 inputs and lines up with a broadcast of 4 input bytes.
pub fn pack<const N: usize>(w: &[[i8; N]]) -> Vec<u8> {
    let mut p = Vec::with_capacity(w.len() * N);
    for g in 0..N / 4 {
        for row in w {
            p.extend((0..4).map(|j| row[4 * g + j] as u8));
        }
    }
    p
}
