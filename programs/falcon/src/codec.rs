// signature decompression into signed 16 bit coefficients (Blueshift's
// blueshift-gg/solana-falcon512 format checks, Firedancer's windowed decoding). adapted from
// simd-falcon/program/src/codec.rs in Blueshift's sbpf-simd experiment (Apache-2.0 / MIT); only the
// buffer handling differs (the caller passes the zero padded copy).

const N: usize = 512;

/// length of the compressed s2 in a 666 byte signature
pub const COMP_LEN: usize = 625;

/// leading zeros of a byte (8 for 0)
static CLZ8: [u8; 256] = {
    let mut t = [8u8; 256];
    let mut i = 1;
    while i < 256 {
        t[i] = (i as u8).leading_zeros() as u8;
        i += 1;
    }
    t
};

/// coefficients whose 12 leading bits hold the whole code (high bits k < 4): `value | length << 26`
/// (value as i16; the length in the top 6 bits needs no masking as a shift amount) indexed by
/// those 12 bits, 0 if the code is longer (or -0, which is invalid)
static SHORT: [u32; 4096] = {
    let mut t = [0u32; 4096];
    let mut i = 0;
    while i < 4096 {
        let unary = i & 0xf;
        if unary != 0 {
            let k = (unary as u8).leading_zeros() - 4;
            let m = (k << 7) | ((i >> 4) & 0x7f) as u32;
            let negative = i & 0x800 != 0;
            if !(negative && m == 0) {
                let v = if negative { (m as i16).wrapping_neg() } else { m as i16 };
                t[i] = v as u16 as u32 | (9 + k) << 26;
            }
        }
        i += 1;
    }
    t
};

/// one coefficient from the top of `w` (at least 24 valid bits): sign, 7 low bits, then the high
/// bits in unary (k zeros and a one, k < 16). returns (signed value, bits used).
#[inline(always)]
fn decode_one(w: u64) -> Option<(u16, u64)> {
    let header = (w >> 56) as u32;
    let b1 = ((w >> 48) & 0xff) as usize;
    let k = if b1 != 0 {
        CLZ8[b1] as u32
    } else {
        let b2 = ((w >> 40) & 0xff) as usize;
        if b2 == 0 {
            return None;
        }
        8 + CLZ8[b2] as u32
    };
    let m = (k << 7) | (header & 0x7f);
    let v = if header & 0x80 != 0 {
        if m == 0 {
            return None; // -0
        }
        (m as u16).wrapping_neg()
    } else {
        m as u16
    };
    Some((v, 9 + k as u64))
}

/// the rare long code (or an invalid one) at bit `pos`, out of line
#[cold]
#[inline(never)]
unsafe fn decode_long(padded: *const u8, pos: u64) -> Option<(u16, u64)> {
    decode_one(unsafe { window(padded, pos) })
}

/// 57 valid bits from bit `pos` (big-endian), at the top of a u64
#[inline(always)]
unsafe fn window(padded: *const u8, pos: u64) -> u64 {
    u64::from_be(unsafe { (padded.add((pos >> 3) as usize) as *const u64).read_unaligned() }) << (pos & 7)
}

/// decodes the compressed s2 (`padded`: COMP_LEN bytes followed by 15 zero bytes) into
/// `out[0..N]` as signed values (|s2| < 2048).
///
/// Firedancer's approach (fd_falcon_signature_parse): a 64 bit big-endian window and a table
/// lookup per coefficient instead of one bit per step. short codes (12 bits at most, nearly all of
/// them) come from one table load; four fit a window. a run past the end fails on the all zero
/// unary part of the padding.
#[inline(always)]
pub unsafe fn decompress_signature(padded: *const u8, out: *mut u16) -> bool {
    unsafe {
        let mut pos = 0u64;
        let mut out = out;
        let end = out.add(N);
        // keeps the table address in a register instead of a 64 bit immediate per lookup
        let short = core::hint::black_box(SHORT.as_ptr());
        while out != end {
            let mut w = window(padded, pos);
            for k in 0..4 {
                let e = *short.add((w >> 52) as usize);
                let (v, n) = if e != 0 {
                    (e as u16, (e >> 26) as u64)
                } else {
                    let Some(r) = decode_long(padded, pos) else { return false };
                    r
                };
                out.add(k).write_volatile(v);
                pos += n;
                // a long code reloads the window, short ones leave enough bits for the rest
                w = if e != 0 { w << n } else { window(padded, pos) };
            }
            out = out.add(4);
        }
        let len_bits = 8 * COMP_LEN as u64;
        if pos > len_bits {
            return false;
        }
        // trailing bits must be zero
        let mut byte = (pos >> 3) as usize;
        if pos & 7 != 0 {
            if *padded.add(byte) & (0xff >> (pos & 7)) != 0 {
                return false;
            }
            byte += 1;
        }
        while byte + 8 <= COMP_LEN {
            if (padded.add(byte) as *const u64).read_unaligned() != 0 {
                return false;
            }
            byte += 8;
        }
        while byte < COMP_LEN {
            if *padded.add(byte) != 0 {
                return false;
            }
            byte += 1;
        }
        true
    }
}
