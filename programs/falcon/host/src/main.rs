//! writes the inputs for the falcon program and an independent host reference.
//!
//! `falcon-host DIR`: for every signature i in ../fixtures/sigs.bin (64 PQClean signatures by
//! Blueshift's key, from Blueshift's sbpf-simd experiment, entries `[signature zero padded to 666
//! bytes][u16 LE message length][message]`):
//! - `DIR/i.in`, `DIR/i.badmsg.in`, `DIR/i.badsig.in`: the input region (`[0u64][ix length][sig]
//!   [message][program id]` and 2064 zero bytes of room for the `debug` entrypoint), the message
//!   or the signature tampered like Blueshift's bench does
//! - `DIR/i.debug.exp`: the input region after the `debug` entrypoint: c, s2, ||s1||^2 + ||s2||^2
//!   and the decoding result appended
//! - `DIR/expected.txt`: one line per input, `name valid norm`
//!
//! the reference is a plain port of PQClean's verify (bit by bit decoding, SHAKE-256 from the
//! sha3 crate, schoolbook s2 h mod x^N + 1) with Blueshift's padded format (trailing bits zero).
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::{fmt::Write, path::Path};

const Q: i64 = 12289;
const N: usize = 512;
const SIG_LEN: usize = 666;
const ROOM: usize = 2064;

fn decode_pubkey(pk: &[u8]) -> Vec<i64> {
    assert!(pk.len() == 897 && pk[0] == 0x09);
    let bits: Vec<u8> = pk[1..].iter().flat_map(|b| (0..8).rev().map(move |i| b >> i & 1)).collect();
    bits.chunks(14).map(|c| c.iter().fold(0, |a, b| a << 1 | *b as i64)).collect()
}

/// PQClean comp_decode, bit by bit, then all remaining bits must be zero
fn decompress(comp: &[u8]) -> Option<Vec<i64>> {
    let bit = |p: usize| comp.get(p / 8).map(|b| b >> (7 - p % 8) & 1);
    let mut pos = 0;
    let mut out = Vec::new();
    for _ in 0..N {
        let mut v = 0;
        for _ in 0..8 {
            v = v << 1 | bit(pos)? as i64;
            pos += 1;
        }
        let (s, mut m) = (v >> 7, v & 127);
        while bit(pos)? == 0 {
            pos += 1;
            m += 128;
            if m > 2047 {
                return None;
            }
        }
        pos += 1;
        if s == 1 && m == 0 {
            return None;
        }
        out.push(if s == 1 { -m } else { m });
    }
    (pos..8 * comp.len()).all(|p| bit(p) == Some(0)).then_some(out)
}

/// unreduced candidates below 5q
fn hash_to_point(nonce: &[u8], msg: &[u8]) -> Vec<i64> {
    let mut h = Shake256::default();
    h.update(nonce);
    h.update(msg);
    let mut r = h.finalize_xof();
    let mut c = Vec::new();
    while c.len() < N {
        let mut b = [0u8; 2];
        r.read(&mut b);
        let w = u16::from_be_bytes(b) as i64;
        if w < 5 * Q {
            c.push(w);
        }
    }
    c
}

/// (decoded, ||s1||^2 + ||s2||^2, c, s2)
fn verify(h: &[i64], sig: &[u8], msg: &[u8]) -> (bool, u64, Vec<i64>, Vec<i64>) {
    let c = hash_to_point(&sig[1..41], msg);
    let Some(s2) = decompress(&sig[41..]) else { return (false, 0, c, vec![0; N]) };
    let mut prod = vec![0i64; N];
    for i in 0..N {
        for j in 0..N {
            let x = s2[i] * h[j];
            if i + j < N { prod[i + j] += x } else { prod[i + j - N] -= x }
        }
    }
    let s1 = |i: usize| {
        let x = (c[i] - prod[i]).rem_euclid(Q);
        if x > Q / 2 { x - Q } else { x }
    };
    let norm = (0..N).map(|i| s1(i).pow(2) + s2[i].pow(2)).sum::<i64>() as u64;
    (true, norm, c, s2)
}

fn input(sig: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&0u64.to_le_bytes());
    d.extend_from_slice(&((SIG_LEN + msg.len()) as u64).to_le_bytes());
    d.extend_from_slice(sig);
    d.extend_from_slice(msg);
    d.extend_from_slice(&[7u8; 32]); // program id
    d.resize(d.len() + ROOM, 0);
    d
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: falcon-host DIR");
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let h = decode_pubkey(&std::fs::read(fixtures.join("falcon.pk")).unwrap());
    let bytes = std::fs::read(fixtures.join("sigs.bin")).unwrap();
    let mut sigs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let len = u16::from_le_bytes([bytes[i + SIG_LEN], bytes[i + SIG_LEN + 1]]) as usize;
        sigs.push((bytes[i..i + SIG_LEN].to_vec(), bytes[i + SIG_LEN + 2..i + SIG_LEN + 2 + len].to_vec()));
        i += SIG_LEN + 2 + len;
    }
    let mut list = String::new();
    for (k, (sig, msg)) in sigs.iter().enumerate() {
        let mut bad_msg = msg.clone();
        if bad_msg.is_empty() { bad_msg.push(1) } else { bad_msg[0] ^= 1 }
        let mut bad_sig = sig.clone();
        bad_sig[100] ^= 0x10;
        for (name, sig, msg) in [(format!("{k}"), sig, msg), (format!("{k}.badmsg"), sig, &bad_msg), (format!("{k}.badsig"), &bad_sig, msg)] {
            let (decoded, norm, c, s2) = verify(&h, sig, msg);
            let valid = decoded && sig[0] == 0x39 && norm <= 34_034_726;
            let mut data = input(sig, msg);
            std::fs::write(dir.join(format!("{name}.in")), &data).unwrap();
            let tail = data.len() - ROOM;
            let mut out: Vec<u8> = c.iter().chain(&s2).flat_map(|x| (*x as u16).to_le_bytes()).collect();
            out.extend_from_slice(&norm.to_le_bytes());
            out.extend_from_slice(&(decoded as u64).to_le_bytes());
            if decoded {
                data[tail..tail + out.len()].copy_from_slice(&out);
                std::fs::write(dir.join(format!("{name}.debug.exp")), &data).unwrap();
            }
            writeln!(list, "{name} {} {norm}", valid as u8).unwrap();
        }
    }
    std::fs::write(dir.join("expected.txt"), list).unwrap();
    println!("{} signatures, inputs in {}", sigs.len(), dir.display());
}
