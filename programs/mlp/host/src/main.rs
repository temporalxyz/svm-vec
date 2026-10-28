// writes mlp input files (scalar and vector mode) and the expected logits/argmax.
// usage: mlp-host <out_dir> [seed]
use std::{fs, path::Path};

#[path = "../../model.rs"]
mod model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let seed = args.get(2).map_or(1, |s| s.parse().unwrap());
    let m = model::model();
    let x = model::input(seed);
    let logits = model::forward(&m, &x);
    let argmax = (0..logits.len()).fold(0, |b, o| if logits[o] > logits[b] { o } else { b });
    assert_eq!(packed_forward(&m, &x), logits, "packed layout disagrees with reference");

    fs::create_dir_all(dir).unwrap();
    for (name, mode) in [("scalar", 0u64), ("vector", 1)] {
        let mut buf = vec![0u8; 384];
        buf[..8].copy_from_slice(&mode.to_le_bytes());
        buf[64..320].copy_from_slice(&x);
        fs::write(dir.join(format!("input-{name}.bin")), &buf).unwrap();
    }
    let mut out = vec![0u8; 384];
    out[64..320].copy_from_slice(&x);
    for (o, l) in logits.iter().enumerate() {
        out[320 + 4 * o..324 + 4 * o].copy_from_slice(&l.to_le_bytes());
    }
    // expected final region for vector mode (mode byte = 1); scalar differs only in byte 0
    out[0] = 1;
    fs::write(dir.join("expected-vector.bin"), &out).unwrap();
    out[0] = 0;
    fs::write(dir.join("expected-scalar.bin"), &out).unwrap();
    println!("{{\"seed\":{seed},\"argmax\":{argmax},\"logits\":{logits:?}}}");
}

// the program's vector algorithm in scalar form: broadcast 4 inputs, dpbusd over the packed weights.
fn packed_forward(m: &model::Model, x: &[u8; model::IN]) -> [i32; model::OUT] {
    fn layer(x: &[u8], packed: &[u8], bias: &[i32]) -> Vec<i32> {
        let mut acc = bias.to_vec();
        for g in 0..x.len() / 4 {
            for (o, a) in acc.iter_mut().enumerate() {
                for j in 0..4 {
                    *a = a.wrapping_add(x[4 * g + j] as i32 * packed[(g * bias.len() + o) * 4 + j] as i8 as i32);
                }
            }
        }
        acc
    }
    let h: Vec<u8> = layer(x, &model::pack(&m.w1), &m.b1)
        .iter()
        .map(|a| (a >> model::SHIFT).clamp(0, 127) as u8)
        .collect();
    layer(&h, &model::pack(&m.w2), &m.b2).try_into().unwrap()
}
