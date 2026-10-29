// writes edge-case inputs for the mlp (the random seeds come from programs/mlp/host): x all 0,
// all 127, and 127 exactly where hidden unit `unit`'s weights are positive (drives it past 127).
// usage: mlp-unrolled-host <out_dir> zero|max|sat:<unit>
use std::{fs, path::Path};

#[path = "../../../mlp/model.rs"]
#[allow(dead_code)]
mod model;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let m = model::model();
    let x: [u8; model::IN] = match args[2].as_str() {
        "zero" => [0; model::IN],
        "max" => [127; model::IN],
        k => {
            let unit: usize = k.strip_prefix("sat:").expect("zero|max|sat:<unit>").parse().unwrap();
            core::array::from_fn(|i| if m.w1[unit][i] > 0 { 127 } else { 0 })
        }
    };
    let logits = model::forward(&m, &x);
    let argmax = (0..logits.len()).fold(0, |b, o| if logits[o] > logits[b] { o } else { b });
    let mut buf = vec![0u8; 384];
    buf[0] = 1;
    buf[64..320].copy_from_slice(&x);
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("input-vector.bin"), &buf).unwrap();
    for (o, l) in logits.iter().enumerate() {
        buf[320 + 4 * o..324 + 4 * o].copy_from_slice(&l.to_le_bytes());
    }
    fs::write(dir.join("expected-vector.bin"), &buf).unwrap();
    println!("{{\"case\":\"{}\",\"argmax\":{argmax},\"logits\":{logits:?}}}", args[2]);
}
