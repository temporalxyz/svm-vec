// writes sort inputs and the expected regions, sorted with slice::sort.
// usage: sort-host <out_dir> <seed>...
// input-<seed>.bin: 256 u32 keys (little-endian) + 1024 zero bytes; expected-<seed>.bin: the keys
// + the sorted keys. seed % 4 picks the distribution: 0 full range, 1 keys mod 5 (many
// duplicates), 2 top bit set plus 0 and u32::MAX, 3 strictly descending.
use std::{fs, path::Path};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    fs::create_dir_all(dir).unwrap();
    for seed in &args[2..] {
        let n: u64 = seed.parse().unwrap();
        let mut x = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            // splitmix64
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as u32
        };
        let mut keys: Vec<u32> = (0..256).map(|_| next()).collect();
        match n % 4 {
            1 => keys.iter_mut().for_each(|k| *k %= 5),
            2 => {
                keys.iter_mut().for_each(|k| *k |= 0x8000_0000);
                keys[3] = 0;
                keys[77] = u32::MAX;
                keys[200] = 0;
            }
            3 => keys = (0..256).map(|i| 0xffff_ff00 - 7 * i).collect(),
            _ => {}
        }
        let mut sorted = keys.clone();
        sorted.sort();
        let bytes = |v: &[u32]| v.iter().flat_map(|k| k.to_le_bytes()).collect::<Vec<u8>>();
        let mut input = bytes(&keys);
        input.resize(2048, 0);
        let mut expected = bytes(&keys);
        expected.extend(bytes(&sorted));
        fs::write(dir.join(format!("input-{seed}.bin")), &input).unwrap();
        fs::write(dir.join(format!("expected-{seed}.bin")), &expected).unwrap();
    }
}
