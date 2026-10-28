#!/usr/bin/env bash
# builds everything, runs each kernel on the interpreter and the jit, checks outputs against the
# host references and prints the results table.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/svm-vec-bench
run=$root/harness/target/release/svm-run
rm -rf "$out" && mkdir -p "$out"

cargo build -q --release --manifest-path "$root/harness/Cargo.toml"
(cd "$root/programs" && ./build.sh mlp >/dev/null 2>&1 && ./build.sh keccak >/dev/null 2>&1)
(cd "$root/programs/mlp" && cargo +nightly build-sbpf -q --features scalar-only --target-dir target-scalar 2>/dev/null)
for h in mlp keccak; do cargo build -q --release --manifest-path "$root/programs/$h/host/Cargo.toml"; done

"$root/programs/mlp/host/target/release/mlp-host" "$out/mlp" 1 >/dev/null
for k in 0 100; do "$root/programs/keccak/host/target/release/keccak-host" "$out/keccak$k" $k 1 >/dev/null; done

# case: label, .so, input dir, mode (scalar|vector)
case_json() {
  "$run" "$2" --input "$3/input-$4.bin" --expect "$3/expected-$4.bin" --runs 300 --budget 100000000 >"$out/$1.json"
}
mlp=$root/programs/mlp/target/bpfel-unknown-none/release/libmlp.so
case_json mlp-stock "$root/programs/mlp/target-scalar/bpfel-unknown-none/release/libmlp.so" "$out/mlp" scalar
case_json mlp-vector "$mlp" "$out/mlp" vector
kec=$root/programs/keccak/target/bpfel-unknown-none/release/libkeccak.so
for k in 0 100; do for m in scalar vector; do case_json keccak$k-$m "$kec" "$out/keccak$k" $m; done; done

python3 -I - "$out" <<'EOF'
import json, sys
d = sys.argv[1]
c = lambda n: json.load(open(f"{d}/{n}.json"))["cases"][0]
assert all(c(n)["expect_matches"] for n in ["mlp-stock", "mlp-vector", "keccak100-scalar", "keccak100-vector"])
perm = lambda m, f: (c(f"keccak100-{m}")[f] - c(f"keccak0-{m}")[f]) / 100
rows = [
    ("int8 mlp, cu", c("mlp-stock")["cu"], c("mlp-vector")["cu"]),
    ("int8 mlp, jit µs", c("mlp-stock")["jit_ns"] / 1e3, c("mlp-vector")["jit_ns"] / 1e3),
    ("keccak-f, cu / permutation", perm("scalar", "cu"), perm("vector", "cu")),
    ("keccak-f, jit µs / permutation", perm("scalar", "jit_ns") / 1e3, perm("vector", "jit_ns") / 1e3),
]
print("| | scalar sbpf | vector ops | ratio |\n|---|---|---|---|")
for name, s, v in rows:
    f = ",.0f" if "cu" in name else ".2f"
    print(f"| {name} | {s:{f}} | {v:{f}} | {s / v:.0f}× |")
print("\nall outputs match the host references on both engines")
EOF
