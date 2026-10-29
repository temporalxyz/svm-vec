#!/usr/bin/env bash
# BLAKE3 of 1 MiB of i % 251, the largest size of Dean's benches/simd_blake3.rs.
#   ours: programs/blake3 on svm-run
#   dean: his generated hash program (simd/blake3_program.rs, Simd8 kernel, register-cached JIT) on
#         his harness, compare/blake3_dean.rs. DEAN_MODE=slot (default, his headline variant E,
#         1 CU per op) or call (static-syscall form, variant G, 2 CU per op)
# both check every case of programs/blake3/host (blake3 crate) on the interpreter and the JIT, then
# time the 1 MiB case. prints {"cu", "jit_ns", "interp_ns", "ok"}.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/target/suite/blake3"
cases="$work/cases"
mkdir -p "$work"

host="$work/host/release/blake3-host"
[ -x "$host" ] || CARGO_TARGET_DIR="$work/host" cargo build -q --release --manifest-path "$root/programs/blake3/host/Cargo.toml" >&2
names=$("$host" "$cases")

case "${1:-}" in
ours)
    so="$work/ours/bpfel-unknown-none/release/libblake3.so"
    if [ ! -f "$so" ]; then
        cargo build -q --release --manifest-path "$root/sbpf-linker/Cargo.toml" >&2
        (cd "$root/programs/blake3" && CARGO_TARGET_DIR="$work/ours" cargo +nightly build-sbpf -q 2>/dev/null)
        rm -rf "$root/programs/blake3/target" # the linker's target/deploy copy
    fi
    run() { "$root/harness/target/release/svm-run" "$so" --heap 36864 --budget 100000000 "$@"; }
    ok=true
    for n in $names; do
        run --runs 1 --input "$cases/input-$n.bin" --expect "$cases/expected-$n.bin" > /dev/null || ok=false
    done
    run --runs 9 --repeat 10 --input "$cases/input-1m.bin" --expect "$cases/expected-1m.bin" > "$work/ours.json" || ok=false
    python3 - "$work/ours.json" $ok <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["cases"][0]
ok = sys.argv[2] == "true" and c["expect_matches"] and c["error"] is None
print(json.dumps({"cu": c["cu"], "jit_ns": c["jit_ns"], "interp_ns": c["interpreter_ns"], "ok": ok}))
PY
    ;;
dean)
    wt="$work/dean"
    [ -d "$wt" ] || git -C "$root/target/compare/dean" worktree add -q "$wt" 7941ce5 >&2
    cp "$root/compare/blake3_dean.rs" "$wt/examples/"
    (cd "$wt" && cargo build -q --release --features simd-experiment --example blake3_dean 2>/dev/null)
    "$wt/target/release/examples/blake3_dean" "${DEAN_MODE:-slot}" "$cases" 1m $names
    ;;
*)
    echo "usage: $0 ours|dean" >&2
    exit 2
    ;;
esac
