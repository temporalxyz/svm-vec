#!/usr/bin/env bash
# perps risk kernel, ours vs Dean's. usage: perps.sh ours|dean
# Checks 5 seeds against perps-host on the interpreter and the JIT, times seed 1: mean of N
# back-to-back runs (N calibrated to ~2 ms of JIT time), median of 9 batches.
# Prints one JSON line: {"cu", "jit_ns", "interp_ns", "ok"}.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
w=$root/target/suite/perps
data=$w/data
seeds="1 2 3 4 5"
batches=9
mkdir -p "$w"
log=$w/build.log

host=$w/host/release/perps-host
[ -x "$host" ] || (cd "$root/programs/perps/host" && CARGO_TARGET_DIR=$w/host cargo build --release -q) >>"$log" 2>&1
for s in $seeds; do
    [ -f "$data/expected-$s.bin" ] || "$host" "$data" $seeds >>"$log" 2>&1
done

repeat_for() { python3 -c "import sys; print(max(1, round(2e6 / float(sys.argv[1]))))" "$1"; }

case "${1:-}" in
ours)
    run=$root/harness/target/release/svm-run
    linker=$root/sbpf-linker/target/release/sbpf-linker
    [ -x "$run" ] || (cd "$root/harness" && cargo build --release -q) >>"$log" 2>&1
    [ -x "$linker" ] || cargo build --release -q --manifest-path "$root/sbpf-linker/Cargo.toml" >>"$log" 2>&1
    # the linker also writes target/deploy/ into the crate directory: remove it
    (cd "$root/programs/perps" && CARGO_TARGET_DIR=$w/ours cargo +nightly build-sbpf && rm -rf target) >>"$log" 2>&1
    so=$w/ours/bpfel-unknown-none/release/libperps.so
    args=()
    for s in $seeds; do args+=(--input "$data/input-$s.bin" --expect "$data/expected-$s.bin"); done
    ok=true
    "$run" "$so" "${args[@]}" --runs 1 --repeat 1 >"$w/ours-check.json" || ok=false
    one=(--input "$data/input-1.bin" --expect "$data/expected-1.bin")
    "$run" "$so" "${one[@]}" --runs 3 --repeat 200 >"$w/ours-cal.json" || ok=false
    n=$(repeat_for "$(python3 -c "import json; print(json.load(open('$w/ours-cal.json'))['cases'][0]['jit_ns'])")")
    "$run" "$so" "${one[@]}" --runs $batches --repeat "$n" >"$w/ours-time.json" || ok=false
    python3 - "$w" "$ok" <<'EOF'
import json, sys
w, ok = sys.argv[1], sys.argv[2] == "true"
check = json.load(open(f"{w}/ours-check.json"))["cases"]
t = json.load(open(f"{w}/ours-time.json"))["cases"][0]
ok = ok and all(c["expect_matches"] and c["error"] is None and c["cu"] == t["cu"] for c in check + [t])
print(json.dumps({"cu": t["cu"], "jit_ns": t["jit_ns"], "interp_ns": t["interpreter_ns"], "ok": ok}))
EOF
    ;;
dean)
    dean=$w/dean
    [ -d "$dean" ] || git -C "$root/target/compare/dean" worktree add "$dean" 7941ce5 >>"$log" 2>&1
    cp "$root/compare/perps_dean.rs" "$dean/examples/"
    (cd "$root/compare/dean-perps" && CARGO_TARGET_DIR=$w/dean-build cargo-build-sbf --arch v3) >>"$log" 2>&1
    so=$w/dean-build/deploy/perps_dean.so
    (cd "$dean" && cargo build -q --release --features simd-experiment --example perps_dean) >>"$log" 2>&1
    ex=$dean/target/release/examples/perps_dean
    args=()
    for s in $seeds; do args+=("$data/input-$s.bin" "$data/expected-$s.bin"); done
    cal=$("$ex" "$so" 200 3 "$data/input-1.bin" "$data/expected-1.bin" 2>>"$log")
    n=$(repeat_for "$(python3 -c "import json, sys; print(json.loads(sys.argv[1])['jit_ns'])" "$cal")")
    "$ex" "$so" "$n" $batches "${args[@]}" 2>>"$log"
    ;;
*)
    echo "usage: $0 ours|dean" >&2
    exit 2
    ;;
esac
