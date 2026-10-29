#!/usr/bin/env bash
# int8 GEMM C = A x B (32 x 256 x 128), ours (programs/matmul) vs Dean's (compare/dean-matmul).
# Checks 5 seeds against matmul-host on the interpreter and the JIT, times seed 1 on each engine:
# mean of N back-to-back runs (N calibrated to ~2 ms per batch), median of 9 batches.
#
# usage: matmul.sh ours|dean   -> one JSON line {"cu", "jit_ns", "interp_ns", "ok"}
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
w=$root/target/suite/matmul
data=$w/data
seeds="1 2 3 4 5"
batches=9
mkdir -p "$w"
log=$w/build.log

host=$w/host/release/matmul-host
[ -x "$host" ] || (cd "$root/programs/matmul/host" && CARGO_TARGET_DIR=$w/host cargo build --release -q) >>"$log" 2>&1
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
    # --deploy=false: the linker would otherwise copy the .so to target/deploy in the crate
    (cd "$root/programs/matmul" && cargo +nightly build-sbpf --target-dir "$w/ours" \
        --config 'target.bpfel-unknown-none.rustflags=["-C", "link-arg=--deploy=false"]') >>"$log" 2>&1
    so=$w/ours/bpfel-unknown-none/release/libmatmul.so
    args=()
    for s in $seeds; do args+=(--input "$data/input-$s.bin" --expect "$data/expected-$s.bin"); done
    ok=true
    # exit code 1: engines disagree, an expectation fails or a run is not deterministic
    "$run" "$so" "${args[@]}" --runs 1 --repeat 1 >"$w/ours-check.json" || ok=false
    one=(--input "$data/input-1.bin" --expect "$data/expected-1.bin")
    "$run" "$so" "${one[@]}" --runs 3 --repeat 20 >"$w/ours-cal.json" || ok=false
    cal() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cases'][0][sys.argv[2]])" "$w/ours-cal.json" "$1"; }
    "$run" "$so" "${one[@]}" --runs $batches --repeat "$(repeat_for "$(cal jit_ns)")" >"$w/ours-jit.json" || ok=false
    "$run" "$so" "${one[@]}" --runs $batches --repeat "$(repeat_for "$(cal interpreter_ns)")" >"$w/ours-interp.json" || ok=false
    python3 - "$w" "$ok" <<'EOF'
import json, sys
w, ok = sys.argv[1], sys.argv[2] == "true"
load = lambda n: json.load(open(f"{w}/{n}"))["cases"]
check, jit, interp = load("ours-check.json"), load("ours-jit.json")[0], load("ours-interp.json")[0]
ok = ok and len(check) == 5 and all(
    c["expect_matches"] and c["error"] is None and c["cu"] == jit["cu"] for c in check + [jit, interp])
print(json.dumps({"cu": jit["cu"], "jit_ns": jit["jit_ns"], "interp_ns": interp["interpreter_ns"], "ok": ok}))
EOF
    ;;
dean)
    dean=$w/dean
    [ -d "$dean" ] || git -C "$root/target/compare/dean" worktree add "$dean" 7941ce5 >>"$log" 2>&1
    cmp -s "$root/compare/matmul_dean.rs" "$dean/examples/matmul_dean.rs" || cp "$root/compare/matmul_dean.rs" "$dean/examples/"
    (cd "$dean" && cargo build -q --release --offline --features simd-experiment --example matmul_dean) >>"$log" 2>&1
    CARGO_TARGET_DIR=$w/dean-target cargo-build-sbf --arch v3 --offline \
        --manifest-path "$root/compare/dean-matmul/Cargo.toml" --sbf-out-dir "$w/dean-deploy" -- --locked >>"$log" 2>&1
    args=()
    for s in $seeds; do args+=("$data/input-$s.bin" "$data/expected-$s.bin"); done
    "$dean/target/release/examples/matmul_dean" "$w/dean-deploy/matmul_dean.so" $batches "${args[@]}" 2>>"$log"
    ;;
*)
    echo "usage: $0 ours|dean" >&2
    exit 2
    ;;
esac
