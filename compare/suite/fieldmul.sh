#!/usr/bin/env bash
# GF(2^255 - 19) field multiplication, 8 lanes x 256 sequential products: ours (programs/fieldmul)
# vs Dean's (compare/dean-fieldmul). Both are checked against programs/fieldmul/host (num-bigint) on
# 5 seeds plus edge and fallback inputs, on the interpreter and the JIT, then timed on seed 1: mean
# of N back-to-back runs (N calibrated to ~2 ms per engine), median of 9 batches.
# Ours takes its general path on the fallback input (more CUs); the reported CU is seed 1's.
#
# usage: fieldmul.sh ours|dean   -> one JSON line {"cu", "jit_ns", "interp_ns", "ok"}
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
S=$root/target/suite/fieldmul
data=$S/data
cases="1 2 3 4 5 edge fallback"
batches=9
mkdir -p "$S"
log=$S/build.log

host=$S/host/release/fieldmul-host
[ -x "$host" ] || (cd "$root/programs/fieldmul/host" && CARGO_TARGET_DIR=$S/host cargo build --release -q --offline) >>"$log" 2>&1
for c in $cases; do
    [ -f "$data/expected-$c.bin" ] || "$host" "$data" $cases >>"$log" 2>&1
done

repeat_for() { python3 -c "import sys; print(max(1, round(2e6 / float(sys.argv[1]))))" "$1"; }

ours() {
    run=$root/harness/target/release/svm-run
    linker=$root/sbpf-linker/target/release/sbpf-linker
    [ -x "$run" ] || (cd "$root/harness" && cargo build --release -q) >>"$log" 2>&1
    [ -x "$linker" ] || cargo build --release -q --manifest-path "$root/sbpf-linker/Cargo.toml" >>"$log" 2>&1
    # --deploy=false: the linker would otherwise copy the .so to target/deploy in the crate
    (cd "$root/programs/fieldmul" && CARGO_TARGET_DIR=$S/ours cargo +nightly build-sbpf -q \
        --config 'target.bpfel-unknown-none.rustflags=["-C", "link-arg=--deploy=false"]') >>"$log" 2>&1
    so=$S/ours/bpfel-unknown-none/release/libfieldmul.so
    args=()
    for c in $cases; do args+=(--input "$data/input-$c.bin" --expect "$data/expected-$c.bin"); done
    ok=true
    # exit code 1: engines disagree, an expectation fails or a run is not deterministic
    "$run" "$so" "${args[@]}" --runs 1 >"$S/ours-check.json" || ok=false
    one=(--input "$data/input-1.bin" --expect "$data/expected-1.bin")
    "$run" "$so" "${one[@]}" --runs 3 --repeat 20 >"$S/ours-cal.json" || ok=false
    nj=$(repeat_for "$(python3 -c "import json; print(json.load(open('$S/ours-cal.json'))['cases'][0]['jit_ns'])")")
    ni=$(repeat_for "$(python3 -c "import json; print(json.load(open('$S/ours-cal.json'))['cases'][0]['interpreter_ns'])")")
    "$run" "$so" "${one[@]}" --runs $batches --repeat "$nj" >"$S/ours-jit.json" || ok=false
    "$run" "$so" "${one[@]}" --runs $batches --repeat "$ni" >"$S/ours-interp.json" || ok=false
    python3 -I - "$S" "$ok" <<'PY'
import json, sys
S, ok = sys.argv[1], sys.argv[2] == "true"
load = lambda n: json.load(open(f"{S}/{n}"))["cases"]
check, jit, interp = load("ours-check.json"), load("ours-jit.json")[0], load("ours-interp.json")[0]
ok = ok and len(check) == 7 and all(c["expect_matches"] and c["error"] is None for c in check + [jit, interp])
print(json.dumps({"cu": jit["cu"], "jit_ns": jit["jit_ns"], "interp_ns": interp["interpreter_ns"], "ok": ok}))
PY
}

dean() {
    w=$S/dean
    [ -d "$w" ] || git -C "$root/target/compare/dean" worktree add "$w" 7941ce5 >>"$log" 2>&1
    cmp -s "$root/compare/fieldmul_dean.rs" "$w/examples/fieldmul_dean.rs" || cp "$root/compare/fieldmul_dean.rs" "$w/examples/"
    (cd "$w" && cargo build -q --release --offline --features simd-experiment --example fieldmul_dean) >>"$log" 2>&1
    CARGO_TARGET_DIR=$S/dean-target cargo-build-sbf --arch v3 --offline \
        --manifest-path "$root/compare/dean-fieldmul/Cargo.toml" --sbf-out-dir "$S/dean-deploy" -- --locked >>"$log" 2>&1
    args=()
    for c in $cases; do args+=("$data/input-$c.bin" "$data/expected-$c.bin"); done
    out=$("$w/target/release/examples/fieldmul_dean" "$S/dean-deploy/fieldmul_dean.so" "${args[@]}" 2>>"$log") ||
        out='{"cu": null, "jit_ns": null, "interp_ns": null, "ok": false}'
    echo "$out"
}

case "${1:-}" in
    ours) ours ;;
    dean) dean ;;
    *) echo "usage: $0 ours|dean" >&2; exit 2 ;;
esac
