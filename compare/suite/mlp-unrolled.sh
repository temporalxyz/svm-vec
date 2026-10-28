#!/usr/bin/env bash
# int8 mlp, fully unrolled: ours (programs/mlp-unrolled) vs Dean's (compare/dean-mlp, unchanged).
# Both are checked against the host references on several inputs, then timed on seed 1: mean of
# back-to-back runs (jit 1200, interpreter 61 per batch), median of 15 batches. Dean's CU varies
# by one with the input (his argmax loop branches), ours does not.
#
# usage: mlp-unrolled.sh ours|dean   -> one JSON line {"cu", "jit_ns", "interp_ns", "ok"}
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
S=$root/target/suite/mlp-unrolled
mkdir -p "$S/data"

# random inputs from programs/mlp/host, edge cases (saturating hidden units) from our host/
cargo build -q --release --manifest-path "$root/programs/mlp/host/Cargo.toml" --target-dir "$S/host-target" >&2
cargo build -q --release --manifest-path "$root/programs/mlp-unrolled/host/Cargo.toml" --target-dir "$S/host-target" >&2
cases="1 3 5 7 9 11 zero max sat:0 sat:77 sat:127"
for c in $cases; do
    d=$S/data/${c/:/-}
    [ -f "$d/expected-vector.bin" ] && [ -s "$d.json" ] && continue
    case $c in
        [0-9]*) "$S/host-target/release/mlp-host" "$d" "$c" > "$d.json" ;;
        *) "$S/host-target/release/mlp-unrolled-host" "$d" "$c" > "$d.json" ;;
    esac
done
dirs=$(for c in $cases; do echo "${c/:/-}"; done)

ours() {
    [ -x "$root/sbpf-linker/target/release/sbpf-linker" ] || cargo build -q --release --manifest-path "$root/sbpf-linker/Cargo.toml" >&2
    [ -x "$root/harness/target/release/svm-run" ] || cargo build -q --release --manifest-path "$root/harness/Cargo.toml" >&2
    # --deploy=false: the linker would otherwise copy the .so to target/deploy in the crate
    (cd "$root/programs/mlp-unrolled" && cargo +nightly build-sbpf -q --target-dir "$S/ours-target" \
        --config 'target.bpfel-unknown-none.rustflags=["-C", "link-arg=--deploy=false"]' 2> "$S/ours-build.log") ||
        { cat "$S/ours-build.log" >&2; exit 1; }
    so=$S/ours-target/bpfel-unknown-none/release/libmlp_unrolled.so
    run=$root/harness/target/release/svm-run
    args=()
    for d in $dirs; do args+=(--input "$S/data/$d/input-vector.bin" --expect "$S/data/$d/expected-vector.bin"); done
    # exit code 1: engines disagree, an expectation fails or a run is not deterministic
    rc=0; "$run" "$so" "${args[@]}" --runs 1 > "$S/ours-check.json" || rc=$?
    "$run" "$so" --input "$S/data/1/input-vector.bin" --expect "$S/data/1/expected-vector.bin" --runs 15 --repeat 1200 > "$S/ours-jit.json" || true
    "$run" "$so" --input "$S/data/1/input-vector.bin" --expect "$S/data/1/expected-vector.bin" --runs 15 --repeat 61 > "$S/ours-interp.json" || true
    python3 -I - "$S" $rc $dirs <<'EOF'
import json, sys
S, rc, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
load = lambda n: json.load(open(f"{S}/{n}"))["cases"]
check, jit, interp = load("ours-check.json"), load("ours-jit.json")[0], load("ours-interp.json")[0]
argmax = [json.load(open(f"{S}/data/{d}.json"))["argmax"] for d in dirs]
ok = (rc == "0" and len(check) == len(dirs) and all(c["expect_matches"] and c["error"] is None for c in check)
      and [c["r0"] for c in check] == argmax and len({c["cu"] for c in check}) == 1
      and jit["expect_matches"] and interp["expect_matches"])
print(json.dumps({"cu": jit["cu"], "jit_ns": jit["jit_ns"], "interp_ns": interp["interpreter_ns"], "ok": ok}))
EOF
}

dean() {
    w=$S/dean
    [ -d "$w" ] || git -C "$root/target/compare/dean" worktree add "$w" 7941ce5 >&2
    cmp -s "$root/compare/mlp_dean.rs" "$w/examples/mlp_dean.rs" || cp "$root/compare/mlp_dean.rs" "$w/examples/"
    (cd "$w" && cargo build -q --release --offline --features simd-experiment --example mlp_dean >&2)
    CARGO_TARGET_DIR=$S/dean-target cargo-build-sbf --arch v3 --offline \
        --manifest-path "$root/compare/dean-mlp/Cargo.toml" --sbf-out-dir "$S/dean-deploy" -- --locked >&2
    ex=$w/target/release/examples/mlp_dean
    so=$S/dean-deploy/mlp_dean.so
    : > "$S/dean-check.txt"
    for d in $dirs; do
        "$ex" "$so" "$S/data/$d/input-vector.bin" "$S/data/$d/expected-vector.bin" 1 >> "$S/dean-check.txt" 2>/dev/null \
            || echo "fail $d" >> "$S/dean-check.txt"
    done
    "$ex" "$so" "$S/data/1/input-vector.bin" "$S/data/1/expected-vector.bin" 1200 > "$S/dean-time.txt" 2>/dev/null || true
    python3 -I - "$S" $dirs <<'EOF'
import json, re, sys
S, dirs = sys.argv[1], sys.argv[2:]
pat = r"^ok argmax (\d+) CU (\d+) jit_ns ([\d.]+) interp_ns ([\d.]+)$"
check = [re.match(pat, l) for l in open(f"{S}/dean-check.txt").read().splitlines()]
t = re.match(pat, open(f"{S}/dean-time.txt").read().strip())
argmax = [json.load(open(f"{S}/data/{d}.json"))["argmax"] for d in dirs]
ok = (t is not None and len(check) == len(dirs) and all(check)
      and [int(m[1]) for m in check] == argmax)
print(json.dumps({"cu": int(t[2]) if t else None, "jit_ns": float(t[3]) if t else None,
                  "interp_ns": float(t[4]) if t else None, "ok": ok}))
EOF
}

case "${1:-}" in
    ours) ours ;;
    dean) dean ;;
    *) echo "usage: $0 ours|dean" >&2; exit 2 ;;
esac
