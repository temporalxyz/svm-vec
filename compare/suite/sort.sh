#!/usr/bin/env bash
# sort: 256 u32 keys ascending (programs/sort, compare/dean-sort, compare/sort_dean.rs).
#
#   compare/suite/sort.sh ours|dean
#
# Builds what it needs, checks the output against sort-host (slice::sort) on 8 inputs on the
# interpreter and the JIT, then times input 1: the mean of N back-to-back runs (fresh VM each, one
# memory mapping), N calibrated to ~2 ms of JIT runs, median of 11 batches. Prints one JSON line
# {"cu", "jit_ns", "interp_ns", "ok"}. Everything it writes goes to target/suite/sort/.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/target/suite/sort"
data="$work/data"
seeds=(1 2 3 4 5 6 7 8)
batches=11
mkdir -p "$work"

host="$work/host/release/sort-host"
if [ ! -x "$host" ]; then
    CARGO_TARGET_DIR="$work/host" cargo build -q --release --manifest-path "$root/programs/sort/host/Cargo.toml" >&2
fi
for s in "${seeds[@]}"; do
    [ -f "$data/expected-$s.bin" ] || "$host" "$data" "${seeds[@]}"
done

case "${1:-}" in
ours)
    run="$root/harness/target/release/svm-run"
    if [ ! -x "$run" ]; then
        CARGO_TARGET_DIR="$work/harness" cargo build -q --release --manifest-path "$root/harness/Cargo.toml" >&2
        run="$work/harness/release/svm-run"
    fi
    [ -x "$root/sbpf-linker/target/release/sbpf-linker" ] ||
        cargo build -q --release --manifest-path "$root/sbpf-linker/Cargo.toml" >&2
    # --deploy=false: the linker would otherwise copy the .so into programs/sort/target/deploy
    (cd "$root/programs/sort" && CARGO_TARGET_DIR="$work/ours" cargo +nightly build-sbpf -q \
        --config 'target.bpfel-unknown-none.rustflags=["-C","link-arg=--deploy=false"]') 2>"$work/ours-build.log" ||
        { cat "$work/ours-build.log" >&2; exit 1; }
    so="$work/ours/bpfel-unknown-none/release/libsort.so"
    args=()
    for s in "${seeds[@]}"; do args+=(--input "$data/input-$s.bin" --expect "$data/expected-$s.bin"); done
    check=0
    "$run" "$so" "${args[@]}" --runs 1 --repeat 1 >"$work/ours-check.json" || check=$?
    pilot=$("$run" "$so" --input "$data/input-1.bin" --runs 1 --repeat 2000 |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["cases"][0]["jit_ns"])')
    n=$(python3 -c "print(max(1, round(2e6 / max($pilot, 1))))")
    timed=0
    "$run" "$so" --input "$data/input-1.bin" --expect "$data/expected-1.bin" --runs "$batches" --repeat "$n" \
        >"$work/ours-time.json" || timed=$?
    python3 - "$work/ours-check.json" "$work/ours-time.json" "$check" "$timed" "$n" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))["cases"]
t = json.load(open(sys.argv[2]))["cases"][0]
ok = sys.argv[3] == "0" and sys.argv[4] == "0" and all(x["expect_matches"] and x["error"] is None for x in c + [t])
ok = ok and len({x["cu"] for x in c + [t]}) == 1
print(f"repeat {sys.argv[5]}", file=sys.stderr)
print(json.dumps({"cu": t["cu"], "jit_ns": t["jit_ns"], "interp_ns": t["interpreter_ns"], "ok": ok}))
EOF
    ;;
dean)
    rev=7941ce5
    wt="$work/dean"
    if [ ! -d "$wt" ]; then
        git -C "$root/target/compare/dean" worktree add -q --detach "$wt" "$rev" >&2
    fi
    cmp -s "$root/compare/sort_dean.rs" "$wt/examples/sort_dean.rs" || cp "$root/compare/sort_dean.rs" "$wt/examples/"
    (cd "$root/compare/dean-sort" && CARGO_TARGET_DIR="$work/dean-build" cargo-build-sbf --arch v3) \
        >"$work/dean-build.log" 2>&1 || { cat "$work/dean-build.log" >&2; exit 1; }
    (cd "$wt" && CARGO_TARGET_DIR="$work/dean-target" cargo build -q --release --features simd-experiment \
        --example sort_dean) >&2
    args=()
    for s in "${seeds[@]}"; do args+=("$data/input-$s.bin" "$data/expected-$s.bin"); done
    "$work/dean-target/release/examples/sort_dean" "$work/dean-build/deploy/sort_dean.so" 0 "$batches" "${args[@]}"
    ;;
*)
    echo "usage: $0 ours|dean" >&2
    exit 2
    ;;
esac
