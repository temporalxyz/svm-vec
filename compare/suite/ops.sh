#!/usr/bin/env bash
# Cost per op: every op both designs share (ours=dean in the op name), plus each design's own
# expensive-looking ops, as a loop of 16 iterations x 256 instances, dependent (one chain) and
# independent (8 chains) variants, minus an empty-loop baseline. Ours: compare/opbench (raw bytecode
# with our encoding), Dean's: compare/ops_dean.rs (his macro assembler, intrinsic call form). Both
# check every program against the host reference in compare/opbench/src/spec.rs on three inputs,
# interpreter and JIT, then time it: mean of back-to-back runs (fresh VM, one memory mapping) in
# batches of about 2 ms, median of 9. Ops above 4 ns/CU on the JIT are listed on stderr.
#
# usage: ops.sh ours|dean   -> one JSON line {"ops": [{"op", "cu", "jit_ns_per_op", ...}], "ok"}
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
S=$root/target/suite/ops
mkdir -p "$S"

ours() {
    cargo build -q --release --offline --manifest-path "$root/compare/opbench/Cargo.toml" --target-dir "$S/opbench-target" >&2
    "$S/opbench-target/release/opbench" 2> "$S/ours.log"
    grep "^above" "$S/ours.log" >&2 || true
}

dean() {
    w=$S/dean
    [ -d "$w" ] || git -C "$root/target/compare/dean" worktree add "$w" 7941ce5 >&2
    cmp -s "$root/compare/ops_dean.rs" "$w/examples/ops_dean.rs" || cp "$root/compare/ops_dean.rs" "$w/examples/"
    cmp -s "$root/compare/opbench/src/spec.rs" "$w/examples/ops_spec.rs" || cp "$root/compare/opbench/src/spec.rs" "$w/examples/ops_spec.rs"
    (cd "$w" && cargo build -q --release --offline --features simd-experiment --example ops_dean >&2)
    "$w/target/release/examples/ops_dean" 2> "$S/dean.log"
    grep "^above" "$S/dean.log" >&2 || true
}

case "${1:-}" in
    ours) ours ;;
    dean) dean ;;
    *) echo "usage: $0 ours|dean" >&2; exit 2 ;;
esac
