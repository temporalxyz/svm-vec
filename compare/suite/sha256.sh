#!/usr/bin/env bash
# SHA-256 compression of 16 independent 64-byte blocks: ours (programs/sha256) vs Dean's
# (compare/dean-sha256). Both run the same rounds (programs/sha256/gen.rs) and are checked against
# sha2::compress256 (programs/sha256/host) on every case, on the interpreter and the JIT, then timed
# on case 1: mean of back-to-back runs in batches of about 2 ms, median of 15 batches.
#
# usage: sha256.sh ours|dean   -> one JSON line {"cu", "jit_ns", "interp_ns", "ok"}
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
S=$root/target/suite/sha256
mkdir -p "$S"
cases="1 2 3 4 5 6 zero ones"

cargo build -q --release --manifest-path "$root/programs/sha256/host/Cargo.toml" --target-dir "$S/host-target" >&2
"$S/host-target/release/sha256-host" "$S/cases" $cases >&2

ours() {
    [ -x "$root/sbpf-linker/target/release/sbpf-linker" ] || cargo build -q --release --manifest-path "$root/sbpf-linker/Cargo.toml" >&2
    [ -x "$root/harness/target/release/svm-run" ] || cargo build -q --release --manifest-path "$root/harness/Cargo.toml" >&2
    # --deploy=false: the linker would otherwise copy the .so to target/deploy in the crate
    (cd "$root/programs/sha256" && cargo +nightly build-sbpf -q --target-dir "$S/ours-target" \
        --config 'target.bpfel-unknown-none.rustflags=["-C", "link-arg=--deploy=false"]' 2> "$S/ours-build.log") ||
        { cat "$S/ours-build.log" >&2; exit 1; }
    so=$S/ours-target/bpfel-unknown-none/release/libsha256.so
    run=$root/harness/target/release/svm-run
    args=()
    for c in $cases; do args+=(--input "$S/cases/input-$c.bin" --expect "$S/cases/expected-$c.bin"); done
    one=(--input "$S/cases/input-1.bin" --expect "$S/cases/expected-1.bin")
    # exit code 1: engines disagree, an expectation fails or a run is not deterministic
    rc=0; "$run" "$so" "${args[@]}" --runs 1 > "$S/ours-check.json" || rc=$?
    # batch sizes for about 2 ms
    "$run" "$so" "${one[@]}" --runs 3 --repeat 100 > "$S/ours-cal.json" || true
    read -r nj ni < <(python3 -I -c '
import json, sys
c = json.load(open(sys.argv[1]))["cases"][0]
print(max(1, round(2e6 / c["jit_ns"])), max(1, round(2e6 / c["interpreter_ns"])))' "$S/ours-cal.json")
    "$run" "$so" "${one[@]}" --runs 15 --repeat "$nj" > "$S/ours-jit.json" || true
    "$run" "$so" "${one[@]}" --runs 15 --repeat "$ni" > "$S/ours-interp.json" || true
    python3 -I - "$S" $rc $cases <<'EOF'
import json, sys
S, rc, cases = sys.argv[1], sys.argv[2], sys.argv[3:]
load = lambda n: json.load(open(f"{S}/{n}"))["cases"]
check, jit, interp = load("ours-check.json"), load("ours-jit.json")[0], load("ours-interp.json")[0]
ok = (rc == "0" and len(check) == len(cases) and all(c["expect_matches"] and c["error"] is None for c in check)
      and len({c["cu"] for c in check}) == 1 and jit["expect_matches"] and interp["expect_matches"])
print(json.dumps({"cu": jit["cu"], "jit_ns": jit["jit_ns"], "interp_ns": interp["interpreter_ns"], "ok": ok}))
EOF
}

dean() {
    w=$S/dean
    [ -d "$w" ] || git -C "$root/target/compare/dean" worktree add "$w" 7941ce5 >&2
    cmp -s "$root/compare/sha256_dean.rs" "$w/examples/sha256_dean.rs" || cp "$root/compare/sha256_dean.rs" "$w/examples/"
    (cd "$w" && cargo build -q --release --offline --features simd-experiment --example sha256_dean >&2)
    CARGO_TARGET_DIR=$S/dean-target cargo-build-sbf --arch v3 --offline \
        --manifest-path "$root/compare/dean-sha256/Cargo.toml" --sbf-out-dir "$S/dean-deploy" -- --locked >&2
    args=()
    for c in $cases; do args+=("$S/cases/input-$c.bin" "$S/cases/expected-$c.bin"); done
    # checks every case on both engines (panics otherwise), times the first
    "$w/target/release/examples/sha256_dean" "$S/dean-deploy/sha256_dean.so" "${args[@]}" > "$S/dean-time.txt" 2> "$S/dean-stats.txt" || true
    python3 -I - "$S" <<'EOF'
import json, re, sys
S = sys.argv[1]
t = re.match(r"^ok CU (\d+) jit_ns ([\d.]+) interp_ns ([\d.]+)$", open(f"{S}/dean-time.txt").read().strip())
print(json.dumps({"cu": int(t[1]) if t else None, "jit_ns": float(t[2]) if t else None,
                  "interp_ns": float(t[3]) if t else None, "ok": t is not None}))
EOF
}

case "${1:-}" in
    ours) ours ;;
    dean) dean ;;
    *) echo "usage: $0 ours|dean" >&2; exit 2 ;;
esac
