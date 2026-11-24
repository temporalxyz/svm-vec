#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
cpu=${CPU:-77}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

clang -O2 -march=native "$dir/example.c" -o "$out/example"
clang -O2 -march=native "$dir/bench_scratch.c" -o "$out/bench_scratch"
clang -O3 -march=native "$dir/bench_mmu.c" -o "$out/bench_mmu"

echo "=== example (hsum.d with 2 scratch ZMMs vs 0 scratch ZMMs) ==="
taskset -c "$cpu" "$out/example"
echo
echo "=== SIMD-0714 scratch-register op benchmark ==="
"$out/bench_scratch" "$cpu"
echo
echo "=== AVX-512 address translation benchmark (0-12 resident + 2 scratch ZMMs) ==="
taskset -c "$cpu" "$out/bench_mmu"
