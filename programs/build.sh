#!/usr/bin/env bash
# Builds sBPF v3 programs that use svm-intrinsics, linked by the forked
# sbpf-linker (which runs the vector pass).
#
# usage: ./build.sh [program-or-workspace-dir...]   (default: toolchain-tests)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cargo build --release --quiet --manifest-path "$here/../sbpf-linker/Cargo.toml"

dirs=("$@")
[ ${#dirs[@]} -gt 0 ] || dirs=("$here/toolchain-tests")
for dir in "${dirs[@]}"; do
    (cd "$dir" && cargo +nightly build-sbpf)
    ls -1 "$dir"/target/deploy/*.so
done
