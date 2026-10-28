#!/usr/bin/env python3
"""Ours vs Blueshift's sBPF SIMD experiment (Dean's), and Blueshift's scalar Falcon-512.

    python3 compare/compare.py [--rounds N]

Clones blueshift-gg/solana-sbpf at REV into target/compare/dean, adds a keccak-only entry to his
Falcon program (dean-keccak-only.patch, needs cargo-build-sbf), then runs in interleaved rounds:
his Falcon bench, ours, his keccak-only example, ours, and Blueshift's scalar program on our
harness. Our side uses svm-run --repeat, the same method as his: the mean of back-to-back runs on
one memory mapping, a fresh VM each, median over samples. Run ./bench.sh first.
"""
import argparse, json, os, re, statistics, struct, subprocess, sys
from pathlib import Path

REV = "7941ce54679e76ed75f96317621d31d884b170d7"
root = Path(__file__).resolve().parent.parent
work = root / "target/compare"
dean = work / "dean"
run = root / "harness/target/release/svm-run"
falcon = root / "programs/falcon"
keccak = root / "programs/keccak/target/bpfel-unknown-none/release/libkeccak.so"
bench = Path(os.environ.get("TMPDIR", "/tmp")) / "svm-vec-bench"  # written by ./bench.sh

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()


def sh(*args, cwd=None, stderr=False):
    out = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"{' '.join(map(str, args))} failed:\n{out.stdout}{out.stderr}")
    return out.stdout + out.stderr if stderr else out.stdout


work.mkdir(parents=True, exist_ok=True)
if not dean.exists():
    sh("git", "clone", "-q", "https://github.com/blueshift-gg/solana-sbpf", dean)
sh("git", "checkout", "-q", "-f", REV, cwd=dean)
sh("git", "apply", root / "compare/dean-keccak-only.patch", cwd=dean)
sh("cp", root / "compare/keccak_only.rs", dean / "examples/")
sh("cargo-build-sbf", "--arch", "v3", cwd=dean / "simd-falcon/program")
sh("cp", dean / "simd-falcon/program/target/deploy/falcon_simd.so", work / "dean-keccak.so")
sh("git", "checkout", "-q", "-f", REV, cwd=dean)

# Blueshift's scalar program takes [0u64][ix length][signature][message][program id]
sigs = (falcon / "fixtures/sigs.bin").read_bytes()
assert sigs == (dean / "simd-falcon/fixtures/sigs.bin").read_bytes()
scalar_inputs, i = [], 0
while i < len(sigs):
    n = struct.unpack_from("<H", sigs, i + 666)[0]
    p = work / f"scalar-{len(scalar_inputs)}.in"
    p.write_bytes(struct.pack("<QQ", 0, 666 + n) + sigs[i:i + 666] + sigs[i + 668:i + 668 + n] + b"\7" * 32)
    scalar_inputs.append(p)
    i += 668 + n


def svm_run(so, inputs, runs, repeat):
    args = [run, so, "--runs", str(runs), "--repeat", str(repeat)]
    for p in inputs:
        args += ["--input", p]
    return json.loads(sh(*args))["cases"]


rows = {k: [] for k in ["falcon", "keccak", "scalar"]}
for r in range(a.rounds):
    out = sh("cargo", "bench", "-q", "--features", "simd-experiment", "--bench", "simd_falcon", cwd=dean, stderr=True)
    d = re.search(r"simd\s+jit\s+CUs median\s+(\d+)\s+time median\s+([\d.]+)", out)
    b = re.search(r"blueshift\s+jit\s+CUs median\s+(\d+)\s+time median\s+([\d.]+)", out)
    sh("python3", falcon / "bench.py", bench / "falcon", "--runs", "9", "--repeat", "160",
       "--out", work / "ours-falcon.json")
    o = json.loads((work / "ours-falcon.json").read_text())
    rows["falcon"].append((int(d[1]), float(d[2]), statistics.median(o["cu"]), o["jit_us"]))

    k = re.search(r"per permutation ([\d.]+) CU ([\d.]+) ns",
                  sh("cargo", "run", "-q", "--release", "--features", "simd-experiment", "--example",
                     "keccak_only", "--", work / "dean-keccak.so", cwd=dean))
    c = [svm_run(keccak, [bench / f"keccak{n}/input-vector.bin"], 15, 64)[0] for n in (0, 100)]
    rows["keccak"].append((float(k[1]), float(k[2]), (c[1]["cu"] - c[0]["cu"]) / 100,
                           (c[1]["jit_ns"] - c[0]["jit_ns"]) / 100))

    s = svm_run(dean / "simd-falcon/fixtures/falcon_blueshift.so", scalar_inputs, 9, 7)
    assert all(x["r0"] == 0 for x in s)
    rows["scalar"].append((int(b[1]), float(b[2]), statistics.median(x["cu"] for x in s),
                           statistics.median(x["jit_ns"] for x in s) / 1e3))
    print(f"round {r + 1}: " + json.dumps(rows), file=sys.stderr)

med = lambda k, i: statistics.median(x[i] for x in rows[k])
spread = lambda k, i: " / ".join(f"{x[i]:.1f}" for x in rows[k])
print(f"Dean = blueshift-gg/solana-sbpf@{REV[:7]}, {a.rounds} interleaved rounds, medians\n")
print("| | Dean | ours | ours is |\n|---|---:|---:|---:|")
print(f"| Falcon-512 verify, CU | {med('falcon', 0):,.0f} | {med('falcon', 2):,.0f} | {med('falcon', 0) / med('falcon', 2):.2f}× fewer |")
print(f"| Falcon-512 verify, JIT µs | {spread('falcon', 1)} | {spread('falcon', 3)} | {med('falcon', 1) / med('falcon', 3):.2f}× faster |")
print(f"| Keccak-f, CU / permutation | {med('keccak', 0):,.0f} | {med('keccak', 2):,.0f} | {med('keccak', 0) / med('keccak', 2):.2f}× fewer |")
print(f"| Keccak-f, JIT ns / permutation | {spread('keccak', 1)} | {spread('keccak', 3)} | {med('keccak', 1) / med('keccak', 3):.2f}× faster |")
print("\nBlueshift's scalar Falcon-512, same inputs, timed by both harnesses:\n")
print("| | Dean's harness | svm-run |\n|---|---:|---:|")
print(f"| CU | {med('scalar', 0):,.0f} | {med('scalar', 2):,.0f} |")
print(f"| JIT µs | {spread('scalar', 1)} | {spread('scalar', 3)} |")
