#!/usr/bin/env python3
"""Runs the falcon program on the inputs written by `host/` and summarizes the result.

    python3 bench.py DIR [--svm-run PATH] [--so PATH] [--runs N] [--repeat N] [--out FILE]

Checks r0 of every input in DIR/expected.txt (64 signatures, tampered messages and signatures)
on both engines (svm-run fails if they disagree), then reports CU (min, median, max) and the
median JIT and interpreter time per verify over the 64 valid signatures (each the median of N runs).
"""
import argparse, json, statistics, subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--svm-run", default=str(here / "../../harness/target/release/svm-run"))
ap.add_argument("--so", default=str(here / "target/bpfel-unknown-none/release/libfalcon.so"))
ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--out")
a = ap.parse_args()
d = Path(a.dir)
expected = {n: int(v) for n, v, _ in (l.split() for l in open(d / "expected.txt"))}


def run(names, runs):
    args = [a.svm_run, a.so, "--runs", str(runs), "--repeat", str(a.repeat if runs > 1 else 1)]
    for n in names:
        args += ["--input", str(d / f"{n}.in")]
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"svm-run failed: {out.stderr}")
    return {Path(c["input"]).name[:-3]: c for c in json.loads(out.stdout)["cases"]}


tampered = run([n for n in expected if "." in n], 1)
valid = run([n for n in expected if "." not in n], a.runs)
bad = [n for n, c in {**tampered, **valid}.items() if c["r0"] != expected[n] or c["error"]]
if bad:
    sys.exit(f"wrong result for {bad}")
cu = sorted(c["cu"] for c in valid.values())
jit = statistics.median(c["jit_ns"] for c in valid.values()) / 1e3
interp = statistics.median(c["interpreter_ns"] for c in valid.values()) / 1e3
print(f"{len(valid)} valid accepted, {len(tampered)} tampered rejected, on the interpreter and the JIT")
print(f"CU median {statistics.median(cu):.0f} (min {cu[0]}, max {cu[-1]})")
print(f"JIT {jit:.1f} us, interpreter {interp:.1f} us (medians)")
if a.out:
    json.dump({"cu": cu, "jit_us": jit, "interpreter_us": interp,
               "cases": {n: {k: c[k] for k in ("cu", "jit_ns", "interpreter_ns")} for n, c in valid.items()}},
              open(a.out, "w"))
