#!/usr/bin/env python3
"""Static checks of the toolchain-test kernels, and (when the runtime's
`svm-run` exists) end-to-end runs against host-computed expected outputs.

usage: check.py [--list KERNEL|FILE.so] [--svm-run PATH]
"""
import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "target", "deploy")

# op -> (name, kind, imm8 limit); limit None = imm8 must be 0
OPS = {}
for code, name, kind, imm in [
    (0x00, "ld", "ls", None), (0x01, "st", "st", None), (0x02, "ldbc.d", "ls", None),
    (0x03, "ldbc.q", "ls", None), (0x06, "splat.d", "ls", None), (0x08, "zero", "r0", None),
    (0x09, "mov", "r2", None),
    (0x11, "add.w", "r3", None), (0x13, "add.q", "r3", None), (0x15, "sub.w", "r3", None),
    (0x17, "sub.q", "r3", None), (0x18, "mullo.w", "r3", None), (0x1a, "mullo.q", "r3", None),
    (0x1d, "mulhi.w", "r3", None), (0x20, "max.d", "r3", None), (0x21, "min.d", "r3", None),
    (0x27, "minu.q", "r3", None), (0x28, "and", "r3", None), (0x2a, "xor", "r3", None),
    (0x2c, "ternlog", "a3i", 256), (0x2f, "minu.w", "r3", None),
    (0x32, "shli.q", "r2i", 64), (0x35, "shri.q", "r2i", 64), (0x37, "sari.d", "r2i", 32),
    (0x38, "sari.q", "r2i", 64), (0x3a, "roli.q", "r2i", 64), (0x40, "rolv.q", "r3", None),
    (0x50, "dpbusd", "a3", None), (0x51, "dpwssd", "a3", None), (0x53, "maddwd", "r3", None),
    (0x5c, "permq", "r3", None), (0x5d, "shufb", "r3", None),
    (0x5e, "perm2.w", "a3", None), (0x5f, "perm2.q", "a3", None),
    (0x60, "narrowus.d.b", "r2", None), (0xb1, "abs.q", "r2", None),
]:
    OPS[code] = (name, kind, imm)

# ld, st, ldbc.d, ldbc.q access r1 + off
MEMORY = (0x00, 0x01, 0x02, 0x03)

# fields used per kind: (out, in0, in1)
FIELDS = {
    "r3": (1, 1, 1), "a3": (1, 1, 1), "a3i": (1, 1, 1), "r2": (1, 1, 0), "r2i": (1, 1, 0),
    "r0": (1, 0, 0), "ls": (1, 0, 0), "st": (0, 1, 0),
}


def text_words(path):
    """The 8-byte words of the executable PT_LOAD segment."""
    data = open(path, "rb").read()
    assert data[:4] == b"\x7fELF", path
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    for i in range(phnum):
        p_type, p_flags, p_offset, _, _, p_filesz = struct.unpack_from(
            "<IIQQQQ", data, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            seg = data[p_offset:p_offset + p_filesz]
            return data, [seg[j:j + 8] for j in range(0, len(seg) - 7, 8)]
    raise SystemExit(f"{path}: no executable segment")


def decode(word):
    imm, = struct.unpack_from("<I", word, 4)
    return imm >> 24, imm >> 16 & 0xff, imm >> 15 & 1, imm >> 10 & 31, imm >> 5 & 31, imm & 31


def vector_ops(words):
    """[(index, op, imm8, out, in0, in1)] for every CALL_IMM src=2."""
    out = []
    lddw_tail = False
    for i, w in enumerate(words):
        if lddw_tail:
            lddw_tail = False
            continue
        if w[0] == 0x18:
            lddw_tail = True
        if w[0] == 0x85 and w[1] >> 4 == 2:
            op, imm8, bit15, o, a, b = decode(w)
            out.append((i, op, imm8, bit15, o, a, b, w))
    return out


def check_static(kernel, expect):
    path = os.path.join(DEPLOY, f"{kernel}.so")
    data, words = text_words(path)
    errors = []
    if b"__svm_" in data:
        errors.append("an __svm_ symbol remains")
    ops = vector_ops(words)
    for i, op, imm8, bit15, o, a, b, w in ops:
        where = f"insn {i} ({w.hex()})"
        off, = struct.unpack_from("<h", w, 2)
        if w[1] & 0xf or (off and op not in MEMORY):
            errors.append(f"{where}: dst set, or off on a non-memory op")
        if op not in OPS:
            errors.append(f"{where}: unknown op {op:#x}")
            continue
        name, kind, lim = OPS[op]
        if bit15:
            errors.append(f"{where}: imm[15] set")
        for used, reg, field in zip(FIELDS[kind], (o, a, b), ("out", "in0", "in1")):
            if reg >= 16:
                errors.append(f"{where}: {name} {field} = v{reg}")
            if not used and reg:
                errors.append(f"{where}: {name} unused {field} = {reg}")
        ok = imm8 == 0 if lim is None else imm8 < lim
        if not ok:
            errors.append(f"{where}: {name} imm8 {imm8:#x} out of range")
    names = [OPS[op][0] for _, op, *_ in ops if op in OPS]
    names += ["off"] * sum(w[2:4] != b"\0\0" for *_, w in ops)
    for name, count in expect.items():
        have = names.count(name)
        if (have < count) if count > 0 else have:
            errors.append(f"expected {'>=' if count > 0 else 'no'} {abs(count)} `{name}`, found {have}")
    if not any(w[0] == 0x95 for w in words):
        errors.append("no exit instruction")
    return ops, errors


def listing(kernel):
    path = kernel if kernel.endswith(".so") else os.path.join(DEPLOY, f"{kernel}.so")
    out = subprocess.run(["llvm-objdump-19", "-d", path], capture_output=True, text=True).stdout
    _, words = text_words(path)
    vops = {i: rest for i, *rest in vector_ops(words)}
    base = None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].rstrip(":").strip().isdigit():
            index = int(parts[0].rstrip(":"))
            base = index if base is None else base
            n = index - base
            if n in vops:
                op, imm8, _, o, a, b, w = vops[n]
                off, = struct.unpack_from("<h", w, 2)
                name, kind, _ = OPS.get(op, (f"op{op:#x}", "r3", None))
                f = FIELDS[kind]
                args = [f"v{r}" for used, r in zip(f, (o, a, b)) if used]
                if kind in ("ls", "st"):
                    args.append((f"[r1{off:+#x}]" if off else "[r1]") if op in MEMORY else "r1")
                if imm8 or kind in ("r2i", "a3i"):
                    args.append(f"{imm8:#x}")
                parts[2] = f"{name} " + ", ".join(args)
            print(f"{n:4}:  {parts[1].strip()}  {parts[2]}")


# ---- host reference implementations of the kernels ----

def u32s(b):
    return list(struct.unpack("<16I", b))


def i32s(b):
    return list(struct.unpack("<16i", b))


def p32(v):
    return struct.pack("<16I", *[x & 0xffffffff for x in v])


def p64(v):
    return struct.pack("<8Q", *[x & 0xffffffffffffffff for x in v])


def u64s(b):
    return list(struct.unpack("<8Q", b))


def rnd(rng, n):
    return bytes(rng.getrandbits(8) for _ in range(n))


def case_vadd(rng):
    inp = bytearray(rnd(rng, 128) + bytes(136))
    a, b = u64s(inp[:64]), u64s(inp[64:128])
    exp = bytearray(inp)
    exp[128:192] = p64([x + y for x, y in zip(a, b)])
    exp[192:256] = p64([x - y for x, y in zip(a, b)])
    exp[256:264] = struct.pack("<Q", sum(a) & 0xffffffffffffffff)
    return inp, exp


def case_dot(rng, n=37):
    inp = bytearray(struct.pack("<Q", n) + bytes(120) + rnd(rng, 128 * n))
    a, b = inp[128:128 + 64 * n], inp[128 + 64 * n:]
    acc = [0] * 16
    for i in range(64 * n):
        acc[(i % 64) // 4] += a[i] * struct.unpack("b", b[i:i + 1])[0]
    exp = bytearray(inp)
    exp[64:128] = p32(acc)
    exp[8:16] = struct.pack("<Q", sum(acc) & 0xffffffff)
    return inp, exp


def case_nested(rng, rows=5, cols=7):
    inp = bytearray(struct.pack("<QQ", rows, cols) + bytes(240) + rnd(rng, 64 * rows * cols))
    m = 2**64 - 1
    best, total = [m] * 8, [0] * 8
    for r in range(rows):
        row = [0] * 8
        for c in range(cols):
            x = u64s(inp[256 + (r * cols + c) * 64:][:64])
            row = [(p + q) & m for p, q in zip(row, x)]
        best = [min(p, q) for p, q in zip(best, row)]
        total = [(p + q) & m for p, q in zip(total, row)]
    exp = bytearray(inp)
    exp[64:128] = p64(best)
    exp[128:192] = p64(total)
    return inp, exp


def case_branchy(rng, n=21):
    flags = rng.getrandbits(64) | 1 << 63
    inp = bytearray(struct.pack("<QQ", n, flags) + bytes(240) + rnd(rng, 64 * n))
    hi, lo = [-2**31] * 16, [2**31 - 1] * 16
    for i in range(n):
        x = i32s(inp[256 + 64 * i:][:64])
        if flags >> i & 1:
            hi = [max(p, q) for p, q in zip(hi, x)]
        else:
            lo = [min(p, q) for p, q in zip(lo, x)]
    exp = bytearray(inp)
    exp[64:128], exp[128:192] = p32(hi), p32(lo)
    exp[192:256] = p32(hi if flags >> 63 else lo)
    return inp, exp


def case_offsets(rng, n=4096):
    inp = bytearray(struct.pack("<Q", n) + rnd(rng, n + 40064 - 8))
    a, b, c = (u64s(inp[k:k + 64]) for k in (64, n + 28000, n + 40000))
    x, = struct.unpack_from("<Q", inp, n - 8)
    exp = bytearray(inp)
    exp[128:192] = p64([p + q for p, q in zip(a, b)])
    exp[n + 32767:n + 32831] = p64([p + x for p in c])
    return inp, exp


def case_pressure(rng):
    inp = bytearray(rnd(rng, 1024) + bytes(1024))
    v = [u64s(inp[64 * k:][:64]) for k in range(16)]
    exp = bytearray(inp)
    for k in range(16):
        j = k + 1 if k < 15 else 15
        exp[1024 + 64 * k:1088 + 64 * k] = p64([x + y for x, y in zip(v[k], v[j])])
    return inp, exp


def case_spill(rng):
    inp = bytearray(rnd(rng, 1024) + bytes(1024))
    v = [u64s(inp[64 * k:][:64]) for k in range(16)]
    exp = bytearray(inp)
    for k in range(16):
        exp[1024 + 64 * k:1088 + 64 * k] = p64([x + y for x, y in zip(v[k], v[(k + 1) % 16])])
    return inp, exp


def case_calls(rng):
    inp = bytearray(rnd(rng, 128) + bytes(128))
    a, b = u64s(inp[:64]), u64s(inp[64:128])
    b2 = [x * 2 for x in b]
    exp = bytearray(inp)
    exp[64:128] = p64(b2)
    exp[128:192] = inp[:64]
    exp[192:256] = p64([x ^ y for x, y in zip(a, b2)])
    return inp, exp


def case_misc(rng):
    inp = bytearray(rnd(rng, 192) + bytes(640))
    a, b, c = inp[:64], inp[64:128], inp[128:192]
    qa, qb, qc = u64s(a), u64s(b), u64s(c)
    exp = bytearray(inp)
    outs = []
    # o0 = dpbusd(a, b, c)
    acc = u32s(a)
    for i in range(64):
        acc[i // 4] += b[i] * struct.unpack("b", c[i:i + 1])[0]
    outs.append(p32(acc))
    # o1 = ternlog 0xca (a ? b : c bitwise)
    ia, ib, ic = (int.from_bytes(x, "little") for x in (a, b, c))
    outs.append(((ia & ib) | (~ia & ic) & (2**512 - 1)).to_bytes(64, "little"))
    outs.append(p64([x << 3 for x in qa]))
    outs.append(struct.pack("<16i", *[x >> 5 for x in i32s(b)]))
    outs.append(p64([(x << 13 | x >> 51) for x in qc]))
    outs.append(struct.pack("<8q", *[x >> 17 for x in struct.unpack("<8q", b)]))
    outs.append(p64([x << (y % 64) | x >> (64 - y % 64) for x, y in zip(qa, qb)]))
    outs.append(p64([qa[qc[i] & 7] for i in range(8)]))
    sh = bytearray(64)
    for i in range(64):
        idx = b[i]
        sh[i] = 0 if idx & 0x80 else a[16 * (i // 16) + (idx & 15)]
    outs.append(bytes(sh))
    outs.append(p64([(qc + qb)[qa[i] & 15] for i in range(8)]))
    for k, o in enumerate(outs):
        exp[192 + 64 * k:256 + 64 * k] = o
    return inp, exp


KERNELS = {
    "vadd": ({"ld": 2, "add.q": 1, "sub.q": 1, "st": 3, "off": 3}, case_vadd),
    "dot": ({"zero": 1, "ld": 2, "dpbusd": 1, "st": 1, "mov": -1}, case_dot),
    "nested": ({"zero": 1, "splat.d": 1, "add.q": 2, "minu.q": 1}, case_nested),
    "offsets": ({"ld": 3, "ldbc.q": 1, "st": 2, "off": 5}, case_offsets),
    "branchy": ({"splat.d": 2, "max.d": 1, "min.d": 1, "st": 3}, case_branchy),
    "misc": ({"dpbusd": 1, "mov": 1, "ternlog": 1, "shli.q": 1, "sari.d": 1, "roli.q": 1,
              "sari.q": 1, "rolv.q": 1, "permq": 1, "shufb": 1, "perm2.q": 1}, case_misc),
    "pressure": ({"ld": 16, "add.q": 16, "st": 16, "mov": -1}, case_pressure),
    "spill": ({"ld": 17, "add.q": 16, "st": 17}, case_spill),
    "calls": ({"ld": 3, "add.q": 1, "xor": 1, "st": 3}, case_calls),
    "all_ops": ({OPS[op][0]: 1 for op in OPS if op != 0x09}, None),
}


def run(svm_run, kernel, inp):
    """Runs a kernel in svm-run (interpreter and JIT, which it checks agree)
    and returns (r0, final input region, error)."""
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(inp)
        f.flush()
        res = subprocess.run(
            [svm_run, os.path.join(DEPLOY, f"{kernel}.so"), "--input", f.name,
             "--runs", "1", "--hex"], capture_output=True, text=True)
    try:
        case = json.loads(res.stdout)["cases"][0]
    except (ValueError, KeyError, IndexError):
        return None, b"", (res.stdout + res.stderr).strip()[-500:]
    out = bytes.fromhex(next(v for k, v in case.items() if "hex" in k))
    error = case["error"] or (res.returncode and res.stderr.strip()[-500:])
    return case["r0"], out, error


def main():
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        listing(args[1])
        return
    svm_run = args[args.index("--svm-run") + 1] if "--svm-run" in args else None
    failed = False
    for kernel, (expect, case) in KERNELS.items():
        ops, errors = check_static(kernel, expect)
        regs = sorted({r for _, op, _, _, o, a, b, _ in ops for r in (o, a, b)})
        status = "ok" if not errors else "FAIL"
        print(f"{kernel:9} static {status}: {len(ops)} vector ops, regs v{regs[0]}..v{regs[-1]}")
        for e in errors:
            print(f"    {e}")
        failed |= bool(errors)
        if svm_run and case:
            inp, exp = case(random.Random(kernel))
            ret, out, error = run(svm_run, kernel, inp)
            ok = ret == 0 and out == exp and not error
            print(f"{kernel:9} run    {'ok' if ok else 'FAIL'} (r0 = {ret})")
            if error:
                print(f"    {error}")
            if not ok:
                diff = [i for i in range(min(len(out), len(exp))) if out[i] != exp[i]]
                print(f"    {len(diff)} bytes differ, first at {diff[:8]}")
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
