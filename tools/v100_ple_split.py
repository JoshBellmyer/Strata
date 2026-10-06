"""V100 fork: the PLE n-gram table in a file of its own - NO GPU USED.

The engine reads layer 1's n-gram table (per_layer_token_embd.weight, 28.8 GB, in Unsloth's shard 2) with
unbuffered 4 KB reads, 30-50 per decode window.  In Unsloth's layout shard 2 ALSO holds model layers, so the engine
keeps a memory mapping of that whole file alive (the experts and dense weights are read from it in place).  On
Windows, unbuffered reads of a file that has a live mapping take a slower path (the file system keeps the mapped,
cached view coherent with every non-cached read).  The engine sees 4-6 ms per read; the same reads without the
mapping took ~0.2 ms (v100_disktest).  ISTA's original layout had the table alone in its shard, which no one maps.

1. measure: tests/ple_reader_test reads random rows with and without a live mapping of the whole shard;
2. if the mapping makes them slower (or --force): copy the table, byte for byte, into a one-tensor GGUF next to the
   shards (architecture "strata-ple", the form the engine already accepts for a table of its own), check rows of
   both files are identical, and point the config at it (--ple-gguf).  The rows are the same bytes, so the answers
   do not change.  Needs ~29 GB of free disk.

    .venv\\Scripts\\python tools\\v100_ple_split.py            (what v100\\13_ple_split.bat runs)
    ... --force                                                 copy even if the measurement shows no difference
    ... --undo                                                  take --ple-gguf out of the config again
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
WIN = os.name == "nt"
CFG = ROOT / "strata-unsloth-ud-q4_k_xl.json"
TABLE = "per_layer_token_embd.weight"

import gguf_reader as GR  # noqa: E402


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def find_table_shard(native: Path):
    for p in sorted(native.parent.glob(re.sub(r"-0000\d-of-", "-0000?-of-", native.name))):
        g = GR.GGUFFile(p)
        for t in g.tensors:
            if t.name == TABLE:
                return p, g, t
    return None, None, None


def build_test(log) -> Path | None:
    import setup as S
    from v100_setup import build_dir
    bdir = build_dir()   # build-cuda12/ since the upstream 0.1.40 merge, else build/
    exe = bdir / ("ple_reader_test.exe" if WIN else "ple_reader_test")
    cmake = S.find_tool("cmake")
    if cmake is None or not (bdir / "CMakeCache.txt").exists():
        log("  no build folder / cmake: run v100\\1_setup_and_build.bat once first")
        return None
    cmd = [cmake, "--build", str(bdir), "--target", "ple_reader_test", "-j", "4"]
    if WIN:
        vcvars = S.find_vcvars()
        bat = ROOT / "build-v100-pletest.bat"
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(cmd)}\r\n', encoding="utf-8")
        cmd = ["cmd", "/c", str(bat)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    for line in (r.stdout + r.stderr).splitlines()[-15:]:
        log("    " + line)
    if r.returncode != 0 or not exe.exists():
        log("  building ple_reader_test failed")
        return None
    return exe


def run_test(exe: Path, gguf: Path, log, label, extra) -> float | None:
    cmd = [str(exe), "--gguf", str(gguf), "--rows", "8000", "--gap-ms", "40", *extra]
    log(f"\n  {label}: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    p50 = None
    for line in (r.stdout + r.stderr).splitlines():
        log("    " + line)
        m = re.search(r"per token .*p50 (\d+) us", line)
        if m:
            p50 = float(m.group(1))
    return p50


def write_table(src: Path, g, t, out: Path, log) -> bool:
    nbytes = t.expected_bytes()
    if nbytes is None:
        log(f"  {TABLE} is {t.type_name}: unknown size")
        return False
    start = g.data_start + t.offset
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("strata-ple")
    header += gguf_str(TABLE) + struct.pack("<I", len(t.shape)) + struct.pack(f"<{len(t.shape)}Q", *t.shape)
    header += struct.pack("<IQ", t.type_id, 0)
    header += b"\0" * ((-len(header)) % 32)
    if out.exists() and out.stat().st_size == len(header) + nbytes:
        log(f"  {out.name} already there ({out.stat().st_size / 1e9:.1f} GB): checking it")
        return True
    free = shutil.disk_usage(out.parent).free
    if free < nbytes + (2 << 30):
        log(f"  not enough free disk: {free / 1e9:.1f} GB free, {nbytes / 1e9:.1f} GB needed")
        return False
    log(f"  copying {nbytes / 1e9:.1f} GB of {src.name} into {out.name} ...")
    tmp = out.with_suffix(".partial")
    t0 = time.time()
    chunk = 64 << 20
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(header)
        fi.seek(start)
        left = nbytes
        done_mark = 0
        while left > 0:
            b = fi.read(min(chunk, left))
            if not b:
                log("  the shard ended early")
                return False
            fo.write(b)
            left -= len(b)
            if (nbytes - left) // (4 << 30) > done_mark:
                done_mark = (nbytes - left) // (4 << 30)
                log(f"    {(nbytes - left) / 1e9:.0f} GB")
    os.replace(tmp, out)
    log(f"  written in {time.time() - t0:.0f} s")
    return True


def check_rows(src: Path, g, t, out: Path, log, n=4000) -> bool:
    nbytes = t.expected_bytes()
    rows = t.shape[1]
    rb = nbytes // rows
    o = GR.GGUFFile(out)
    ot = o.tensors[0]
    if ot.name != TABLE or ot.shape != t.shape or ot.type_id != t.type_id or out.stat().st_size != o.data_start + nbytes:
        log("  the new file's header does not match the table")
        return False
    rnd = random.Random(5)
    picks = [0, rows - 1] + [rnd.randrange(rows) for _ in range(n)]
    with open(src, "rb") as fa, open(out, "rb") as fb:
        for r in picks:
            fa.seek(g.data_start + t.offset + r * rb)
            fb.seek(o.data_start + r * rb)
            if fa.read(rb) != fb.read(rb):
                log(f"  row {r} differs")
                return False
    log(f"  {len(picks)} rows (first, last and random) are byte-identical")
    return True


def set_config(path: Path | None, log):
    raw = CFG.read_bytes()
    c = json.loads(raw.decode("utf-8-sig"))
    a = c["args"]
    if "--ple-gguf" in a:
        i = a.index("--ple-gguf")
        del a[i:i + 2]
    if path is not None:
        a += ["--ple-gguf", str(path)]
    txt = json.dumps(c, indent=1)
    if b"\r\n" in raw:
        txt = txt.replace("\n", "\r\n")
    CFG.write_text(txt, encoding="utf-8", newline="")
    log(f"  config: {'--ple-gguf ' + str(path) if path else 'no --ple-gguf (the table is read from the shard)'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "v100_ple_split.log"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--retest", action="store_true", help="only time the reads again: the shard alone, the shard "
                    "mapped, the table's own file (after the copy has settled on the drive)")
    a = ap.parse_args()
    log = Log(a.log)
    log(f"==== v100_ple_split {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if a.undo:
        set_config(None, log)
        return 0
    c = json.loads(CFG.read_text(encoding="utf-8-sig"))
    native = Path(c["args"][c["args"].index("--native") + 1])
    src, g, t = find_table_shard(native)
    if src is None:
        log(f"  no shard beside {native} holds {TABLE}")
        return 1
    others = [x.name for x in g.tensors if x.name != TABLE]
    log(f"  the table: {src.name}, {t.type_name} {t.shape}, {t.expected_bytes() / 1e9:.1f} GB; the shard also holds "
        f"{len(others)} other tensors ({src.stat().st_size / 1e9:.1f} GB in all)")
    out = src.parent / (re.sub(r"-0000\d-of-0000\d$", "", src.stem) + "-ple-table.gguf")
    if a.retest:
        exe = build_test(log)
        if exe is None:
            return 1
        r = {"shard alone": run_test(exe, src, log, "the shard, alone", ["--direct-only"]),
             "shard mapped": run_test(exe, src, log, "the shard, mapped", ["--map-only"])}
        if out.exists():
            r["own file"] = run_test(exe, out, log, "the table's own file", ["--direct-only"])
        log("\n  per token p50 (us): " + ", ".join(f"{k} {v}" for k, v in r.items()))
        return 0
    go = a.force
    exe = None if a.skip_test else build_test(log)
    if exe is not None:
        alone = run_test(exe, src, log, "1. the reads alone (no mapping of the file in the process)", ["--direct-only"])
        mapped = run_test(exe, src, log, "2. the same reads with the whole shard mapped (as in the engine)",
                          ["--map-only"])
        touched = run_test(exe, src, log, "3. mapped, and 512 MB of the shard's layers touched",
                           ["--map-only", "--touch-mb", "512"])
        log(f"\n  per token p50: alone {alone} us, mapped {mapped} us, mapped+touched {touched} us")
        worst = max(x for x in (mapped, touched) if x is not None) if (mapped or touched) else None
        if alone and worst and worst >= 1.5 * alone:
            log("  => the live mapping slows the reads: the table gets a file of its own")
            go = True
        elif not a.force:
            log("  => no clear slowdown from the mapping; nothing copied (--force copies anyway)")
    if not go:
        return 0
    if not write_table(src, g, t, out, log) or not check_rows(src, g, t, out, log):
        return 1
    if exe is not None:
        run_test(exe, out, log, "4. the new file, alone", ["--direct-only"])
    set_config(out, log)
    log("\n  done: the engine reads the table from its own file from the next start on (--undo reverts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
