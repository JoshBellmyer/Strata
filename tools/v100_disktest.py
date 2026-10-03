"""V100 fork: how fast the drive with the model files answers small random reads - NO GPU USED (safe any time).

The engine reads layer 1's n-gram (PLE) table from the second GGUF shard with unbuffered 4 KB reads: ~16 per token,
before every decode window.  On this PC the engine logged ~4-6 ms per read (p50), where an NVMe SSD should take
~0.05-0.2 ms.  This measures the same kind of reads on that file outside the engine, and lists the drives, so the log
shows whether the drive, the folder (OneDrive / Desktop redirection, a filter driver) or the engine is slow.

    .venv\\Scripts\\python tools\\v100_disktest.py                 (what v100\\9_disk_test.bat runs)
    ... --file D:\\models\\x.gguf                                  (another file / drive to compare)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
BLOCK = 4096


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


if WIN:
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                             ctypes.c_void_p]
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
    k32.GetFileAttributesW.restype = wintypes.DWORD
    k32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    GENERIC_READ, SHARE_RW, OPEN_EXISTING = 0x80000000, 0x3, 3
    NO_BUFFERING, RANDOM_ACCESS = 0x20000000, 0x10000000

    class Reader:
        """One unbuffered handle (FILE_FLAG_NO_BUFFERING, as the engine's DirectFile) and a 4 KB-aligned buffer."""
        def __init__(self, path):
            self.h = k32.CreateFileW(str(path), GENERIC_READ, SHARE_RW, None, OPEN_EXISTING,
                                     NO_BUFFERING | RANDOM_ACCESS, None)
            if self.h in (None, wintypes.HANDLE(-1).value):
                raise OSError(ctypes.get_last_error(), "CreateFileW failed")
            self.buf = k32.VirtualAlloc(None, BLOCK, 0x3000, 0x04)
            self.n = wintypes.DWORD(0)

        def read(self, off):
            k32.SetFilePointerEx(self.h, off, None, 0)
            if not k32.ReadFile(self.h, self.buf, BLOCK, ctypes.byref(self.n), None):
                raise OSError(ctypes.get_last_error(), "ReadFile failed")

        def close(self):
            k32.CloseHandle(self.h)
else:
    class Reader:
        def __init__(self, path):
            self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
            import mmap
            self.buf = mmap.mmap(-1, BLOCK)

        def read(self, off):
            os.preadv(self.fd, [self.buf], off)

        def close(self):
            os.close(self.fd)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def test_file(path: Path, log, n_qd1=1500, batches=40, batch=48):
    size = path.stat().st_size
    blocks = size // BLOCK - 1
    log(f"\n--- {path}  ({size / 1e9:.1f} GB)")
    if WIN:
        attrs = k32.GetFileAttributesW(str(path))
        flags = {0x400: "REPARSE_POINT", 0x1000: "OFFLINE", 0x40000: "RECALL_ON_OPEN", 0x400000: "RECALL_ON_DATA_ACCESS",
                 0x80000: "PINNED", 0x100000: "UNPINNED", 0x800: "COMPRESSED", 0x4000: "ENCRYPTED",
                 0x20000: "INTEGRITY_STREAM"}
        log(f"    attributes 0x{attrs:x}: " + (", ".join(v for k, v in flags.items() if attrs & k) or "plain"))
    rnd = random.Random(1)
    r = Reader(path)
    try:
        # QD1: one read at a time (the latency of a single row)
        lat = []
        for _ in range(n_qd1):
            off = rnd.randrange(blocks) * BLOCK
            t0 = time.perf_counter()
            r.read(off)
            lat.append((time.perf_counter() - t0) * 1e3)
        log(f"    one read at a time ({n_qd1}): p50 {pct(lat, .5):.3f} ms, p90 {pct(lat, .9):.3f} ms, p99 "
            f"{pct(lat, .99):.3f} ms, mean {statistics.mean(lat):.3f} ms")
    finally:
        r.close()
    # a decode window's worth: `batch` reads at once (threads, one handle each), the time until the last one lands
    # (a pool of threads started beforehand, so thread start-up is not timed)
    from concurrent.futures import ThreadPoolExecutor
    readers = [Reader(path) for _ in range(batch)]
    pool = ThreadPoolExecutor(max_workers=batch)
    try:
        list(pool.map(lambda i: readers[i].read(0), range(batch)))   # warm the pool
        walls = []
        for _ in range(batches):
            offs = [rnd.randrange(blocks) * BLOCK for _ in range(batch)]
            t0 = time.perf_counter()
            list(pool.map(lambda i: readers[i].read(offs[i]), range(batch)))
            walls.append((time.perf_counter() - t0) * 1e3)
        log(f"    {batch} reads at once ({batches} batches): until the last lands p50 {pct(walls, .5):.2f} ms, "
            f"p90 {pct(walls, .9):.2f} ms, max {max(walls):.2f} ms")
        # the engine's pattern while decoding: a burst every ~40 ms, the drive idle in between.  An NVMe drive that
        # drops into a power-saving state in the gaps pays its wake-up latency on the first read of every burst.
        for gap in (0.010, 0.040, 0.150):
            walls = []
            for _ in range(40):
                time.sleep(gap)
                offs = [rnd.randrange(blocks) * BLOCK for _ in range(batch)]
                t0 = time.perf_counter()
                list(pool.map(lambda i: readers[i].read(offs[i]), range(batch)))
                walls.append((time.perf_counter() - t0) * 1e3)
            log(f"    {batch} at once after {gap * 1000:.0f} ms idle: p50 {pct(walls, .5):.2f} ms, p90 "
                f"{pct(walls, .9):.2f} ms, max {max(walls):.2f} ms")
    finally:
        pool.shutdown()
        for x in readers:
            x.close()
    # one read after an idle gap: the drive's wake-up latency alone
    r = Reader(path)
    try:
        for gap in (0.005, 0.020, 0.040, 0.100, 0.300):
            lat = []
            for _ in range(40):
                time.sleep(gap)
                off = rnd.randrange(blocks) * BLOCK
                t0 = time.perf_counter()
                r.read(off)
                lat.append((time.perf_counter() - t0) * 1e3)
            log(f"    one read after {gap * 1000:.0f} ms idle: p50 {pct(lat, .5):.3f} ms, p90 {pct(lat, .9):.3f} ms, "
                f"max {max(lat):.3f} ms")
    finally:
        r.close()


def overlapped_test(path: Path, log, bursts=60, batch=48, gap=0.040):
    """The engine's way of reading (src/platform/direct_file.cpp): ONE handle opened OVERLAPPED + NO_BUFFERING,
    every read a ReadFile with an OVERLAPPED, completions through an I/O completion port.  If a ReadFile returns
    only when its data is there (synchronously) instead of at once with ERROR_IO_PENDING, the engine's four issuing
    threads read four pages at a time, not 48 - a whole decode window's rows would then take ~12 drive round trips."""
    import ctypes
    from ctypes import wintypes

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    class ENTRY(ctypes.Structure):
        _fields_ = [("key", ctypes.c_size_t), ("ov", ctypes.c_void_p), ("internal", ctypes.c_size_t),
                    ("bytes", wintypes.DWORD)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)   # its own prototypes (ReadFile with an OVERLAPPED)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CreateIoCompletionPort.restype = wintypes.HANDLE
    k32.CreateIoCompletionPort.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD]
    k32.GetQueuedCompletionStatusEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.ULONG,
                                                ctypes.POINTER(wintypes.ULONG), wintypes.DWORD, wintypes.BOOL]
    k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    OVERLAPPED_FLAG = 0x40000000
    h = k32.CreateFileW(str(path), GENERIC_READ, 0x1, None, OPEN_EXISTING, NO_BUFFERING | OVERLAPPED_FLAG | RANDOM_ACCESS,
                        None)
    if h in (None, wintypes.HANDLE(-1).value):
        log("    overlapped open failed:", ctypes.get_last_error())
        return
    port = k32.CreateIoCompletionPort(h, None, 0, 1)
    bufs = k32.VirtualAlloc(None, batch * BLOCK, 0x3000, 0x04)
    ovs = (OVERLAPPED * batch)()
    entries = (ENTRY * 64)()
    got = wintypes.ULONG(0)
    blocks = path.stat().st_size // BLOCK - 1
    rnd = random.Random(7)
    sync_n = 0
    call_ms, burst_ms = [], []
    log(f"\n--- the engine's way: one overlapped handle + a completion port, {batch} ReadFile calls per burst, "
        f"{gap * 1000:.0f} ms apart")
    try:
        for _ in range(bursts):
            time.sleep(gap)
            t0 = time.perf_counter()
            for i in range(batch):
                off = rnd.randrange(blocks) * BLOCK
                ctypes.memset(ctypes.byref(ovs[i]), 0, ctypes.sizeof(OVERLAPPED))
                ovs[i].Offset = off & 0xFFFFFFFF
                ovs[i].OffsetHigh = off >> 32
                c0 = time.perf_counter()
                ok = k32.ReadFile(h, bufs + i * BLOCK, BLOCK, None, ctypes.byref(ovs[i]))
                call_ms.append((time.perf_counter() - c0) * 1e3)
                if ok:
                    sync_n += 1
                elif ctypes.get_last_error() != 997:   # ERROR_IO_PENDING
                    log("    ReadFile failed:", ctypes.get_last_error())
                    return
            done = 0
            while done < batch:   # every read queues one packet, synchronous or not
                if not k32.GetQueuedCompletionStatusEx(port, entries, 64, ctypes.byref(got), 5000, False):
                    log("    completion wait failed:", ctypes.get_last_error())
                    return
                done += got.value
            burst_ms.append((time.perf_counter() - t0) * 1e3)
        n = bursts * batch
        log(f"    ReadFile calls that returned with the data already there (synchronous): {sync_n} of {n} "
            f"({100.0 * sync_n / n:.0f}%)")
        log(f"    time inside one ReadFile call: p50 {pct(call_ms, .5):.3f} ms, p90 {pct(call_ms, .9):.3f} ms, "
            f"max {max(call_ms):.3f} ms")
        log(f"    whole burst ({batch} reads issued from one thread, all completions collected): p50 "
            f"{pct(burst_ms, .5):.2f} ms, p90 {pct(burst_ms, .9):.2f} ms, max {max(burst_ms):.2f} ms")
    finally:
        k32.CloseHandle(port)
        k32.CloseHandle(h)


def spinners(n, log):
    """n busy processes on logical CPUs 0..n-1 (the i5-13600K's P-core threads come first), as the engine's expert
    pool spins on the P-cores while it decodes."""
    procs = []
    for i in range(n):
        p = subprocess.Popen([sys.executable, "-c", "while True: pass"])
        if WIN:
            h = k32.OpenProcess(0x0200 | 0x0400, False, p.pid)   # SET_INFORMATION | QUERY_INFORMATION
            k32.SetProcessAffinityMask(h, ctypes.c_size_t(1 << i))
            k32.CloseHandle(h)
        procs.append(p)
    log(f"\n--- with {n} busy processes pinned to logical CPUs 0-{n - 1} (like the engine's CPU expert pool)")
    return procs


def ps(log, cmd):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        for line in (out.stdout or out.stderr).strip().splitlines():
            log("    " + line.rstrip())
    except Exception as e:  # noqa: BLE001
        log("    powershell failed:", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "strata-unsloth-ud-q4_k_xl.json"))
    ap.add_argument("--file", action="append", help="another file to test (repeatable)")
    ap.add_argument("--log", default=str(ROOT / "v100_disktest.log"))
    ap.add_argument("--no-load", action="store_true", help="skip the run with busy CPU threads")
    ap.add_argument("--full", action="store_true", help="also the earlier tests (thread pool, idle gaps, busy CPU)")
    a = ap.parse_args()
    log = Log(a.log)
    log(f"==== v100_disktest {time.strftime('%Y-%m-%d %H:%M:%S')}")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    args = cfg["args"]
    native = Path(args[args.index("--native") + 1])
    log("    model folder:", native.parent)
    for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(k):
            inside = str(native).lower().startswith(os.environ[k].lower())
            log(f"    {k} = {os.environ[k]}  (model folder inside it: {'YES' if inside else 'no'})")
    files = sorted(native.parent.glob(native.name.replace("00001-of", "0000?-of")))
    ple = [f for f in files if "-00002-of-" in f.name] or files[:1]
    targets = ple + [Path(f) for f in (a.file or [])]
    if WIN:
        log("\n--- drives")
        ps(log, "Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,MediaType,BusType,"
                "@{n='GB';e={[math]::Round($_.Size/1e9)}} | Format-Table -AutoSize | Out-String -Width 200")
        ps(log, "Get-Partition | Where-Object DriveLetter | Select-Object DriveLetter,DiskNumber,"
                "@{n='GB';e={[math]::Round($_.Size/1e9)}} | Format-Table -AutoSize | Out-String -Width 200")
        ps(log, "Get-BitLockerVolume -ErrorAction SilentlyContinue | Select-Object MountPoint,ProtectionStatus,"
                "EncryptionMethod | Format-Table -AutoSize | Out-String -Width 200")
        ps(log, "powercfg /query SCHEME_CURRENT SUB_DISK 2>$null | Select-String -Pattern 'Name|Current AC' | "
                "Out-String -Width 200")
    if WIN and ple:
        try:
            overlapped_test(ple[0], log)
            overlapped_test(ple[0], log, gap=0.0)
        except Exception as e:  # noqa: BLE001
            log(f"    overlapped test failed: {e!r}")
    if not a.full:
        log("\ndone (--full repeats the earlier tests)")
        return 0
    for t in targets:
        try:
            test_file(t, log)
        except Exception as e:  # noqa: BLE001
            log(f"    {t}: failed: {e!r}")
    if ple and not a.no_load:
        procs = spinners(11, log)
        try:
            test_file(ple[0], log, n_qd1=500, batches=20)
        except Exception as e:  # noqa: BLE001
            log(f"    failed: {e!r}")
        finally:
            for p in procs:
                p.kill()
    log("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
