"""V100 fork: run the server for your real work, with everything logged - USES BOTH GPUs (it is the server).

Starts the server exactly as run-unsloth-ud-q4_k_xl.bat does (same config, port 8080, the browser page), and while
you work it records into runs\\workload-<date-time>\\:
  engine.log   the engine's own log, with the timing switches on: per request the prompt and writing speed, where
               each verify window's time goes (STRATA_DECODE_TIMING / STRATA_SPLIT_TIMING), the prompt path's phases
               per chunk (STRATA_PREFILL_TIMING), the expert tiers, the PLE reads
  server.log   the server window's lines (the 80-character preview of your last message is blanked out)
  gpu.csv      both GPUs every 2 s: load, memory, temperature, power, clocks, PCIe link
  system.csv   CPU load, free RAM and disk reads every 2 s (Windows typeperf)
  routing.bin  which experts every decode step routed to (for an expert ranking built from your own work; no text)
  summary.txt  made when you stop: every request (context length, tokens reused, prompt and writing speed) and
               averages by context length
No prompt or answer text is written to any of these files.  Stop with Ctrl+C in this window when you are done; the
summary and a zip of everything except routing.bin (which stays on your PC) are written then.

    .venv\\Scripts\\python tools\\v100_workload.py                    (what v100\\15_workload.bat runs)
    ... --no-routing      do not record the routing (saves ~4 KB per generated token)
    ... --no-prefill-timing
    ... -- --api-key KEY  everything after -- goes to the server
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
CFG = ROOT / "strata-unsloth-ud-q4_k_xl.json"
REQ = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), "
                 r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
DEC = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify "
                 r"([\d.]+) .*?\+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+); per layer-window: CPU "
                 r"experts ([\d.]+)")
HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%")
REDACT = re.compile(r"last=.*$")


def start_loggers(run: Path):
    procs = []
    try:
        f = open(run / "gpu.csv", "w", encoding="utf-8")
        procs.append((subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,"
             "memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,"
             "pcie.link.width.current", "--format=csv", "-l", "2"], stdout=f, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL), f))
    except OSError as e:
        print("[workload] nvidia-smi logging off:", e)
    if WIN:
        try:
            procs.append((subprocess.Popen(
                ["typeperf", r"\Processor(_Total)\% Processor Time", r"\Memory\Available MBytes",
                 r"\Memory\Pages/sec", r"\PhysicalDisk(_Total)\Disk Reads/sec",
                 r"\PhysicalDisk(_Total)\Disk Read Bytes/sec", r"\PhysicalDisk(_Total)\Avg. Disk sec/Read",
                 "-si", "2", "-o", str(run / "system.csv"), "-f", "CSV", "-y"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL), None))
        except OSError as e:
            print("[workload] typeperf logging off:", e)
    return procs


def stop(procs):
    for p, f in procs:
        try:
            p.terminate()
            p.wait(10)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        if f:
            f.close()


def bucket(n):
    for lim, name in ((8_000, "<8K"), (32_000, "8-32K"), (64_000, "32-64K"), (128_000, "64-128K"),
                      (200_000, "128-200K")):
        if n < lim:
            return name
    return "200K+"


def summarize(run: Path) -> str:
    log = run / "engine.log"
    if not log.exists():
        return "no engine.log"
    reqs, dec, hit = [], None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = DEC.search(line)
        if m:
            dec = m
            continue
        m = REQ.search(line)
        if m:
            reqs.append({"prompt": int(m[1]), "reused": int(m[2]), "read": int(m[3]), "read_ms": float(m[4]),
                         "pp": float(m[5]), "gen": int(m[6]), "gen_ms": float(m[7]), "tg": float(m[8]),
                         "acc": int(m[9]), "off": int(m[10]), "dec": dec, "hit": None})
            dec = None
            continue
        m = HIT.search(line)
        if m and reqs and reqs[-1]["hit"] is None:
            reqs[-1]["hit"] = float(m[1])
    out = [f"workload run {run.name}: {len(reqs)} requests", ""]
    out.append(f"{'#':>3} {'context':>8} {'reused':>8} {'read':>8} {'read s':>7} {'prompt t/s':>10} {'gen':>6} "
               f"{'write t/s':>9} {'ms/win':>7} {'tok/win':>7} {'PLE ms':>6} {'CPU exp':>7} {'GPU hit':>7}")
    for i, r in enumerate(reqs, 1):
        d = r["dec"]
        out.append(f"{i:>3} {r['prompt']:>8} {r['reused']:>8} {r['read']:>8} {r['read_ms'] / 1000:>7.1f} "
                   f"{r['pp']:>10.1f} {r['gen']:>6} {r['tg']:>9.1f} "
                   f"{(float(d[4]) if d else 0):>7.1f} {(float(d[3]) if d else 0):>7.2f} "
                   f"{(float(d[6]) if d else 0):>6.1f} {(float(d[9]) if d else 0):>7.2f} "
                   f"{(str(r['hit']) + '%') if r['hit'] is not None else '-':>7}")
    out += ["", "by context length (requests that wrote >= 32 tokens / read >= 1000 new tokens):"]
    groups = {}
    for r in reqs:
        groups.setdefault(bucket(r["prompt"]), []).append(r)
    for name in ("<8K", "8-32K", "32-64K", "64-128K", "128-200K", "200K+"):
        g = groups.get(name)
        if not g:
            continue
        w = [r["tg"] for r in g if r["gen"] >= 32]
        p = [r["pp"] for r in g if r["read"] >= 1000]
        reuse = sum(r["reused"] for r in g) / max(1, sum(r["prompt"] for r in g))
        out.append(f"  {name:>9}: {len(g)} requests; writing median "
                   f"{statistics.median(w) if w else float('nan'):.1f} tok/s ({len(w)}); prompt reading median "
                   f"{statistics.median(p) if p else float('nan'):.1f} tok/s ({len(p)}); {100 * reuse:.0f}% of "
                   f"prompt tokens reused from the cache")
    tot_gen = sum(r["gen"] for r in reqs)
    tot_gen_s = sum(r["gen_ms"] for r in reqs) / 1000
    tot_read = sum(r["read"] for r in reqs)
    tot_read_s = sum(r["read_ms"] for r in reqs) / 1000
    out += ["", f"total: {tot_gen} tokens written in {tot_gen_s:.0f} s; {tot_read} prompt tokens read in "
                f"{tot_read_s:.0f} s; {sum(r['reused'] for r in reqs)} reused"]
    rb = run / "routing.bin"
    if rb.exists():
        out.append(f"routing.bin: {rb.stat().st_size / 1e6:.0f} MB (kept on this PC)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-routing", action="store_true")
    ap.add_argument("--no-prefill-timing", action="store_true")
    ap.add_argument("--summarize", help="only (re)write summary.txt for this run folder")
    ap.add_argument("server_args", nargs="*")
    a = ap.parse_args()
    if a.summarize:
        run = Path(a.summarize)
        (run / "summary.txt").write_text(summarize(run), encoding="utf-8")
        print((run / "summary.txt").read_text(encoding="utf-8"))
        return 0
    run = ROOT / "runs" / time.strftime("workload-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    port = a.port or int(cfg.get("port") or 8080)
    env = cfg.setdefault("env", {})
    env.update({"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1"})
    if not a.no_prefill_timing:
        env["STRATA_PREFILL_TIMING"] = "1"
    if not a.no_routing:
        args = cfg["args"]
        if "--dump-routing" in args:
            del args[args.index("--dump-routing"):args.index("--dump-routing") + 2]
        args += ["--dump-routing", str(run / "routing.bin")]
    cfg["log"] = str(run / "engine.log")
    (run / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"[workload] logging into {run}")
    print(f"[workload] the server: http://127.0.0.1:{port}  -  use it as usual; press Ctrl+C here when you are done")
    loggers = start_loggers(run)
    cmd = [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(run / "config.json"),
           "--port", str(port)] + ([] if a.no_open else ["--open"]) + list(a.server_args)
    slog = open(run / "server.log", "w", encoding="utf-8")
    t0 = time.time()
    srv = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace", bufsize=1)

    def pump():
        for line in srv.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            slog.write(f"{time.strftime('%H:%M:%S')} " + REDACT.sub("last=<not logged>", line))
            slog.flush()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        while srv.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[workload] stopping ...")
        try:
            srv.terminate()
            srv.wait(60)
        except Exception:  # noqa: BLE001
            srv.kill()
    finally:
        stop(loggers)
        time.sleep(1)
        slog.close()
        s = summarize(run)
        (run / "summary.txt").write_text(s + f"\n\nrun length {(time.time() - t0) / 60:.0f} min\n", encoding="utf-8")
        z = run / f"{run.name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("summary.txt", "engine.log", "server.log", "gpu.csv", "system.csv", "config.json"):
                if (run / name).exists():
                    zf.write(run / name, name)
        print("\n" + s)
        print(f"\n[workload] done. Send {z} back for review (routing.bin stays here).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
