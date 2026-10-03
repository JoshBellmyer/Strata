"""V100 fork: where the decode (writing) time goes on the 4070 Super + V100 split, and what changes it.

Every variant restarts the engine with one change against the config (strata-unsloth-ud-q4_k_xl.json, 256K context)
and measures writing speed two ways:
  - short chats: three prompts, 256 greedy tokens each, twice (the second run of a prompt has an adapted cache);
  - long context: a ~60K-token document, then a 300-token answer about it (the KV of a long conversation is read
    every step, most of it from VRAM, some from RAM).
Every variant runs with STRATA_DECODE_TIMING and STRATA_SPLIT_TIMING (host clocks, no GPU syncs): per request the
engine logs ms per verify window split into waiting for the GPUs, the CPU experts, the draft, and per layer-window how
many experts the CPU computed, VRAM hits and PCIe sends.  Those lines are copied from the engine log into this log.

    .venv\\Scripts\\python tools\\v100_decodebench.py                    (what v100\\11_decode_ple.bat runs)
    ... --variants baseline,v100_only --long-tokens 100000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import calibrate as CAL  # noqa: E402
import needle_bench as NB  # noqa: E402
from v100_bench import Log, load_tokenizer, nvsmi, one_gpu  # noqa: E402

TIMERS = {"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1"}
SHORT_PROMPTS = CAL.PROMPTS
SHORT_NEW = 256
LONG_NEW = 300

# name -> (what it tests, extra args {flag: value|None}, extra env, cfg edit)
VARIANTS = {
    "baseline": ("the config as it is (4070S layers 0-16, V100 the rest)", {}, {}, None),
    "profile": ("baseline + the GPU stage stamps of every window (STRATA_VERIFY_PROFILE)", {},
                {"STRATA_VERIFY_PROFILE": "1"}, None),
    "v100_only": ("the V100 alone, same config (the comparison you asked about)", {}, {}, one_gpu(1)),
    "kq256": ("multi-token AVX2 kernels for the CPU's Q4_K / Q5_1 / Q8_0 experts", {}, {"STRATA_KQ256": "1"}, None),
    "pcie0": ("CUDA0's misses all on the CPU (--pcie-frac 0)", {"--pcie-frac": "0"}, {}, None),
    "pcie80": ("more of CUDA0's misses sent to the 4070S over PCIe (--pcie-frac 0.8)", {"--pcie-frac": "0.8"}, {}, None),
    "minp03": ("longer draft windows (--spec-min-p 0.3)", {"--spec-min-p": "0.3"}, {}, None),
    "minp07": ("shorter draft windows (--spec-min-p 0.7)", {"--spec-min-p": "0.7"}, {}, None),
    "miss300": ("layer split placement with a higher CPU-miss cost", {}, {"STRATA_SPLIT_MISS_MS": "300"}, None),
    # round 2 (2026-10-03): the adaptive tier's swaps on a worker thread (the default now) against 0.1.32's
    "sync": ("the adaptive swaps as before (STRATA_ADAPT_SYNC=1: the decode loop waits for them)", {},
             {"STRATA_ADAPT_SYNC": "1"}, None),
    "adapt_off": ("no adaptive swaps at all (--adapt-every 0): the profile's experts stay put", {"--adapt-every": "0"},
                  {}, None),
    "swaps32": ("at most 32 swaps per adaptive round (--adapt-swaps 32)", {"--adapt-swaps": "32"}, {}, None),
    "every2": ("an adaptive round every 2 windows instead of 4 (--adapt-every 2)", {"--adapt-every": "2"}, {}, None),
}
# round 3 (2026-10-03): the config now has --adapt-swaps 32; combinations, other split points, the page-locked RAM copy
# (lets CUDA0 take misses over PCIe), a bigger PLE row cache.  "baseline" runs first and last (the drift between them
# is the noise floor).
def split_at(k):
    def f(cfg):
        cfg["layer_split"] = str(k)
    return f


VARIANTS.update({
    "baseline_end": ("the config again, last (how much the numbers drift over the run)", {}, {}, None),
    "every2_minp07": ("--adapt-every 2 and --spec-min-p 0.7", {"--adapt-every": "2", "--spec-min-p": "0.7"}, {}, None),
    "k12": ("layers 0-11 on the 4070S, 12-47 on the V100", {}, {}, split_at(12)),
    "k14": ("layers 0-13 on the 4070S, 14-47 on the V100", {}, {}, split_at(14)),
    "k20": ("layers 0-19 on the 4070S, 20-47 on the V100", {}, {}, split_at(20)),
    "pagelock": ("the RAM copy page-locked (STRATA_ARENA_PIN_GIB=0): CUDA0 may take misses over PCIe", {},
                 {"STRATA_ARENA_PIN_GIB": "0"}, None),
    "plecache": ("a PLE row cache of 8M rows (~750 MB) instead of 1M", {"--ple-row-cache": "8388608"}, {}, None),
})
# round 4 (2026-10-03): the split point (k20 was best after 60K in round 3) and --spec-min-p 0.7, with two long-context
# samples per run (the second is a follow-up turn on the same document)
VARIANTS.update({
    "k20_minp07": ("layers 0-19 on the 4070S, and --spec-min-p 0.7", {"--spec-min-p": "0.7"}, {}, split_at(20)),
    "k22": ("layers 0-21 on the 4070S, 22-47 on the V100", {}, {}, split_at(22)),
    "k24": ("layers 0-23 on the 4070S, 24-47 on the V100", {}, {}, split_at(24)),
    "pagelock_p20": ("the RAM copy page-locked, and only 20% of CUDA0's misses over PCIe (--pcie-frac 0.2)",
                     {"--pcie-frac": "0.2"}, {"STRATA_ARENA_PIN_GIB": "0"}, None),
})
# round 5 (2026-10-03): the PLE row reads (4-6 ms per window inside the engine; the drive answers a 48-read burst in
# under 2 ms outside it).  The config is now K=20 + --spec-min-p 0.7.
VARIANTS.update({
    "io48": ("48 threads issue the PLE reads (STRATA_IO_THREADS=48; default 4)", {}, {"STRATA_IO_THREADS": "48"}, None),
    "io16": ("16 threads issue the PLE reads (STRATA_IO_THREADS=16)", {}, {"STRATA_IO_THREADS": "16"}, None),
    "plesync": ("the PLE reads submitted on the decode thread, no I/O worker (--ple-sync-submit)", {}, {},
                lambda cfg: cfg["args"].append("--ple-sync-submit")),
    "keepalive5": ("the SSD kept awake with a read after 5 ms without one (STRATA_SSD_KEEPALIVE=5; default 100)", {},
                   {"STRATA_SSD_KEEPALIVE": "5"}, None),
})
# Applied to the running engine once it has loaded (Windows): whether Windows slows the engine's sleeping threads
# (the PLE reader wakes ~5 threads per decode window).  The disk test read the same file 3-10x faster from a console
# window than the engine does.
def _proc(pid, access):
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(access, False, pid)
    if not h:
        raise OSError(ctypes.get_last_error(), "OpenProcess")
    return k, h


def no_throttle(pid):
    """Power throttling (EcoQoS) off for the engine: SetProcessInformation(ProcessPowerThrottling), the execution
    speed and timer-resolution controls set, their states cleared."""
    import ctypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    k, h = _proc(pid, 0x0200)   # PROCESS_SET_INFORMATION
    st = PPTS(1, 0x1 | 0x4, 0)
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    ok = k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
    err = ctypes.get_last_error()
    k.CloseHandle(ctypes.c_void_p(h))
    return f"power throttling off for pid {pid}: {'ok' if ok else f'FAILED ({err})'}"


def high_priority(pid):
    import ctypes
    k, h = _proc(pid, 0x0200)
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    ok = k.SetPriorityClass(h, 0x80)   # HIGH_PRIORITY_CLASS
    err = ctypes.get_last_error()
    k.CloseHandle(ctypes.c_void_p(h))
    return f"high priority class for pid {pid}: {'ok' if ok else f'FAILED ({err})'}"


POST_START = {"nothrottle": no_throttle, "highprio": high_priority}
VARIANTS.update({
    "nothrottle": ("Windows power throttling (EcoQoS) turned off for the engine process", {}, {}, None),
    "highprio": ("the engine process at high priority", {}, {}, None),
})
DEFAULT_ORDER = ["baseline", "io48", "io16", "plesync", "keepalive5", "baseline_end"]   # step 11; step 12 names its own
# round 7 (2026-10-03): the PLE table from its own file (config --ple-gguf, tools/v100_ple_split.py) against the
# table read from Unsloth's shard 2, which the engine also maps
VARIANTS.update({
    "ple_shard": ("the PLE table read from the model's shard 2 again (no --ple-gguf: the mapped file)",
                  {"--ple-gguf": None}, {}, None),
    "ple_shard_end": ("the PLE table from shard 2 again, last", {"--ple-gguf": None}, {}, None),
})
# round 8 (2026-10-03): the adaptive tier's swap budget per card (--adapt-stage-swaps CUDA0,CUDA1).  Replaying the
# workload runs' routing (tools/v100_cachesim.py): the 4070 Super's layers answer ~55% of their lookups on the GPU,
# the V100's ~94%, and the shared budget of 32 gives the 4070 Super only ~17 swaps a round; a budget per card
# replayed 77.0 -> 80.0% (96,32).  The 4070 Super's swaps cross its x16 link; the V100 keeps its 32.
VARIANTS.update({
    "stage64_32": ("swap budget per card: 64 for the 4070S, 32 for the V100 (--adapt-stage-swaps 64,32)",
                   {"--adapt-stage-swaps": "64,32"}, {}, None),
    "stage96_32": ("swap budget per card: 96 for the 4070S, 32 for the V100", {"--adapt-stage-swaps": "96,32"}, {}, None),
    "stage160_32": ("swap budget per card: 160 for the 4070S, 32 for the V100", {"--adapt-stage-swaps": "160,32"}, {},
                    None),
    "stage96_16": ("swap budget per card: 96 for the 4070S, 16 for the V100", {"--adapt-stage-swaps": "96,16"}, {}, None),
})
ENGINE_LINE_KEYS = ("ple io", "page-locked", "locked resident", "decode timing", "decode GPU stages", "strata serve: stage ", "hit rate", "layer split auto",
                    "expert cache ", "layer split: CUDA", "strata serve: prompt ", "pcie_frac", "PCIe probe",
                    "resident RAM mode", "adaptive", "ERROR", "error", "failed")


def chat(tok, text):
    return CAL.chat_ids(tok, text)


def copy_engine_lines(elog, at, log):
    if elog is None or not elog.exists():
        return
    try:
        with open(elog, "r", encoding="utf-8", errors="replace") as f:
            f.seek(at)
            lines = [l.rstrip() for l in f if any(k in l for k in ENGINE_LINE_KEYS)]
    except OSError as e:
        log("    (engine log not readable:", e, ")")
        return
    log("    engine log:")
    for l in lines[-150:]:
        log("      " + l[:900])


def run_variant(name, base_cfg, tok, log, long_tokens):
    what, extra_args, extra_env, edit = VARIANTS[name]
    cfg = copy.deepcopy(base_cfg)
    if edit:
        edit(cfg)
    args = list(cfg["args"])
    for flag, val in extra_args.items():
        args = CAL.with_arg(args, flag, val)
    cfg["args"] = args
    cfg.setdefault("env", {}).update({**TIMERS, **extra_env})
    from serve.server import StrataEngine, child_env, engine_args
    log(f"\n=== variant {name}: {what}")
    log("    gpu:", cfg.get("gpu"), " env:", json.dumps(extra_env), " extra args:", json.dumps(extra_args))
    res = {"variant": name, "what": what}
    elog = Path(cfg["log"]) if cfg.get("log") else None
    elog_at = elog.stat().st_size if elog is not None and elog.exists() else 0
    t0 = time.time()
    try:
        eng = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    except Exception as e:  # noqa: BLE001
        log("    ENGINE FAILED TO START:", e)
        res["error"] = str(e)
        copy_engine_lines(elog, elog_at, log)
        return res
    res["load_s"] = round(time.time() - t0, 1)
    if name in POST_START:
        try:
            log("    " + POST_START[name](eng.proc.pid))
        except Exception as e:  # noqa: BLE001
            log("    (could not apply:", repr(e), ")")
    log(f"    loaded in {res['load_s']} s; INFO", json.dumps(eng.info))
    nvsmi(log)

    def ask(ids, max_new, label):
        out = [t for t in eng.generate(ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
        last = dict(eng.last or {})
        dms = last.get("decode_ms") or 0.0
        rate = len(out) / (dms / 1000) if dms > 0 and len(out) > 8 else 0.0
        hit = 100.0 * last["hits"] / last["lookups"] if last.get("lookups") else None
        acc, off = last.get("drafts_accepted"), last.get("drafts_offered")
        log(f"    {label}: {rate:6.1f} tok/s ({len(out)} tokens, {dms:.0f} ms); drafts {acc}/{off}"
            + (f"; GPU hits {hit:.1f}%" if hit is not None else "")
            + f"; prompt {last.get('prompt_tokens')} tokens in {(last.get('prompt_ms') or 0) / 1000:.1f} s")
        return out, {"tok_s": round(rate, 2), "n": len(out), "hit": hit, "acc": acc, "off": off}

    try:
        ask(chat(tok, SHORT_PROMPTS[0]), 32, "warm-up")
        short = []
        for rep in range(2):
            for i, p in enumerate(SHORT_PROMPTS):
                _, r = ask(chat(tok, p), SHORT_NEW, f"short {i + 1} run {rep + 1}")
                short.append(r["tok_s"])
        res["short_tok_s"] = round(statistics.median(short), 2)
        res["short_all"] = short
        text = NB.haystack(int(long_tokens * NB.CHARS_PER_TOKEN))
        q = (f"Report {random.Random(7).randrange(10**9)}.\n" + text +
             "\n\nExplain in detail what this software does and how its parts fit together.")
        ids1 = chat(tok, q)
        out1, r = ask(ids1, LONG_NEW, f"long ~{long_tokens // 1000}K")
        # a second sample at the same length: a follow-up turn (the document is reused from the prompt cache)
        ids2 = ids1 + out1 + tok.encode("<|im_end|>\n<|im_start|>user\nNow list the main limitations the text "
                                        "mentions, with a short explanation of each.<|im_end|>\n<|im_start|>assistant\n"
                                        "<think>\n\n</think>\n\n", parse_special=True)
        _, r2 = ask(ids2, LONG_NEW, f"long ~{long_tokens // 1000}K follow-up")
        res["long_tok_s"] = round((r["tok_s"] + r2["tok_s"]) / 2, 2)
        res["long"] = [r, r2]
        log(f"    SHORT median {res['short_tok_s']} tok/s; LONG {res['long_tok_s']} tok/s")
    except Exception as e:  # noqa: BLE001
        log("    VARIANT FAILED:", repr(e))
        res["error"] = repr(e)
    finally:
        CAL.close(eng)
        time.sleep(5)
    copy_engine_lines(elog, elog_at, log)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "strata-unsloth-ud-q4_k_xl.json"))
    ap.add_argument("--log", default=str(ROOT / "v100_decodebench.log"))
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER), help="also: " + ",".join(VARIANTS))
    ap.add_argument("--long-tokens", type=int, default=60000)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    log = Log(Path(a.log))
    log(f"==== v100_decodebench {time.strftime('%Y-%m-%d %H:%M:%S')}  config {a.config}")
    log("    args:", " ".join(cfg["args"]))
    tok = load_tokenizer(cfg)
    results = []
    for name in (v.strip() for v in a.variants.split(",") if v.strip()):
        if name not in VARIANTS:
            log(f"unknown variant {name}")
            continue
        results.append(run_variant(name, cfg, tok, log, a.long_tokens))
        Path(a.log).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("\n==== summary (writing speed, tok/s)")
    for r in results:
        if r.get("error"):
            log(f"    {r['variant']:>10}: ERROR {r['error'][:200]}")
        else:
            log(f"    {r['variant']:>10}: short chats {r.get('short_tok_s')}  (all {r.get('short_all')}), "
                f"after ~{a.long_tokens // 1000}K tokens {r.get('long_tok_s')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
