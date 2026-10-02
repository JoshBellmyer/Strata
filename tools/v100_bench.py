"""V100 fork: measure UD-Q4_K_XL on the RTX 4070 Super + Tesla V100 + 96 GB PC, one engine setting per run.

Every variant restarts the engine with one change against the config setup wrote (strata-unsloth-ud-q4_k_xl.json),
then measures:
  - decode: greedy tok/s on three short chat prompts (twice each, median), with the expert tiers per request
    (GPU hits, RAM blobs, file blobs/MB: file reads should be 0 once the RAM copy holds every non-GPU expert);
  - prompt: tok/s reading a ~4K and a ~16K token prompt (text from this repository's docs, a fresh slice each
    time so the prompt cache cannot help);
  - a quality check: the greedy answer to a fixed coding prompt, written out in full so it can be read, with the
    position where its tokens first differ from the first variant's (a small late difference is a near-tie flip;
    garbage or an early split would mean a broken kernel).
Everything goes to one log file (and a JSON next to it).  The engine's own log is the config's "log" file.

    .venv\\Scripts\\python tools\\v100_bench.py strata-unsloth-ud-q4_k_xl.json --log v100_bench.log
    ... --variants baseline,kq256       (a subset; --list shows them)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import calibrate as CAL  # noqa: E402  (chat_ids, with_arg, close)

DECODE_PROMPTS = CAL.PROMPTS
QUALITY_PROMPT = ("Write a Python class LRUCache with get and put in O(1), using an OrderedDict, "
                  "with a short docstring and three asserts that test it.")
DECODE_NEW = 192
QUALITY_NEW = 320


def swap_gpus(cfg):
    if isinstance(cfg.get("gpu"), list) and len(cfg["gpu"]) == 2:
        cfg["gpu"] = list(reversed(cfg["gpu"]))


def one_gpu(which):
    def f(cfg):
        if isinstance(cfg.get("gpu"), list):
            cfg["gpu"] = cfg["gpu"][which]          # an int: the server runs it on that card alone
            cfg.pop("layer_split", None)
    return f


# name -> (what it tests, extra args {flag: value|None to remove}, extra env, cfg edit)
VARIANTS = {
    "baseline": ("the config as setup wrote it (layer split auto, 4070S first)", {}, {}, None),
    "kq256": ("multi-token AVX2 kernels for the Q4_K / Q5_1 / Q8_0 CPU experts", {}, {"STRATA_KQ256": "1"}, None),
    "pcores": ("CPU pool on the P-cores and their SMT siblings only (11 workers)",
               {"--pool-affinity": "auto", "--pool-workers": "11"}, {}, None),
    "pcores6": ("CPU pool on the P-cores only, one per core (5 workers + the host thread)",
                {"--pool-affinity": "p-cores"}, {}, None),
    "own_buffers": ("every GPU keeps its own prompt buffers (no loan from the expert caches)", {},
                    {"STRATA_SPLIT_OWN": "1"}, None),
    "no_mmq_kq": ("prompt experts by FP16 dequantize + cuBLAS instead of the MMQ kernels", {},
                  {"STRATA_MMQ_KQUANTS": "0"}, None),
    "miss300": ("layer split placement with a higher CPU-miss cost (Q4 experts on a 6P+8E AVX2 CPU)", {},
                {"STRATA_SPLIT_MISS_MS": "300"}, None),
    "v100_first": ("the V100 as the first GPU (the 4070S then holds the head and the draft layer)", {}, {},
                   swap_gpus),
    "v100_only": ("the V100 alone (what upstream Strata would do: one GPU)", {}, {}, one_gpu(1)),
}
DEFAULT_ORDER = ["baseline", "kq256", "pcores", "own_buffers", "no_mmq_kq", "miss300", "v100_first", "v100_only"]


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def load_tokenizer(cfg):
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((tpath / "token_type.json").read_text()))


def corpus_ids(tok, n_tokens: int, offset: int) -> list[int]:
    """~n_tokens of this repository's own text (docs, then sources), starting `offset` characters in."""
    text = ""
    for p in sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "src").rglob("*.cpp")):
        text += p.read_text(encoding="utf-8", errors="replace") + "\n"
        if len(text) > offset + n_tokens * 6:
            break
    body = text[offset:offset + n_tokens * 5]
    ids = tok.encode("<|im_start|>user\nSummarize the following in three sentences.\n\n", parse_special=True)
    ids += tok.encode(body)
    ids = ids[:n_tokens]
    ids += tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    return ids


def gen(eng, ids, max_new):
    out = [t for t in eng.generate(ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
    return out, dict(eng.last or {})


def nvsmi(log):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_model.current,memory.used,"
                            "memory.total,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm",
                            "--format=csv"], capture_output=True, text=True, timeout=30)
        for line in r.stdout.strip().splitlines():
            log("   ", line)
    except Exception as e:  # noqa: BLE001
        log("    nvidia-smi failed:", e)


def run_variant(name, base_cfg, tok, log, first_quality):
    what, extra_args, extra_env, edit = VARIANTS[name]
    cfg = copy.deepcopy(base_cfg)
    if edit:
        edit(cfg)
    args = list(cfg["args"])
    for flag, val in extra_args.items():
        args = CAL.with_arg(args, flag, val)
    cfg["args"] = args
    cfg.setdefault("env", {}).update(extra_env)
    from serve.server import StrataEngine, child_env, engine_args
    log(f"\n=== variant {name}: {what}")
    log("    gpu:", cfg.get("gpu"), " env:", json.dumps(extra_env), " extra args:", json.dumps(extra_args))
    res = {"variant": name, "what": what}
    t0 = time.time()
    try:
        eng = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    except Exception as e:  # noqa: BLE001
        log("    ENGINE FAILED TO START:", e)
        res["error"] = str(e)
        return res
    res["load_s"] = round(time.time() - t0, 1)
    res["info"] = dict(eng.info)
    log(f"    loaded in {res['load_s']} s; INFO", json.dumps(eng.info))
    nvsmi(log)
    try:
        gen(eng, CAL.chat_ids(tok, DECODE_PROMPTS[0]), 32)          # warm-up
        rates, tiers = [], []
        for rep in range(2):
            for p in DECODE_PROMPTS:
                out, last = gen(eng, CAL.chat_ids(tok, p), DECODE_NEW)
                ms = last.get("decode_ms") or 0
                r = len(out) / (ms / 1000) if ms > 0 and len(out) > 8 else 0.0
                rates.append(r)
                tiers.append({k: last.get(k) for k in ("hits", "lookups", "ram_blobs", "file_blobs", "file_mb",
                                                        "drafts_accepted", "drafts_offered")})
                acc = last.get("drafts_accepted"), last.get("drafts_offered")
                hit = (100.0 * last["hits"] / last["lookups"]) if last.get("lookups") else None
                log(f"    decode rep{rep} {r:6.1f} tok/s  ({len(out)} tok, {ms:.0f} ms)  drafts {acc[0]}/{acc[1]}"
                    + (f"  GPU hit {hit:.1f}%" if hit is not None else "")
                    + f"  RAM blobs {last.get('ram_blobs')}  file blobs {last.get('file_blobs')}"
                      f" ({last.get('file_mb')} MB)")
        res["decode_tok_s"] = round(statistics.median(rates), 2)
        res["decode_all"] = [round(x, 2) for x in rates]
        res["tiers"] = tiers
        log(f"    DECODE median {res['decode_tok_s']} tok/s")
        for n, off in ((4096, 1000 + 7919 * len(name)), (16384, 200000 + 7919 * len(name))):
            ids = corpus_ids(tok, n, off)
            out, last = gen(eng, ids, 8)
            pms = last.get("prompt_ms") or 0
            pt = last.get("prompt_tokens") or len(ids)
            rate = pt / (pms / 1000) if pms > 0 else 0.0
            res[f"prompt_{n}_tok_s"] = round(rate, 1)
            log(f"    PROMPT {pt} tokens: {rate:7.1f} tok/s ({pms / 1000:.1f} s); reused {last.get('reused')}")
        out, last = gen(eng, CAL.chat_ids(tok, QUALITY_PROMPT), QUALITY_NEW)
        res["quality_ids_sha"] = hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16]
        res["quality_ids"] = out
        if first_quality is not None:
            same = next((i for i, (a, b) in enumerate(zip(first_quality, out)) if a != b), min(len(out), len(first_quality)))
            res["quality_first_diff"] = same
            log(f"    QUALITY: first token that differs from the first variant's answer: {same} of {len(out)}")
        log("    QUALITY answer:\n" + "\n".join("      | " + l for l in tok.decode(out).splitlines()))
    except Exception as e:  # noqa: BLE001
        log("    VARIANT FAILED:", repr(e))
        res["error"] = repr(e)
    finally:
        CAL.close(eng)
        time.sleep(5)                                   # the driver gives the memory back
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--log", default="v100_bench.log")
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in VARIANTS.items():
            print(f"{k:12s} {v[0]}")
        return 0
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    log = Log(Path(a.log))
    log(f"==== v100_bench {time.strftime('%Y-%m-%d %H:%M:%S')}  config {a.config}")
    log("    host:", platform.platform(), platform.processor(), "python", platform.python_version())
    log("    config:", json.dumps(cfg))
    nvsmi(log)
    tok = load_tokenizer(cfg)
    results, first_q = [], None
    for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            log("    unknown variant", name)
            continue
        r = run_variant(name, cfg, tok, log, first_q)
        if first_q is None and r.get("quality_ids"):
            first_q = r["quality_ids"]
        results.append(r)
    log("\n==== summary (decode median tok/s | prompt 4K | prompt 16K | quality diff | load s)")
    for r in results:
        log(f"    {r['variant']:12s} {r.get('decode_tok_s', '-'):>8} | {r.get('prompt_4096_tok_s', '-'):>7} | "
            f"{r.get('prompt_16384_tok_s', '-'):>7} | {r.get('quality_first_diff', '-'):>5} | {r.get('load_s', '-')}"
            + (f"  ERROR {r['error']}" if r.get("error") else ""))
    Path(a.log).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
