"""V100 fork: prompt reading speed with per-stage chunks (the V100 reads the prompt in bigger chunks than the 4070S).

Every variant restarts the engine with one change against the config (strata-unsloth-ud-q4_k_xl.json, long context),
then:
  1. warm-up: a ~16K-token prompt (the first prompt after a start is slower; also the short-prompt speed);
  2. a long prompt (default ~120K tokens) with a code word hidden half way, asking for it: prompt reading tok/s and
     whether the word is found;
  3. a follow-up turn in the same conversation (the long prompt + the answer + a new question): how much of the
     prompt is reused, how long the follow-up takes, and the writing speed at that length.
The engine's own lines for each variant (per-stage chunks, the prompt paths' expert streams, file reads, the
STRATA_PREFILL_TIMING breakdown) are copied from the engine log into this log.

    .venv\\Scripts\\python tools\\v100_prefillbench.py                  (what v100\\5_prompt_chunks.bat runs)
    ... --variants same8k,v100_32k --tokens 200000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import calibrate as CAL  # noqa: E402
import needle_bench as NB  # noqa: E402
from v100_bench import Log, load_tokenizer, nvsmi  # noqa: E402

TIMING = {"STRATA_PREFILL_TIMING": "1"}
# name -> (what it tests, extra args {flag: value|None}, extra env)
VARIANTS = {
    "same8k": ("before: every stage reads 8192-token chunks (the 4070 Super's limit)",
               {"--prefill": "auto"}, {"STRATA_SPLIT_STAGE_CHUNKS": "0", **TIMING}),
    "v100_16k": ("the V100 reads 16384-token chunks (two of the 4070S's gathered)", {"--prefill": "auto:16384"}, TIMING),
    "v100_32k": ("the V100 reads 32768-token chunks (four of the 4070S's gathered)", {"--prefill": "auto:32768"}, TIMING),
    "v100_32k_pagelock": ("v100_32k, and the RAM copy page-locked (DMA without the host copies; WDDM may refuse it)",
                          {"--prefill": "auto:32768"}, {"STRATA_ARENA_PIN_GIB": "0", **TIMING}),
}
DEFAULT_ORDER = ["same8k", "v100_16k", "v100_32k"]
ENGINE_LINE_KEYS = ("prompt chunk", "reads the prompt in", "every stage reads", "borrows", "prompt path CUDA",
                    "prefill timing", "expert reads from the model files", "strata serve: prompt ", "hand-off",
                    "cache complement ready", "FileExpertSource: allocating", "do not fit", "error", "ERROR", "failed")


def chat(tok, text):
    return CAL.chat_ids(tok, text)


def run_variant(name, base_cfg, tok, log, n_tokens, seed):
    what, extra_args, extra_env = VARIANTS[name]
    cfg = copy.deepcopy(base_cfg)
    args = list(cfg["args"])
    for flag, val in extra_args.items():
        args = CAL.with_arg(args, flag, val)
    cfg["args"] = args
    cfg.setdefault("env", {}).update(extra_env)
    from serve.server import StrataEngine, child_env, engine_args
    log(f"\n=== variant {name}: {what}")
    log("    env:", json.dumps(extra_env), " extra args:", json.dumps(extra_args))
    res = {"variant": name, "what": what}
    elog = Path(cfg["log"]) if cfg.get("log") else None
    if elog is not None and not elog.is_absolute():
        elog = Path(cfg.get("cwd") or ROOT) / elog
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
    log(f"    loaded in {res['load_s']} s; INFO", json.dumps(eng.info))
    nvsmi(log)

    def ask(ids, max_new, label):
        out = [t for t in eng.generate(ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
        last = dict(eng.last or {})
        pms, dms = last.get("prompt_ms") or 0.0, last.get("decode_ms") or 0.0
        fresh = (last.get("prompt_tokens") or len(ids)) - (last.get("reused") or 0)
        row = {"prompt_tokens": last.get("prompt_tokens"), "reused": last.get("reused"),
               "prompt_s": round(pms / 1000, 1), "prompt_tok_s": round(fresh / (pms / 1000), 1) if pms > 0 else None,
               "generated": len(out), "decode_tok_s": round(len(out) / (dms / 1000), 1) if dms > 0 and len(out) > 8 else None,
               "file_mb": last.get("file_mb")}
        log(f"    {label}: {row['prompt_tokens']} tokens, reused {row['reused']}: prompt {row['prompt_tok_s']} tok/s "
            f"({row['prompt_s']} s); wrote {len(out)} tokens at {row['decode_tok_s']} tok/s; file MB {row['file_mb']}")
        return out, row

    try:
        warm = NB.haystack(int(16000 * NB.CHARS_PER_TOKEN))
        _, res["warmup"] = ask(chat(tok, f"Notes {random.Random(seed + 1).randrange(10**9)}.\n" + warm +
                                    "\n\nIn one sentence, what is this text about?"), 40, "warm-up ~16K")
        rnd = random.Random(seed)
        word = f"{rnd.choice(NB.WORDS)}-{rnd.choice(NB.WORDS)}-{rnd.randint(100, 999)}"
        text = NB.haystack(int(n_tokens * NB.CHARS_PER_TOKEN))
        cut = text.rfind("\n", 0, len(text) // 2) + 1 or len(text) // 2
        body = (f"Document set {rnd.randrange(10**9)}.\n" + text[:cut] +
                f"\nThe secret code word for this text is: {word}. Remember it.\n" + text[cut:])
        q1 = body + "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only."
        ids1 = chat(tok, q1)
        out1, res["long"] = ask(ids1, 24, f"long ~{n_tokens // 1000}K")
        ans1 = tok.decode(out1)
        res["long"]["found"] = word in ans1
        log(f"    code word {word}: {'FOUND' if word in ans1 else 'MISSED'} (answer {ans1.strip()[:60]!r})")
        # the follow-up turn: what a chat client sends next (the whole conversation again)
        ids2 = ids1 + out1 + tok.encode("<|im_end|>\n<|im_start|>user\nNow summarize the text above in three "
                                        "bullet points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                        parse_special=True)
        out2, res["followup"] = ask(ids2, 160, "follow-up turn")
        log("    follow-up answer: " + tok.decode(out2).strip().replace("\n", " / ")[:400])
    except Exception as e:  # noqa: BLE001
        log("    VARIANT FAILED:", repr(e))
        res["error"] = repr(e)
    finally:
        CAL.close(eng)
        time.sleep(5)
    copy_engine_lines(elog, elog_at, log)
    return res


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
    for l in lines[-120:]:
        log("      " + l[:600])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "strata-unsloth-ud-q4_k_xl.json"))
    ap.add_argument("--log", default=str(ROOT / "v100_prefillbench.log"))
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER), help="also: " + ",".join(VARIANTS))
    ap.add_argument("--tokens", type=int, default=120000, help="the long prompt's length")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    log = Log(Path(a.log))
    log(f"==== v100_prefillbench {time.strftime('%Y-%m-%d %H:%M:%S')}  config {a.config}  long prompt ~{a.tokens}")
    log("    args:", " ".join(cfg["args"]))
    tok = load_tokenizer(cfg)
    results = []
    for i, name in enumerate(v.strip() for v in a.variants.split(",") if v.strip()):
        if name not in VARIANTS:
            log(f"unknown variant {name}")
            continue
        results.append(run_variant(name, cfg, tok, log, a.tokens, 1000 + i))
        Path(a.log).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("\n==== summary")
    for r in results:
        if r.get("error"):
            log(f"    {r['variant']:>18}: ERROR {r['error'][:200]}")
            continue
        L, F, W = r.get("long", {}), r.get("followup", {}), r.get("warmup", {})
        log(f"    {r['variant']:>18}: long prompt {L.get('prompt_tok_s')} tok/s ({L.get('prompt_s')} s), code word "
            f"{'found' if L.get('found') else 'MISSED'}; 16K {W.get('prompt_tok_s')} tok/s; follow-up reused "
            f"{F.get('reused')} in {F.get('prompt_s')} s, writing {F.get('decode_tok_s')} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
