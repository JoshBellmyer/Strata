"""V100 fork: long-context benchmark - starts the Strata server on the config, then measures through its API.

1. Speed: one long prompt per length (the repository's own text, with a unique first line so no prompt cache can
   help), "summarize it", 256 greedy tokens, thinking off.  Logged per length: prompt tokens, prompt reading tok/s,
   writing tok/s at that context, wall time.
2. Recall: needle-in-a-haystack at the longest length - a code word hidden at several depths, asked for afterwards
   (tools/needle_bench.py's text and words).  The depths run deepest first, so later ones reuse the shared prefix.
Everything goes to one log (default v100_longbench.log) and a JSON next to it; the engine's own log is the config's.

    .venv\\Scripts\\python tools\\v100_longbench.py                                 (what v100\\4_long_context.bat runs)
    ... --lengths 32k,128k,240k --needle-length 240k --depths 90,50,10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import needle_bench as NB  # noqa: E402  (haystack, WORDS, CHARS_PER_TOKEN)


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def tokens_of(s: str) -> int:
    s = s.lower()
    return int(float(s.rstrip("k")) * 1000) if s.endswith("k") else int(s)


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def chat(url, prompt, max_tokens, timeout):
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def smi(log):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,pcie.link.gen.current,"
                            "pcie.link.width.current", "--format=csv"], capture_output=True, text=True, timeout=30)
        for line in r.stdout.strip().splitlines():
            log("    " + line)
    except Exception as e:  # noqa: BLE001
        log("    nvidia-smi failed:", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "strata-unsloth-ud-q4_k_xl.json"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--log", default=str(ROOT / "v100_longbench.log"))
    ap.add_argument("--lengths", default="32k,128k,240k", help="speed runs (k = 1000 tokens)")
    ap.add_argument("--needle-length", default="240k")
    ap.add_argument("--depths", default="90,50,10")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    log = Log(Path(a.log))
    url = f"http://127.0.0.1:{a.port}"
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    ctx = int(cfg["args"][cfg["args"].index("--max-context") + 1]) if "--max-context" in cfg["args"] else 0
    log(f"==== v100_longbench {time.strftime('%Y-%m-%d %H:%M:%S')}  config {a.config}  context {ctx}")
    log("    args:", " ".join(cfg["args"]))
    log("    env:", json.dumps(cfg.get("env", {})), " gpu:", cfg.get("gpu"))
    smi(log)
    server_log = open(Path(a.log).with_suffix(".server.txt"), "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", a.config,
                             "--port", str(a.port)], cwd=str(ROOT), stdout=server_log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    results = {"speed": [], "needles": []}
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                log(f"    the server exited (code {proc.returncode}) - see {server_log.name} and the engine log")
                return 1
            try:
                h = get(url + "/health")
                if h.get("loaded") and h.get("max_context"):
                    break
            except (OSError, ValueError):
                pass
            if time.time() - t0 > 900:
                log("    the server did not get ready in 15 minutes")
                return 1
            time.sleep(5)
        log(f"    server ready in {time.time() - t0:.0f} s, max_context {h.get('max_context')}")
        try:
            m = get(url + "/metrics")
            log("    engine:", json.dumps(m.get("engine", {}))[:1500])
        except (OSError, ValueError):
            pass
        smi(log)
        max_ctx = int(h["max_context"])
        # 1. speed
        for i, L in enumerate(x.strip() for x in a.lengths.split(",") if x.strip()):
            n = tokens_of(L)
            if n + 600 > max_ctx:
                log(f"\n--- {L}: skipped (context {max_ctx})")
                continue
            text = NB.haystack(int(n * NB.CHARS_PER_TOKEN))
            prompt = (f"Document set {random.randrange(10**9)} ({L}).\n" + text +
                      "\n\nSummarize the text above in five bullet points, one line each.")
            log(f"\n--- speed {L}: sending ~{n:,} tokens ...")
            try:
                out, secs = chat(url, prompt, 256, a.timeout)
            except (OSError, urllib.error.HTTPError) as e:
                log(f"    request failed: {e}")
                results["speed"].append({"length": L, "error": str(e)})
                continue
            t = out.get("timings") or {}
            u = out.get("usage") or {}
            row = {"length": L, "prompt_tokens": u.get("prompt_tokens"), "cache_n": t.get("cache_n"),
                   "prompt_tok_s": t.get("prompt_per_second"), "prompt_s": round((t.get("prompt_ms") or 0) / 1000, 1),
                   "decode_tok_s": t.get("predicted_per_second"), "generated": t.get("predicted_n"),
                   "wall_s": round(secs, 1)}
            results["speed"].append(row)
            log(f"    prompt {row['prompt_tokens']} tokens (cached {row['cache_n']}): {row['prompt_tok_s']} tok/s, "
                f"{row['prompt_s']} s;  writing {row['decode_tok_s']} tok/s ({row['generated']} tokens);  "
                f"wall {row['wall_s']} s")
            answer = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            log("    answer: " + answer.strip().replace("\n", " / ")[:600])
        # 2. needles
        n = tokens_of(a.needle_length)
        if n + 300 > max_ctx:
            log(f"\n--- needles {a.needle_length}: skipped (context {max_ctx})")
        else:
            text = NB.haystack(int(n * NB.CHARS_PER_TOKEN))
            rnd = random.Random(11)
            for d in (int(x) for x in a.depths.split(",")):
                word = f"{rnd.choice(NB.WORDS)}-{rnd.choice(NB.WORDS)}-{rnd.randint(100, 999)}"
                cut = int(len(text) * d / 100)
                cut = text.rfind("\n", 0, cut) + 1 or cut
                needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
                prompt = (text[:cut] + needle + text[cut:] +
                          "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
                log(f"\n--- needle {a.needle_length} at depth {d}% ...")
                try:
                    out, secs = chat(url, prompt, 40, a.timeout)
                except (OSError, urllib.error.HTTPError) as e:
                    log(f"    request failed: {e}")
                    results["needles"].append({"depth": d, "error": str(e)})
                    continue
                answer = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                t = out.get("timings") or {}
                ok = word in answer
                results["needles"].append({"depth": d, "found": ok, "word": word, "answer": answer.strip()[:200],
                                           "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
                                           "cache_n": t.get("cache_n"), "prompt_tok_s": t.get("prompt_per_second"),
                                           "wall_s": round(secs, 1)})
                log(f"    {'FOUND' if ok else 'MISSED'} ({word}); answer {answer.strip()[:80]!r}; "
                    f"{(out.get('usage') or {}).get('prompt_tokens')} tokens, cached {t.get('cache_n')}, "
                    f"{t.get('prompt_per_second')} tok/s, wall {secs:.0f} s")
        smi(log)
    finally:
        try:
            proc.terminate()
            proc.wait(60)
        except Exception:  # noqa: BLE001
            proc.kill()
    log("\n==== summary")
    for r in results["speed"]:
        log(f"    {r['length']:>6}: prompt {r.get('prompt_tok_s')} tok/s ({r.get('prompt_s')} s), "
            f"writing {r.get('decode_tok_s')} tok/s" + (f"  ERROR {r['error']}" if r.get("error") else ""))
    for r in results["needles"]:
        log(f"    needle {a.needle_length} depth {r['depth']}%: "
            + ("FOUND" if r.get("found") else "MISSED" if "found" in r else "ERROR " + r.get("error", "")))
    Path(a.log).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
