"""V100 fork: replay your recorded decode routing against the GPU expert caches - NO GPU USED (CPU only, a few minutes).

Every workload run (tools/v100_workload.py) leaves runs\\workload-<date-time>\\routing.bin: the experts each layer
routed to, token by token, in order.  This tool replays those traces through a model of the engine's expert caches
(which experts each card holds at the start - from an expert profile - and the adaptive tier: every 4th verify window,
the most-routed missing experts of a layer swap in for its least-routed resident ones, usage decayed by 0.7, at most
--adapt-swaps of them, or a budget per card) and reports the share of expert lookups the GPUs answer.  The rest go
to the CPU, which is the largest part of a decode window here.

  1. the model against the engine: the replayed hit rate next to the one the engine logged in that run;
  2. an expert profile built from your largest run's trace against the shipped one, scored on the OTHER runs (a
     ranking that only fits the conversation it was made from is no use);
  3. the adaptive tier's swap budget: shared (--adapt-swaps 32, the config now) against a budget per card
     (--adapt-stage-swaps CUDA0,CUDA1);
  4. what more VRAM for the 4070 Super's cache would be worth.

    .venv\\Scripts\\python tools\\v100_cachesim.py                    (what v100\\16_expert_cache.bat runs first)
    ... --write-profile      also write data\\expert-profile-workload.bin (from all runs) when it scores better
"""
from __future__ import annotations

import argparse
import re
import struct
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NL, NE, K = 48, 512, 10
REC = 2 + 2 * K                       # layer, k, k ids, k weights (int32 / float32)


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def read_profile(path):
    b = Path(path).read_bytes()
    if b[:4] != b"STRP":
        raise SystemExit(f"{path}: not a Strata profile")
    _, nl, ne, _, n = struct.unpack_from("<5I", b, 4)
    if (nl, ne) != (NL, NE):
        raise SystemExit(f"{path}: {nl}x{ne}")
    return [struct.unpack_from("<HH", b, 24 + 4 * i) for i in range(n)]


def write_profile(path, ranked):
    table = np.full((NL, NE), -1, np.int32)
    for slot, (l, e) in enumerate(ranked):
        table[l, e] = slot
    with open(path, "wb") as f:
        f.write(b"STRP" + struct.pack("<5I", 1, NL, NE, len(ranked), len(ranked)))
        for l, e in ranked:
            f.write(struct.pack("<HH", l, e))
        f.write(table.astype("<i4").tobytes())


class Trace:
    def __init__(self, path):
        r = np.fromfile(path, dtype=np.int32)
        n = len(r) // REC
        r = r[: n * REC].reshape(n, REC)
        if n and not (r[:, 1] == K).all():
            raise SystemExit(f"{path}: records with k != {K}")
        self.L = r[:, 0].copy()
        self.ids = r[:, 2:2 + K].copy()
        self.w = np.r_[0, np.cumsum(self.L[1:] < self.L[:-1])] if n else np.zeros(0, np.int64)
        self.windows = int(self.w[-1]) + 1 if n else 0
        self.tokens = n // NL

    def freq(self):
        f = np.zeros((NL, NE), np.int64)
        np.add.at(f, (np.repeat(self.L, K), self.ids.ravel()), 1)
        return f


def initial(profile, cards):
    res = np.zeros((NL, NE), bool)
    for lo, hi, n in cards:
        k = 0
        for l, e in profile:
            if k >= n:
                break
            if lo <= l < hi:
                res[l, e] = True
                k += 1
    return res


def simulate(t: Trace, res0, cards, budget, every=4, decay=0.7):
    """budget: an int (shared, as --adapt-swaps) or a tuple per card (--adapt-stage-swaps); 0 = no swaps"""
    res = res0.copy()
    usage = np.zeros((NL, NE), np.float32)
    starts = np.searchsorted(t.w, np.arange(t.windows))
    ends = np.r_[starts[1:], len(t.w)]
    card_of = np.zeros(NL, np.int64)
    for c, (lo, hi, _) in enumerate(cards):
        card_of[lo:hi] = c
    hits = np.zeros(len(cards))
    looks = np.zeros(len(cards))
    nswap = 0
    for i in range(t.windows):
        a, b = starts[i], ends[i]
        ll = np.repeat(t.L[a:b], K)
        ee = t.ids[a:b].ravel()
        h = res[ll, ee]
        cl = card_of[ll]
        hits += np.bincount(cl, weights=h, minlength=len(cards))
        looks += np.bincount(cl, minlength=len(cards))
        if budget == 0 or budget == (0,) * len(cards):
            continue
        np.add.at(usage, (ll, ee), 1.0)
        if (i + 1) % every:
            continue
        cand = np.where(~res & (usage >= 2.0), usage, -np.inf)
        vict = np.where(res, usage, np.inf)
        W = 192
        ci = np.argsort(-cand, axis=1, kind="stable")[:, :W]
        cs = np.take_along_axis(cand, ci, 1)
        vi = np.argsort(vict, axis=1, kind="stable")[:, :W]
        vs = np.take_along_axis(vict, vi, 1)
        ok = np.cumprod(np.isfinite(cs) & np.isfinite(vs) & (cs >= vs + 1.5), axis=1).astype(bool)
        lay, pos = np.nonzero(ok)
        if len(lay):
            gain = (cs - vs)[lay, pos]
            order = np.argsort(-gain, kind="stable")
            if isinstance(budget, int):
                order = order[:budget]
            else:
                left = list(budget)
                keep = []
                for j in order:
                    c = card_of[lay[j]]
                    if left[c] > 0:
                        left[c] -= 1
                        keep.append(j)
                order = keep
            for j in order:
                l, p = lay[j], pos[j]
                res[l, ci[l, p]] = True
                res[l, vi[l, p]] = False
            nswap += len(order)
        usage *= decay
    tot = hits.sum() / max(1, looks.sum())
    per = [hits[c] / max(1, looks[c]) for c in range(len(cards))]
    return tot, per, nswap / max(1, t.windows // every)


def rank_by(freq, tiebreak):
    tb = np.full((NL, NE), NL * NE, np.int64)
    for r, (l, e) in enumerate(tiebreak):
        tb[l, e] = r
    order = np.lexsort((tb.ravel(), -freq.ravel()))
    return [(int(i // NE), int(i % NE)) for i in order]


def engine_facts(log: Path):
    """(cards, logged decode hit rate) from a run's engine.log"""
    if not log.exists():
        return None, None
    txt = log.read_text(encoding="utf-8", errors="replace")
    c0 = re.search(r"pre-filled (\d+) of (\d+) slots", txt)
    c1 = re.search(r"layer split: CUDA1 runs layers (\d+)-(\d+), expert cache (\d+) slots", txt)
    cards = None
    if c0 and c1:
        k = int(c1.group(1))
        cards = [(0, k, int(c0.group(2))), (k, NL, int(c1.group(3)))]
    h = l = 0
    for m in re.finditer(r"decode expert cache hit rate: [\d.]+% \((\d+) hits / (\d+) lookups\)", txt):
        h += int(m.group(1))
        l += int(m.group(2))
    return cards, (h / l if l else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--log", default=str(ROOT / "v100_cachesim.log"))
    ap.add_argument("--write-profile", action="store_true")
    a = ap.parse_args()
    log = Log(a.log)
    log(f"==== v100_cachesim {time.strftime('%Y-%m-%d %H:%M:%S')}")
    base = read_profile(a.profile)
    runs = []
    for d in sorted(Path(a.runs).glob("workload-*")):
        rb = d / "routing.bin"
        if not rb.exists() or rb.stat().st_size < REC * 4 * NL * 100:
            continue
        t = Trace(rb)
        cards, eng = engine_facts(d / "engine.log")
        runs.append((d.name, t, cards, eng))
        log(f"  {d.name}: {t.tokens} tokens in {t.windows} verify windows ({t.tokens / max(1, t.windows):.2f} per window)"
            f"; caches {cards}; engine's decode hit rate {'%.1f%%' % (100 * eng) if eng else '-'}")
    if not runs:
        log("  no runs\\workload-*\\routing.bin found")
        return 1
    cards = next((c for _, _, c, _ in runs if c), [(0, 20, 1736), (20, 48, 8334)])
    log(f"  the caches modelled: " + ", ".join(f"CUDA{i} layers {lo}-{hi - 1}: {n} slots ({100 * n / ((hi - lo) * NE):.0f}% "
                                              f"of their experts)" for i, (lo, hi, n) in enumerate(cards)))
    res_base = initial(base, cards)

    log("\n1. the model against the engine (the shipped profile, --adapt-swaps 32 shared)")
    for name, t, c, eng in runs:
        tot, per, _ = simulate(t, initial(base, c or cards), c or cards, 32)
        log(f"  {name}: replayed {100 * tot:.1f}% (CUDA0's layers {100 * per[0]:.1f}%, CUDA1's {100 * per[1]:.1f}%); "
            f"engine {'%.1f%%' % (100 * eng) if eng else '-'}")

    log("\n2. an expert profile from your own routing, scored on the runs it was NOT made from")
    big = max(runs, key=lambda r: r[1].tokens)
    trained = rank_by(big[1].freq(), base)
    res_tr = initial(trained, cards)
    gains = []
    for name, t, _, _ in runs:
        tag = "(made from this run)" if name == big[0] else "(held out)"
        for bud in (32, (96, 32)):
            b0 = simulate(t, res_base, cards, bud)[0]
            b1 = simulate(t, res_tr, cards, bud)[0]
            if name != big[0]:
                gains.append(b1 - b0)
            log(f"  {name} {tag}, swaps {bud}: shipped {100 * b0:.1f}%, from {big[0]} {100 * b1:.1f}% ({100 * (b1 - b0):+.1f})")
    held = 100 * float(np.mean(gains)) if gains else None
    log(f"  => held-out change from the trained profile: {'%+.2f points' % held if held is not None else 'no held-out run'}")

    log("\n3. the adaptive swap budget (per 4-window round)")
    for name, t, _, _ in runs:
        for bud in (32, (32, 32), (64, 32), (96, 32), (128, 32), (160, 32), (96, 16)):
            tot, per, ns = simulate(t, res_base, cards, bud)
            log(f"  {name} {str(bud):>10}: {100 * tot:.1f}% (CUDA0's layers {100 * per[0]:.1f}%, CUDA1's "
                f"{100 * per[1]:.1f}%), {ns:.0f} swaps per round")

    log("\n4. more slots for CUDA0 (each ~2.9 MB), --adapt-stage-swaps 96,32")
    t = big[1]
    for extra in (0, 350, 700, 1050):
        c2 = [(cards[0][0], cards[0][1], cards[0][2] + extra), cards[1]]
        tot, per, _ = simulate(t, initial(base, c2), c2, (96, 32))
        log(f"  {big[0]}: CUDA0 {c2[0][2]} slots (+{extra * 2.92 / 1024:.1f} GiB): {100 * tot:.1f}% "
            f"(CUDA0's layers {100 * per[0]:.1f}%)")

    if a.write_profile:
        if held is not None and held >= 0.5:
            f = sum((r[1].freq() for r in runs), np.zeros((NL, NE), np.int64))
            out = ROOT / "data" / "expert-profile-workload.bin"
            write_profile(out, rank_by(f, base))
            log(f"\n  wrote {out} (all runs' routing; point --expert-profile at it to use it)")
        else:
            log("\n  no profile written: the trained one does not score better on runs it was not made from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
