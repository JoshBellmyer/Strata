"""V100 fork: the GPU tests, for when the GPUs are free (stop any other GPU program first).

1. The kernel parity programs (synthetic data, no model) on each card alone - the V100 first: GPU kernels against
   their CPU / FP64 references, plus the prompt path's BF16 GEMM through cuBLAS and through the Volta fallback.
2. tools/v100_bench.py on the config setup wrote: decode / prompt speed and a quality check per setting.
Everything goes to one log (default v100_gputest.log); the bench adds v100_bench.log / .json.

    .venv\\Scripts\\python tools\\v100_gputest.py                     (what v100\\2_gpu_tests.bat runs)
    ... --parity-only   |   --bench-only --variants baseline,kq256
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""

# (program, arguments) - every one synthetic; exit code 0 = pass
PARITY = [
    ("strata-device", []),
    ("native_expert_parity", ["--synthetic", "q4_K/q5_1"]),
    ("native_expert_parity", ["--synthetic", "q4_K/q8_0"]),
    ("native_expert_parity", ["--synthetic", "q5_K/q8_0"]),
    ("native_expert_parity", ["--synthetic", "q8_0/q8_0"]),
    ("native_expert_parity", ["--q5_1-min"]),
    ("mmvq_multi_parity", []),
    ("gr_parity", ["--selftest"]),
    ("gdn_parity", ["--selftest"]),
    ("qsa_parity", ["--selftest"]),
    ("kv_q8_parity", []),
    ("elementwise_parity", ["--selftest"]),
    ("sampler_parity", ["--selftest"]),
    ("router_top10_parity", ["--selftest"]),
    ("rope_parity", ["--selftest"]),
    ("quantize_act_parity", ["--selftest"]),
    ("shared_expert_parity", ["--selftest"]),
    ("bf16_gemv_parity", ["--selftest"]),
    ("s_gemv_q8k_parity", ["--selftest"]),
]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", errors="replace")

    def __call__(self, line=""):
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def smi(log):
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_model.current,memory.used,"
                        "memory.total,utilization.gpu,pcie.link.gen.max,pcie.link.width.max,"
                        "pcie.link.gen.current,pcie.link.width.current", "--format=csv"],
                       capture_output=True, text=True)
    for line in r.stdout.strip().splitlines():
        log("  " + line)
    return r.stdout


def find_cards():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,compute_cap,name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    cards = []
    for line in r.stdout.strip().splitlines():
        i, cc, name = [x.strip() for x in line.split(",", 2)]
        cards.append((int(i), cc, name))
    # the V100 first: it is the card this fork adds
    return sorted(cards, key=lambda c: (c[1] != "7.0", c[0]))


def parity(log, build: Path) -> list:
    results = []
    for idx, cc, name in find_cards():
        log(f"\n==== parity tests on GPU {idx} ({name}, compute {cc}) alone")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(idx))
        for prog, args in PARITY:
            exe = build / (prog + EXE)
            label = f"{prog} {' '.join(args)}".strip()
            if not exe.exists():
                log(f"  SKIP {label}: {exe} not built")
                results.append((idx, label, "missing"))
                continue
            t0 = time.time()
            try:
                p = subprocess.run([str(exe), *args], env=env, capture_output=True, text=True, timeout=900,
                                   encoding="utf-8", errors="replace")
                rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
            except subprocess.TimeoutExpired:
                rc, out = -999, "TIMEOUT after 900 s"
            verdict = "PASS" if rc == 0 else f"FAIL ({rc})"
            log(f"  {verdict:10s} {label}  [{time.time() - t0:.1f} s]")
            for line in out.strip().splitlines()[-25:]:
                log("      | " + line)
            results.append((idx, label, verdict))
    log("\n==== parity summary")
    for idx, label, v in results:
        log(f"  GPU {idx}  {v:10s} {label}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "v100_gputest.log"))
    ap.add_argument("--config", default=str(ROOT / "strata-unsloth-ud-q4_k_xl.json"))
    from v100_setup import build_dir
    ap.add_argument("--build", default=str(build_dir()))
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--variants", help="passed to v100_bench.py")
    a = ap.parse_args()
    log = Tee(a.log)
    log(f"==== v100_gputest {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("  GPUs now (memory.used should be near 0 - nothing else running on them):")
    smi(log)
    if not a.bench_only:
        parity(log, Path(a.build))
    if not a.parity_only:
        if not Path(a.config).exists():
            log(f"  no config {a.config}: run v100\\1_setup_and_build.bat first")
            return 1
        cmd = [sys.executable, str(ROOT / "tools" / "v100_bench.py"), a.config,
               "--log", str(ROOT / "v100_bench.log")]
        if a.variants:
            cmd += ["--variants", a.variants]
        log("\n==== benchmark: " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        log(f"  benchmark exit code {rc} (details in v100_bench.log)")
    log("  GPUs after:")
    smi(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
