"""V100 fork: set up UD-Q4_K_XL on an RTX 4070 Super + Tesla V100 PC, and build the GPU test programs.

Uses NO GPU: it only asks nvidia-smi which cards are there, then runs setup (download/pack if needed, compile the
engine for sm_70 + sm_89 with the CUDA 12.x toolkit, write the config) with --no-start, and compiles the parity
test programs into the same build folder.  Everything is written to the log file (default v100_setup.log).

    .venv\\Scripts\\python tools\\v100_setup.py                       (what v100\\1_setup_and_build.bat runs)
    ... --context 65536 --gguf-dir D:\\models\\unsloth               (your own choices, passed to setup)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"

# the GPU test programs the second script runs (all synthetic: no model needed); built next to the engine
TEST_TARGETS = ["bf16_gemm_fallback_test", "native_expert_parity", "mmvq_multi_parity", "gr_parity", "gdn_parity",
                "qsa_parity", "kv_q8_parity", "elementwise_parity", "sampler_parity", "router_top10_parity",
                "rope_parity", "quantize_act_parity", "shared_expert_parity", "bf16_gemv_parity",
                "s_gemv_q8k_parity", "prefill_mmq_kquant_test", "strata-device"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", errors="replace")

    def __call__(self, line=""):
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def run_logged(cmd, log, env=None, cwd=None) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                         encoding="utf-8", errors="replace", env=env, cwd=cwd)
    for line in p.stdout:
        log(line.rstrip("\n"))
    return p.wait()


def gpus(log):
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_model.current,"
                        "pci.bus_id", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    out = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4:
            out.append({"index": int(f[0]), "name": f[1], "cc": f[2], "mib": f[3],
                        "model": f[4] if len(f) > 4 else "?", "bus": f[5] if len(f) > 5 else "?"})
            log(f"  GPU {f[0]}: {f[1]}, compute {f[2]}, {f[3]} MiB, driver model {out[-1]['model']}, bus {out[-1]['bus']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "v100_setup.log"))
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--gguf-dir")
    ap.add_argument("--data-dir")
    ap.add_argument("--gpus", help="override the order, e.g. 0,1 (the first is the main card)")
    ap.add_argument("--skip-setup", action="store_true", help="only build the test programs")
    a = ap.parse_args()
    log = Tee(a.log)
    log(f"==== v100_setup {time.strftime('%Y-%m-%d %H:%M:%S')}")
    found = gpus(log)
    if a.gpus:
        order = a.gpus
    else:
        ada = [g for g in found if g["cc"].startswith("8.9")]
        volta = [g for g in found if g["cc"] == "7.0"]
        if not ada or not volta:
            log("  could not find both an sm_89 card (RTX 4070 Super) and an sm_70 card (V100); pass --gpus A,B")
            return 1
        # the 4070 Super first: the V100 (32 GB) then carries the output head and the MTP draft layer, so the 12 GB
        # card keeps more of its VRAM for experts (docs/V100.md)
        order = f"{ada[0]['index']},{volta[0]['index']}"
    log(f"  GPU order for the layer split: {order} (the first one is CUDA0)")
    py = sys.executable
    if not a.skip_setup:
        cmd = [py, str(ROOT / "setup.py"), "--setup", "--build", "--yes", "--no-start", "--family", "unsloth",
               "--model", "UD-Q4_K_XL", "--gpus", order, "--context", str(a.context), "--vision", "no",
               "--experimental-speed-projection", "off"]
        gguf_dir = a.gguf_dir
        old_cfg = ROOT / "strata-unsloth-ud-q4_k_xl.json"
        if not gguf_dir and old_cfg.exists():
            # a run again (another --context): the model files stay where the last setup found them - without this,
            # setup looks in Strata-data\models and would download the 111 GB again
            import json
            args = json.loads(old_cfg.read_text(encoding="utf-8-sig")).get("args", [])
            if "--native" in args:
                native = Path(args[args.index("--native") + 1])
                if native.exists():
                    gguf_dir = str(native.parent)
                    log(f"  model files: {gguf_dir} (from the existing config)")
        if gguf_dir:
            cmd += ["--gguf-dir", gguf_dir]
        if a.data_dir:
            cmd += ["--data-dir", a.data_dir]
        rc = run_logged(cmd, log, cwd=str(ROOT))
        log(f"  setup exit code {rc}")
        if rc != 0:
            return rc
    # the test programs, in setup's build folder (same configuration, so only they are compiled)
    sys.path.insert(0, str(ROOT))
    import setup as S                                      # noqa: E402
    cmake, ninja = S.find_tool("cmake"), S.find_tool("ninja")
    if not (ROOT / "build" / "CMakeCache.txt").exists() or cmake is None:
        log("  no build/CMakeCache.txt (setup did not compile the engine?) - the test programs are not built")
        return 1
    jobs = str(max(2, (os.cpu_count() or 4) // 2))
    build = [cmake, "--build", str(ROOT / "build"), "-j", jobs, "--target", *TEST_TARGETS]
    if WIN:
        vcvars = S.find_vcvars()
        if vcvars is None:
            log("  Visual Studio Build Tools not found")
            return 1
        bat = ROOT / "build-v100-tests.bat"
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(build)}\r\n', encoding="utf-8")
        rc = run_logged(["cmd", "/c", str(bat)], log, cwd=str(ROOT))
    else:
        rc = run_logged(build, log, cwd=str(ROOT))
    log(f"  test programs build exit code {rc}")
    exe = ".exe" if WIN else ""
    for t in TEST_TARGETS:
        p = ROOT / "build" / (t + exe)
        log(f"    {'ok     ' if p.exists() else 'MISSING'} {p}")
    cfg = ROOT / "strata-unsloth-ud-q4_k_xl.json"
    log(f"  config: {cfg} {'(present)' if cfg.exists() else '(MISSING)'}")
    if cfg.exists():
        # measured on this PC (2026-10-02, v100_bench run 2): the CPU expert pool on the P-cores and their SMT
        # siblings, without the E-cores - 43.3 vs 41.0 tok/s, ahead in 4 of 6 prompts
        import json
        c = json.loads(cfg.read_text(encoding="utf-8-sig"))
        # v100_prefillbench (2026-10-03): the V100 reads long prompts in 32768-token chunks - 814 vs 543 tok/s on a
        # 114K prompt (16384: 758)
        # v100_decodebench2 (2026-10-03): at most 32 adaptive swaps per round - 55.6 / 46.3 tok/s (short / after 60K)
        # against 52.4 / 40.9 with 96 (the swaps share the V100's x4 link with decoding)
        # v100_decodebench3/4 (2026-10-03): --spec-min-p 0.7 was a little ahead in every round; layers 0-19 on the
        # 4070 Super decode as fast as 0-16 (17-22 are within the noise) and read prompts ~7% faster (the V100
        # streams fewer layers' experts)
        for flag, val in (("--pool-affinity", "auto"), ("--pool-workers", "11"), ("--prefill", "auto:32768"),
                          ("--adapt-swaps", "32"), ("--spec-min-p", "0.7")):
            if flag in c["args"]:
                c["args"][c["args"].index(flag) + 1] = val
            else:
                c["args"] += [flag, val]
        if isinstance(c.get("gpu"), list) and len(c["gpu"]) == 2:
            c["layer_split"] = "20"
        # the PLE table in a file of its own (tools/v100_ple_split.py), when it has been made: setup's config
        # reads it from the shard, which the engine also maps
        if "--native" in c["args"]:
            nat = Path(c["args"][c["args"].index("--native") + 1])
            own = sorted(nat.parent.glob("*-ple-table.gguf"))
            if own:
                if "--ple-gguf" in c["args"]:
                    c["args"][c["args"].index("--ple-gguf") + 1] = str(own[0])
                else:
                    c["args"] += ["--ple-gguf", str(own[0])]
                log(f"  the PLE table from its own file: {own[0].name}")
        cfg.write_text(json.dumps(c, indent=1), encoding="utf-8")
        log("  tuned for this PC: --pool-affinity auto --pool-workers 11 --prefill auto:32768 --adapt-swaps 32 "
            "--spec-min-p 0.7, layer_split 20")
    if cfg.exists():
        log(cfg.read_text(encoding="utf-8"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
