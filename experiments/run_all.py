"""Run every experiment in order and report wall-clock cost.

    python experiments/run_all.py            full scale  (~15-25 min)
    python experiments/run_all.py --quick    smoke test  (~2 min)
    python experiments/run_all.py --only 4 5 run a subset

Each experiment is a separate process so that a failure in one does not lose
the results of the others, and so that per-experiment memory is released.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _common import FIGURES, OUT, RESULTS, banner

EXPERIMENTS = {
    1: ("exp1_latency.py", "Admission latency S1-S5 + overhead vs baseline"),
    2: ("exp2_burst.py", "Allocation sub-pipeline burst behaviour"),
    3: ("exp3_replay.py", "Replay / freshness / tampering, eight classes"),
    4: ("exp4_fairness.py", "Policy ladder B0-B4 vs proposed, flood sweep"),
    5: ("exp5_fieldhiding.py", "Single-request field-hiding verification"),
    6: ("exp6_multirequest.py", "Multi-request leakage stress test"),
    7: ("exp7_timing_oracle.py", "Minimal disclosure under a timing observer"),
    8: ("exp8_envelope_comparison.py", "Dual-tag vs AEAD-with-AAD envelope"),
    9: ("exp4b_robustness.py", "Recovery, ablation, equal tuning, adaptive attackers"),
}


def main() -> int:
    args = sys.argv[1:]
    quick = "--quick" in args
    only = [int(a) for a in args if a.isdigit()]
    chosen = only or sorted(EXPERIMENTS)

    here = Path(__file__).resolve().parent
    passthrough = ["--quick"] if quick else []

    banner(f"csa6g experiment suite ({'quick' if quick else 'full'} scale)"
           f"\noutput tree: {OUT}")
    results, t_start = [], time.time()

    for n in chosen:
        script, title = EXPERIMENTS[n]
        print(f"\n>>> [{n}/{len(EXPERIMENTS)}] {title}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(here / script),
                               *passthrough], cwd=here)
        dt = time.time() - t0
        results.append((n, title, proc.returncode, dt))
        print(f"<<< experiment {n} finished in {dt:.1f} s "
              f"(exit {proc.returncode})")

    banner("Summary")
    failed = 0
    for n, title, rc, dt in results:
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        failed += rc != 0
        print(f"  {n}  {title:52s} {dt:7.1f} s  {status}")
    print(f"\nTotal wall-clock: {time.time() - t_start:.1f} s")
    print(f"Output  -> {OUT}")
    print(f"Tables  -> {RESULTS}")
    print(f"Figures -> {FIGURES}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
