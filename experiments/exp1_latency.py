"""Experiment 1 -- admission latency, scenarios S1-S5 (paper the latency experiment).

Five authentication/admission patterns over an identical benign workload:

    S1  first request only            (local bootstrap + 1 secondary-auth compute, unamortised)
    S2  requests 1..N                 (local bootstrap + first secondary-auth compute, amortised)
    S3  requests 2..N                 (same run, first request excluded)
    S4  requests 1..N                 (established context, no secondary-auth compute)  -- baseline
    S5  requests 1..N                 (local bootstrap amortised + secondary-auth compute on every request)

The reported comparison is the paired overhead of each scenario over S4 with a
confidence interval; see the overhead section of ``main``.

Each scenario is measured as its own configuration on the same seeded stream,
so the seeds pair across scenarios and the S3-vs-S4 comparison is a paired one.

The pool is kept empty by releasing each slice immediately after the timed
decision, so that every request traverses the full admit path.  Capacity
effects are the subject of experiments 2 and 4, not this one.
"""
from __future__ import annotations

import gc
import time
from collections import Counter

import numpy as np
import pandas as pd

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g import crypto
from csa6g.eap import STAGES, EdgeAdmissionProxy
from csa6g.harness import make_pool
from csa6g.keys import SessionKeys, run_anchor_bootstrap
from csa6g.model import AdmissionRequest, Reason
from csa6g.params import DEFAULT_EXPERIMENT, QUICK_EXPERIMENT
from csa6g.stats import mean_ci, tost
from csa6g.workload import benign_stream

SCENARIOS = {
    "S1": dict(label="First req. (BOOT + 1 AUTH)", aka=True,
               auth="first", scope="first"),
    "S2": dict(label="Reqs. 1..N (BOOT + first AUTH, amort.)", aka=True,
               auth="first", scope="all"),
    "S3": dict(label="Reqs. 2..N (first request excluded)", aka=True,
               auth="first", scope="tail"),
    "S4": dict(label="Reqs. 1..N (established, no AUTH)", aka=False,
               auth="never", scope="all"),
    "S5": dict(label="Reqs. 1..N (BOOT + N AUTH)", aka=True,
               auth="every", scope="all"),
}

# Paper scenario labels.  The paper orders S2/S3 by exposition (S2 = the
# cheapest steady state, S3 = the amortised-bootstrap case) while this artifact
# orders them by construction, so the paper's S2 is this file's S3 and vice
# versa.  The paper's macro generator applies this remap when it reads
# exp1_latency_summary.csv, and the figures apply the same one.  Only the x-axis
# labels of the two figures are affected -- every CSV/JSON keeps the artifact
# names, and nothing else here reads MANUSCRIPT_LABELS.
MANUSCRIPT_LABELS = {"S1": "S1", "S2": "S3", "S3": "S2", "S4": "S4", "S5": "S5"}


def _measure_bootstrap(eap, rng, slice_type, service_id, tenant_id
                       ) -> tuple[SessionKeys, float]:
    """Local bootstrap surrogate plus slice-key derivation, in seconds.

    ``run_anchor_bootstrap`` samples a fresh post-authentication anchor; it does not
    execute 5G-AKA/EAP-AKA' signaling.  This timing therefore covers only
    local anchor generation and HKDF work and must not be interpreted as
    end-to-end primary-authentication latency.

    Both local components are included, despite the previous version's
    docstring claiming both while timing only the first.  ``register_session`` runs ``SessionKeys.create`` ->
    ``derive_slice_keys``, two HKDF expansions, and on this machine that costs
    slightly more than the AKA stand-in itself -- so roughly half of the
    one-time bootstrap was silently excluded, and it was excluded from exactly
    the scenarios (S1, S3, S5) whose overheads the paper's claims rest on.
    """
    t0 = time.perf_counter_ns()
    k_anchor = run_anchor_bootstrap(rng)
    sk = eap.register_session(k_anchor, slice_type, service_id, tenant_id)
    return sk, (time.perf_counter_ns() - t0) / 1e9


RAW_ROWS: list[dict] = []     # every timed request, exported raw


def run_scenario(name: str, seed: int, n_requests: int, n_tenants: int
                 ) -> tuple[np.ndarray, dict]:
    """Return per-request total latencies (s) and mean per-stage costs (s).

    Python's cyclic garbage collector is paused for the scenario and run
    explicitly before it (``timeit`` does the same), so a collection cannot
    land inside a timed request.  Every timed request is appended to
    ``RAW_ROWS`` with its wall-clock time and the collector's counters, so a
    stalled seed can be attributed after the fact.
    """
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _run_scenario(name, seed, n_requests, n_tenants)
    finally:
        if gc_was_enabled:
            gc.enable()


def _run_scenario(name: str, seed: int, n_requests: int, n_tenants: int
                  ) -> tuple[np.ndarray, dict]:
    cfg = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    # Warm the generator. A fresh numpy Generator's first integers() draw costs
    # noticeably more than its steady state (one-off internal setup), and that
    # spike would land on the very first tenant's bootstrap measurement. S1
    # counts only first-requests -- n_tenants samples per seed -- so a single
    # warm-up draw would be a large fraction of its sample.
    rng.integers(0, 256, size=32, dtype="uint8")
    raw_stream = benign_stream(seed, n_tenants=n_tenants, n_requests=n_requests)
    # One slice/service security context per tenant in this latency experiment.
    # Reusing a key derived for one S-NSSAI/service to encrypt another context
    # would defeat the binding the framework claims.  The broader fairness
    # experiments vary slice types without exercising the crypto/session layer.
    context_by_tenant = {}
    stream = []
    for req in raw_stream:
        if req.tenant_id not in context_by_tenant:
            context_by_tenant[req.tenant_id] = (req.slice_type, req.service_id)
        stype, service_id = context_by_tenant[req.tenant_id]
        stream.append(AdmissionRequest(
            tenant_id=req.tenant_id, pseudonym=req.pseudonym,
            slice_type=stype, service_id=service_id,
            sla_latency_ms=req.sla_latency_ms, demand=req.demand,
            arrival_index=req.arrival_index, counter=req.counter,
            malicious=False))

    # A tenant obtains the local anchor bootstrap surrogate once, and that derivation serves *that tenant's* own
    # requests -- roughly n_requests / n_tenants of them, not the whole stream.
    # Dividing by the stream length under-charged the one-time cost by a factor
    # of n_tenants, which lands directly on S1, S3, S5 and on the S3-vs-S4
    # equivalence claim.
    served_by_tenant = Counter(r.tenant_id for r in stream)

    pool = make_pool()
    eap = EdgeAdmissionProxy(pool)

    sessions: dict[int, SessionKeys] = {}
    aka_cost: dict[int, float] = {}
    counters: dict[int, int] = {}

    totals, stage_sums = [], {s: 0.0 for s in STAGES}
    n_counted = 0

    for req in stream:
        t = req.tenant_id
        first = t not in sessions
        if first:
            sk, cost = _measure_bootstrap(eap, rng, req.slice_type,
                                          req.service_id, t)
            sessions[t] = sk
            aka_cost[t] = cost if cfg["aka"] else 0.0
            counters[t] = 0
        sk = sessions[t]
        counters[t] += 1

        run_auth = (cfg["auth"] == "every") or (cfg["auth"] == "first" and first)

        sealed_req = AdmissionRequest(
            tenant_id=t, pseudonym=req.pseudonym, slice_type=req.slice_type,
            service_id=req.service_id, sla_latency_ms=req.sla_latency_ms,
            demand=req.demand, arrival_index=req.arrival_index,
            counter=counters[t], malicious=False)
        env = crypto.seal(sk, sealed_req)

        out = eap.handle(env.raw, tenant_id=t, run_secondary_auth=run_auth)
        if not out.decision.admitted:
            raise RuntimeError(f"unexpected denial in {name}: "
                               f"{out.internal_reason}")

        # Keep the pool empty so every request traverses the full admit path.
        # The tenant keeps its session -- wiping keys here would re-run AKA on
        # every request and S3/S4 would stop being established-session scenarios.  Outside
        # the timed region either way.
        assert eap.release_slice(out.session_id) == Reason.OK

        per_request = dict(out.decision.timings)
        # The one-time local bootstrap/derivation cost is amortised across the requests it
        # serves, except in S1 where it is charged to the single first request.
        if cfg["scope"] == "first":
            if not first:
                continue
            per_request["auth"] += aka_cost[t]
        elif cfg["scope"] == "tail":
            # S3 is the steady-state tail of the *same* run as S2: the first
            # request (including local bootstrap, key derivation, and first secondary
            # authentication) is executed to establish state but excluded from
            # the reported mean.  Its cost is therefore not amortised here.
            if first:
                continue
        else:
            per_request["auth"] += aka_cost[t] / served_by_tenant[t]

        for k, v in per_request.items():
            stage_sums[k] += v
        totals.append(sum(per_request.values()))
        n_counted += 1
        RAW_ROWS.append(dict(scenario=name, seed=seed, request=len(totals) - 1,
                             tenant=t, first=first,
                             total_ms=1e3 * totals[-1],
                             wall_time=time.time(),
                             gc_counts=str(gc.get_count()),
                             **{f"{k}_ms": 1e3 * v
                                for k, v in per_request.items()}))

    stage_means = {k: v / max(1, n_counted) for k, v in stage_sums.items()}
    return np.asarray(totals, dtype=float), stage_means


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    banner(f"Experiment 1: admission latency, {P.seeds} seeds x "
           f"{P.requests_per_seed} requests x {P.tenants} tenants")

    per_seed = {s: [] for s in SCENARIOS}
    per_stage = {s: [] for s in SCENARIOS}

    for seed in range(P.seeds):
        for name in SCENARIOS:
            totals, stages = run_scenario(name, seed, P.requests_per_seed,
                                          P.tenants)
            per_seed[name].append(float(totals.mean()) * 1e3)   # ms
            per_stage[name].append({k: v * 1e3 for k, v in stages.items()})
        if (seed + 1) % 10 == 0:
            print(f"  seed {seed + 1}/{P.seeds}")

    # ---------------------------------------------------------- summary ---
    rows = []
    base = np.asarray(per_seed["S4"])
    for name, cfg in SCENARIOS.items():
        x = np.asarray(per_seed[name])
        m, lo, hi = mean_ci(x)
        rows.append(dict(scenario=name, configuration=cfg["label"],
                         mean_ms=m, std_ms=float(x.std(ddof=1)),
                         ci_lo=lo, ci_hi=hi,
                         overhead_vs_S4=float(m / base.mean() - 1.0)))
    summary = pd.DataFrame(rows)
    save_table(summary, "exp1_latency_summary")
    save_table(pd.DataFrame(per_seed), "exp1_latency_per_seed")
    save_table(pd.DataFrame(RAW_ROWS), "exp1_latency_raw")

    # Primary statistic: per-seed *median* request latency.  The raw
    # per-request export shows a heavy right tail of
    # single-request stalls (up to ~74 ms against a 0.17 ms median) from
    # operating-system preemption, with the garbage collector already paused;
    # one such stall moves a per-seed mean by ~30 %.  The median of each seed's requests is insensitive to them.
    # Means are kept above as the secondary statistic.  This choice was fixed
    # before the reported run.
    raw = pd.DataFrame(RAW_ROWS)
    med = raw.groupby(["scenario", "seed"]).total_ms.median().unstack(0)
    save_table(med.reset_index(), "exp1_latency_per_seed_median")
    mrows = []
    for name, cfg in SCENARIOS.items():
        m, lo, hi = mean_ci(med[name].values)
        mrows.append(dict(scenario=name, configuration=cfg["label"],
                          median_of_seed_medians_ms=float(med[name].median()),
                          mean_of_seed_medians_ms=m, ci_lo=lo, ci_hi=hi,
                          sd_ms=float(med[name].std(ddof=1))))
    save_table(pd.DataFrame(mrows), "exp1_latency_summary_median")
    base_m = med["S4"].values
    orow = []
    for name, cfg in SCENARIOS.items():
        if name == "S4":
            continue
        d = med[name].values - base_m
        m, lo, hi = mean_ci(d)
        b = base_m.mean()
        orow.append(dict(comparison=f"{name} - S4",
                         configuration=cfg["label"], overhead_ms=m,
                         ci_lo_ms=lo, ci_hi_ms=hi,
                         overhead_pct=100 * m / b, pct_ci_lo=100 * lo / b,
                         pct_ci_hi=100 * hi / b, n_seeds=int(d.size)))
    save_table(pd.DataFrame(orow), "exp1_overhead_median")
    spikes = (raw.total_ms > 5 * raw.groupby("scenario").total_ms
              .transform("median"))
    save_json({"requests": int(len(raw)),
               "requests_over_5x_scenario_median": int(spikes.sum()),
               "max_request_ms": float(raw.total_ms.max())},
              "exp1_latency_tail")

    stage_rows = []
    for name in SCENARIOS:
        d = pd.DataFrame(per_stage[name]).mean().to_dict()
        d["scenario"] = name
        stage_rows.append(d)
    stages_df = pd.DataFrame(stage_rows)[["scenario", *STAGES]]
    save_table(stages_df, "exp1_stage_breakdown")

    # --------------------------------------------- overhead vs baseline ---
    # Reported as a paired difference with a confidence interval, not as an
    # equivalence test.
    #
    # A TOST margin of +/-3% is 3% *of a machine-dependent baseline*, so it
    # moves with the hardware while the quantity it bounds -- one HKDF
    # derivation amortised over a tenant's requests -- does not.  The overhead
    # is therefore reported directly, with its CI, and a reader can apply
    # whatever threshold they consider material.
    over_rows = []
    base = np.asarray(per_seed["S4"], dtype=float)
    for name, cfg in SCENARIOS.items():
        if name == "S4":
            continue
        d = np.asarray(per_seed[name], dtype=float) - base
        m, lo, hi = mean_ci(d)
        rel = mean_ci(100.0 * d / base.mean())
        over_rows.append(dict(
            comparison=f"{name} - S4", configuration=cfg["label"],
            overhead_ms=m, ci_lo_ms=lo, ci_hi_ms=hi,
            overhead_pct=rel[0], pct_ci_lo=rel[1], pct_ci_hi=rel[2],
            n_seeds=int(d.size)))
    over_df = pd.DataFrame(over_rows)
    save_table(over_df, "exp1_overhead")

    # TOST is kept as a secondary diagnostic.
    tost_rows = []
    for margin in P.tost_margins:
        for name in ("S3", "S5"):
            r = tost(per_seed[name], per_seed["S4"], margin, paired=True)
            row = dict(comparison=f"{name} vs S4", margin_pct=margin * 100,
                       **r.as_row())
            row["margin_used_pct"] = (100.0 * r.ci_hi / r.margin
                                      if r.margin else float("nan"))
            tost_rows.append(row)
    tost_df = pd.DataFrame(tost_rows)
    save_table(tost_df, "exp1_tost_diagnostic")

    print("\n" + summary.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
    print("\nOverhead vs S4 (paired mean difference, 95% CI):")
    print(over_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nTOST diagnostic only -- not the reported result. 'margin_used_pct'"
          "\nis how much of the equivalence margin the CI consumes; near 100"
          "\nmeans the verdict is a coin-flip on hardware, which is why the"
          "\nequivalence framing was dropped.")
    print(tost_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nPer-stage mean cost (ms):")
    print(stages_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---------------------------------------------------------- figures ---
    import matplotlib.pyplot as plt

    # Plot in paper order (S1..S5 under the paper's labels): reorder the
    # artifact rows so that the bar labelled "S2" is the paper's S2, etc.
    art_names = list(SCENARIOS)
    names = sorted(art_names, key=lambda a: MANUSCRIPT_LABELS[a])
    order = [art_names.index(a) for a in names]
    names = [MANUSCRIPT_LABELS[a] for a in names]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    # Plot the primary statistic (mean of per-seed medians), as in the paper.
    msum = pd.DataFrame(mrows)
    means = msum["mean_of_seed_medians_ms"].to_numpy()[order]
    err = np.vstack([means - msum["ci_lo"].to_numpy()[order],
                     msum["ci_hi"].to_numpy()[order] - means])
    ax.bar(names, means, yerr=err, capsize=3, color="#3b6ea5", width=0.6)
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Per-seed median latency (ms)")
    ax.set_xlabel("Scenario")
    ax.set_title("Admission latency (mean of per-seed medians, 95% CI)")
    save_fig(fig, "fig2_latency_by_scenario")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    # Hatch per stage in addition to colour so the stack is legible in
    # grayscale, not colour-only.
    _hatches = ["", "///", "\\\\\\", "xxx", "...", "ooo", "+++", "---",
                "|||", "***", "OOO", "//", "\\\\", "xx"]
    _cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(names))
    _hi = 0
    for _i, stage in enumerate(STAGES):
        vals = stages_df[stage].to_numpy()[order]
        if vals.max() < 1e-6:
            continue
        ax.bar(names, vals, bottom=bottom, label=stage, width=0.6,
               color=_cmap(_i % 20), hatch=_hatches[_hi % len(_hatches)],
               edgecolor="white", linewidth=0.4)
        bottom += vals
        _hi += 1
    ax.set_ylabel("Mean cost (ms)")
    ax.set_xlabel("Scenario")
    ax.set_title("Admission pipeline component breakdown")
    ax.legend(ncol=2, fontsize=7, loc="upper center")
    save_fig(fig, "fig3_stage_breakdown")

    save_json({"scenarios": summary.to_dict("records"),
               "overhead_vs_S4": over_df.to_dict("records"),
               "tost_diagnostic": tost_df.to_dict("records"),
               "note": "The reported result is the paired overhead with its "
                       "CI. TOST at +/-3% is kept as a secondary "
                       "diagnostic."}, "exp1_latency")


if __name__ == "__main__":
    main()
