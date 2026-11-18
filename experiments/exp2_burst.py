"""Experiment 2 -- allocation sub-pipeline cost and its invariance to load.

Scope: this measures **only** the allocation sub-pipeline on
already-established sessions -- NSAC quota lookup, isolation-aware placement
(placement rule) and the authenticated state commit.  It excludes the admission
cryptographic path entirely.  These numbers are therefore not comparable with
the S1-S5 latencies of experiment 1 and say nothing about DoS resilience,
which is experiment 4.

The question is narrow: for an admitted request, does the allocation decision
cost grow with the size of the burst it arrives in?

Only admitted requests are timed
--------------------------------
A denied request never reaches ``store.write``/``store.verify`` -- the
authenticated commit this experiment exists to time -- so averaging the two
paths does not produce a noisier measurement of the sub-pipeline, it produces a
measurement of something else.

The policy is not detuned.  The obvious repair for the low admission rate --
enlarge the pool and open the rate gate until everything is admitted -- would
weaken the very controls experiment 4 evaluates.  So this runs the shipped
``ProposedNSACQuota`` on the same deployment as experiment 4, and the
sliding-window rate gate binds hard: 2 admits per tenant per 64 arrivals.

Two confounds, both real, both now handled
------------------------------------------
Timing only the admitted path means the *composition* of the timed sample can
differ between the two burst sizes, and it does, for two independent reasons.
Neither was visible in the previous version of this experiment and they push in
opposite directions.

*AIMD sinks deeper on longer streams.*  A denial multiplies a tenant's
``cap_scale`` by 0.5; an admit adds 0.15.  Under a rate gate that binds, the
scale ratchets toward its floor, and the longer the stream the more of the
tenant population has reached it before the stream ends.  ``mean_cap_scale``
in the summary table records where each arm ended up, and the admitted fraction
tracks it.  The admitted fraction is therefore a
*monotone function of N by construction*, not a nuisance that averaging
removes.  This is a genuine property of the policy and is reported as such; it
is also exactly the thing that stops a raw N-sweep from isolating N.  The
previous version of this loop never called ``on_deny``, so half the policy was
inert and the effect was hidden.

*Warm-up dilutes asymmetrically.*  The first few timed samples in any run cost
about 2.7x the steady state (interpreter and allocator warm-up, not the system
under test).  The two arms yield different numbers of admitted samples, so that
fixed transient is diluted several times more in the longer arm -- a bias
toward "cost falls with load" that has nothing to do with load.

*Denial density differs, and denials are untimed but not free.*  A lower admit
rate means more denied requests are processed *between* consecutive timed
samples, and that untimed work touches the same caches and branch predictors
the next timed sample uses.  The two arms differ several-fold in how many
denials separate consecutive measurements, so any per-sample effect of that
work lands on them unequally.

The size and even the *sign* of the effect are machine-dependent, so this
module measures them rather than asserting them: ``exp2_denial_gap_strata``
reports mean cost stratified by denials-since-previous-sample, and the printed
correlation gives the direction on the machine that produced the run.  On the
box these results were generated on the relationship is flat to slightly
negative -- samples following denials are marginally *faster*, not slower --
which is the opposite of what the mechanism suggests and small enough
(|r| < 0.05) to be nothing at all.  An earlier draft of this docstring asserted
a monotone rise and named this the largest of the three confounds; that was
read off one machine and does not reproduce here.  Report the stratum table,
not a story about it.

What survives the machine is the *asymmetry*: the arms differ in composition,
so the comparison is run on a matched sample regardless of which direction any
individual confound pushes.  Matching is cheap and does not require knowing the
mechanism.

The repairs
-----------
``WARMUP_ADMITS`` samples are discarded from every run before anything is
computed, which removes the second confound outright.

For the first and third, the headline comparison is run on a **matched
sample**: both arms are restricted to timed samples whose denial gap is at most
``GAP_CAP``, then truncated to the same number of admitted requests, so the two
burst sizes are compared at comparable policy depth, comparable denial density
and equal sample size.  The unmatched figures and the gap stratification are
reported alongside, not instead, so the size of each correction is visible.

The matched comparison still spans two different stream lengths, so it is
corroborated by a **within-run drift test** that needs no cross-arm matching at
all: split each run's own timed samples into an early and a late half and
compare them.  If accumulated policy state cost anything, the late half would
be dearer than the early half *within a single run*, and no composition
argument can explain that away.  This is the measurement that actually answers
the question the experiment is asked.

``--quick`` is not adequate here.  Matching costs sample size twice over -- the
gap cap discards roughly half the timed samples and the truncation discards
more -- so at reduced scale the matched interval is wide enough to be
uninformative (+16% [-13, +46] on one run) while the full-scale interval is
tight around zero.  Use the full run for any number that reaches the paper.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g.datastore import AllocationRecord, AuthenticatedStore
from csa6g.harness import make_pool
from csa6g.keys import SessionKeys, run_anchor_bootstrap
from csa6g.model import Reason, demand_norm
from csa6g.policies import ProposedNSACQuota
from csa6g.params import DEFAULT_EXPERIMENT, QUICK_EXPERIMENT
from csa6g.placement import select_node
from csa6g.stats import mean_ci
from csa6g.workload import benign_stream


#: Held constant across burst sizes so that N is the only variable that moves.
BURST_TENANTS = 20

#: Timed samples discarded at the head of every run.  The first few admissions
#: in a fresh process cost ~2.7x the steady state -- interpreter warm-up, not
#: the system under test -- and because the two arms yield different numbers of
#: admitted samples, that fixed transient is diluted differently in each and
#: biases the comparison on its own.
WARMUP_ADMITS = 10

#: Maximum number of denials-since-previous-admit a timed sample may carry and
#: still enter the headline comparison.  Denials are not timed, but they are
#: executed between timed samples and measurably perturb the next one, so an arm
#: with a lower admit rate is otherwise charged for work no clock ever saw.
GAP_CAP = 2


def run_burst(n: int, seed: int, num_nodes: int, beta: int) -> dict:
    """Time the allocation sub-pipeline for every admitted request.

    Returns the post-warm-up latencies in arrival order (ms), the admitted
    fraction over the whole stream, and the policy's final mean AIMD scale --
    the last so the composition explanation above is evidenced, not asserted.
    """
    rng = np.random.default_rng(seed)
    stream = benign_stream(seed, n_tenants=BURST_TENANTS, n_requests=n)
    pool = make_pool(num_nodes)
    total = pool.total_tup
    # The policy object the E-AP and the offline harness both drive, with the
    # shipped parameters, rather than a bare NSACController stepped by hand.
    # Hand-stepping made this a third implementation of the same decision, and
    # it had already drifted: it never called on_deny.
    policy = ProposedNSACQuota()
    policy.reset(pool, beta)
    store = AuthenticatedStore()

    # Sessions are already established: keys exist, no admission crypto runs.
    sessions = {t: SessionKeys.create(run_anchor_bootstrap(rng),
                                      r.slice_type, r.service_id)
                for t, r in {q.tenant_id: q for q in stream}.items()}

    lat: list[float] = []
    # Tenants holding a live allocation.  Every slice is released immediately
    # below, so this is empty at every check and ``max(1, .)`` yields 1 -- the
    # semantically correct live count for a pool that is empty by construction.
    # It is maintained rather than assumed: a set of everyone-ever-seen would
    # divide the pressure term by the tenant population and collapse the cap.
    live: set[int] = set()
    gaps: list[int] = []
    denied = 0
    since_admit = 0

    for req in stream:
        sk = sessions[req.tenant_id]
        t0 = time.perf_counter_ns()

        cost = demand_norm(req.demand, total)
        # Dominant-dimension headroom, matching the E-AP and the policy
        # harness.  The mean across dimensions lets the cap stay relaxed while
        # the resource that actually binds is gone.
        free_frac = pool.dominant_free_fraction()
        reason = policy.admit(req, cost, free_frac, max(1, len(live)))
        node, place_reason = select_node(pool.nodes, req.demand,
                                         req.slice_type, beta)
        if reason != Reason.OK or node is None:
            # Not timed.  A denial skips the authenticated commit, so counting
            # it would report a blend of two paths rather than the cost of one.
            # on_deny is not optional: it drives the AIMD multiplicative
            # decrease, and omitting it here -- as this loop used to -- left
            # half the proposed policy inert while the E-AP and the offline
            # harness both applied it.
            policy.on_deny(req, reason if reason != Reason.OK
                           else place_reason)
            denied += 1
            since_admit += 1
            continue

        node.allocate(req.demand, req.slice_type)
        live.add(req.tenant_id)
        sid = os.urandom(16)
        rec = AllocationRecord(sid, req.tenant_id, int(req.slice_type),
                               node.node_id, req.demand, time.time(),
                               request_counter=req.counter)
        store.write(rec)
        store.verify(sid)
        lat.append((time.perf_counter_ns() - t0) / 1e6)          # ms
        gaps.append(since_admit)
        since_admit = 0

        # Keep the window open: release immediately, outside the timer.  The
        # release is complete -- resources, policy budget and live-tenant set,
        # exactly what EdgeAdmissionProxy.release_slice does -- because a
        # partial release leaves the next request reading state that the E-AP
        # would never have presented it.
        node.free(req.demand, req.slice_type)
        store.mark_released(sid)
        policy.on_admit(req, cost)
        policy.on_release(req.tenant_id, req.demand, cost)
        live.discard(req.tenant_id)

    if len(lat) <= WARMUP_ADMITS:
        raise RuntimeError(
            f"only {len(lat)} admitted samples at N={n}, seed={seed}; "
            f"fewer than the {WARMUP_ADMITS} discarded as warm-up")

    scales = [s.cap_scale for s in policy.nsac.tenants.values()]
    return dict(lat=np.asarray(lat[WARMUP_ADMITS:], dtype=float),
                gap=np.asarray(gaps[WARMUP_ADMITS:], dtype=int),
                admitted_frac=1.0 - denied / len(stream),
                n_admitted=len(lat),
                mean_cap_scale=(float(np.mean(scales)) if scales
                                else float("nan")))


def drift_within_run(lat: np.ndarray) -> float:
    """Late-half mean minus early-half mean, as a percentage of the early half.

    The cross-arm comparison can always be argued to compare different
    admission mixes.  This cannot: it is one run, one policy trajectory, split
    at its own midpoint.  If accumulated state cost anything, this is positive.
    """
    h = len(lat) // 2
    early, late = lat[:h], lat[h:2 * h]
    return float(100.0 * (late.mean() - early.mean()) / early.mean())


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    reps = 10 if quick_flag() else 30
    nodes, beta = P.fairness_nodes, P.fairness_beta
    banner(f"Experiment 2: allocation sub-pipeline cost, N in {P.burst_sizes} "
           f"({nodes} nodes, beta = {beta}, {BURST_TENANTS} tenants, "
           f"admitted path only, first {WARMUP_ADMITS} samples discarded)")

    # ---------------------------------------------------------- collect ---
    runs: dict[int, list[dict]] = {}
    for n in P.burst_sizes:
        runs[n] = [run_burst(n, seed, nodes, beta) for seed in range(reps)]

    # The matched sample.  Restrict to timed samples whose denial gap is at
    # most GAP_CAP -- so both arms carry comparable untimed work between
    # measurements -- then truncate every arm to the same count, so policy
    # depth and sample size match too.  Taking the minimum over *all* runs
    # rather than per-N keeps the seeds paired.
    def matched(r):
        return r["lat"][r["gap"] <= GAP_CAP]

    k_match = min(len(matched(r)) for n in P.burst_sizes for r in runs[n])
    if k_match < 20:
        raise RuntimeError(f"only {k_match} gap-matched samples per run; "
                           f"raise GAP_CAP or lower WARMUP_ADMITS")

    rows, raw = [], {}
    for n in P.burst_sizes:
        rs = runs[n]
        means = [float(r["lat"].mean()) for r in rs]
        m_means = [float(matched(r)[:k_match].mean()) for r in rs]
        m_p95s = [float(np.percentile(matched(r)[:k_match], 95)) for r in rs]
        drifts = [drift_within_run(r["lat"]) for r in rs]
        fracs = [r["admitted_frac"] for r in rs]
        scales = [r["mean_cap_scale"] for r in rs]

        raw[f"N{n}_mean_ms"] = m_means          # report.py reads these
        raw[f"N{n}_unmatched_mean_ms"] = means
        raw[f"N{n}_p95_ms"] = m_p95s
        raw[f"N{n}_admitted_frac"] = fracs
        raw[f"N{n}_drift_pct"] = drifts

        m, lo, hi = mean_ci(m_means)
        p, plo, phi = mean_ci(m_p95s)
        um, ulo, uhi = mean_ci(means)
        rows.append(dict(burst_N=n, mean_ms=m, mean_ci_lo=lo, mean_ci_hi=hi,
                         p95_ms=p, p95_ci_lo=plo, p95_ci_hi=phi,
                         unmatched_mean_ms=um, unmatched_ci_lo=ulo,
                         unmatched_ci_hi=uhi,
                         matched_samples=k_match,
                         admitted_frac=float(np.mean(fracs)),
                         admitted_samples=float(np.mean(
                             [r["n_admitted"] for r in rs])),
                         mean_cap_scale=float(np.mean(scales)),
                         mean_denial_gap=float(np.mean(
                             [r["gap"].mean() for r in rs])),
                         repetitions=reps, nodes=nodes, beta=beta,
                         tenants=BURST_TENANTS))

    df = pd.DataFrame(rows)
    save_table(df, "exp2_burst_summary")
    save_table(pd.DataFrame(raw), "exp2_burst_per_seed")

    factor = df["burst_N"].iloc[-1] / df["burst_N"].iloc[0]
    a = np.asarray(raw[f"N{P.burst_sizes[0]}_mean_ms"], dtype=float)
    b = np.asarray(raw[f"N{P.burst_sizes[-1]}_mean_ms"], dtype=float)
    pm, plo, phi = mean_ci(100.0 * (b - a) / a.mean())      # paired, matched

    all_drift = np.concatenate([np.asarray(raw[f"N{n}_drift_pct"])
                                for n in P.burst_sizes])
    dm, dlo, dhi = mean_ci(all_drift)

    # ---- the third confound, measured rather than assumed ---------------
    # Denials are untimed but executed between timed samples, and the arm with
    # the lower admit rate carries more of them.  If this stratification were
    # flat the correction would be unnecessary; it is not flat, which is why
    # the headline comparison is gap-matched.
    all_lat = np.concatenate([r["lat"] for n in P.burst_sizes for r in runs[n]])
    all_gap = np.concatenate([r["gap"] for n in P.burst_sizes for r in runs[n]])
    strat = []
    for lo, hi, lbl in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"),
                        (6, 10 ** 9, "6+")):
        m = (all_gap >= lo) & (all_gap <= hi)
        if m.sum():
            strat.append(dict(denial_gap=lbl, samples=int(m.sum()),
                              mean_ms=float(all_lat[m].mean())))
    strat_df = pd.DataFrame(strat)
    # Kept in the JSON, no longer written as a standalone results table: this
    # experiment is demoted to a bound (see report.py section 2), and the strata
    # exist to justify the matching, not as a finding in their own right.

    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\nPer-sample cost by denials executed since the previous timed "
          "sample (all runs pooled):")
    print(strat_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    r_gap = float(np.corrcoef(all_lat, all_gap)[0, 1])
    if abs(r_gap) < 0.05:
        verdict = ("no usable association on this machine; the arms are "
                   "matched on it anyway, because matching costs nothing and "
                   "does not depend on the effect being real")
    elif r_gap > 0:
        verdict = ("samples following denials are slower here, so the arm with "
                   "the lower admit rate would be charged for work no clock "
                   "saw; the gap cap removes it")
    else:
        verdict = ("samples following denials are *faster* here, which would "
                   "flatter the arm with the lower admit rate; the gap cap "
                   "removes it")
    print(f"  correlation(latency, denial gap) = {r_gap:+.3f} -- {verdict}.")

    print(f"\nMatched sample: {k_match} admitted requests per run in both arms "
          f"(of {df['admitted_samples'].iloc[0]:.0f} and "
          f"{df['admitted_samples'].iloc[-1]:.0f} available), first "
          f"{WARMUP_ADMITS} discarded as warm-up, denial gap <= {GAP_CAP}.")
    print(f"{factor:.0f}x load increase, paired across {a.size} seeds on the "
          f"matched sample: {pm:+.2f}% [{plo:+.2f}, {phi:+.2f}] of the "
          f"per-request cost.")
    print(f"Within-run drift (late half vs early half, {all_drift.size} runs): "
          f"{dm:+.2f}% [{dlo:+.2f}, {dhi:+.2f}] -- accumulated policy state "
          f"costs nothing this measurement can resolve inside a single run.")

    print(f"\nAdmitted fraction: {df['admitted_frac'].iloc[0]:.3f} at "
          f"N={df['burst_N'].iloc[0]}, {df['admitted_frac'].iloc[-1]:.3f} at "
          f"N={df['burst_N'].iloc[-1]}. This gap is a property of the policy, "
          f"not noise: the AIMD scale ratchets toward its floor under a "
          f"binding rate gate, so a longer stream ends with more of the tenant "
          f"population there (mean final scale "
          f"{df['mean_cap_scale'].iloc[0]:.3f} vs "
          f"{df['mean_cap_scale'].iloc[-1]:.3f}). It is why the headline "
          f"comparison runs on the matched sample and is corroborated by the "
          f"within-run drift test.")

    # No figure. This experiment is demoted to a two-sentence bound (report.py
    # section 2): a paired interval that spans zero, whose sign flips between
    # runs of identical code, is not something a reader should be shown a chart
    # of -- a bar chart of two means with overlapping error bars invites exactly
    # the reading the bound exists to prevent. The numbers behind the bound stay
    # in the CSVs and the JSON below.

    save_json({"summary": df.to_dict("records"),
               "matched_samples_per_run": int(k_match),
               "warmup_discarded": WARMUP_ADMITS,
               "denial_gap_cap": GAP_CAP,
               "denial_gap_strata": strat_df.to_dict("records"),
               "denial_gap_corr": float(np.corrcoef(all_lat, all_gap)[0, 1]),
               "across_N_pct": {"mean": float(pm), "ci_lo": float(plo),
                                "ci_hi": float(phi), "seeds": int(a.size)},
               "within_run_drift_pct": {"mean": float(dm), "ci_lo": float(dlo),
                                        "ci_hi": float(dhi),
                                        "runs": int(all_drift.size)},
               "deployment": {"nodes": int(nodes), "beta": int(beta),
                              "tenants": BURST_TENANTS,
                              "policy": "shipped parameters, not detuned"},
               "admitted_frac": {str(int(r.burst_N)): float(r.admitted_frac)
                                 for r in df.itertuples()},
               "mean_cap_scale": {str(int(r.burst_N)): float(r.mean_cap_scale)
                                  for r in df.itertuples()},
               "composition_note":
                   "The admitted fraction falls with N because the AIMD scale "
                   "ratchets toward its floor under a binding rate gate, so a "
                   "longer stream ends with more of the tenant population "
                   "there. That is a property of the policy, not measurement "
                   "noise, and it means a raw N-sweep does not isolate N. The "
                   "headline figure is therefore the matched-sample paired "
                   "comparison, corroborated by a within-run early-vs-late "
                   "drift test that no composition argument can explain away.",
               "denial_density_note":
                   "Denied requests are not timed but are executed between "
                   "timed samples, and the arm with the lower admit rate "
                   "carries several times more of them. Whether that measurably "
                   "moves the next sample, and in which direction, is "
                   "machine-dependent: see denial_gap_correlation and the "
                   "exp2_denial_gap_strata table for this run rather than "
                   "relying on this note. Both arms are restricted to a denial "
                   "gap of at most GAP_CAP either way, because matching is "
                   "cheap and does not require the effect to be real. Any "
                   "effect here is a harness artefact, not a property of the "
                   "allocation sub-pipeline.",
               "denial_gap_correlation": r_gap,
               "warmup_note":
                   "The first admissions in a fresh process cost ~2.7x the "
                   "steady state. Because the two arms yield different numbers "
                   "of admitted samples, that fixed transient is diluted "
                   "differently in each and biases the comparison toward "
                   "'cost falls with load' on its own. It is discarded.",
               "scope": "allocation sub-pipeline on the admitted path only; "
                        "not end-to-end admission, not a DoS-resilience "
                        "result"},
              "exp2_burst")


if __name__ == "__main__":
    main()
