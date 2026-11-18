"""Experiment 4 -- comparative fairness and DoS resilience (the fairness experiment).

The proposed admission policy against a ladder of recognised allocation schemes on an identical
adversarial workload, rather than a quotas-on/off mechanism check.

    B0  uncontrolled FCFS                (the floor)
    B1  static equal per-tenant cap
    B2  per-tenant token bucket
    B3  scalar progressive-fill approximation
    B4  offline discretised DRF approximation
    P   proposed NSAC quota + dynamic caps

Protocol.  Seed set, request stream (arrival order, demand vectors,
benign/malicious labels), resource pool and isolation constraint are held
identical across policies; only the policy varies, so comparisons are paired
per seed.  Parameter selection and evaluation use disjoint seed blocks: the
(q_base, kappa) sensitivity grid runs on a development block, and every
reported comparison -- the sweep, the headline table, the paired tests, the benign
reference, the invariance and isolation sweeps -- runs on a separate
evaluation block it never touched, so no headline number is reported on a
seed used to characterise the operating point.  The malicious:benign ratio is swept over {1, 2, 5, 10, 20} so that
no single flattering operating point is reported.

Two metrics of DoS resilience are reported and the second is the one to read.
The malicious *rejection rate* is confounded by saturation: when the offered
attack load is twenty times capacity, even a policy with no defence rejects
~96% of malicious requests simply because it runs out of resources.  The
malicious *capacity share* -- how much of the granted pool the attackers
actually walk away with -- is not confounded that way.

Outputs also include two sweeps that make the result falsifiable rather than
merely favourable: a sensitivity sweep over the proposed policy's two tunables,
and an isolation-budget sweep pricing the anti-co-location constraint.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g.harness import run_policy
from csa6g.params import (DEFAULT_EXPERIMENT, DEFAULT_POLICY_PARAMS,
                          QUICK_EXPERIMENT)
from csa6g.policies import ProposedNSACQuota, UncontrolledFCFS, build_ladder
from csa6g.stats import holm_bonferroni, mean_ci, paired_comparison
from csa6g.workload import StreamSpec, generate_stream

METRICS = ("fi_benign", "fi_all", "fi_benign_norm", "benign_acceptance",
           "malicious_rejection", "malicious_capacity_share",
           "dominant_utilisation", "utilisation", "decision_ms")


# Held-out seed discipline.  The two tunables
# (q_base, kappa) are a fixed a-priori engineering point, but the sensitivity
# grid that *characterises* that point and every *reported* comparison must
# not share seeds, or the diagnostic and the headline are drawn from the same
# sample.  The grid runs on the development block; the sweep, benign
# reference, invariance and isolation results run on a disjoint evaluation
# block.  Both blocks are the full ``fairness_seeds`` size.
DEV_SEED_BASE = 0            # sensitivity / operating-point diagnostic
EVAL_SEED_BASE = 100_000     # every reported result
EVAL_BENIGN_SEED_BASE = 110_000   # benign-only reference, disjoint from both


def _spec(P, ratio: float, **kw) -> StreamSpec:
    return StreamSpec(n_benign_tenants=P.fairness_benign_tenants,
                      n_attacker_tenants=P.fairness_attacker_tenants,
                      n_benign_requests=P.fairness_benign_requests,
                      malicious_ratio=ratio, **kw)


def sweep(P, ratios=None) -> pd.DataFrame:
    """Sweep flood intensity, including a true zero-attack point.

    ``ratio = 0`` is generated exactly like every other point: same seeds, same
    ``_spec``, same 60-tenant population (50 benign + 10 attacker) -- only the
    attack *volume* goes to zero, so the attacker tenants are present and
    silent.  That is what makes the sweep a single continuous axis from no
    attack to 20:1.

    The separate ``benign_reference`` below is a different question and is not a
    substitute for this point: it drops the attacker tenants entirely, uses a
    disjoint seed block and a smaller seed count, so it cannot be plotted on the
    same axis.
    """
    rows = []
    # ``ratios`` lets a driver run the sweep one intensity at a
    # time on a shell with a wall-clock cap and concatenate; the seeds,
    # streams and policy order per intensity are unchanged.
    for ratio in (SWEEP_RATIOS(P) if ratios is None else ratios):
        spec = _spec(P, ratio)
        for i in range(P.fairness_seeds):
            seed = EVAL_SEED_BASE + i          # evaluation block
            stream = generate_stream(seed, spec)
            for policy in build_ladder():
                run = run_policy(policy, stream, seed,
                                 num_nodes=P.fairness_nodes,
                                 beta=P.fairness_beta)
                row = run.as_row()
                row["ratio"] = ratio
                rows.append(row)
        print(f"  intensity {ratio:g}:1 done")
    return pd.DataFrame(rows)


def SWEEP_RATIOS(P) -> tuple:
    """Configured intensities with the true 0:1 point prepended."""
    return (0.0,) + tuple(P.malicious_ratios)


def benign_reference(P) -> pd.DataFrame:
    """Behaviour with no adversary present *at all*, to show the adversarial-case
    result is not bought with benign-stream distortion.

    Distinct from the sweep's 0:1 point, which keeps the attacker tenants in the
    population and merely silences them.  Here they are removed outright and the
    demand shape is varied, which is a question about benign-stream distortion
    rather than about the flood axis.  Run at the full seed count -- it is
    deterministic and cheap, and a reference measured on fewer seeds than the
    thing it is a reference for is awkward to defend.
    """
    rows = []
    for name, kw in (("balanced", dict(demand_spread=0.0)),
                     ("heterogeneous", dict(demand_spread=0.7))):
        spec = StreamSpec(n_benign_tenants=P.fairness_benign_tenants,
                          n_attacker_tenants=0,
                          n_benign_requests=P.fairness_benign_requests,
                          malicious_ratio=0.0, **kw)
        for i in range(P.fairness_seeds):
            seed = EVAL_BENIGN_SEED_BASE + i
            stream = generate_stream(seed, spec)
            for policy in build_ladder():
                run = run_policy(policy, stream, seed,
                                 num_nodes=P.fairness_nodes,
                                 beta=P.fairness_beta)
                row = run.as_row()
                row["workload"] = name
                rows.append(row)
    return pd.DataFrame(rows)


SENS_METRICS = ("fi_benign", "fi_benign_norm", "benign_acceptance",
                "malicious_capacity_share", "dominant_utilisation")

# ---------------------------------------------------------------------------
# Selection criterion, stated before any result is computed.
#
# The shipped point (q_base=0.025, kappa=4) is a fixed engineering operating
# point, not an unconstrained optimum.  The full grid is published so readers
# can see the fairness/security/utilisation trade-off.  In particular, argmax
# Phi alone moves to the largest q_base tested and roughly doubles attacker
# capacity share, so the shipped point is not a global optimum and no single
# scalar metric prices the security objective.
SELECTION_METRIC = "fi_benign_norm"
SELECTION_RATIO_IS_HEADLINE = True

# kappa is swept past the incumbent and the obvious candidate so the chosen
# value is not at the grid edge. A parameter sitting on the boundary of its own
# sensitivity sweep has not been shown to be optimal -- only to be the largest
# value tried.
SENS_QBASE = (0.015, 0.020, 0.025, 0.030, 0.045)
SENS_KAPPAS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)


def sensitivity(P) -> pd.DataFrame:
    """Sweep the proposed policy's two tunables.

    Included because reporting a single tuned operating point is not
    falsifiable.  If the advantage exists only in a narrow corner of the
    parameter space, this table says so.

    Run at the full seed count and reported with paired confidence intervals
    against the shipped default, for a specific reason.  At 20 seeds and bare
    means, the default (kappa = 2) led kappa = 4 on ``fi_benign`` by 0.9% while
    giving up 6.5% of benign acceptance, 7.7% of attacker share and 8.3% of
    utilisation -- and there was no way to tell whether that 0.9% was a real
    ordering or seed noise.  Each cell is paired with the default on the *same*
    seeds and the same streams, so the difference has a CI and the question is
    answerable either way.  The experiment is deterministic and costs ~2
    minutes, so there is no reason to answer it at 20 seeds.

    ``fi_benign_norm`` is carried alongside ``fi_benign`` because it is the
    demand-normalised index implied by the utility Phi of the feasibility problem (see
    ``metrics``).  Selecting a default on ``fi_benign`` alone while the paper's
    stated objective is Phi is the kind of mismatch that only shows up when both
    are in the table.
    """
    rows = []
    spec = _spec(P, P.headline_ratio)
    # Development block: the grid that characterises the operating point runs
    # on seeds disjoint from every reported comparison (which use EVAL_*).
    seeds = list(range(DEV_SEED_BASE, DEV_SEED_BASE + P.fairness_seeds))
    streams = [generate_stream(s, spec) for s in seeds]

    per_cell: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for q in SENS_QBASE:
        for k in SENS_KAPPAS:
            pp = dataclasses.replace(DEFAULT_POLICY_PARAMS,
                                     p_base_quota_frac=q, p_pressure_kappa=k)
            runs = [run_policy(ProposedNSACQuota(pp), st, s,
                               num_nodes=P.fairness_nodes,
                               beta=P.fairness_beta)
                    for s, st in zip(seeds, streams)]
            per_cell[(q, k)] = {m: np.array([getattr(r, m) for r in runs],
                                            dtype=float)
                                for m in SENS_METRICS}
        print(f"  q_base {q:g} done")

    default_key = (DEFAULT_POLICY_PARAMS.p_base_quota_frac,
                   DEFAULT_POLICY_PARAMS.p_pressure_kappa)
    base = per_cell[default_key]

    for (q, k), vals in per_cell.items():
        row = dict(q_base=q, kappa=k, n_seeds=len(seeds),
                   is_default=((q, k) == default_key))
        for m in SENS_METRICS:
            mu, lo, hi = mean_ci(vals[m])
            row[m], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = mu, lo, hi
            # Paired against the default cell on identical seeds and streams:
            # a CI that spans zero means this cell is not distinguishable from
            # the shipped default on this metric, however the means order.
            d = vals[m] - base[m]
            dm, dlo, dhi = mean_ci(d)
            row[f"{m}_vs_default"] = dm
            row[f"{m}_vs_default_ci_lo"] = dlo
            row[f"{m}_vs_default_ci_hi"] = dhi
            row[f"{m}_differs"] = bool(dlo > 0.0 or dhi < 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def select_operating_point(sens: pd.DataFrame) -> dict:
    """Report the unconstrained Phi corner as a sensitivity diagnostic.

    The shipped parameters are fixed engineering settings, not the result of a
    scalar optimisation.  This function deliberately shows what *would* happen
    under unconstrained argmax Phi: the winner moves to a grid boundary and
    accepts substantially more attacker capacity.  It is a diagnostic against
    overclaiming optimality, not a parameter-selection rule.
    """
    best = sens.loc[sens[SELECTION_METRIC].idxmax()]
    shipped = sens[sens["is_default"]].iloc[0]
    at_kappa_edge = best["kappa"] in (min(SENS_KAPPAS), max(SENS_KAPPAS))
    at_q_edge = best["q_base"] in (min(SENS_QBASE), max(SENS_QBASE))
    return dict(
        criterion=f"diagnostic unconstrained argmax {SELECTION_METRIC} over "
                  f"the full (q_base, kappa) grid at the headline point; "
                  f"not used to select the shipped parameters",
        selected_q_base=float(best["q_base"]),
        selected_kappa=float(best["kappa"]),
        selected_phi=float(best[SELECTION_METRIC]),
        shipped_q_base=float(shipped["q_base"]),
        shipped_kappa=float(shipped["kappa"]),
        shipped_phi=float(shipped[SELECTION_METRIC]),
        phi_gain=float(best[SELECTION_METRIC] - shipped[SELECTION_METRIC]),
        selection_changes_default=bool(
            best["q_base"] != shipped["q_base"]
            or best["kappa"] != shipped["kappa"]),
        winner_on_kappa_boundary=bool(at_kappa_edge),
        winner_on_q_boundary=bool(at_q_edge),
        kappa_grid=list(SENS_KAPPAS),
        q_base_grid=list(SENS_QBASE))


def kappa_saturation(sens: pd.DataFrame) -> pd.DataFrame:
    """Where along q_base does kappa stop doing anything?

    At small q_base the pressure term is not the binding constraint -- the base
    quota is -- so cells that differ only in kappa return bit-identical results.
    A kappa optimum quoted without saying where the control is live is a claim
    about a parameter that, over part of the grid, has no effect at all.
    """
    rows = []
    for q, g in sens.groupby("q_base"):
        vals = g.sort_values("kappa")[SELECTION_METRIC].to_numpy()
        distinct = len(np.unique(np.round(vals, 10)))
        rows.append(dict(q_base=float(q), kappa_values=len(vals),
                         distinct_phi=distinct,
                         phi_spread=float(vals.max() - vals.min()),
                         kappa_is_live=bool(distinct > 1)))
    return pd.DataFrame(rows)


def invariance(P, points: dict) -> pd.DataFrame:
    """Behaviour across the whole flood axis at each candidate operating point.

    Reported as a measured outcome of the selection, never as an input to it.
    A higher pressure exponent reacts harder to load, so it is entirely possible
    for the better 10:1 point to have a steeper 0:1 -> 20:1 curve; if it does,
    that is a finding about the trade and the paper says so, rather than the
    criterion being quietly widened until the incumbent wins again.
    """
    rows = []
    for label, pp in points.items():
        for ratio in SWEEP_RATIOS(P):
            spec = _spec(P, ratio)
            runs = [run_policy(ProposedNSACQuota(pp),
                               generate_stream(EVAL_SEED_BASE + i, spec),
                               EVAL_SEED_BASE + i,
                               num_nodes=P.fairness_nodes,
                               beta=P.fairness_beta)
                    for i in range(P.fairness_seeds)]
            row = dict(point=label, q_base=pp.p_base_quota_frac,
                       kappa=pp.p_pressure_kappa, ratio=ratio)
            for m in SENS_METRICS:
                mu, lo, hi = mean_ci(np.array([getattr(r, m) for r in runs],
                                              dtype=float))
                row[m], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = mu, lo, hi
            rows.append(row)
        print(f"  invariance: {label} done")
    return pd.DataFrame(rows)


def invariance_summary(inv: pd.DataFrame) -> pd.DataFrame:
    """0:1 -> 20:1 change at each candidate point, per metric."""
    lo_r, hi_r = inv["ratio"].min(), inv["ratio"].max()
    rows = []
    for label, g in inv.groupby("point", sort=False):
        a = g[g["ratio"] == lo_r].iloc[0]
        b = g[g["ratio"] == hi_r].iloc[0]
        row = dict(point=label, q_base=a["q_base"], kappa=a["kappa"],
                   ratio_lo=lo_r, ratio_hi=hi_r)
        for m in SENS_METRICS:
            row[f"{m}_lo"] = a[m]
            row[f"{m}_hi"] = b[m]
            row[f"{m}_abs_change"] = b[m] - a[m]
            # attacker share is 0 at the 0:1 point by construction, so a
            # relative change is undefined there rather than infinite
            row[f"{m}_rel_pct"] = (100.0 * (b[m] / a[m] - 1.0)
                                   if a[m] not in (0.0,) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def isolation_sweep(P) -> pd.DataFrame:
    """Price the anti-co-location constraint of the placement rule.

    beta = 1 is strict anti-co-location; beta = 999 effectively disables the
    constraint.  The sweep measures what the constraint costs.
    """
    rows = []
    spec = _spec(P, P.headline_ratio)
    seeds = [EVAL_SEED_BASE + i for i in range(min(P.fairness_seeds, 20))]
    streams = [generate_stream(s, spec) for s in seeds]
    for beta in P.isolation_budgets:
        for cls in (UncontrolledFCFS, ProposedNSACQuota):
            runs = [run_policy(cls(), st, s, num_nodes=P.fairness_nodes,
                               beta=beta)
                    for s, st in zip(seeds, streams)]
            rows.append(dict(
                beta=beta, policy=runs[0].policy,
                fi_benign=float(np.mean([r.fi_benign for r in runs])),
                benign_acceptance=float(np.mean([r.benign_acceptance
                                                 for r in runs])),
                dominant_utilisation=float(np.mean([r.dominant_utilisation
                                                    for r in runs])),
                malicious_capacity_share=float(np.mean(
                    [r.malicious_capacity_share for r in runs]))))
    return pd.DataFrame(rows)


SLOT_CAPS = (1, 2, 3, 4, 6, 8, 10**9)


def slot_cap_sweep(P) -> pd.DataFrame:
    """Sensitivity of the shipped policy to the per-tenant active-allocation
    cap ``p_slot_cap``, on the development seed block, at the
    headline intensity.  ``10**9`` is effectively uncapped.  Paired against
    the shipped cap on identical seeds and streams."""
    rows = []
    spec = _spec(P, P.headline_ratio)
    seeds = list(range(DEV_SEED_BASE, DEV_SEED_BASE + P.fairness_seeds))
    streams = [generate_stream(s, spec) for s in seeds]
    per_cap = {}
    for cap in SLOT_CAPS:
        pp = dataclasses.replace(DEFAULT_POLICY_PARAMS, p_slot_cap=cap)
        runs = [run_policy(ProposedNSACQuota(pp), st, s,
                           num_nodes=P.fairness_nodes, beta=P.fairness_beta)
                for s, st in zip(seeds, streams)]
        per_cap[cap] = {m: np.array([getattr(r, m) for r in runs], dtype=float)
                        for m in SENS_METRICS}
    base = per_cap[DEFAULT_POLICY_PARAMS.p_slot_cap]
    for cap, vals in per_cap.items():
        row = dict(slot_cap=cap, n_seeds=len(seeds),
                   is_default=(cap == DEFAULT_POLICY_PARAMS.p_slot_cap))
        for m in SENS_METRICS:
            mu, lo, hi = mean_ci(vals[m])
            row[m], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = mu, lo, hi
            dm, dlo, dhi = mean_ci(vals[m] - base[m])
            row[f"{m}_vs_default"] = dm
            row[f"{m}_vs_default_ci_lo"] = dlo
            row[f"{m}_vs_default_ci_hi"] = dhi
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    banner(f"Experiment 4: policy ladder, {P.fairness_seeds} paired seeds x "
           f"{len(SWEEP_RATIOS(P))} intensities (incl. a true 0:1 point) "
           f"x 6 policies "
           f"({P.fairness_nodes} nodes, beta = {P.fairness_beta})")

    raw = sweep(P)
    save_table(raw, "exp4_fairness_raw")

    agg = []
    for (ratio, pol), g in raw.groupby(["ratio", "policy"], sort=False):
        d = dict(ratio=ratio, policy=pol, label=g["label"].iloc[0])
        for m in METRICS:
            mu, lo, hi = mean_ci(g[m].to_numpy())
            d[m], d[f"{m}_ci_lo"], d[f"{m}_ci_hi"] = mu, lo, hi
        agg.append(d)
    summary = pd.DataFrame(agg).sort_values(["ratio", "policy"])
    save_table(summary, "exp4_fairness_summary")

    head = summary[summary["ratio"] == P.headline_ratio].copy()
    cols = ["policy", "label", "fi_benign", "fi_benign_ci_lo",
            "fi_benign_ci_hi", "benign_acceptance", "malicious_rejection",
            "malicious_capacity_share", "dominant_utilisation", "decision_ms"]
    print(f"\nHeadline operating point "
          f"(malicious:benign = {P.headline_ratio:g}:1)")
    print(head[cols].to_string(index=False,
                               float_format=lambda v: f"{v:.4f}"))
    save_table(head, "exp4_headline_table5")

    hh = raw[raw["ratio"] == P.headline_ratio]
    p_rows = hh[hh["policy"] == "P"].sort_values("seed")
    tests = []
    for pol in ("B0", "B1", "B2", "B3", "B4"):
        b_rows = hh[hh["policy"] == pol].sort_values("seed")
        for m in ("fi_benign", "benign_acceptance",
                  "malicious_capacity_share", "decision_ms"):
            t = paired_comparison(p_rows[m].to_numpy(), b_rows[m].to_numpy(),
                                  label=f"P - {pol}")
            t["metric"] = m
            tests.append(t)
    tests_df = pd.DataFrame(tests)
    for m, g in tests_df.groupby("metric"):
        adj, rej = holm_bonferroni(g["p_raw"].to_numpy())
        tests_df.loc[g.index, "p_holm"] = adj
        tests_df.loc[g.index, "significant"] = rej
    save_table(tests_df, "exp4_paired_tests")
    print("\nPaired Wilcoxon (P minus each baseline), Holm-corrected per metric")
    print(tests_df[["metric", "comparison", "mean_diff", "p_raw", "p_holm",
                    "significant", "cliffs_delta", "magnitude"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ben = benign_reference(P)
    ben_summary = (ben.groupby(["workload", "policy"])[list(METRICS)]
                      .mean().reset_index())
    save_table(ben_summary, "exp4_benign_reference")
    print("\nBenign-workload reference (no adversary present)")
    print(ben_summary[["workload", "policy", "fi_benign", "fi_benign_norm",
                       "benign_acceptance", "dominant_utilisation"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sens = sensitivity(P)
    save_table(sens, "exp4_sensitivity")
    print(f"\nSensitivity of the proposed policy to its two tunables "
          f"({P.fairness_seeds} paired seeds)")
    print(sens[["q_base", "kappa", "is_default", "fi_benign",
                "fi_benign_norm", "benign_acceptance",
                "malicious_capacity_share", "dominant_utilisation"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  paired against the shipped default "
          f"(q_base = {DEFAULT_POLICY_PARAMS.p_base_quota_frac:g}, "
          f"kappa = {DEFAULT_POLICY_PARAMS.p_pressure_kappa:g}); "
          "'differs' = CI excludes zero")
    cmp_cols = ["q_base", "kappa"]
    for m in ("fi_benign", "fi_benign_norm"):
        cmp_cols += [f"{m}_vs_default", f"{m}_vs_default_ci_lo",
                     f"{m}_vs_default_ci_hi", f"{m}_differs"]
    print(sens[cmp_cols].to_string(index=False,
                                   float_format=lambda v: f"{v:+.4f}"))

    sat = kappa_saturation(sens)
    save_table(sat, "exp4_kappa_saturation")
    print("\nWhere the pressure term is live (Phi distinct across kappa?)")
    print(sat.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    dead = sat[~sat["kappa_is_live"]]
    if len(dead):
        print(f"  kappa has no effect at q_base in "
              f"{sorted(dead['q_base'].tolist())}: at a small base quota the "
              f"quota, not the pressure term, is the binding constraint")

    sel = select_operating_point(sens)
    save_json(sel, "exp4_selection")
    print(f"\nSensitivity diagnostic: {sel['criterion']}")
    print(f"  shipped  q_base = {sel['shipped_q_base']:g}, "
          f"kappa = {sel['shipped_kappa']:g}  -> Phi {sel['shipped_phi']:.4f}")
    print(f"  argmax   q_base = {sel['selected_q_base']:g}, "
          f"kappa = {sel['selected_kappa']:g}  -> Phi {sel['selected_phi']:.4f}"
          f"  ({sel['phi_gain']:+.4f})")
    if sel["winner_on_kappa_boundary"] or sel["winner_on_q_boundary"]:
        print("  WARNING: the winner sits on a grid boundary -- extend the "
              "sweep before quoting it as an optimum")

    # Invariance at the fixed shipped point and the unconstrained-Phi corner.
    # The latter is a stress-test comparator, not a recommended setting.
    cand = {}
    cand[f"shipped (q={sel['shipped_q_base']:g}, k={sel['shipped_kappa']:g})"] \
        = dataclasses.replace(DEFAULT_POLICY_PARAMS,
                              p_base_quota_frac=sel["shipped_q_base"],
                              p_pressure_kappa=sel["shipped_kappa"])
    if sel["selection_changes_default"]:
        cand[f"unconstrained-Phi (q={sel['selected_q_base']:g}, "
             f"k={sel['selected_kappa']:g})"] = dataclasses.replace(
                 DEFAULT_POLICY_PARAMS,
                 p_base_quota_frac=sel["selected_q_base"],
                 p_pressure_kappa=sel["selected_kappa"])
    inv = invariance(P, cand)
    save_table(inv, "exp4_invariance")
    inv_sum = invariance_summary(inv)
    save_table(inv_sum, "exp4_invariance_summary")
    print(f"\nInvariance across the flood axis "
          f"({inv_sum['ratio_lo'].iloc[0]:g}:1 -> "
          f"{inv_sum['ratio_hi'].iloc[0]:g}:1), measured at each candidate")
    print(inv_sum[["point", "fi_benign_norm_lo", "fi_benign_norm_hi",
                   "fi_benign_norm_rel_pct", "benign_acceptance_lo",
                   "benign_acceptance_hi", "benign_acceptance_rel_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    iso = isolation_sweep(P)
    save_table(iso, "exp4_isolation_budget")
    print("\nCost of the anti-co-location budget beta (placement rule)")
    print(iso.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sc = slot_cap_sweep(P)
    save_table(sc, "exp4_slot_cap")
    print("\nSensitivity to the per-tenant active-allocation cap")
    print(sc[["slot_cap", "fi_benign", "benign_acceptance",
              "malicious_capacity_share", "dominant_utilisation"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib.pyplot as plt
    # Distinct (colour, marker, linestyle) per policy so the six series are
    # separable in grayscale; the proposed policy is drawn bold black.
    _STYLE = {
        "B0": dict(color="C0", marker="o", ls="-"),
        "B1": dict(color="C1", marker="s", ls="--"),
        "B2": dict(color="C2", marker="^", ls="-."),
        "B3": dict(color="C3", marker="D", ls=":"),
        "B4": dict(color="C4", marker="v", ls=(0, (3, 1, 1, 1))),
        "P":  dict(color="k",  marker="*", ls="-"),
        "PS": dict(color="0.45", marker="P", ls="--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for pol, g in summary.groupby("policy", sort=False):
        g = g.sort_values("ratio")
        st = _STYLE.get(pol, dict(marker="o", ls="-"))
        big = pol == "P"
        lbl = f"{pol} {g['label'].iloc[0]}"
        axes[0].plot(g["ratio"], g["fi_benign"], ms=6 if big else 4,
                     lw=2.0 if big else 1.3, label=lbl, **st)
        axes[1].plot(g["ratio"], g["malicious_capacity_share"] * 100,
                     ms=6 if big else 4, lw=2.0 if big else 1.3, label=pol, **st)
        axes[2].plot(g["ratio"], g["benign_acceptance"] * 100,
                     ms=6 if big else 4, lw=2.0 if big else 1.3, label=pol, **st)
    ratios = SWEEP_RATIOS(P)
    for ax, ylab, title in zip(
            axes,
            ["Jain's FI (benign tenants)", "Attacker capacity share (%)",
             "Benign acceptance (%)"],
            ["Fairness across flood intensity",
             "Capacity captured by attackers",
             "Service delivered to legitimate tenants"]):
        # symlog, not log: the axis now starts at a genuine zero-attack point
        # and a log axis cannot represent it. Linear below 1:1, log above, so
        # the whole sweep is one continuous axis instead of a plot that begins
        # after the attack has already started.
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xticks(ratios,
                      ["0" if r == 0 else f"{int(r)}:1" for r in ratios])
        ax.set_xlabel("malicious : benign request ratio")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=6.5, loc="lower left")
    save_fig(fig, "fig6_fairness_sweep")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for _, r in head.iterrows():
        ax.scatter(r["decision_ms"], r["fi_benign"], s=45)
        ax.annotate(r["policy"], (r["decision_ms"], r["fi_benign"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Per-request decision cost (ms, log scale)")
    ax.set_ylabel("Jain's FI (benign tenants)")
    ax.set_title(f"Fairness vs decision cost at {P.headline_ratio:g}:1")
    save_fig(fig, "fig7_fairness_vs_cost")

    save_json({"headline_ratio": P.headline_ratio,
               "deployment": {"nodes": P.fairness_nodes,
                              "beta": P.fairness_beta},
               "headline": head.to_dict("records"),
               "paired_tests": tests_df.to_dict("records")},
              "exp4_fairness")


if __name__ == "__main__":
    main()
