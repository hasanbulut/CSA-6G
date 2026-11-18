"""Experiment 4b -- robustness of the allocation comparison.

Four parts, all on the experiment-4 harness (24 nodes, beta = 3):

R. Recovery.  A congestion burst (every benign tenant submits its requests
   back to back) followed by a quiet phase with expiring leases.  Counts benign
   tenants that are never admitted again in the quiet phase even though
   capacity is free -- the starvation admission-only AIMD recovery allows --
   for the admission-only rule, the shipped rule, and the baselines.  Optionally with a continuing
   10:1 flood in the quiet phase, to check the fix does not free attackers.

A. Ablation of P at 10:1: each mechanism switched off in turn.

T. Equal tuning.  B1, B2 and P each get a parameter grid on the development
   seeds (0-99) and the same selection rule -- maximise benign Jain FI subject
   to attacker capacity share <= 15 % -- then the selected point is evaluated
   on the evaluation seeds (100000-100099).  B0, B3 and B4 have no tunables.
   Every grid point is exported (Pareto frontier).

X. Adaptive attackers at 10:1, all policies: (paced) each attacker identity
   self-limits to the rate gate's 2 requests per 64 arrivals; (sybil) each
   attacker splits into 5 identities; (paced+sybil) both.

All per-seed rows are exported as raw CSVs.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g.harness import make_pool, run_policy
from csa6g.metrics import (acceptance_rate, fairness_summary,
                           malicious_capacity_share)
from csa6g.model import Reason, SliceType, demand_norm
from csa6g.params import DEFAULT_POLICY_PARAMS as D
from csa6g.placement import select_node
from csa6g.policies import (POLICY_LADDER, DominantResourceFairness,
                            ProposedNSACQuota, StaticEqualCap, TokenBucket,
                            WithProfileCheck, build_ladder)
from csa6g.workload import StreamSpec, generate_stream

NODES, BETA = 24, 3
from csa6g.params import ADMISSION_ONLY_RECOVERY_PARAMS as ADMIT_ONLY
DEV = range(0, 100)
EVAL = range(100000, 100100)


def _metrics(run) -> dict:
    return dict(fi_benign=run.fi_benign,
                benign_acceptance=run.benign_acceptance,
                attacker_share=run.malicious_capacity_share,
                attacker_principal_share=run.attacker_principal_share,
                capture_ratio=run.capture_ratio,
                decision_ms=run.decision_ms)


def _reindex(stream):
    return [dataclasses.replace(r, arrival_index=i)
            for i, r in enumerate(stream)]


# ------------------------------------------------------------ R: recovery --

def recovery_stream(seed: int, attack: bool):
    """Phase 1 (arrivals 0..999): 50 benign tenants, 20 requests each, in
    per-tenant bursts.  Phase 2: each tenant 10 requests spread over 5000
    arrival units (one request every 5 units), optionally interleaved with a
    10:1 attacker flood from 10 attacker identities."""
    base = generate_stream(seed, StreamSpec(
        n_benign_tenants=50, n_attacker_tenants=10 if attack else 0,
        n_benign_requests=1500,
        malicious_ratio=(10.0 * 1000 / 1500) if attack else 0.0))
    rng = np.random.default_rng(seed + 7)
    ben = [r for r in base if not r.malicious]
    mal = [r for r in base if r.malicious]
    by_t: dict[int, list] = {}
    for r in ben:
        by_t.setdefault(r.tenant_id, []).append(r)
    p1, p2 = [], []
    tenants = list(by_t)
    rng.shuffle(tenants)
    for t in tenants:                       # bursts, tenant after tenant
        p1 += by_t[t][:20]
    rest = [r for t in tenants for r in by_t[t][20:30]]
    rng.shuffle(rest)
    out = [dataclasses.replace(r, arrival_index=i) for i, r in enumerate(p1)]
    t0 = len(out)
    ticks = t0 + 5 * np.arange(len(rest))
    events = [(int(k), 0, r) for k, r in zip(ticks, rest)]
    if attack and mal:
        mt = np.sort(rng.integers(t0, int(ticks[-1]) + 1, size=len(mal)))
        events += [(int(k), 1, r) for k, r in zip(mt, mal)]
    events.sort(key=lambda e: (e[0], e[1]))
    out += [dataclasses.replace(r, arrival_index=k) for k, _, r in events]
    return out, t0


def run_with_leases(policy, stream, lease: int, phase2_start: int):
    """The harness loop of experiment 4 plus lease expiry: an admitted
    allocation is released ``lease`` arrival units after admission."""
    pool = make_pool(NODES)
    policy.reset(pool, BETA)
    policy.prepare(stream)
    total = pool.total_tup
    accepted = np.zeros(len(stream), bool)
    live: list[tuple[int, object, object]] = []   # (expiry, node, req)
    active: dict[int, int] = {}
    for i, req in enumerate(stream):
        now = req.arrival_index
        keep = []
        for exp, node, r in live:
            if exp <= now:
                node.free(r.demand, r.slice_type)
                policy.on_release(r.account_id, r.demand,
                                  demand_norm(r.demand, total),
                                  slice_type=r.slice_type)
                active[r.account_id] -= 1
                if not active[r.account_id]:
                    del active[r.account_id]
            else:
                keep.append((exp, node, r))
        live = keep
        cost = demand_norm(req.demand, total)
        reason = policy.admit(req, cost, pool.dominant_free_fraction(),
                              max(1, len(active)))
        if reason != Reason.OK:
            policy.on_deny(req, reason)
            continue
        node, pr = select_node(pool.nodes, req.demand, req.slice_type, BETA)
        if node is None:
            policy.on_deny(req, pr)
            continue
        node.allocate(req.demand, req.slice_type)
        policy.on_admit(req, cost)
        active[req.account_id] = active.get(req.account_id, 0) + 1
        live.append((now + lease, node, req))
        accepted[i] = True
    idx = np.array([r.arrival_index >= phase2_start for r in stream])
    mal = np.array([r.malicious for r in stream])
    ben2 = idx & ~mal
    ten2 = {}
    for r, a, b in zip(stream, accepted, ben2):
        if b:
            ten2[r.tenant_id] = ten2.get(r.tenant_id, 0) + int(a)
    sub = [r for r, k in zip(stream, idx) if k]
    return dict(
        phase2_benign_acceptance=acceptance_rate(accepted, ben2),
        starved_tenants=int(sum(v == 0 for v in ten2.values())),
        phase2_tenants=len(ten2),
        phase2_attacker_share=(malicious_capacity_share(
            sub, accepted[idx], total) if mal.any() else 0.0))


def part_recovery(seeds, lease=200):
    variants = {"P (v6 rule)": lambda: ProposedNSACQuota(ADMIT_ONLY),
                "P (v7 rule)": lambda: ProposedNSACQuota(D)}
    for cls in POLICY_LADDER:
        if cls.name in ("B1", "B2", "B4"):
            variants[cls.name] = (lambda c=cls: c(D))
    rows = []
    for attack in (False, True):
        for seed in seeds:
            stream, t0 = recovery_stream(seed, attack)
            for name, mk in variants.items():
                rows.append(dict(attack=attack, seed=seed, policy=name,
                                 **run_with_leases(mk(), stream, lease, t0)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ A: ablation --

_QUOTA_ONLY = dict(p_use_aimd=False, p_use_pressure=False,
                   p_use_rate_gate=False, p_slot_mode="none", p_profile_z=0.0)
ABLATIONS = {
    "P (full, v7)": D,
    "P without profile check": dataclasses.replace(D, p_profile_z=0.0),
    "P without AIMD": dataclasses.replace(D, p_use_aimd=False),
    "P without rate gate": dataclasses.replace(D, p_use_rate_gate=False),
    "P without slot cap": dataclasses.replace(D, p_slot_mode="none"),
    "P with pressure cap (v6 term)": dataclasses.replace(D,
                                                        p_use_pressure=True),
    "Quota only": dataclasses.replace(D, **_QUOTA_ONLY),
    "Quota + profile check only": dataclasses.replace(
        D, **dict(_QUOTA_ONLY, p_profile_z=3.0)),
    "P (v6 rules)": ADMIT_ONLY,
}


def part_ablation(seeds, ratio=10.0):
    """Run under the greedy (2x-inflated) flood and under the in-profile
    flood: against the greedy flood the profile check rejects nearly every
    attacker request and masks the other mechanisms."""
    rows = []
    for seed in seeds:
        base = generate_stream(seed, StreamSpec(malicious_ratio=ratio))
        for attack, s in (("greedy flood", base),
                          ("in-profile flood", in_profile(base, seed))):
            for name, p in ABLATIONS.items():
                rows.append(dict(attack=attack, variant=name, seed=seed,
                                 **_metrics(run_policy(ProposedNSACQuota(p),
                                                       s, seed, NODES,
                                                       BETA))))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- T: tuning --

GRIDS = {
    "B1": [dict(b1_max_tenants=n)
           for n in (10, 20, 30, 40, 60, 80, 120, 200)],
    "B2": [dict(b2_rate=r, b2_burst=b)
           for r in (5e-6, 1e-5, 2e-5, 4e-5, 8e-5)
           for b in (0.005, 0.01, 0.02, 0.04)],
    # The pressure term is off, so kappa is inert; P's grid is q_base x
    # the request-index gate limit (20 points, as B2).
    "P": [dict(p_base_quota_frac=q, p_rate_max=m)
          for q in (0.015, 0.020, 0.025, 0.030, 0.045)
          for m in (1, 2, 3, 4)],
}
CLS = {"B1": StaticEqualCap, "B2": TokenBucket, "P": ProposedNSACQuota}
SHARE_LIMIT = 0.15


def part_tuning(dev, ev, ratio=10.0):
    grid_rows = []
    streams = {s: generate_stream(s, StreamSpec(malicious_ratio=ratio))
               for s in dev}
    for pol, grid in GRIDS.items():
        for gi, kw in enumerate(grid):
            p = dataclasses.replace(D, **kw)
            m = [_metrics(run_policy(CLS[pol](p), streams[s], s, NODES,
                                     BETA)) for s in dev]
            grid_rows.append(dict(policy=pol, point=gi, params=str(kw),
                                  **pd.DataFrame(m).mean().to_dict()))
    g = pd.DataFrame(grid_rows)
    chosen = {}
    for pol in GRIDS:
        sub = g[g.policy == pol]
        ok = sub[sub.attacker_share <= SHARE_LIMIT]
        pick = (ok if len(ok) else sub).sort_values("fi_benign").iloc[-1]
        chosen[pol] = dict(GRIDS[pol][int(pick.point)],
                           feasible=bool(len(ok)))
    ev_rows = []
    for s in ev:
        st = generate_stream(s, StreamSpec(malicious_ratio=ratio))
        for cls in POLICY_LADDER:
            if cls.name == "PS":
                continue
            if cls.name in chosen:
                kw = {k: v for k, v in chosen[cls.name].items()
                      if k != "feasible"}
                pol = cls(dataclasses.replace(D, **kw))
            else:
                pol = cls(D)
            ev_rows.append(dict(policy=cls.name, seed=s,
                                **_metrics(run_policy(pol, st, s, NODES,
                                                      BETA))))
    return g, chosen, pd.DataFrame(ev_rows)


# ------------------------------------------------- X: adaptive attackers --

def paced(stream, window=64, k=2):
    """Each attacker identity submits at most k requests in any window of
    ``window`` arrivals (it self-limits to the rate gate); benign unchanged."""
    out, hist = [], {}
    for r in stream:
        if r.malicious:
            h = [a for a in hist.get(r.tenant_id, []) if a > len(out) - window]
            if len(h) >= k:
                hist[r.tenant_id] = h
                continue
            h.append(len(out))
            hist[r.tenant_id] = h
        out.append(r)
    return _reindex(out)


def split(stream, k=5, independent=True, base=1000):
    """Each attacker splits its requests over k identities (round robin).

    independent=True: k separately registered principals (k credentials).
    independent=False: k sessions under the attacker's one principal -- what
    an attacker holding a single subscription can do, since the E-AP charges
    quota to the tenant bound at registration."""
    cnt, out = {}, []
    for r in stream:
        if r.malicious:
            c = cnt.get(r.tenant_id, 0)
            cnt[r.tenant_id] = c + 1
            new = base + r.tenant_id * k + c % k
            r = dataclasses.replace(
                r, tenant_id=new,
                principal=new if independent else r.tenant_id)
        out.append(r)
    return out


def in_profile(stream, seed):
    """Attackers stop inflating: demand is redrawn at the benign profile
    (same slice type, inflation 1.0), so every request passes the check."""
    from csa6g.workload import sample_demand
    rng = np.random.default_rng(seed + 11)
    return [dataclasses.replace(r, demand=sample_demand(
                rng, r.slice_type, r.entitlement_scale))
            if r.malicious else r for r in stream]


ATTACKS = {
    "greedy flood": lambda s, seed: s,
    "paced": lambda s, seed: paced(s),
    "in-profile": in_profile,
    "paced + in-profile": lambda s, seed: paced(in_profile(s, seed)),
    # Identity attacks use in-profile demand so the profile check cannot
    # mask them: what is measured is the effect of extra identities alone.
    "in-profile, 5 sessions, 1 credential":
        lambda s, seed: split(in_profile(s, seed), 5, False),
    "in-profile, 2 credentials":
        lambda s, seed: split(in_profile(s, seed), 2, True),
    "in-profile, 5 credentials":
        lambda s, seed: split(in_profile(s, seed), 5, True),
    "in-profile, 10 credentials":
        lambda s, seed: split(in_profile(s, seed), 10, True),
}


def part_adaptive(seeds, ratio=10.0):
    rows = []
    for seed in seeds:
        base = generate_stream(seed, StreamSpec(malicious_ratio=ratio))
        for aname, fn in ATTACKS.items():
            s = fn(base, seed)
            nmal = sum(r.malicious for r in s)
            ladder = build_ladder(D) + [
                WithProfileCheck(StaticEqualCap(D)),
                WithProfileCheck(DominantResourceFairness(D))]
            for pol in ladder:
                rows.append(dict(attack=aname, policy=pol.name, seed=seed,
                                 attacker_requests=nmal,
                                 **_metrics(run_policy(pol, s, seed, NODES,
                                                       BETA))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main --

def _ci(df, by, cols):
    g = df.groupby(by)[cols]
    m, s, n = g.mean(), g.std(ddof=1), g.count()
    from scipy.stats import t
    h = s / np.sqrt(n) * t.ppf(0.975, n - 1)
    out = m.copy()
    for c in cols:
        out[c + "_ci"] = h[c]
    return out.reset_index()


def main() -> None:
    q = quick_flag()
    dev = range(0, 10) if q else DEV
    ev = range(100000, 100010) if q else EVAL
    banner(f"Experiment 4b: robustness ({len(ev)} evaluation seeds)")

    rec = part_recovery(ev)
    save_table(rec, "exp4b_recovery_raw")
    rs = _ci(rec, ["attack", "policy"], ["phase2_benign_acceptance",
                                        "starved_tenants",
                                        "phase2_attacker_share"])
    save_table(rs, "exp4b_recovery_summary")
    print("\nR. recovery\n", rs.to_string(index=False,
                                         float_format=lambda v: f"{v:.3f}"))

    ab = part_ablation(ev)
    save_table(ab, "exp4b_ablation_raw")
    abs_ = _ci(ab, ["attack", "variant"], ["fi_benign", "benign_acceptance",
                                 "attacker_share", "capture_ratio",
                                 "decision_ms"])
    save_table(abs_, "exp4b_ablation_summary")
    print("\nA. ablation\n", abs_.to_string(index=False,
                                           float_format=lambda v: f"{v:.4f}"))

    grid, chosen, tun = part_tuning(dev, ev)
    save_table(grid, "exp4b_tuning_grid_dev")
    save_table(tun, "exp4b_tuned_raw")
    ts = _ci(tun, ["policy"], ["fi_benign", "benign_acceptance",
                               "attacker_share", "capture_ratio",
                               "decision_ms"])
    save_table(ts, "exp4b_tuned_summary")
    print("\nT. chosen points:", chosen)
    print(ts.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ad = part_adaptive(ev)
    save_table(ad, "exp4b_adaptive_raw")
    ads = _ci(ad, ["attack", "policy"], ["fi_benign", "benign_acceptance",
                                         "attacker_share",
                                         "attacker_principal_share",
                                         "capture_ratio"])
    save_table(ads, "exp4b_adaptive_summary")
    print("\nX. adaptive attackers\n",
          ads.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = {"B1": "#3b6ea5", "B2": "#c08a2b", "P": "#2e8b57"}
    for pol, c in colors.items():
        sub = grid[grid.policy == pol]
        ax.scatter(100 * sub.attacker_share, 100 * sub.benign_acceptance,
                   s=14, color=c, alpha=0.7, label=f"{pol} grid")
    for pol, mk in (("B0", "x"), ("B3", "s"), ("B4", "D")):
        r = ts[ts.policy == pol]
        ax.scatter(100 * r.attacker_share, 100 * r.benign_acceptance,
                   marker=mk, color="k", s=28, label=pol)
    ax.axvline(100 * SHARE_LIMIT, ls=":", color="grey", lw=1)
    ax.set_xlabel("Attacker capacity share (%)")
    ax.set_ylabel("Benign acceptance (%)")
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, "fig_tuning_frontier")

    save_json({"chosen": chosen, "share_limit": SHARE_LIMIT,
               "dev_seeds": [min(dev), max(dev)],
               "eval_seeds": [min(ev), max(ev)],
               "recovery_lease_units": 200}, "exp4b_robustness")


if __name__ == "__main__":
    main()
