"""Evaluation metrics.

The fairness definition matters more than anything else in the fairness experiment, so it
is stated explicitly rather than left to the reader.

Jain's index over an allocation vector x is

    J(x) = (sum x_i)^2 / (n * sum x_i^2)

but *over which tenants* is a modelling choice that changes the answer
completely.  Computed over all tenants, a policy that correctly rejects an
attacker is punished for it (the attacker's allocation is zero, so the vector
becomes less equal).  We therefore report:

  ``fi_benign``      Jain's index over benign tenants' granted shares.  This is
                     the headline number: fairness *among legitimate tenants*,
                     which is what an operator cares about under attack.
  ``fi_all``         Jain's index over every tenant.  Reported alongside for
                     transparency; it is not the claim.
  ``fi_benign_norm`` Jain's index over x_u / d_u, the demand-normalised form
                     implied by the utility Phi in the feasibility problem.

A comparison that reports only one of these is not falsifiable.  All three are
written to the results CSV.
"""
from __future__ import annotations

import numpy as np


def jain(x) -> float:
    """Jain's fairness index.  Returns NaN on a degenerate vector.

    J(x) is 0/0 when nothing was allocated, and the convenient convention --
    return 1.0 -- awards the *best possible* fairness score to a policy that
    granted nothing at all.  Under a heavy flood that is precisely the failure
    mode the metric has to be able to see, so the degenerate case returns NaN
    and propagates visibly into any aggregate instead of flattering the run.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    s = x.sum()
    if s <= 0:
        return float("nan")
    return float(s * s / (x.size * float((x * x).sum())))


def acceptance_rate(accepted: np.ndarray, mask: np.ndarray) -> float:
    sel = accepted[mask]
    return float(sel.mean()) if sel.size else float("nan")


def rejection_rate(accepted: np.ndarray, mask: np.ndarray) -> float:
    sel = accepted[mask]
    return float(1.0 - sel.mean()) if sel.size else float("nan")


def per_tenant_allocation(stream, accepted, total) -> dict[int, float]:
    """Aggregate-normalised granted share per tenant."""
    from .model import demand_norm
    out: dict[int, float] = {}
    for req, ok in zip(stream, accepted):
        out.setdefault(req.tenant_id, 0.0)
        if ok:
            out[req.tenant_id] += demand_norm(req.demand, total)
    return out


def per_tenant_demand(stream, total) -> dict[int, float]:
    from .model import demand_norm
    out: dict[int, float] = {}
    for req in stream:
        out[req.tenant_id] = out.get(req.tenant_id, 0.0) + demand_norm(
            req.demand, total)
    return out


def attacker_principal_share(stream) -> float:
    """Fraction of submitting principals that are attackers -- the share a
    per-principal fair allocator would give them.  Capacity share divided by
    this is the capture ratio: < 1 means attackers get less than their fair
    share per principal."""
    p = {r.account_id: r.malicious for r in stream}
    return float(sum(p.values()) / len(p)) if p else 0.0


def malicious_capacity_share(stream, accepted, total) -> float:
    """Fraction of *granted* capacity captured by attackers.

    This, not the malicious rejection rate, is the meaningful DoS metric under
    heavy flooding.  When the offered malicious load is ten or twenty times
    capacity, even a policy with no defence at all rejects ~96% of malicious
    requests simply because it runs out of resources, so a high rejection rate
    is mostly a saturation artefact.  What distinguishes a resilient policy is
    how much of the pool the attackers actually walk away with.
    """
    from .model import demand_norm
    mal = ben = 0.0
    for req, ok in zip(stream, accepted):
        if not ok:
            continue
        c = demand_norm(req.demand, total)
        if req.malicious:
            mal += c
        else:
            ben += c
    tot = mal + ben
    return float(mal / tot) if tot > 0 else 0.0


def fairness_summary(stream, accepted, total) -> dict[str, float]:
    alloc = per_tenant_allocation(stream, accepted, total)
    dem = per_tenant_demand(stream, total)
    malicious = {r.tenant_id for r in stream if r.malicious}
    benign = [t for t in alloc if t not in malicious]

    x_all = np.array([alloc[t] for t in sorted(alloc)], dtype=float)
    x_ben = np.array([alloc[t] for t in sorted(benign)], dtype=float)
    x_ben_norm = np.array([alloc[t] / max(dem[t], 1e-12)
                           for t in sorted(benign)], dtype=float)
    return {
        "fi_benign": jain(x_ben),
        "fi_all": jain(x_all),
        "fi_benign_norm": jain(x_ben_norm),
    }
