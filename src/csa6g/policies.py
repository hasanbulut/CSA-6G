"""Allocation policy ladder for the comparative evaluation (the fairness experiment).

Six interchangeable policies over one shared admission harness.  They differ
*only* in the admission/allocation rule; the request stream, seed set, resource
pool and the isolation constraint of the placement rule are held identical, so per-seed
comparisons are paired.

    B0  uncontrolled FCFS to capacity, no per-tenant limit
    B1  static equal per-tenant cap, q = R / n_max
    B2  per-tenant token-bucket admission limiting (rate rho, burst sigma)
    B3  max-min fair allocation by progressive fill (aggregate-resource form
        of the feasibility problem)
    B4  Dominant Resource Fairness over the CPU/memory/storage profiles
    P   the proposed NSAC-style quota with AIMD dynamic caps

B3 and B4 are *offline* allocators: progressive fill and DRF both need the full
demand set before they can equalise anything.  Treating them as online
per-request rules would handicap them, so they are given the whole stream up
front in ``prepare()``, the cost of which is measured and amortised across the
requests it serves.  This is deliberately generous to the baselines -- it is
the reason their per-request decision cost is reported separately from P's.
"""
from __future__ import annotations

import numpy as np

from .model import (AdmissionRequest, Demand, Reason, ResourcePool,
                    demand_norm, slot_share)
from .nsac import NSACController
from .params import COLOCATION_BUDGET_BETA, DEFAULT_POLICY_PARAMS, PolicyParams

def within_profile(req: AdmissionRequest, z: float) -> bool:
    """True if every demand dimension is within the tenant's entitlement:
    (mean + z*sd of its slice type's declared profile) x entitlement_scale.
    Storage profile is in GB."""
    from .params import SLICE_PROFILES, STORAGE_GB_PER_TB
    p = SLICE_PROFILES[req.slice_type.label]
    d, e = req.demand, req.entitlement_scale
    return (d.cpu <= e * (p["cpu"][0] + z * p["cpu"][1])
            and d.mem <= e * (p["mem"][0] + z * p["mem"][1])
            and d.storage * STORAGE_GB_PER_TB
            <= e * (p["storage"][0] + z * p["storage"][1]))


# Denials produced by placement (placement rule), not by the policy itself.
PLACEMENT_REASONS = frozenset({Reason.ISOLATION, Reason.INFEASIBLE})


class AllocationPolicy:
    """Interface shared by every policy under test."""

    name = "base"
    label = "base"

    def __init__(self, params: PolicyParams = DEFAULT_POLICY_PARAMS):
        self.p = params
        self.prepare_seconds = 0.0

    def reset(self, pool: ResourcePool, beta: int = COLOCATION_BUDGET_BETA
              ) -> None:
        self.pool = pool
        self.total = pool.total_tup
        self.beta = beta
        self.prepare_seconds = 0.0

    def prepare(self, stream: list[AdmissionRequest]) -> None:
        """Offline pre-computation. Timed and amortised by the driver."""
        return None

    def admit(self, req: AdmissionRequest, cost: float,
              free_frac: float, active: int) -> str:
        raise NotImplementedError

    def on_admit(self, req: AdmissionRequest, cost: float) -> None:
        return None

    def on_deny(self, req: AdmissionRequest, reason: str | None = None
                ) -> None:
        """``reason`` is the denial's internal code; a policy may treat
        placement failures (congestion) differently from its own denials."""
        return None

    def on_release(self, tenant_id: int, demand: Demand, cost: float,
                   slice_type=None) -> None:
        """Return a released slice's budget to its tenant.

        Slice lifetimes are not modelled in the offline harness -- experiment 4
        holds every admitted slice for the whole stream, so nothing there ever
        releases.  The E-AP does release (Step 6), and without this hook a
        tenant's consumed quota only ever grew: a long-lived session would
        starve its own tenant after enough admit/release cycles even though the
        resources had gone back to the pool.  Policies whose state is purely
        per-arrival (B0, B2) have nothing to give back.
        """
        return None


# ------------------------------------------------------------------- B0 -----

class UncontrolledFCFS(AllocationPolicy):
    """No per-tenant limit; first come, first served until capacity runs out."""
    name = "B0"
    label = "Uncontrolled (FCFS)"

    def admit(self, req, cost, free_frac, active):
        return Reason.OK


# ------------------------------------------------------------------- B1 -----

class StaticEqualCap(AllocationPolicy):
    """Fixed per-tenant cap q = R / n_max, independent of load."""
    name = "B1"
    label = "Static equal cap"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.cap = 1.0 / float(self.p.b1_max_tenants)
        self.used: dict[int, float] = {}

    def admit(self, req, cost, free_frac, active):
        if self.used.get(req.account_id, 0.0) + cost > self.cap:
            return Reason.QUOTA
        return Reason.OK

    def on_admit(self, req, cost):
        self.used[req.account_id] = self.used.get(req.account_id, 0.0) + cost

    def on_release(self, tenant_id, demand, cost, slice_type=None):
        self.used[tenant_id] = max(0.0, self.used.get(tenant_id, 0.0) - cost)


# ------------------------------------------------------------------- B2 -----

class TokenBucket(AllocationPolicy):
    """Per-tenant token bucket: tokens accrue at rho per arrival tick, capped
    at burst sigma; a request costs its aggregate-normalised demand."""
    name = "B2"
    label = "Token bucket"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.tokens: dict[int, float] = {}
        self.last_tick: dict[int, int] = {}

    def _refill(self, tenant_id: int, tick: int) -> float:
        last = self.last_tick.get(tenant_id, 0)
        tok = self.tokens.get(tenant_id, self.p.b2_burst)
        tok = min(self.p.b2_burst, tok + self.p.b2_rate * max(0, tick - last))
        self.tokens[tenant_id] = tok
        self.last_tick[tenant_id] = tick
        return tok

    def admit(self, req, cost, free_frac, active):
        tok = self._refill(req.account_id, req.arrival_index)
        return Reason.OK if tok >= cost else Reason.RATE

    def on_admit(self, req, cost):
        self.tokens[req.account_id] = self.tokens.get(
            req.account_id, self.p.b2_burst) - cost


# ------------------------------------------------------------------- B3 -----

def progressive_fill(demands: np.ndarray, capacity: float) -> np.ndarray:
    """Classic max-min fair allocation by progressive fill.

    ``demands[i]`` is tenant i's total requested aggregate share; ``capacity``
    is the total available share (1.0 for a full pool).  Returns the max-min
    fair allocation vector.
    """
    n = len(demands)
    alloc = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    remaining = float(capacity)
    while active.any() and remaining > 1e-12:
        share = remaining / active.sum()
        # Tenants whose residual demand is below the equal share are satisfied.
        residual = demands - alloc
        satisfied = active & (residual <= share + 1e-12)
        if satisfied.any():
            alloc[satisfied] = demands[satisfied]
            remaining -= residual[satisfied].sum()
            active &= ~satisfied
        else:
            alloc[active] += share
            remaining = 0.0
    return alloc


class MaxMinFair(AllocationPolicy):
    """Offline max-min fair budgets, enforced online per tenant.

    Scalarised, not multi-resource.  Each tenant's demand is collapsed to the
    sum of its per-request dominant shares and progressive fill runs against a
    scalar capacity of 1.0.  Summing dominant shares is not a valid capacity
    constraint -- it double-counts whichever resource happens to dominate each
    request and ignores slack on the others -- so B3 is an aggregate
    approximation of max-min fairness, not the real thing.

    B3 is therefore an approximation on this workload.
    :class:`DominantResourceFairness` is the exact multi-resource comparator -- it allocates over the CPU/memory/storage
    vector against the real per-resource capacities -- and B4, not B3, is the
    baseline the proposed policy should be judged against.
    """
    name = "B3"
    label = "Max-min fair"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.budget: dict[int, float] = {}
        self.used: dict[int, float] = {}

    def prepare(self, stream):
        tenants = sorted({r.account_id for r in stream})
        idx = {t: i for i, t in enumerate(tenants)}
        d = np.zeros(len(tenants), dtype=float)
        for r in stream:
            d[idx[r.account_id]] += demand_norm(r.demand, self.total)
        alloc = progressive_fill(d, 1.0)
        self.budget = {t: float(alloc[idx[t]]) for t in tenants}

    def admit(self, req, cost, free_frac, active):
        b = self.budget.get(req.account_id, 0.0)
        if self.used.get(req.account_id, 0.0) + cost > b + 1e-12:
            return Reason.QUOTA
        return Reason.OK

    def on_admit(self, req, cost):
        self.used[req.account_id] = self.used.get(req.account_id, 0.0) + cost

    def on_release(self, tenant_id, demand, cost, slice_type=None):
        self.used[tenant_id] = max(0.0, self.used.get(tenant_id, 0.0) - cost)


# ------------------------------------------------------------------- B4 -----

def drf_shares(demand_vectors: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Dominant Resource Fairness (Ghodsi et al., NSDI 2011).

    ``demand_vectors[i]`` is tenant i's total demand across the resource
    dimensions.  Returns a per-tenant scale factor in [0, 1]: tenant i is
    granted ``scale[i] * demand_vectors[i]``.  Dominant shares are equalised
    subject to every resource staying within capacity, with tenants whose
    demand is exhausted dropping out (water-filling on dominant share).
    """
    n = demand_vectors.shape[0]
    frac = demand_vectors / capacity            # per-resource fractions
    dominant = frac.max(axis=1)                 # dominant share at full demand
    dominant = np.where(dominant <= 0, 1e-12, dominant)

    scale = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    target = 0.0
    step = 1.0 / 4096.0
    while active.any():
        nxt = target + step
        trial = scale.copy()
        trial[active] = np.minimum(1.0, nxt / dominant[active])
        proposed = (trial[:, None] * demand_vectors).sum(axis=0)
        if np.any(proposed > capacity + 1e-9):
            break
        scale = trial
        target = nxt
        # Tenants whose full demand is met leave the active set.
        active &= scale < 1.0 - 1e-12
        if target > 1.0 + 1e-9:
            break
    return scale


class DominantResourceFairness(AllocationPolicy):
    """DRF budgets computed offline over the three resource dimensions."""
    name = "B4"
    label = "DRF"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.budget: dict[int, np.ndarray] = {}
        self.used: dict[int, np.ndarray] = {}

    def prepare(self, stream):
        tenants = sorted({r.account_id for r in stream})
        idx = {t: i for i, t in enumerate(tenants)}
        D = np.zeros((len(tenants), 3), dtype=float)
        for r in stream:
            D[idx[r.account_id]] += r.demand.as_array()
        scale = drf_shares(D, np.asarray(self.total, dtype=float))
        self.budget = {t: scale[idx[t]] * D[idx[t]] for t in tenants}

    def admit(self, req, cost, free_frac, active):
        b = self.budget.get(req.account_id)
        if b is None:
            return Reason.QUOTA
        u = self.used.get(req.account_id, np.zeros(3))
        if np.any(u + req.demand.as_array() > b + 1e-9):
            return Reason.QUOTA
        return Reason.OK

    def on_admit(self, req, cost):
        u = self.used.get(req.account_id, np.zeros(3))
        self.used[req.account_id] = u + req.demand.as_array()

    def on_release(self, tenant_id, demand, cost, slice_type=None):
        u = self.used.get(tenant_id, np.zeros(3))
        self.used[tenant_id] = np.maximum(0.0, u - demand.as_array())


# -------------------------------------------------------------------- P -----

class ProposedNSACQuota(AllocationPolicy):
    """NSAC-style quota with pressure-adaptive AIMD caps, a rate gate and
    placement-slot accounting.

    Shipped variant (``p_slot_mode == "cap"``): the resource quota is the
    dominant share over CPU/memory/storage as before, and a tenant may hold
    at most ``p_slot_cap`` active allocations per slice type.  A tenant with
    negligible demand is therefore bounded to ``p_slot_cap`` of the
    ``beta * nodes`` same-slice slots; ``k`` colluding registered identities
    are bounded to ``k * p_slot_cap`` (Sybil resistance is out of scope).
    """
    name = "P"
    label = "Proposed (quota+caps)"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.nsac = NSACController(self.p)
        self.slot = slot_share(len(pool.nodes), beta)
        if self.p.p_slot_mode == "share":
            # One-slot entitlement: the quota prices placement slots, so the
            # cap must always afford one, or a small deployment (few nodes,
            # beta = 1) would deny every request.
            self.nsac.min_cap = self.slot
        self.nsac.reset()

    def _cost(self, cost: float) -> float:
        return max(cost, self.slot) if self.p.p_slot_mode == "share" else cost

    def admit(self, req, cost, free_frac, active):
        if self.p.p_profile_z > 0 and not within_profile(req,
                                                        self.p.p_profile_z):
            return Reason.PROFILE
        if (self.p.p_slot_mode == "cap"
                and not self.nsac.slot_ok(req.account_id, req.slice_type)):
            return Reason.SLOT_CAP
        return self.nsac.check(req.account_id, self._cost(cost),
                               req.arrival_index, free_frac, active)

    def on_admit(self, req, cost):
        self.nsac.on_admit(req.account_id, self._cost(cost), req.arrival_index)
        self.nsac.on_slot_admit(req.account_id, req.slice_type)

    def on_deny(self, req, reason=None):
        # Placement failures (capacity / isolation) are congestion, not the
        # tenant's behaviour; penalise them only if configured to.
        placement = reason in PLACEMENT_REASONS
        self.nsac.on_deny(req.account_id, req.arrival_index,
                          penalise=(not placement
                                    or self.p.p_penalize_placement))

    def on_release(self, tenant_id, demand, cost, slice_type=None):
        self.nsac.on_release(tenant_id, self._cost(cost))
        if slice_type is not None:
            self.nsac.on_slot_release(tenant_id, slice_type)


class ProposedSlotPricedQuota(ProposedNSACQuota):
    """Variant evaluated alongside the shipped policy: the placement slot is a
    fourth resource in the dominant-share cost (share ``1/(beta*nodes)``),
    with a one-slot entitlement.  Same parameters otherwise."""
    name = "PS"
    label = "Proposed (slot-priced quota)"

    def __init__(self, params: PolicyParams = DEFAULT_POLICY_PARAMS):
        import dataclasses
        super().__init__(dataclasses.replace(params, p_slot_mode="share"))


class WithProfileCheck(AllocationPolicy):
    """Any policy plus the slice-profile admission filter used by P.

    Lets the comparison credit the filter to the filter and the allocator to
    the allocator: a profile denial is returned before the wrapped policy is
    consulted, and is reported to it as a denial like any other."""

    def __init__(self, inner: AllocationPolicy, z: float = 3.0):
        super().__init__(inner.p)
        self.inner, self.z = inner, z
        self.name = inner.name + "+prof"
        self.label = inner.label + " + profile check"

    def reset(self, pool, beta=COLOCATION_BUDGET_BETA):
        super().reset(pool, beta)
        self.inner.reset(pool, beta)

    def prepare(self, stream):
        self.inner.prepare(stream)

    def admit(self, req, cost, free_frac, active):
        if not within_profile(req, self.z):
            return Reason.PROFILE
        return self.inner.admit(req, cost, free_frac, active)

    def on_admit(self, req, cost):
        self.inner.on_admit(req, cost)

    def on_deny(self, req, reason=None):
        self.inner.on_deny(req, reason)

    def on_release(self, tenant_id, demand, cost, slice_type=None):
        self.inner.on_release(tenant_id, demand, cost, slice_type)


POLICY_LADDER = (UncontrolledFCFS, StaticEqualCap, TokenBucket,
                 MaxMinFair, DominantResourceFairness, ProposedNSACQuota,
                 ProposedSlotPricedQuota)


def build_ladder(params: PolicyParams = DEFAULT_POLICY_PARAMS
                 ) -> list[AllocationPolicy]:
    return [cls(params) for cls in POLICY_LADDER]
