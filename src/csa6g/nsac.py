"""NSAC-style capacity control with dynamic caps (paper Step 4).

3GPP Network Slice Admission Control (NSAC) maintains admission limits for UEs
and PDU sessions associated with an S-NSSAI.  The proposed policy P is an
application-layer resource controller inspired by that counting principle; it
does not implement the standardized NSACF procedure.  It adds three local
admission-time controls:

*Pressure-adaptive tenant cap.*  Tenant ``u`` has an AIMD scale ``s_u``.  The
instantaneous cap, expressed as a fraction of the aggregate pool, is exactly::

    cap_u(t) = min(q_base * s_u(t),
                   max(kappa * R_free(t) / max(1, |active(t)|),
                       floor * q_base))

Thus the cap is bounded by the tenant's scaled base quota, responds to remaining
capacity, and never drops below the configured floor.

*AIMD scale update.*  Repeated denial multiplicatively decreases ``s_u``; an
admission additively restores it::

    denial  ->  s_u <- max(floor, md * s_u)
    admit   ->  s_u <- min(1, s_u + ai)

*A sliding-window gate* bounds the number of admitted requests for each tenant.
The proposed path therefore uses only counters and deque operations per request.
B3 and B4 are comparison policies implemented as scalar progressive-fill and
offline discretized DRF approximations, respectively; neither should be read as
a production scheduler or as an exact per-request optimization oracle.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .model import Reason
from .params import DEFAULT_POLICY_PARAMS, PolicyParams


@dataclass
class TenantState:
    tenant_id: int
    allocated: float = 0.0          # cumulative aggregate-normalised share
    cap_scale: float = 1.0          # AIMD multiplier on q_base
    admitted: int = 0
    denied: int = 0
    recent: deque = field(default_factory=deque)   # arrival indices of admits
    last_event: int | None = None   # arrival index of last scale update


class NSACController:
    """Quota bookkeeping shared by the proposed policy."""

    def __init__(self, params: PolicyParams = DEFAULT_POLICY_PARAMS):
        self.p = params
        self.tenants: dict[int, TenantState] = {}
        self.q_base = params.p_base_quota_frac
        # Minimum cap: one placement slot's share of the pool (set by the
        # policy from the deployment's node count and beta). Guarantees every
        # tenant can hold at least one allocation; see ``admission_cost``.
        self.min_cap = 0.0
        # Active allocations per (tenant, slice type), for the "cap" variant.
        self.active_slots: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self.tenants.clear()
        self.active_slots.clear()

    # ---------------------------------------------------------- slot cap
    def slots_held(self, tenant_id: int, slice_type: int) -> int:
        return self.active_slots.get((tenant_id, int(slice_type)), 0)

    def slot_ok(self, tenant_id: int, slice_type: int) -> bool:
        return self.slots_held(tenant_id, slice_type) < self.p.p_slot_cap

    def on_slot_admit(self, tenant_id: int, slice_type: int) -> None:
        k = (tenant_id, int(slice_type))
        self.active_slots[k] = self.active_slots.get(k, 0) + 1

    def on_slot_release(self, tenant_id: int, slice_type: int) -> None:
        k = (tenant_id, int(slice_type))
        self.active_slots[k] = max(0, self.active_slots.get(k, 0) - 1)

    def state(self, tenant_id: int) -> TenantState:
        st = self.tenants.get(tenant_id)
        if st is None:
            st = TenantState(tenant_id)
            self.tenants[tenant_id] = st
        return st

    # ------------------------------------------------------------------ cap
    def scale(self, tenant_id: int, arrival_index: int | None) -> float:
        """AIMD scale at ``arrival_index``, including time-based recovery.

        The stored ``cap_scale`` is the value at the tenant's last scale
        event; between events it recovers linearly at ``p_recover_step`` per
        ``p_rate_window`` arrivals.  Pure function of the state -- repeated
        checks do not compound recovery.
        """
        st = self.state(tenant_id)
        if not self.p.p_use_aimd:
            return 1.0
        s = st.cap_scale
        if (self.p.p_recover_step > 0 and st.last_event is not None
                and arrival_index is not None and s < 1.0):
            elapsed = max(0, arrival_index - st.last_event)
            s = min(1.0, s + self.p.p_recover_step * elapsed
                    / self.p.p_rate_window)
        return s

    def _materialise(self, st: TenantState, arrival_index: int | None) -> None:
        st.cap_scale = self.scale(st.tenant_id, arrival_index)
        if arrival_index is not None:
            st.last_event = arrival_index

    def dynamic_cap(self, tenant_id: int, free_frac: float,
                    active: int, arrival_index: int | None = None) -> float:
        """cap_u(t), expressed as a fraction of the total pool."""
        base = self.q_base * self.scale(tenant_id, arrival_index)
        if not self.p.p_use_pressure:
            return max(max(0.0, base), self.min_cap)
        pressure = self.p.p_pressure_kappa * free_frac / max(1, active)
        cap = max(0.0, min(base, max(pressure, self.p.p_cap_floor * self.q_base)))
        return max(cap, self.min_cap)

    # ---------------------------------------------------------------- gates
    def rate_ok(self, tenant_id: int, arrival_index: int) -> bool:
        """Sliding-window admitted-request-rate gate."""
        if not self.p.p_use_rate_gate:
            return True
        st = self.state(tenant_id)
        lo = arrival_index - self.p.p_rate_window
        while st.recent and st.recent[0] < lo:
            st.recent.popleft()
        return len(st.recent) < self.p.p_rate_max

    def check(self, tenant_id: int, cost: float, arrival_index: int,
              free_frac: float, active: int) -> str:
        if not self.rate_ok(tenant_id, arrival_index):
            return Reason.RATE
        st = self.state(tenant_id)
        cap = self.dynamic_cap(tenant_id, free_frac, active, arrival_index)
        if st.allocated + cost > cap:
            return Reason.QUOTA
        return Reason.OK

    # --------------------------------------------------------------- update
    def on_admit(self, tenant_id: int, cost: float, arrival_index: int) -> None:
        st = self.state(tenant_id)
        st.allocated += cost
        st.admitted += 1
        st.recent.append(arrival_index)
        self._materialise(st, arrival_index)
        st.cap_scale = min(1.0, st.cap_scale + self.p.p_ai_step)

    def on_deny(self, tenant_id: int, arrival_index: int | None = None,
                penalise: bool = True) -> None:
        st = self.state(tenant_id)
        st.denied += 1
        if not penalise:
            return
        self._materialise(st, arrival_index)
        st.cap_scale = max(self.p.p_cap_floor, st.cap_scale * self.p.p_md_factor)

    def on_release(self, tenant_id: int, cost: float) -> None:
        st = self.state(tenant_id)
        st.allocated = max(0.0, st.allocated - cost)

    # ------------------------------------------------------------- readouts
    def allocation_vector(self) -> np.ndarray:
        return np.array([s.allocated for s in self.tenants.values()], dtype=float)
