"""Data model for admission requests and resources (paper Section 4.1)."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from .params import DEFAULT_NODE, ResourceSpec


class SliceType(IntEnum):
    """S-NSSAI slice/service type. Values match 3GPP TS 23.501 SST codepoints."""
    URLLC = 2
    EMBB = 1
    MMTC = 3

    @property
    def label(self) -> str:
        return {SliceType.URLLC: "URLLC",
                SliceType.EMBB: "eMBB",
                SliceType.MMTC: "mMTC"}[self]


ALL_SLICE_TYPES = (SliceType.URLLC, SliceType.EMBB, SliceType.MMTC)


@dataclass(frozen=True)
class Demand:
    """Resource demand vector d_u. CPU cores, memory GB, storage TB."""
    cpu: float
    mem: float
    storage: float

    def as_array(self) -> np.ndarray:
        return np.array([self.cpu, self.mem, self.storage], dtype=float)

    @property
    def tup(self) -> tuple[float, float, float]:
        """Scalar view used on the admission hot path (no array allocation)."""
        return (self.cpu, self.mem, self.storage)

    def scaled(self, f: float) -> "Demand":
        return Demand(self.cpu * f, self.mem * f, self.storage * f)

    def is_admissible(self) -> bool:
        """Every component finite and non-negative.

        The decrypted request is attacker-chosen data: the AEAD says a holder
        of ``K_slice`` wrote it, not that what they wrote is meaningful, and a
        registered tenant is exactly the adversary experiment 4 models.  A
        *negative* component is the dangerous case, because it passes every
        downstream test by being smaller than any bound: ``Node.fits`` accepts
        it, ``Node.allocate`` then *subtracts* it from the node's usage, and
        ``demand_norm`` returns a negative cost that a quota check reads as
        free credit.  One request for -100000 cores drove the pool to 33x its
        real free capacity and left the tenant's consumed share negative --
        both of the accounting structures the DoS-resilience result rests on,
        broken from inside a legitimate session.

        Non-finite values happen to be rejected downstream already (every
        comparison against NaN is False, so ``fits`` fails and the request is
        denied ``resource_infeasible``), but by accident rather than by rule,
        and an accident is not a property.  This makes all three explicit.
        Large *positive* demands need no bound here: they are infeasible, and
        being denied as infeasible is the correct outcome.

        An *all-zero* vector is rejected too.  It costs nothing
        against the dominant-share quota, yet a successful placement still
        appends it to a node's ``hosted`` list and so consumes one of the
        node's ``beta`` anti-co-location slots.  A registered tenant could
        therefore hold every same-slice slot in the pool at zero quota cost
        and deny same-slice placement to everyone else, limited only by the
        rate gate.  A request that asks for nothing has no admissible
        meaning, so it is denied ``malformed_request`` before any freshness
        or allocation state is touched.  At least one strictly positive
        component is required; the quota then prices the request.
        """
        vals = (self.cpu, self.mem, self.storage)
        return (all(math.isfinite(v) and v >= 0.0 for v in vals)
                and any(v > 0.0 for v in vals))

    def __add__(self, other: "Demand") -> "Demand":
        return Demand(self.cpu + other.cpu, self.mem + other.mem,
                      self.storage + other.storage)


class Node:
    """A cloud/edge node with residual capacity r_v and hosted slice types.

    Deliberately built on plain floats rather than numpy arrays: the admission
    path touches this object several times per request, and numpy small-array
    allocation would otherwise dominate the very latencies experiment 1 is
    trying to measure.
    """

    __slots__ = ("node_id", "capacity", "used", "hosted", "_cap")

    def __init__(self, node_id: int, capacity: ResourceSpec = DEFAULT_NODE):
        self.node_id = node_id
        self.capacity = capacity
        self._cap = (capacity.cpu, capacity.mem, capacity.storage)
        self.used = [0.0, 0.0, 0.0]
        self.hosted: list[SliceType] = []

    @property
    def cap_array(self) -> np.ndarray:
        return np.asarray(self._cap, dtype=float)

    @property
    def cap_tup(self) -> tuple[float, float, float]:
        return self._cap

    @property
    def residual(self) -> np.ndarray:
        """r_v in the placement rule."""
        return self.cap_array - np.asarray(self.used, dtype=float)

    @property
    def load(self) -> float:
        """Scalar load used to order nodes for first-fit (ascending)."""
        u, c = self.used, self._cap
        return (u[0] / c[0] + u[1] / c[1] + u[2] / c[2]) / 3.0

    def fits(self, d: Demand) -> bool:
        u, c = self.used, self._cap
        return (u[0] + d.cpu <= c[0] + 1e-9
                and u[1] + d.mem <= c[1] + 1e-9
                and u[2] + d.storage <= c[2] + 1e-9)

    def allocate(self, d: Demand, stype: SliceType) -> None:
        u = self.used
        u[0] += d.cpu
        u[1] += d.mem
        u[2] += d.storage
        self.hosted.append(stype)

    def free(self, d: Demand, stype: SliceType) -> None:
        u = self.used
        u[0] = max(0.0, u[0] - d.cpu)
        u[1] = max(0.0, u[1] - d.mem)
        u[2] = max(0.0, u[2] - d.storage)
        if stype in self.hosted:
            self.hosted.remove(stype)

    def reset(self) -> None:
        self.used = [0.0, 0.0, 0.0]
        self.hosted = []


class ResourcePool:
    """Aggregate resource pool R (feasibility problem) realised as a set of nodes."""

    def __init__(self, num_nodes: int, node_spec: ResourceSpec = DEFAULT_NODE):
        self.nodes = [Node(i, node_spec) for i in range(num_nodes)]
        self._total_tup = (node_spec.cpu * num_nodes, node_spec.mem * num_nodes,
                           node_spec.storage * num_nodes)
        self._total = np.asarray(self._total_tup, dtype=float)

    @property
    def total(self) -> np.ndarray:
        return self._total

    @property
    def total_tup(self) -> tuple[float, float, float]:
        return self._total_tup

    @property
    def used_tup(self) -> tuple[float, float, float]:
        a = b = c = 0.0
        for n in self.nodes:
            u = n.used
            a += u[0]
            b += u[1]
            c += u[2]
        return (a, b, c)

    @property
    def used(self) -> np.ndarray:
        return np.asarray(self.used_tup, dtype=float)

    @property
    def free(self) -> np.ndarray:
        return self._total - self.used

    def free_fraction(self) -> float:
        """Mean fractional headroom across the three dimensions (scalar path)."""
        u, t = self.used_tup, self._total_tup
        return (3.0 - u[0] / t[0] - u[1] / t[1] - u[2] / t[2]) / 3.0

    def dominant_free_fraction(self) -> float:
        """Headroom on the most-loaded dimension.

        This is the pressure signal the dynamic cap reacts to.  Using the mean
        free fraction instead lets a policy stay relaxed while the resource
        that actually binds is already exhausted."""
        return 1.0 - self.dominant_utilisation()

    def utilisation(self) -> float:
        """Mean fractional utilisation across the three dimensions."""
        return 1.0 - self.free_fraction()

    def dominant_utilisation(self) -> float:
        """Utilisation of the most-loaded dimension.

        Reported alongside the mean because the mean is misleading here: the
        storage dimension is far from binding in this parameterisation, so a
        pool whose CPU is completely full still shows a mean utilisation near
        one third.  The dominant figure is the one that says whether capacity
        actually ran out."""
        u, t = self.used_tup, self._total_tup
        return max(u[0] / t[0], u[1] / t[1], u[2] / t[2])

    def reset(self) -> None:
        for n in self.nodes:
            n.reset()


@dataclass(frozen=True)
class AdmissionRequest:
    """Cleartext admission request m_u = (u*, s, c, l_u, d_u, n_u, tau_u).

    ``malicious`` is ground-truth labelling used only for evaluation; it is
    never visible to any policy under test.
    """
    tenant_id: int
    pseudonym: bytes            # u*
    slice_type: SliceType       # s
    service_id: int             # c
    sla_latency_ms: float       # l_u
    demand: Demand              # d_u
    arrival_index: int
    counter: int = 0
    malicious: bool = False
    # The authenticated principal (subscription / slice customer) that
    # quota is charged to.  None means the tenant is its own principal.  In
    # the E-AP the registered tenant bound at primary authentication plays
    # this role; in the offline harness it lets one principal open several
    # sessions (identities) that share one quota.  Not serialised: it is
    # registration state, never sender-supplied.
    principal: int | None = None
    # The tenant's registered resource entitlement, as a multiple of its
    # slice type's declared profile (subscription state, not serialised).
    # The profile check bounds demand by profile x entitlement, so a
    # legitimately large tenant is not mistaken for an inflating one.
    entitlement_scale: float = 1.0

    @property
    def account_id(self) -> int:
        return self.tenant_id if self.principal is None else self.principal

    # -- serialisation used as the AEAD plaintext ---------------------------
    # Doubles, not singles.  The plaintext is padded to a fixed 256 bytes
    # regardless, so the 16 bytes the wider fields cost are free on the wire,
    # and in exchange the request that comes back out of ``open_envelope`` is
    # bit-identical to the one that went in.  With float32 the demand vector
    # round-tripped with ~1e-7 relative error, which was enough to move
    # ``demand_norm`` and so to flip a marginal quota decision between the
    # E-AP path (which sees the decrypted request) and the offline harness
    # (which sees the original) -- see ``test_eap_and_harness_agree``.
    # ``arrival_index`` is an integer and is now stored as one.
    STRUCT = struct.Struct("<8sBIddddII")

    def serialise(self) -> bytes:
        return self.STRUCT.pack(
            self.pseudonym.ljust(8, b"\x00")[:8],
            int(self.slice_type),
            self.service_id,
            self.sla_latency_ms,
            self.demand.cpu, self.demand.mem, self.demand.storage,
            self.arrival_index,
            self.counter,
        )

    @classmethod
    def deserialise(cls, blob: bytes, tenant_id: int = -1) -> "AdmissionRequest":
        (pseud, stype, svc, sla, cpu, mem, sto, arr, ctr) = cls.STRUCT.unpack(
            blob[: cls.STRUCT.size]
        )
        return cls(
            tenant_id=tenant_id,
            pseudonym=pseud,
            slice_type=SliceType(stype),
            service_id=svc,
            sla_latency_ms=sla,
            demand=Demand(cpu, mem, sto),
            arrival_index=arr,
            counter=ctr,
        )


@dataclass(frozen=True)
class Decision:
    """Outcome of the admission decision A(u, s, c, d_u) (admission decision)."""
    admitted: bool
    reason: str = "ok"
    node_id: int | None = None
    granted: Demand | None = None
    # Timing breakdown, seconds. Keys are pipeline stages.
    timings: dict = None       # type: ignore[assignment]

    def with_timings(self, t: dict) -> "Decision":
        return replace(self, timings=t)


# Denial reason codes. Only ``NOT_AUTHORIZED`` and ``TRANSIENT`` are ever
# returned to the requester (Step 5 minimal-disclosure property); the finer
# codes stay inside the network for evaluation and logging.
class Reason:
    OK = "ok"
    NOT_AUTHORIZED = "not_authorized"      # externally visible
    TRANSIENT = "transient_error"          # externally visible
    # internal-only
    AUTH_FAIL = "auth_fail"
    CONTEXT_MISMATCH = "context_mismatch"
    STALE = "stale_timestamp"
    FUTURE = "future_timestamp"
    NONCE_REPLAY = "nonce_replay"
    MALFORMED_REQUEST = "malformed_request"
    COUNTER_ROLLBACK = "counter_rollback"
    SEQUENCE_ROLLBACK = "sequence_rollback"
    EPOCH_MISMATCH = "epoch_mismatch"
    TOKEN_EXPIRED = "token_expired"
    INTEGRITY = "integrity_fail"
    QUOTA = "quota_exceeded"
    RATE = "rate_limited"
    INFEASIBLE = "resource_infeasible"
    ISOLATION = "isolation_violation"
    SLOT_CAP = "slot_cap_exceeded"       # per-tenant active-allocation cap
    PROFILE = "profile_exceeded"         # demand outside slice profile
    COMMIT_FAIL = "commit_fail"

    EXTERNAL = {OK: OK, TRANSIENT: TRANSIENT, COMMIT_FAIL: TRANSIENT}

    @classmethod
    def externalise(cls, reason: str) -> str:
        """Map an internal reason to the single coarse category the requester
        sees. Impostors and quota-exhausted tenants are indistinguishable."""
        return cls.EXTERNAL.get(reason, cls.NOT_AUTHORIZED)


def demand_norm(d: Demand, pool_total) -> float:
    """Dominant-resource share of a demand vector: the *largest* of its
    per-dimension fractions of the pool.

    The obvious alternative -- the mean of the three fractions -- was tried
    first and is wrong here.  The three dimensions are wildly unbalanced in
    this parameterisation (storage demand is two orders of magnitude below the
    storage pool), so a mean dilutes the binding resource and any quota
    expressed in those units stops binding on the resource that actually runs
    out.  Taking the max makes the scalar quota policies (B1, B2, B3, P) budget
    against the resource that constrains the deployment, which is also what
    makes them comparable with DRF.

    Accepts either a tuple or an ndarray of totals; the tuple path avoids array
    allocation on the admission hot path.
    """
    return max(d.cpu / pool_total[0], d.mem / pool_total[1],
               d.storage / pool_total[2])


def slot_share(num_nodes: int, beta: int) -> float:
    """Share of the pool's same-slice placement capacity taken by one
    allocation: one of ``beta * num_nodes`` anti-co-location slots."""
    return 1.0 / (beta * num_nodes)


def admission_cost(d: Demand, pool_total, num_nodes: int, beta: int) -> float:
    """Slot-priced quota cost (``p_slot_mode == "share"``): the
    dominant share over CPU, memory, storage *and* the placement slot the
    allocation will occupy.  The shipped variant (``"cap"``) charges the
    resource share only and bounds active allocations per tenant instead.

    Every admitted allocation occupies exactly one of a node's ``beta``
    same-slice slots (placement rule) whatever its resource demand, so the slot is a
    fourth, indivisible resource with pool capacity ``beta * num_nodes`` and
    per-request demand 1.  Charging it as a share in the same dominant-share
    quota means a tenant issuing tiny positive demands is bounded by its
    quota exactly as one issuing large ones: it cannot hold every
    same-slice slot for a negligible charge.  The same cost is refunded at
    release and on rollback, so the accounting stays balanced.
    """
    return max(demand_norm(d, pool_total), slot_share(num_nodes, beta))
