"""Seeded request-stream generation.

The single most important property here is that the stream is a function of the
seed *alone* -- not of the policy, not of whether security is enabled.  The
prior codebase had a bug in exactly this place (security decisions drawing from
the arrival RNG, so turning security on changed the tenant population and any
measured difference was partly a different workload).  We prevent it
structurally: :func:`generate_stream` takes a seed and returns a frozen list,
and every policy under test is handed the same list object.

Slice-conditional demand distributions follow the paper's synthetic field distributions table.  Storage is
given there in GB and converted to TB to match the resource pool units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ALL_SLICE_TYPES, AdmissionRequest, Demand, SliceType
from .params import SLICE_PROFILES, STORAGE_GB_PER_TB


def _profile(stype: SliceType) -> dict:
    return SLICE_PROFILES[stype.label]


def sample_demand(rng: np.random.Generator, stype: SliceType,
                  inflate: float = 1.0) -> Demand:
    p = _profile(stype)
    cpu = max(0.5, rng.normal(*p["cpu"]))
    mem = max(0.5, rng.normal(*p["mem"]))
    sto = max(0.1, rng.normal(*p["storage"])) / STORAGE_GB_PER_TB
    return Demand(cpu * inflate, mem * inflate, sto * inflate)


def sample_sla(rng: np.random.Generator, stype: SliceType) -> float:
    mu, sigma = _profile(stype)["latency"]
    return float(rng.lognormal(mu, sigma))


@dataclass(frozen=True)
class StreamSpec:
    """Shape of a generated request stream.

    The benign load is held *constant* and the malicious load scales with
    ``malicious_ratio``.  Splitting a fixed total by the ratio instead -- the
    obvious first implementation -- makes the benign population vanish at high
    intensity (19 benign requests spread over 50 tenants at 20:1), which turns
    Jain's index over benign tenants into noise.  Holding the benign load fixed
    keeps the benign experience comparable across the whole sweep, which is the
    only way the sweep answers the question it is asked.
    """
    n_benign_tenants: int = 50
    n_attacker_tenants: int = 10
    n_benign_requests: int = 250
    malicious_ratio: float = 10.0     # malicious requests per benign request
    attacker_inflation: float = 2.0   # demand multiplier used by attackers
    demand_spread: float = 0.0        # per-tenant log-normal demand scaling
    service_ids: int = 4

    @property
    def n_tenants(self) -> int:
        return self.n_benign_tenants + self.n_attacker_tenants

    @property
    def n_malicious_requests(self) -> int:
        return int(round(self.malicious_ratio * self.n_benign_requests))


def generate_stream(seed: int, spec: StreamSpec = StreamSpec()
                    ) -> list[AdmissionRequest]:
    """Return a deterministic list of :class:`AdmissionRequest`.

    Attackers are a fixed minority of tenants emitting a disproportionate share
    of the requests with inflated demand vectors.  Benign tenants emit typical
    requests at a typical rate.  Arrival order is interleaved, not blocked, so
    no policy can exploit a contiguous attack burst.

    Nothing here depends on the policy under test or on whether security is
    enabled -- the stream is a pure function of ``(seed, spec)``.  The previous
    codebase had a bug in exactly this place;
    ``test_enabling_security_does_not_perturb_the_workload`` will fail if it
    ever returns.
    """
    rng = np.random.default_rng(seed)

    n_ben_t, n_att_t = spec.n_benign_tenants, spec.n_attacker_tenants
    benign_ids = list(range(n_ben_t))
    attacker_ids = list(range(n_ben_t, n_ben_t + n_att_t))

    n_ben = spec.n_benign_requests
    n_mal = spec.n_malicious_requests if attacker_ids else 0

    owners = np.concatenate([
        rng.choice(attacker_ids, size=n_mal) if n_mal else np.zeros(0, int),
        rng.choice(benign_ids, size=n_ben) if n_ben else np.zeros(0, int),
    ]).astype(int)
    flags = np.concatenate([np.ones(n_mal, bool), np.zeros(n_ben, bool)])
    order = rng.permutation(len(owners))
    owners, flags = owners[order], flags[order]

    n_all = n_ben_t + n_att_t
    tenant_type = {t: ALL_SLICE_TYPES[int(rng.integers(0, 3))]
                   for t in range(n_all)}
    pseudonym = {t: bytes(rng.integers(0, 256, size=8, dtype="uint8"))
                 for t in range(n_all)}
    scale = {t: (float(rng.lognormal(0.0, spec.demand_spread))
                 if spec.demand_spread > 0 else 1.0) for t in range(n_all)}
    counter = {t: 0 for t in range(n_all)}

    stream: list[AdmissionRequest] = []
    for i, (t, mal) in enumerate(zip(owners.tolist(), flags.tolist())):
        stype = tenant_type[t]
        inflate = scale[t] * (spec.attacker_inflation if mal else 1.0)
        counter[t] += 1
        stream.append(AdmissionRequest(
            tenant_id=int(t),
            pseudonym=pseudonym[t],
            slice_type=stype,
            service_id=int(rng.integers(0, spec.service_ids)),
            sla_latency_ms=sample_sla(rng, stype),
            demand=sample_demand(rng, stype, inflate),
            arrival_index=i,
            counter=counter[t],
            malicious=bool(mal),
            # Every tenant, attacker or not, is entitled to its own base
            # scale; attackers then inflate beyond it.
            entitlement_scale=scale[t],
        ))
    return stream


def benign_stream(seed: int, n_tenants: int = 5, n_requests: int = 100
                  ) -> list[AdmissionRequest]:
    """Uniform benign stream used by the latency and burst experiments."""
    return generate_stream(seed, StreamSpec(
        n_benign_tenants=n_tenants, n_attacker_tenants=0,
        n_benign_requests=n_requests, malicious_ratio=0.0))


def privacy_dataset(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditional admission fields for the field-hiding experiment.

    Columns: cpu, mem, storage_GB, latency_ms, sla_class.  Labels are the slice
    type index.  Deliberately well separated, per the paper's field distributions table: real
    URLLC / eMBB / mMTC requests do have distinct profiles, so this is the
    realistic *upper bound* on pre-encryption leakage.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        k = int(rng.integers(0, 3))
        stype = ALL_SLICE_TYPES[k]
        p = _profile(stype)
        X.append([
            rng.normal(*p["cpu"]),
            rng.normal(*p["mem"]),
            rng.normal(*p["storage"]),
            rng.lognormal(*p["latency"]),
            float(p["sla_class"]),
        ])
        y.append(k)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
