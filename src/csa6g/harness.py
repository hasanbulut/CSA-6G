"""Shared allocation harness used by the comparative experiment.

Every policy sees the same stream object, the same pool construction and the
same isolation constraint (placement rule).  Only ``policy.admit`` differs.  The
per-request decision cost reported for each policy is

    (offline prepare time + sum of online admit-call times) / n_requests

so an offline allocator is charged for the optimisation it needs, amortised
across the requests that optimisation serves.  Placement and commit time are
excluded because they are identical across policies by construction.

Relationship to :mod:`csa6g.eap`
--------------------------------
This loop is deliberately *not* the E-AP.  It runs the same decision --
``policy.admit`` then isolation-aware placement -- with the admission cryptography
removed, because experiment 4 evaluates 6 policies x 5 flood intensities x 100
seeds x ~2750 requests and sealing and opening several million envelopes would
add hours to measure a cost that is identical across every arm and already
measured, per stage, by experiment 1.

The risk is that two implementations of one decision drift apart, so that the
comparative evaluation no longer reflects the code path the system ships.
``test_eap_and_harness_agree`` guards against this: it drives an identical stream through both and asserts the admit/deny
vectors match request for request.  If the paths ever diverge, that test fails
before any number reaches a table.  Keep them in step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .metrics import (acceptance_rate, attacker_principal_share,
                      fairness_summary, malicious_capacity_share,
                      rejection_rate)
from .model import Reason, ResourcePool, demand_norm
from .params import COLOCATION_BUDGET_BETA, DEFAULT_NODE, DEFAULT_NUM_NODES
from .placement import select_node
from .policies import AllocationPolicy


@dataclass
class PolicyRun:
    policy: str
    label: str
    seed: int
    accepted: np.ndarray
    reasons: list
    fi_benign: float
    fi_all: float
    fi_benign_norm: float
    benign_acceptance: float
    malicious_rejection: float
    malicious_capacity_share: float
    utilisation: float
    dominant_utilisation: float
    decision_ms: float
    prepare_ms: float
    nodes: int = 0
    beta: int = 0
    attacker_principal_share: float = 0.0

    @property
    def capture_ratio(self) -> float:
        s = self.attacker_principal_share
        return self.malicious_capacity_share / s if s > 0 else float("nan")

    def as_row(self) -> dict:
        return dict(policy=self.policy, label=self.label, seed=self.seed,
                    fi_benign=self.fi_benign, fi_all=self.fi_all,
                    fi_benign_norm=self.fi_benign_norm,
                    benign_acceptance=self.benign_acceptance,
                    malicious_rejection=self.malicious_rejection,
                    malicious_capacity_share=self.malicious_capacity_share,
                    utilisation=self.utilisation,
                    dominant_utilisation=self.dominant_utilisation,
                    decision_ms=self.decision_ms, prepare_ms=self.prepare_ms,
                    nodes=self.nodes, beta=self.beta,
                    attacker_principal_share=self.attacker_principal_share,
                    capture_ratio=self.capture_ratio)


def make_pool(num_nodes: int = DEFAULT_NUM_NODES) -> ResourcePool:
    return ResourcePool(num_nodes, DEFAULT_NODE)


def run_policy(policy: AllocationPolicy, stream, seed: int,
               num_nodes: int = DEFAULT_NUM_NODES,
               beta: int = COLOCATION_BUDGET_BETA) -> PolicyRun:
    """Run one policy over one stream. ``num_nodes`` and ``beta`` define the
    deployment; both are held identical across policies within a comparison."""
    pool = make_pool(num_nodes)
    policy.reset(pool, beta)

    t0 = time.perf_counter()
    policy.prepare(stream)
    prepare_s = time.perf_counter() - t0

    total = pool.total_tup
    accepted = np.zeros(len(stream), dtype=bool)
    reasons: list[str] = []
    active: set[int] = set()
    decision_s = 0.0

    for i, req in enumerate(stream):
        cost = demand_norm(req.demand, total)
        free_frac = pool.dominant_free_fraction()

        t0 = time.perf_counter()
        reason = policy.admit(req, cost, free_frac, max(1, len(active)))
        decision_s += time.perf_counter() - t0

        if reason != Reason.OK:
            policy.on_deny(req, reason)
            reasons.append(reason)
            continue

        node, place_reason = select_node(pool.nodes, req.demand,
                                         req.slice_type, beta)
        if node is None:
            policy.on_deny(req, place_reason)
            reasons.append(place_reason)
            continue

        node.allocate(req.demand, req.slice_type)
        policy.on_admit(req, cost)
        active.add(req.account_id)
        accepted[i] = True
        reasons.append(Reason.OK)

    mal = np.array([r.malicious for r in stream], dtype=bool)
    fair = fairness_summary(stream, accepted, total)
    n = max(1, len(stream))
    return PolicyRun(
        policy=policy.name, label=policy.label, seed=seed,
        accepted=accepted, reasons=reasons,
        fi_benign=fair["fi_benign"], fi_all=fair["fi_all"],
        fi_benign_norm=fair["fi_benign_norm"],
        benign_acceptance=acceptance_rate(accepted, ~mal),
        malicious_rejection=rejection_rate(accepted, mal),
        malicious_capacity_share=malicious_capacity_share(stream, accepted,
                                                          total),
        utilisation=pool.utilisation(),
        dominant_utilisation=pool.dominant_utilisation(),
        decision_ms=1e3 * (decision_s + prepare_s) / n,
        prepare_ms=1e3 * prepare_s / n,
        nodes=num_nodes, beta=beta,
        attacker_principal_share=attacker_principal_share(stream),
    )
