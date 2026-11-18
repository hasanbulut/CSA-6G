"""AIMD recovery, congestion-vs-abuse penalties, ablation switches.

If a tenant's AIMD scale recovers only on admission, four penalised denials
take the scale to the 0.1 floor, the cap to 0.1 * q_base, and a tenant whose
smallest request exceeds that cap is never admitted again.  These tests pin the starvation, the fix, and that the fix does not
let a persistent flooder recover.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csa6g.harness import make_pool, run_policy
from csa6g.model import Reason
from csa6g.nsac import NSACController
from csa6g.params import DEFAULT_POLICY_PARAMS
from csa6g.policies import ProposedNSACQuota
from csa6g.workload import StreamSpec, generate_stream

from csa6g.params import ADMISSION_ONLY_RECOVERY_PARAMS as ADMIT_ONLY
P_DEFAULT = DEFAULT_POLICY_PARAMS
Q = DEFAULT_POLICY_PARAMS.p_base_quota_frac
REQ = 0.5 * Q          # larger than floor*q_base (0.1 Q), well under q_base


def _drive_to_floor(nsac, t=1, start=0):
    for i in range(4):
        nsac.on_deny(t, start + i)
    return start + 4


def test_admission_only_recovery_starves_a_tenant_forever():
    nsac = NSACController(ADMIT_ONLY)
    nxt = _drive_to_floor(nsac)
    assert nsac.scale(1, nxt) == ADMIT_ONLY.p_cap_floor
    # however long the tenant waits, with an idle pool:
    for later in (nxt + 10, nxt + 10_000, nxt + 10**7):
        assert nsac.check(1, REQ, later, free_frac=1.0, active=1) \
            == Reason.QUOTA


def test_v7_recovery_readmits_after_waiting():
    nsac = NSACController(P_DEFAULT)
    nxt = _drive_to_floor(nsac)
    assert nsac.check(1, REQ, nxt, 1.0, 1) == Reason.QUOTA
    # (0.5 - 0.1) / 0.15 windows of no denial is enough
    wait = int(np.ceil((0.5 - P_DEFAULT.p_cap_floor) / P_DEFAULT.p_recover_step
                       * P_DEFAULT.p_rate_window)) + 1
    assert nsac.check(1, REQ, nxt + wait, 1.0, 1) == Reason.OK


def test_recovery_is_not_compounded_by_repeated_checks():
    nsac = NSACController(P_DEFAULT)
    nxt = _drive_to_floor(nsac)
    for _ in range(1000):
        nsac.check(1, REQ, nxt + 1, 1.0, 1)
    assert abs(nsac.scale(1, nxt + 1)
               - (P_DEFAULT.p_cap_floor + 2 * P_DEFAULT.p_recover_step / P_DEFAULT.p_rate_window)
               ) < 1e-12   # last denial at arrival nxt-1: two arrivals elapsed


def test_a_persistent_flooder_stays_at_the_floor():
    """Each penalised denial restarts the recovery clock: a tenant denied
    every few arrivals cannot climb out."""
    nsac = NSACController(P_DEFAULT)
    a = 0
    for _ in range(500):
        nsac.on_deny(1, a)
        a += 4
    assert nsac.scale(1, a) < P_DEFAULT.p_cap_floor + 4 * P_DEFAULT.p_recover_step \
        / P_DEFAULT.p_rate_window + 1e-12


def test_placement_denials_are_not_penalised_by_default():
    pol = ProposedNSACQuota(P_DEFAULT)
    pol.reset(make_pool(4))
    req = generate_stream(0, StreamSpec(n_benign_tenants=1,
                                        n_attacker_tenants=0,
                                        n_benign_requests=1,
                                        malicious_ratio=0.0))[0]
    for _ in range(10):
        pol.on_deny(req, Reason.INFEASIBLE)
        pol.on_deny(req, Reason.ISOLATION)
    assert pol.nsac.scale(req.tenant_id, req.arrival_index) == 1.0
    pol.on_deny(req, Reason.QUOTA)
    assert pol.nsac.scale(req.tenant_id, req.arrival_index) \
        == P_DEFAULT.p_md_factor


def test_admission_only_switches_penalise_placement():
    pol = ProposedNSACQuota(ADMIT_ONLY)
    pol.reset(make_pool(4))
    req = generate_stream(0, StreamSpec(n_benign_tenants=1,
                                        n_attacker_tenants=0,
                                        n_benign_requests=1,
                                        malicious_ratio=0.0))[0]
    pol.on_deny(req, Reason.INFEASIBLE)
    assert pol.nsac.scale(req.tenant_id, req.arrival_index) \
        == ADMIT_ONLY.p_md_factor


def test_ablation_switches_disable_their_mechanism():
    nsac = NSACController(dataclasses.replace(P_DEFAULT, p_use_aimd=False))
    _drive_to_floor(nsac)
    assert nsac.scale(1, 5) == 1.0
    nsac = NSACController(dataclasses.replace(P_DEFAULT, p_use_rate_gate=False))
    for i in range(50):
        nsac.on_admit(1, 0.0, i)
    assert nsac.rate_ok(1, 50)
    nsac = NSACController(dataclasses.replace(P_DEFAULT, p_use_pressure=False))
    assert nsac.dynamic_cap(1, free_frac=0.0, active=100) == Q


def test_v7_policy_runs_end_to_end_and_still_limits_attackers():
    stream = generate_stream(3, StreamSpec(malicious_ratio=10.0))
    run = run_policy(ProposedNSACQuota(P_DEFAULT), stream, 3, 24, 3)
    assert run.benign_acceptance > 0.3
    assert run.malicious_capacity_share < 0.3


# ----------------------------------------------- principal, profile, metric --

def _one(stype_name="eMBB", scale=1.0, tenant=1, principal=None):
    from csa6g.model import AdmissionRequest, Demand, SliceType
    from csa6g.params import SLICE_PROFILES
    st = next(t for t in SliceType if t.label == stype_name)
    p = SLICE_PROFILES[stype_name]
    return AdmissionRequest(
        tenant_id=tenant, pseudonym=b"x" * 8, slice_type=st, service_id=0,
        sla_latency_ms=1.0,
        demand=Demand(p["cpu"][0] * scale, p["mem"][0] * scale,
                      p["storage"][0] * scale / 1000.0),
        arrival_index=0, counter=1, principal=principal)


def test_profile_check_admits_typical_and_rejects_doubled_demand():
    from csa6g.policies import within_profile
    for name in ("URLLC", "eMBB", "mMTC"):
        assert within_profile(_one(name, 1.0), 3.0)
        assert not within_profile(_one(name, 2.0), 3.0)


def test_profile_denial_is_the_tenants_own_and_is_penalised():
    pol = ProposedNSACQuota(P_DEFAULT)
    pol.reset(make_pool(24))
    req = _one("eMBB", 2.0)
    assert pol.admit(req, 0.01, 1.0, 1) == Reason.PROFILE
    pol.on_deny(req, Reason.PROFILE)
    assert pol.nsac.scale(req.account_id, 0) == P_DEFAULT.p_md_factor


def test_sessions_of_one_principal_share_one_quota():
    """Five sessions (tenant ids) under one principal are one quota holder."""
    pol = ProposedNSACQuota(dataclasses.replace(P_DEFAULT, p_use_rate_gate=False,
                                                p_slot_mode="none"))
    pol.reset(make_pool(24))
    cost = 0.4 * Q
    admitted = 0
    for k in range(5):
        r = dataclasses.replace(_one("URLLC", tenant=100 + k, principal=7),
                                arrival_index=k)
        if pol.admit(r, cost, 1.0, 1) == Reason.OK:
            pol.on_admit(r, cost)
            admitted += 1
    assert admitted == 2          # 2 * 0.4 Q fits, a third would exceed Q


def test_attacker_principal_share_counts_principals_not_sessions():
    from csa6g.metrics import attacker_principal_share
    ben = [_one(tenant=t) for t in range(9)]
    mal = [dataclasses.replace(_one(tenant=100 + k, principal=50),
                               malicious=True) for k in range(5)]
    assert attacker_principal_share(ben + mal) == 0.1


def test_pressure_term_is_off_by_default_in_v7():
    assert P_DEFAULT.p_use_pressure is False
    assert ADMIT_ONLY.p_use_pressure is True and ADMIT_ONLY.p_profile_z == 0.0


def test_profile_wrapper_filters_then_delegates():
    from csa6g.policies import StaticEqualCap, WithProfileCheck
    pol = WithProfileCheck(StaticEqualCap(P_DEFAULT))
    pol.reset(make_pool(24))
    assert pol.admit(_one("eMBB", 2.0), 0.001, 1.0, 1) == Reason.PROFILE
    assert pol.admit(_one("eMBB", 1.0), 0.001, 1.0, 1) == Reason.OK
    assert pol.name == "B1+prof"


def test_profile_check_scales_with_the_tenants_entitlement():
    from csa6g.policies import within_profile
    big = dataclasses.replace(_one("eMBB", 1.8), entitlement_scale=2.0)
    assert within_profile(big, 3.0)                      # large but entitled
    assert not within_profile(
        dataclasses.replace(big, entitlement_scale=1.0), 3.0)
    assert not within_profile(
        dataclasses.replace(_one("eMBB", 4.0), entitlement_scale=2.0), 3.0)


def test_heterogeneous_benign_tenants_are_not_profile_rejected():
    s = generate_stream(5, StreamSpec(n_attacker_tenants=0,
                                      malicious_ratio=0.0,
                                      demand_spread=0.7))
    from csa6g.policies import within_profile
    assert np.mean([within_profile(r, 3.0) for r in s]) > 0.99
