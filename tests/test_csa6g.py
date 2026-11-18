"""Test suite.

These are the properties the paper relies on.  If one of them stops holding,
a claim in the paper no longer holds.

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import dataclasses
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csa6g import crypto
from csa6g.datastore import (AuthenticatedStore, issue_admit_token,
                             open_admit_token)
from csa6g.eap import EdgeAdmissionProxy, replay_key
from csa6g.freshness import CounterTracker, ReplayWindow, TokenStore
from csa6g.harness import make_pool, run_policy
from csa6g.keys import (SessionKeys, derive_slice_keys, derive_token_key,
                        run_anchor_bootstrap, snssai_bytes)
from csa6g.metrics import jain
from csa6g.model import (AdmissionRequest, Demand, Node, Reason, ResourcePool,
                         SliceType, demand_norm)
from csa6g.params import (ADMIT_TOKEN_TTL_S, CLOCK_TOLERANCE_S, DEFAULT_NODE,
                          DEFAULT_POLICY_PARAMS, ENVELOPE_PLAINTEXT_BYTES,
                          ENVELOPE_WIRE_BYTES)
from csa6g.placement import can_place, select_node
from csa6g.policies import (POLICY_LADDER, ProposedNSACQuota,
                            UncontrolledFCFS, build_ladder, drf_shares,
                            progressive_fill)
from csa6g.stats import cliffs_delta, holm_bonferroni, tost
from csa6g.workload import StreamSpec, benign_stream, generate_stream


# --------------------------------------------------------------- fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def session(rng):
    return SessionKeys.create(run_anchor_bootstrap(rng),
                              SliceType.URLLC, 7)


@pytest.fixture
def request_():
    return AdmissionRequest(
        tenant_id=1, pseudonym=b"abcdefgh", slice_type=SliceType.URLLC,
        service_id=7, sla_latency_ms=1.0,
        demand=Demand(20.0, 16.0, 0.05), arrival_index=0, counter=1)


# ------------------------------------------------------- key derivation -----

def test_slice_keys_are_domain_separated(rng):
    k_anchor = run_anchor_bootstrap(rng)
    ks, km = derive_slice_keys(k_anchor, SliceType.URLLC, 7)
    assert ks != km, "K_slice and K_mac must be independent (key hierarchy)"
    assert len(ks) == len(km) == 32


def test_slice_keys_depend_on_snssai_and_service(rng):
    k_anchor = run_anchor_bootstrap(rng)
    a, _ = derive_slice_keys(k_anchor, SliceType.URLLC, 7)
    b, _ = derive_slice_keys(k_anchor, SliceType.EMBB, 7)
    c, _ = derive_slice_keys(k_anchor, SliceType.URLLC, 8)
    assert len({a, b, c}) == 3, "keys must bind to both S-NSSAI and service"


def test_snssai_is_four_bytes():
    assert len(snssai_bytes(SliceType.EMBB, 42)) == 4


def test_key_wipe_zeroises_and_blocks_reuse(session):
    session.wipe()
    assert set(bytes(session.kslice)) == {0}
    assert set(bytes(session.kmac)) == {0}
    assert session.epoch == 1
    with pytest.raises(RuntimeError):
        session.assert_live()


# ------------------------------------------------------------- envelope -----

def test_envelope_size_is_constant_across_slice_types(rng):
    sizes = set()
    for stype in (SliceType.URLLC, SliceType.EMBB, SliceType.MMTC):
        sk = SessionKeys.create(run_anchor_bootstrap(rng), stype, 1)
        req = AdmissionRequest(1, b"abcdefgh", stype, 1, 1.0,
                               Demand(999.0, 999.0, 9.0), 0, 1)
        sizes.add(crypto.seal(sk, req).size)
    assert sizes == {ENVELOPE_WIRE_BYTES}, \
        "fixed-size padding must make the wire length content-independent"


def test_roundtrip_preserves_the_request(session, request_):
    env = crypto.seal(session, request_)
    got = crypto.open_envelope(session, env, tenant_id=1)
    assert got.slice_type == request_.slice_type
    assert got.counter == request_.counter
    assert got.demand.cpu == pytest.approx(request_.demand.cpu, rel=1e-6)


def test_outer_hmac_detects_tampering(session, request_):
    env = crypto.seal(session, request_)
    assert crypto.verify_outer(session, env)
    assert not crypto.verify_outer(session, crypto.tamper_outer_tag(env))
    assert not crypto.verify_outer(session, crypto.tamper_ciphertext(env))


def test_rewriting_the_timestamp_breaks_the_outer_hmac(session, request_):
    """The outer HMAC covers the freshness fields, so a replay that refreshes
    the timestamp cannot survive it."""
    env = crypto.seal(session, request_)
    forged = crypto.parse(crypto.retime(env, time.time() + 1000))
    assert not crypto.verify_outer(session, forged)


def test_aead_rejects_a_wrong_key(rng, request_):
    a = SessionKeys.create(run_anchor_bootstrap(rng), SliceType.URLLC, 1)
    b = SessionKeys.create(run_anchor_bootstrap(rng), SliceType.URLLC, 1)
    env = crypto.seal(a, request_)
    with pytest.raises(crypto.EnvelopeError):
        crypto.open_envelope(b, env)


def test_padding_roundtrip():
    for n in (0, 1, 37, 250):
        msg = os.urandom(n)
        assert crypto.unpad_fixed(crypto.pad_fixed(msg)) == msg


def test_padding_is_exact_at_the_largest_representable_message():
    """The two-byte length field means the last legal message is size - 2.

    Sampling only round numbers missed this: at size - 1 the old bound let the
    message through and emitted size + 1 bytes, which unpadding then rejected
    as an integrity failure.
    """
    limit = ENVELOPE_PLAINTEXT_BYTES - 2
    for n in (limit - 1, limit):
        msg = os.urandom(n)
        padded = crypto.pad_fixed(msg)
        assert len(padded) == ENVELOPE_PLAINTEXT_BYTES
        assert crypto.unpad_fixed(padded) == msg
    with pytest.raises(crypto.EnvelopeError):
        crypto.pad_fixed(os.urandom(limit + 1))


def test_padding_rejects_oversized_plaintext():
    with pytest.raises(crypto.EnvelopeError):
        crypto.pad_fixed(os.urandom(4096))


def test_request_survives_the_envelope_bit_exactly(session):
    """No lossy field in the AEAD plaintext.

    The demand vector used to round-trip through float32, so the request the
    E-AP decided on differed from the one the offline harness saw by ~1e-7 --
    enough to move ``demand_norm`` across a quota boundary and desynchronise
    the two decision paths.
    """
    req = AdmissionRequest(
        tenant_id=3, pseudonym=b"abcdefgh", slice_type=SliceType.EMBB,
        service_id=11, sla_latency_ms=9.876543210987654,
        demand=Demand(80.12345678901234, 128.98765432109876, 0.5432109876543),
        arrival_index=1234567, counter=42)
    got = crypto.open_envelope(session, crypto.seal(session, req), tenant_id=3)
    assert got.demand == req.demand
    assert got.sla_latency_ms == req.sla_latency_ms
    assert got.arrival_index == req.arrival_index
    assert demand_norm(got.demand, (1200.0, 3072.0, 60.0)) == \
        demand_norm(req.demand, (1200.0, 3072.0, 60.0))


# ------------------------------------------------------------ freshness -----

def test_replay_window_rejects_a_repeated_nonce():
    w = ReplayWindow()
    now = time.time()
    n = os.urandom(16)
    assert w.check_and_record(n, now, now) == Reason.OK
    assert w.check_and_record(n, now, now) == Reason.NONCE_REPLAY


def test_replay_window_enforces_the_clock_tolerance():
    w = ReplayWindow()
    now = time.time()
    assert w.check(os.urandom(16), now - CLOCK_TOLERANCE_S - 1, now) \
        == Reason.STALE
    assert w.check(os.urandom(16), now + CLOCK_TOLERANCE_S + 1, now) \
        == Reason.FUTURE


def test_replay_window_is_bounded():
    w = ReplayWindow(capacity=64)
    now = time.time()
    for _ in range(500):
        w.record(os.urandom(16), now, now)
    assert len(w) <= 64
    assert w.evictions > 0


def test_counter_tracker_blocks_rollback():
    c = CounterTracker()
    kid = os.urandom(8)
    assert c.check(kid, 0, 5) == Reason.OK
    c.record(kid, 0, 5)
    assert c.check(kid, 0, 5) == Reason.COUNTER_ROLLBACK
    assert c.check(kid, 0, 4) == Reason.COUNTER_ROLLBACK
    assert c.check(kid, 0, 6) == Reason.OK


def test_counter_epoch_isolation():
    """After release the epoch changes, so old counter values are unknown
    rather than replayable."""
    c = CounterTracker()
    kid = os.urandom(8)
    c.record(kid, 0, 100)
    assert c.check(kid, 1, 1) == Reason.OK


def test_token_expiry():
    t = TokenStore(ttl_s=10.0)
    now = time.time()
    tok = os.urandom(16)
    t.issue(tok, now)
    assert t.validate(tok, now + 1) == Reason.OK
    assert t.validate(tok, now + 11) == Reason.TOKEN_EXPIRED
    assert t.validate(os.urandom(16), now) == Reason.AUTH_FAIL


# ------------------------------------------------------------ placement -----

def test_isolation_budget_is_enforced():
    node = Node(0, DEFAULT_NODE)
    d = Demand(1.0, 1.0, 0.01)
    assert can_place(node, d, SliceType.URLLC, beta=1)
    node.allocate(d, SliceType.URLLC)
    assert not can_place(node, d, SliceType.URLLC, beta=1), \
        "beta = 1 must forbid a second slice of the same type (placement rule)"
    assert can_place(node, d, SliceType.EMBB, beta=1), \
        "a different slice type is not a co-location conflict"


def test_placement_distinguishes_capacity_from_isolation():
    pool = ResourcePool(1, DEFAULT_NODE)
    d = Demand(1.0, 1.0, 0.01)
    pool.nodes[0].allocate(d, SliceType.URLLC)
    _, reason = select_node(pool.nodes, d, SliceType.URLLC, beta=1)
    assert reason == Reason.ISOLATION
    _, reason = select_node(pool.nodes, Demand(1e9, 1e9, 1e9),
                            SliceType.EMBB, beta=1)
    assert reason == Reason.INFEASIBLE


def test_placement_prefers_the_least_loaded_node():
    pool = ResourcePool(3, DEFAULT_NODE)
    pool.nodes[0].allocate(Demand(150.0, 1.0, 0.01), SliceType.EMBB)
    node, _ = select_node(pool.nodes, Demand(10.0, 1.0, 0.01),
                          SliceType.URLLC, beta=1)
    assert node.node_id != 0


# ------------------------------------------------------------ datastore -----

def test_commit_mac_detects_record_substitution(request_):
    store = AuthenticatedStore()
    from csa6g.datastore import AllocationRecord
    rec = AllocationRecord(os.urandom(16), 1, int(SliceType.URLLC), 0,
                           request_.demand, time.time())
    rec = store.write(rec)
    assert store.verify(rec.session_id)
    forged = AllocationRecord(rec.session_id, 999, int(SliceType.EMBB), 3,
                              Demand(1.0, 1.0, 1.0), time.time(),
                              transaction_id=rec.transaction_id,
                              key_epoch=rec.key_epoch)
    store._records[rec.session_id] = (forged,
                                      store._records[rec.session_id][1])
    assert not store.verify(rec.session_id)


def test_commit_failure_rolls_back_and_leaks_nothing(rng):
    """Step 5: a failed commit must release the resources and return only a
    transient code."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    stream = benign_stream(0, n_tenants=1, n_requests=1)
    req = stream[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    eap.store.fail_next = True
    env = crypto.seal(sk, req)
    out = eap.handle(env.raw, tenant_id=req.tenant_id)

    assert not out.decision.admitted
    assert out.internal_reason == Reason.COMMIT_FAIL
    assert out.external_reason == Reason.TRANSIENT
    assert eap.store.rollbacks == 1
    assert pool.used_tup == (0.0, 0.0, 0.0), "resources were not released"


def test_admit_token_roundtrip_and_expiry():
    tk = derive_token_key(os.urandom(32), os.urandom(8), 0)
    sid = os.urandom(16)
    tok = issue_admit_token(tk, sid, ttl_s=5.0)
    got, reason = open_admit_token(tk, tok)
    assert reason == Reason.OK and got == sid
    _, reason = open_admit_token(tk, tok, now=time.time() + 10)
    assert reason == Reason.TOKEN_EXPIRED


# ------------------------------------------------------------------ eap -----

def test_full_admit_and_release_cycle(rng):
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(1, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id,
                     run_secondary_auth=True)
    assert out.decision.admitted
    assert pool.used_tup != (0.0, 0.0, 0.0)

    assert eap.release(out.session_id, out.token).ok
    assert pool.used_tup == (0.0, 0.0, 0.0)
    assert sk.wiped, "K_slice must be wiped on release (Step 6)"


def test_envelope_replay_after_release_fails(rng):
    """Replay-after-release: the session key is gone, so the handle no longer
    resolves and the envelope is unusable."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(2, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    env = crypto.seal(sk, req)
    out = eap.handle(env.raw, tenant_id=req.tenant_id)
    eap.release(out.session_id, out.token)

    replay = eap.handle(env.raw, tenant_id=req.tenant_id)
    assert not replay.decision.admitted
    assert replay.external_reason == Reason.NOT_AUTHORIZED


def test_minimal_disclosure_is_two_categories_not_one():
    """Two externally visible outcomes, and the second one is deliberate.

    RESULTS.md used to say denials collapse to the single category
    `not_authorized`. They do not: a commit failure externalises to
    `transient_error` so the requester retries instead of giving up. Every
    other internal reason -- impostor, stale, replay, quota, isolation -- maps
    to `not_authorized` and is mutually indistinguishable, which is the actual
    Step 5 property.
    """
    internal = [v for k, v in vars(Reason).items()
                if k.isupper() and isinstance(v, str) and v != Reason.OK]
    external = {Reason.externalise(r) for r in internal}
    assert external == {Reason.NOT_AUTHORIZED, Reason.TRANSIENT}
    assert Reason.externalise(Reason.COMMIT_FAIL) == Reason.TRANSIENT
    for r in (Reason.AUTH_FAIL, Reason.STALE, Reason.NONCE_REPLAY,
              Reason.COUNTER_ROLLBACK, Reason.INTEGRITY, Reason.QUOTA,
              Reason.RATE, Reason.INFEASIBLE, Reason.ISOLATION):
        assert Reason.externalise(r) == Reason.NOT_AUTHORIZED, r


def test_denials_never_leak_the_internal_reason(rng):
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(3, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    env = crypto.tamper_outer_tag(crypto.seal(sk, req))
    out = eap.handle(env.raw, tenant_id=req.tenant_id)
    assert out.internal_reason == Reason.INTEGRITY
    assert out.external_reason == Reason.NOT_AUTHORIZED


def test_aead_counter_observes_the_admission_path(rng):
    """Positive control for the assertion below.

    Without this, a zero decryption count proves only that the counter is not
    plugged in.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(4, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    env = crypto.seal(sk, req)
    with crypto.count_aead_operations() as ops:
        out = eap.handle(env.raw, tenant_id=req.tenant_id)
    assert out.decision.admitted
    assert ops["decrypt"] == 1, "an admitted request opens the envelope once"


def test_hmac_failure_costs_no_aead(rng):
    """The two-tag design exists so a bad envelope is rejected without an AEAD
    decryption. If this fails, the outer HMAC is redundant.

    Counted, not timed.  ``timings["aead"] == 0.0`` -- what this asserted
    before -- is what ``_Timer`` initialises the stage to, so it stayed true
    even with a real ``open_envelope`` spliced onto the HMAC-failure path: the
    assertion could not fail for the reason it existed to guard.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(4, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    forged = crypto.tamper_outer_tag(crypto.seal(sk, req)).raw
    with crypto.count_aead_operations() as ops:
        out = eap.handle(forged, tenant_id=req.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.INTEGRITY
    assert ops["decrypt"] == 0, "AEAD ran on an envelope that failed the HMAC"


def test_the_aead_counter_can_actually_fail(rng):
    """Guard on the guard: splice AEAD work onto the rejection path and the
    counter must notice.  This is the check the timer-based assertion could
    not make."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(4, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    forged = crypto.tamper_outer_tag(crypto.seal(sk, req)).raw

    real_verify = crypto.verify_outer

    def leaky_verify(sk_, env_):
        ok = real_verify(sk_, env_)
        if not ok:                       # a decryption the design forbids
            try:
                crypto.open_envelope(sk_, env_)
            except crypto.EnvelopeError:
                pass
        return ok

    crypto.verify_outer = leaky_verify
    try:
        with crypto.count_aead_operations() as ops:
            out = eap.handle(forged, tenant_id=req.tenant_id)
    finally:
        crypto.verify_outer = real_verify

    assert out.decision.timings["aead"] == 0.0, \
        "the timer stays at zero -- which is exactly why it cannot be the test"
    assert ops["decrypt"] == 1, "the counter must see the spliced decryption"


def test_secondary_auth_failure_is_reachable(rng):
    """Stage 7's AUTH_FAIL branch must be exercisable.

    The default authenticator recomputes an HMAC against the key it just used,
    so it cannot return False and the denial branch was unreachable dead code
    that no test could cover. The E-AP takes the authenticator as a parameter
    so a failing one can be injected.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool, authenticator=lambda sk: False)
    req = benign_stream(8, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id,
                     run_secondary_auth=True)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.AUTH_FAIL
    assert out.external_reason == Reason.NOT_AUTHORIZED
    assert pool.used_tup == (0.0, 0.0, 0.0), "a failed NSSAA must allocate no"\
        " resources"


def test_a_failed_secondary_auth_burns_no_freshness_state(rng):
    """Freshness and counter state commit only after authentication.

    Otherwise a request that fails NSSAA would consume its own nonce, and the
    legitimate retry would be rejected as a replay.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool, authenticator=lambda sk: False)
    req = benign_stream(9, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    env = crypto.seal(sk, req)
    assert eap.handle(env.raw, tenant_id=req.tenant_id,
                      run_secondary_auth=True).internal_reason \
        == Reason.AUTH_FAIL
    assert len(eap.replay) == 0, "a rejected request must not burn its nonce"
    # The identical envelope now succeeds once authentication is restored.
    eap.authenticator = lambda sk: True
    assert eap.handle(env.raw, tenant_id=req.tenant_id,
                      run_secondary_auth=True).decision.admitted


def test_integrity_is_checked_before_the_replay_window(rng):
    """No unauthenticated input may reach the nonce cache.

    The replay window is keyed on an attacker-supplied nonce, so consulting it
    before verifying the outer HMAC is a membership oracle: the two outcomes
    return the same external code but do measurably different amounts of work.
    Experiment 7 quantifies it; this pins the ordering.
    """
    pool = make_pool(4)
    req = benign_stream(10, n_tenants=1, n_requests=1)[0]

    for integrity_first, expected in ((True, Reason.INTEGRITY),
                                      (False, Reason.NONCE_REPLAY)):
        eap = EdgeAdmissionProxy(pool, integrity_first=integrity_first)
        sk = eap.register_session(run_anchor_bootstrap(rng),
                                  req.slice_type, req.service_id,
                                  req.tenant_id)
        now = time.time()
        good = crypto.seal(sk, req, timestamp=now).raw
        assert eap.handle(good, now=now,
                          tenant_id=req.tenant_id).decision.admitted
        eap.release_slice(
            next(iter(eap.sessions)))          # keep the pool empty

        # A forgery carrying a nonce the E-AP has already seen. Unauthenticated
        # -- the tag is broken -- so it must be rejected on integrity, not on
        # the cache lookup, or the cache answered before anyone proved anything.
        probe = bytearray(good)
        probe[-1] ^= 0xFF
        out = eap.handle(bytes(probe), now=now, tenant_id=req.tenant_id)
        assert out.internal_reason == expected, (
            f"integrity_first={integrity_first}: expected {expected}, "
            f"got {out.internal_reason}")
        assert out.external_reason == Reason.NOT_AUTHORIZED


def test_release_returns_the_quota_to_the_policy(rng):
    """Step 6 gives budget back to the policy, not only to the pool.

    Without it a tenant's consumed quota only ever grew, so a tenant cycling
    admit/release would starve itself while the resources sat idle.  The rate
    gate is opened by configuration so that this fails on the quota if it fails
    at all -- it is a separate control with its own test.

    It used to be isolated instead by spacing ``arrival_index`` past the
    rate-gate window, which worked only because the gate was indexed on a
    client-supplied field.  Now that the proxy stamps the arrival ordinal
    itself -- see ``test_a_client_cannot_inflate_arrival_index_past_the_rate_
    gate`` -- a sender cannot move that window, so the control under test has
    to be isolated the honest way.
    """
    pool = make_pool(24)
    policy = ProposedNSACQuota(
        dataclasses.replace(DEFAULT_POLICY_PARAMS, p_rate_max=10_000))
    eap = EdgeAdmissionProxy(pool, policy=policy)
    tenant = 1
    demand = Demand(80.0, 128.0, 0.5)          # a typical eMBB slice
    cost = demand_norm(demand, pool.total_tup)
    assert 2 * cost > DEFAULT_POLICY_PARAMS.p_base_quota_frac, \
        "the slice must be big enough that two unreleased ones exhaust q_base"

    for cycle in range(12):
        sk = eap.register_session(run_anchor_bootstrap(rng),
                                  SliceType.EMBB, 3, tenant)
        req = AdmissionRequest(
            tenant_id=tenant, pseudonym=b"abcdefgh", slice_type=SliceType.EMBB,
            service_id=3, sla_latency_ms=10.0, demand=demand,
            arrival_index=cycle * 128, counter=1)
        out = eap.handle(crypto.seal(sk, req).raw, tenant_id=tenant)
        assert out.decision.admitted, \
            f"starved on cycle {cycle} ({out.internal_reason})"
        assert policy.nsac.state(tenant).allocated == pytest.approx(cost)
        assert eap.release(out.session_id, out.token).ok
        assert policy.nsac.state(tenant).allocated == pytest.approx(0.0)

    assert eap.live_sessions == {}, "released sessions must leave the count"
    assert pool.used_tup == (0.0, 0.0, 0.0)


def test_release_revokes_the_admit_token(rng):
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(6, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert eap.release(out.session_id, out.token).ok
    assert eap.release(out.session_id,
                       out.token).internal_reason == Reason.AUTH_FAIL


def test_release_rejects_an_expired_admit_token(rng):
    """The expired-token class of experiment 3, through the real Step 6 path."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(7, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    now = time.time()
    out = eap.handle(crypto.seal(sk, req, timestamp=now).raw, now=now,
                     tenant_id=req.tenant_id)
    assert eap.release(out.session_id, out.token,
                       now=now + ADMIT_TOKEN_TTL_S + 1.0) \
        .internal_reason == Reason.TOKEN_EXPIRED


def test_tenant_attribution_is_session_bound(rng):
    """Quota and fairness state must key on the tenant bound at registration,
    not on a label the requester supplies per call: an adversary free to claim
    an arbitrary tenant evades per-tenant caps, or drains a victim's."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(12, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=999)
    assert out.decision.admitted
    assert set(eap.live_sessions) == {req.tenant_id}, \
        "the caller-claimed tenant label must never reach quota accounting"



def test_slice_and_service_context_are_enforced(rng):
    """Key derivation is not enough: the decrypted claims must be checked."""
    pool = make_pool()
    eap = EdgeAdmissionProxy(pool)
    tenant_id = 77
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              SliceType.URLLC, 3, tenant_id)
    req = AdmissionRequest(
        tenant_id=tenant_id, pseudonym=b"CTXTEST1",
        slice_type=SliceType.EMBB, service_id=4,
        sla_latency_ms=10.0, demand=Demand(1.0, 1.0, 0.01),
        arrival_index=0, counter=1)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.CONTEXT_MISMATCH
    assert out.external_reason == Reason.NOT_AUTHORIZED
    assert len(eap.store) == 0
    assert pool.nodes[0].used == [0.0, 0.0, 0.0]

def test_release_externalises_like_admission(rng):
    """The minimal-disclosure property must hold on the release path too.

    ``release`` used to return the internal reason bare and left the
    externalisation to its callers; an invariant enforced by every caller
    separately is not an invariant.  An impostor probing with a random
    session_id and a requester holding an expired token must see the same
    external code."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(13, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    now = time.time()
    out = eap.handle(crypto.seal(sk, req, timestamp=now).raw, now=now,
                     tenant_id=req.tenant_id)

    impostor = eap.release(os.urandom(16), out.token, now=now)
    assert impostor.internal_reason == Reason.AUTH_FAIL
    expired = eap.release(out.session_id, out.token,
                          now=now + ADMIT_TOKEN_TTL_S + 1.0)
    assert expired.internal_reason == Reason.TOKEN_EXPIRED
    assert {impostor.external_reason, expired.external_reason} \
        == {Reason.NOT_AUTHORIZED}, \
        "an impostor and an expired token must be indistinguishable outside"

    good = eap.release(out.session_id, out.token, now=now)
    assert good.ok and good.external_reason == Reason.OK


def test_admit_token_key_is_operator_side_not_derivable_by_the_ue(session):
    """The admit token is an E-AP-issued release capability, so its key lives
    in an E-AP-only key space: it is derived from K_EAP, never from the
    UE-held K_slice/K_mac.  A tenant holding the session keys therefore cannot
    reconstruct the token key, and a token it seals under anything it can
    derive does not open under the real one."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    k_eap = os.urandom(32)
    tk = derive_token_key(k_eap, session.key_id, session.key_epoch)
    # The token key is not any key the UE holds, nor anything it can derive
    # from them (the UE never sees K_EAP).
    assert bytes(tk) not in (bytes(session.kslice), bytes(session.kmac))
    tok = issue_admit_token(tk, os.urandom(16))
    with pytest.raises(InvalidTag):
        AESGCM(bytes(session.kslice)).decrypt(tok[:12], tok[12:], b"admit")


def test_tenant_cannot_mint_a_token_the_release_path_accepts(rng):
    """End-to-end: a registered tenant holding K_slice/K_mac cannot forge a
    release token.  Even the old K_slice-derived construction it might attempt
    is rejected, and so is any self-sealed token, because the release path
    gates on the E-AP's issued set and opens under the operator-side key."""
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted
    sid = out.session_id

    # Attacker attempts, all from keys/values the UE actually holds:
    from csa6g.keys import _expand
    forged_key_old = _expand(bytes(sk.kslice), b"token", 32)   # the v21 design
    forgeries = [
        issue_admit_token(forged_key_old, sid),
        issue_admit_token(bytes(sk.kslice), sid),
        issue_admit_token(bytes(sk.kmac), sid),
        issue_admit_token(os.urandom(32), sid),
    ]
    for tok in forgeries:
        rel = eap.release(sid, tok)
        assert not rel.ok, "a UE-forged token was accepted at release"
        assert rel.external_reason == Reason.NOT_AUTHORIZED
    # The genuine E-AP-issued token still works.
    assert eap.release(sid, out.token).ok


def test_unknown_key_handle_pays_the_mac_before_denial(rng):
    """Key-handle liveness must not be readable off the rejection time.

    A handle that does not resolve is denied with the same external code as a
    bad tag; without equalisation it returned one HMAC earlier, so the
    response time answered "is this session still live?" for anyone replaying
    a captured envelope.  The E-AP burns a decoy MAC on that path."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(14, n_tenants=1, n_requests=1)[0]
    unregistered = SessionKeys.create(run_anchor_bootstrap(rng),
                                      req.slice_type, req.service_id)
    out = eap.handle(crypto.seal(unregistered, req).raw,
                     tenant_id=req.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.AUTH_FAIL
    assert out.external_reason == Reason.NOT_AUTHORIZED
    assert out.decision.timings["hmac"] > 0.0, \
        "the unknown-handle rejection must include the decoy MAC work"


def test_token_store_prunes_expired_entries():
    """Tokens whose sessions simply let them expire must not accumulate
    forever; the store sweeps them on an amortised cadence."""
    store = TokenStore(ttl_s=10.0)
    now = 1000.0
    for _ in range(TokenStore.PRUNE_EVERY - 1):
        store.issue(os.urandom(8), now)
    assert len(store._issued) == TokenStore.PRUNE_EVERY - 1
    store.issue(os.urandom(8), now + 20.0)   # every earlier token is expired
    assert len(store._issued) == 1, "the sweep must drop expired tokens"


def test_admit_token_ttl_has_one_source(rng, session):
    """The sealed token and the revocation list must expire together.

    They used to carry independent 300.0 defaults; retuning either alone would
    have left experiment 3's expired-token class passing for the wrong reason.
    """
    now = time.time()
    tk = derive_token_key(os.urandom(32), session.key_id, session.key_epoch)
    tok = issue_admit_token(tk, os.urandom(16))
    store = TokenStore()
    store.issue(tok, now)
    just_inside = now + ADMIT_TOKEN_TTL_S - 1.0
    just_outside = now + ADMIT_TOKEN_TTL_S + 1.0
    assert store.validate(tok, just_inside) == Reason.OK
    assert open_admit_token(tk, tok, now=just_inside)[1] == Reason.OK
    assert store.validate(tok, just_outside) == Reason.TOKEN_EXPIRED
    assert open_admit_token(tk, tok,
                            now=just_outside)[1] == Reason.TOKEN_EXPIRED


def test_admit_token_expiry_follows_the_proxys_clock_not_the_wall_clock(rng):
    """A token minted on a simulated clock must be live on that same clock.

    The E-AP stamps the revocation-list entry and the record's ``expires_at``
    from its injectable ``now``; the sealed token used to read the wall clock
    instead.  Driven an hour ahead, the proxy therefore issued tokens that
    ``open_admit_token`` rejected as expired the instant they were created --
    a one-second-old session could not be released.  The offset here exceeds
    the TTL, which is what makes the two timelines separable at all.
    """
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=2)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)

    sim_now = time.time() + 3600.0            # > ADMIT_TOKEN_TTL_S
    env = crypto.seal(sk, req, timestamp=sim_now)
    out = eap.handle(env.raw, now=sim_now, tenant_id=req.tenant_id)
    assert out.decision.admitted, out.internal_reason

    tk = derive_token_key(eap._k_eap, sk.key_id, sk.key_epoch)
    assert open_admit_token(tk, out.token, now=sim_now)[1] == Reason.OK
    rel = eap.release(out.session_id, out.token, now=sim_now)
    assert rel.ok, \
        f"a freshly issued token was not accepted: {rel.internal_reason}"

    # The TTL still binds on that clock -- the fix moves the timeline, it does
    # not disable expiry.
    eap2 = EdgeAdmissionProxy(make_pool(8))
    sk2 = eap2.register_session(run_anchor_bootstrap(rng), req.slice_type,
                                req.service_id, req.tenant_id)
    env2 = crypto.seal(sk2, req, timestamp=sim_now)
    out2 = eap2.handle(env2.raw, now=sim_now, tenant_id=req.tenant_id)
    expired_at = sim_now + ADMIT_TOKEN_TTL_S + 1.0
    tk2 = derive_token_key(eap2._k_eap, sk2.key_id, sk2.key_epoch)
    assert open_admit_token(tk2, out2.token,
                            now=expired_at)[1] == Reason.TOKEN_EXPIRED


# --------------------------------------------------------------- policies ---

def test_progressive_fill_is_max_min_fair():
    alloc = progressive_fill(np.array([0.1, 0.5, 0.5]), 1.0)
    assert alloc[0] == pytest.approx(0.1, abs=1e-6)
    assert alloc[1] == pytest.approx(alloc[2], abs=1e-6)
    assert alloc.sum() == pytest.approx(1.0, abs=1e-6)


def test_progressive_fill_never_exceeds_demand_or_capacity():
    d = np.array([0.05, 0.05, 0.05])
    alloc = progressive_fill(d, 1.0)
    assert np.all(alloc <= d + 1e-9)


def test_drf_equalises_dominant_shares():
    """Ghodsi et al.'s worked example: 9 CPU, 18 GB; user A wants (1, 4) per
    task, user B wants (3, 1). DRF equalises dominant shares."""
    D = np.array([[100.0, 400.0], [300.0, 100.0]])
    cap = np.array([9.0, 18.0])
    scale = drf_shares(D, cap)
    a, b = scale[0] * D[0], scale[1] * D[1]
    dom_a = max(a[0] / cap[0], a[1] / cap[1])
    dom_b = max(b[0] / cap[0], b[1] / cap[1])
    assert dom_a == pytest.approx(dom_b, abs=0.02)
    assert np.all(a + b <= cap + 1e-6)


def test_every_policy_respects_capacity():
    stream = generate_stream(0, StreamSpec(malicious_ratio=5.0))
    for policy in build_ladder():
        run = run_policy(policy, stream, 0, num_nodes=24, beta=3)
        assert 0.0 <= run.dominant_utilisation <= 1.0 + 1e-9, policy.name


def test_uncontrolled_policy_is_the_fairness_floor():
    """B0 must be worse than the proposed policy under flood, or the whole
    comparison is vacuous."""
    stream = generate_stream(0, StreamSpec(malicious_ratio=10.0))
    b0 = run_policy(UncontrolledFCFS(), stream, 0, num_nodes=24, beta=3)
    p = run_policy(ProposedNSACQuota(), stream, 0, num_nodes=24, beta=3)
    assert p.fi_benign > b0.fi_benign
    assert p.malicious_capacity_share < b0.malicious_capacity_share


# -------------------------------------------------------------- metrics -----

def test_jain_index_bounds():
    assert jain([1, 1, 1, 1]) == pytest.approx(1.0)
    assert jain([1, 0, 0, 0]) == pytest.approx(0.25)


def test_jain_is_undefined_when_nothing_was_allocated():
    """Not 1.0.  Returning the best possible fairness score for a run that
    granted nothing hides exactly the failure the metric exists to catch."""
    assert np.isnan(jain([]))
    assert np.isnan(jain([0.0, 0.0, 0.0]))


def test_demand_norm_is_the_dominant_share():
    total = (100.0, 1000.0, 10.0)
    d = Demand(50.0, 100.0, 0.1)
    assert demand_norm(d, total) == pytest.approx(0.5)


# ---------------------------------------------------------------- stats -----

def test_tost_declares_equivalence_for_identical_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.001, 200)
    b = rng.normal(1.0, 0.001, 200)
    assert tost(a, b, 0.03, paired=True).equivalent


def test_tost_rejects_equivalence_for_a_real_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(1.5, 0.01, 200)
    b = rng.normal(1.0, 0.01, 200)
    assert not tost(a, b, 0.03, paired=True).equivalent


def test_cliffs_delta_signs():
    d, mag = cliffs_delta([5, 6, 7], [1, 2, 3])
    assert d == pytest.approx(1.0) and mag == "large"
    d, _ = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert d == pytest.approx(0.0)


def test_holm_is_monotone_and_conservative():
    adj, rej = holm_bonferroni([0.001, 0.02, 0.03, 0.5])
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj >= np.array([0.001, 0.02, 0.03, 0.5]) - 1e-12)


# ---------------------------------------------------------- determinism -----

def test_stream_is_a_pure_function_of_the_seed():
    a = generate_stream(7, StreamSpec(malicious_ratio=5.0))
    b = generate_stream(7, StreamSpec(malicious_ratio=5.0))
    assert [(r.tenant_id, r.slice_type, r.demand, r.malicious) for r in a] \
        == [(r.tenant_id, r.slice_type, r.demand, r.malicious) for r in b]


def test_enabling_security_does_not_perturb_the_workload():
    """The regression the previous codebase had: security decisions drawing
    from the arrival RNG, so turning security on changed the tenant population
    and any measured difference was partly a different workload."""
    spec = StreamSpec(malicious_ratio=5.0)
    stream = generate_stream(3, spec)
    before = [(r.tenant_id, r.arrival_index) for r in stream]

    pool = make_pool(24)
    eap = EdgeAdmissionProxy(pool)
    rng = np.random.default_rng(0)
    for req in stream[:50]:
        sk = eap.register_session(run_anchor_bootstrap(rng),
                                  req.slice_type, req.service_id,
                                  req.tenant_id)
        eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id,
                   run_secondary_auth=True)

    assert [(r.tenant_id, r.arrival_index) for r in stream] == before


@pytest.mark.parametrize("policy_cls", POLICY_LADDER,
                         ids=lambda c: c.name)
def test_eap_and_harness_agree(policy_cls):
    """The two admission loops must decide identically.

    Experiment 4 evaluates the policy ladder through ``harness.run_policy``,
    which is the E-AP's decision with the cryptography removed -- millions of
    envelopes would add hours to measure a cost that is identical across every
    arm.  That is valid only while the two paths cannot drift, which this test
    checks.

    So drive one stream through both and compare request for request.
    Parametrised per policy so a failure names the one that drifted, including
    the offline allocators, whose ``prepare`` the E-AP has to be given as well.
    """
    spec = StreamSpec(n_benign_tenants=12, n_attacker_tenants=4,
                      n_benign_requests=60, malicious_ratio=3.0)
    stream = generate_stream(21, spec)
    nodes, beta = 8, 2

    harness_run = run_policy(policy_cls(), stream, 21, num_nodes=nodes,
                             beta=beta)

    online = policy_cls()
    pool = make_pool(nodes)
    eap = EdgeAdmissionProxy(pool, policy=online, beta=beta)
    online.prepare(stream)                # B3/B4 need the whole stream
    rng = np.random.default_rng(21)
    anchors, keys, now = {}, {}, time.time()
    eap_accepted = np.zeros(len(stream), dtype=bool)

    for i, req in enumerate(stream):
        if req.tenant_id not in anchors:
            anchors[req.tenant_id] = run_anchor_bootstrap(rng)
        context = (req.tenant_id, req.slice_type, req.service_id)
        if context not in keys:
            keys[context] = eap.register_session(
                anchors[req.tenant_id], req.slice_type,
                req.service_id, req.tenant_id)
        env = crypto.seal(keys[context], req, timestamp=now)
        out = eap.handle(env.raw, now=now, tenant_id=req.tenant_id)
        eap_accepted[i] = out.decision.admitted

    mismatch = np.flatnonzero(eap_accepted != harness_run.accepted)
    assert mismatch.size == 0, (
        f"{online.name}: {mismatch.size} of {len(stream)} decisions differ "
        f"between the E-AP and the offline harness, first at request "
        f"{int(mismatch[0])}")
    # Same decisions must leave the pool in the same state, which also catches
    # a divergence in placement rather than in admission.
    assert pool.dominant_utilisation() == pytest.approx(
        harness_run.dominant_utilisation, abs=1e-12), online.name


def test_policies_all_see_the_identical_stream():
    stream = generate_stream(11, StreamSpec(malicious_ratio=10.0))
    snapshot = [id(r) for r in stream]
    for policy in build_ladder():
        run_policy(policy, stream, 11, num_nodes=24, beta=3)
    assert [id(r) for r in stream] == snapshot


# ----------------------------------------- structured IV / nonce (SP 800-38D)


def test_iv_unique_across_one_million_seals(session):
    """No two freshness tokens allocated under one key share an AEAD IV.

    Uniqueness is structural -- epoch || atomically-allocated sequence -- so
    this holds deterministically, not merely with high probability.  The
    check runs a million allocations to exercise the allocator itself.
    """
    from csa6g.keys import iv_from_nonce, parse_nonce
    n = 1_000_000
    start = session._seq_next
    seen = set()
    for _ in range(n):
        seen.add(iv_from_nonce(session.next_nonce()))
    assert len(seen) == n, "duplicate AEAD IV under one K_slice"
    assert session._seq_next == start + n, "sequence must advance once per seal"
    epoch, seq = parse_nonce(session.next_nonce())
    assert epoch == session.key_epoch and seq == start + n


def test_concurrent_sealing_never_duplicates_a_sequence(session):
    """Sequence allocation is atomic: racing threads never share a number."""
    import threading
    from csa6g.keys import parse_nonce
    per_thread, n_threads = 2000, 8
    out: list[list[bytes]] = [[] for _ in range(n_threads)]

    def worker(i: int) -> None:
        out[i] = [session.next_nonce() for _ in range(per_thread)]

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [parse_nonce(nc)[1] for chunk in out for nc in chunk]
    assert len(set(seqs)) == per_thread * n_threads, \
        "two concurrent seals drew the same encryption sequence"


def test_replay_after_cache_eviction_is_rejected_before_the_aead(rng):
    """The review's attack: reuse a nonce once the bounded replay cache has
    evicted it.  The sequence tracker rejects the rolled-back sequence before
    any AEAD work, so the IV is never presented to GCM twice."""
    from csa6g.freshness import ReplayWindow
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool, replay_window=ReplayWindow(capacity=1))
    stream = benign_stream(0, n_tenants=1, n_requests=2)
    req = stream[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    now = time.time()

    first = crypto.seal(sk, req, timestamp=now)
    out1 = eap.handle(first.raw, now=now, tenant_id=req.tenant_id)
    assert out1.decision.admitted
    eap.release_slice(out1.session_id)

    second = AdmissionRequest(
        tenant_id=req.tenant_id, pseudonym=req.pseudonym,
        slice_type=req.slice_type, service_id=req.service_id,
        sla_latency_ms=req.sla_latency_ms, demand=req.demand,
        arrival_index=1, counter=req.counter + 1)
    out2 = eap.handle(crypto.seal(sk, second, timestamp=now + 0.5).raw,
                      now=now + 0.5, tenant_id=req.tenant_id)
    assert out2.decision.admitted
    # Probe the composite cache key the E-AP actually writes. Asserting on the
    # bare nonce here would be vacuously true -- no bare nonce is ever a key --
    # so the eviction precondition would silently stop being checked and this
    # test would no longer be about eviction at all.
    assert replay_key(sk.key_id, first.nonce) not in eap.replay._seen, \
        "eviction precondition"

    with crypto.count_aead_operations() as ops:
        out3 = eap.handle(first.raw, now=now + 1.0, tenant_id=req.tenant_id)
    assert not out3.decision.admitted
    assert out3.internal_reason == Reason.SEQUENCE_ROLLBACK
    assert ops["decrypt"] == 0, \
        "a replayed IV must never reach the AEAD, even post-eviction"


def test_restart_yields_a_fresh_key_not_a_resumed_sequence(rng):
    """A restarted client cannot resume the old sequence: key material is
    session-scoped, so re-registration -- even from the *same* anchor --
    derives a fresh K_slice, which is the 'rotate the key' arm of the
    SP 800-38D requirement.  The sequence restarting at zero is then safe."""
    anchor = run_anchor_bootstrap(rng)
    a = SessionKeys.create(anchor, SliceType.URLLC, 7)
    b = SessionKeys.create(anchor, SliceType.URLLC, 7)   # "after restart"
    assert bytes(a.kslice) != bytes(b.kslice), \
        "re-registration must never reproduce a K_slice that has sealed data"
    assert bytes(a.kmac) != bytes(b.kmac)
    # Both sequences start at zero -- allowed, precisely because the keys
    # differ: equal IV prefixes under distinct keys are harmless.
    from csa6g.keys import parse_nonce
    assert parse_nonce(a.next_nonce())[1] == parse_nonce(b.next_nonce())[1] == 0


def test_key_epoch_rotation_invalidates_previous_epoch(rng, request_):
    """Envelopes minted before a key-epoch rotation die afterwards, and a
    forged current-key envelope claiming an old epoch is caught by the
    receiver's epoch check."""
    from csa6g.keys import build_nonce
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    anchor = run_anchor_bootstrap(rng)
    sk = eap.register_session(anchor, request_.slice_type,
                              request_.service_id, request_.tenant_id)
    stale = crypto.seal(sk, request_)

    sk.rotate_key_epoch(anchor)
    assert sk.key_epoch == 1

    # (a) the pre-rotation envelope fails outright: its MACs were minted
    # under the previous epoch's keys.
    out = eap.handle(stale.raw, tenant_id=request_.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.INTEGRITY

    # (b) a holder of the *current* keys who forges an old-epoch prefix into
    # the freshness token passes the HMAC but is stopped by the epoch check
    # -- before any AEAD work.
    forged_nonce = build_nonce(0, 12345)
    env = crypto.seal(sk, request_, nonce=forged_nonce)
    with crypto.count_aead_operations() as ops:
        out = eap.handle(env.raw, tenant_id=request_.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.EPOCH_MISMATCH
    assert ops["decrypt"] == 0

    # (c) the current epoch still works end to end.
    fresh = crypto.seal(sk, request_)
    out = eap.handle(fresh.raw, tenant_id=request_.tenant_id)
    assert out.decision.admitted, out.internal_reason


def test_sequence_exhaustion_raises_instead_of_wrapping(session):
    from csa6g.keys import MAX_ENCRYPTION_SEQUENCE, SequenceExhausted
    session._seq_next = MAX_ENCRYPTION_SEQUENCE + 1
    with pytest.raises(SequenceExhausted):
        session.next_nonce()
    session._seq_next = MAX_ENCRYPTION_SEQUENCE
    session.next_nonce()                     # the last sequence is usable
    with pytest.raises(SequenceExhausted):
        session.next_nonce()


def test_two_sessions_sharing_a_freshness_token_are_both_admitted(rng):
    """A colliding n_u across *different* sessions must not deny either one.

    Since the token became epoch || sequence || random(4), two sessions at the
    same point in their sequence differ only in 4 random bytes, so a shared
    replay window keyed on the bare token denies a legitimate request with
    ``nonce_replay`` at 2^-32 per pair -- a coin flip over a full 2^16-entry
    window, and ~1 in 240 runs of experiment 5.  The forced collision below is
    the deterministic form of that birthday event.
    """
    from csa6g.keys import build_nonce
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    stream = benign_stream(0, n_tenants=2, n_requests=4)
    a, b = stream[0], stream[1]
    ska = eap.register_session(run_anchor_bootstrap(rng), a.slice_type,
                               a.service_id, tenant_id=1)
    skb = eap.register_session(run_anchor_bootstrap(rng), b.slice_type,
                               b.service_id, tenant_id=2)
    assert ska.key_id != skb.key_id
    now = time.time()

    tail = b"\xde\xad\xbe\xef"
    n_a = build_nonce(ska.key_epoch, 0, tail)
    n_b = build_nonce(skb.key_epoch, 0, tail)
    assert n_a == n_b, "precondition: the two sessions draw the same token"

    env_a = crypto.seal(ska, a, nonce=n_a, timestamp=now)
    out_a = eap.handle(env_a.raw, now=now, tenant_id=a.tenant_id)
    assert out_a.decision.admitted, out_a.internal_reason

    env_b = crypto.seal(skb, b, nonce=n_b, timestamp=now)
    out_b = eap.handle(env_b.raw, now=now, tenant_id=b.tenant_id)
    assert out_b.decision.admitted, (
        "a distinct session's genuine first request was denied because "
        f"another session drew the same token: {out_b.internal_reason}")

    # ...and the widened key must not have cost same-session replay detection.
    again = eap.handle(env_a.raw, now=now, tenant_id=a.tenant_id)
    assert not again.decision.admitted
    assert again.internal_reason == Reason.NONCE_REPLAY


def test_malformed_plaintext_under_a_valid_key_denies_cleanly(rng):
    """An authentic envelope carrying an undecodable request is a denial.

    The AEAD proves who sealed the plaintext, not that the plaintext is a
    well-formed request.  A client holding real session keys can seal a pad
    length that consumes the whole block, or a slice octet that is not a
    TS 23.501 codepoint; both used to raise out of ``handle`` as
    ``struct.error`` / ``ValueError``, producing no decision, no reason code
    and no denial record -- an unhandled exception per attempt, reachable by
    any tenant that finished registration.
    """
    from csa6g.crypto import _HEADER, _outer_mac, VERSION
    from csa6g.keys import iv_from_nonce
    from csa6g.params import ENVELOPE_PLAINTEXT_BYTES, GCM_TAG_BYTES
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=2)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = time.time()

    def seal_raw(plaintext: bytes) -> bytes:
        """Seal an arbitrary block, bypassing pad_fixed -- a hostile client."""
        nonce = sk.next_nonce()
        blob = AESGCM(bytes(sk.kslice)).encrypt(iv_from_nonce(nonce),
                                                plaintext, b"")
        ct, tag = blob[:-GCM_TAG_BYTES], blob[-GCM_TAG_BYTES:]
        hdr = _HEADER.pack(VERSION, sk.key_id, nonce, now)
        return hdr + ct + tag + _outer_mac(bytes(sk.kmac), hdr, ct, tag)

    # (a) pad length consumes the whole block -> zero-byte request body.
    whole_block = (b"\x00" * (ENVELOPE_PLAINTEXT_BYTES - 2)
                   + struct.pack("<H", ENVELOPE_PLAINTEXT_BYTES))
    out = eap.handle(seal_raw(whole_block), now=now, tenant_id=req.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.INTEGRITY
    assert out.external_reason == Reason.NOT_AUTHORIZED

    # (b) a well-sized body whose slice octet is not a valid SliceType.
    body = bytearray(AdmissionRequest.STRUCT.size)
    body[8] = 99                                    # slice_type octet
    padded = crypto.pad_fixed(bytes(body))
    out = eap.handle(seal_raw(padded), now=now, tenant_id=req.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.INTEGRITY

    # The proxy is still alive and still admits a well-formed request.
    good = eap.handle(crypto.seal(sk, req, timestamp=now).raw, now=now,
                      tenant_id=req.tenant_id)
    assert good.decision.admitted, good.internal_reason


def test_a_client_cannot_inflate_arrival_index_past_the_rate_gate(rng):
    """The Step 4 rate gate must not be indexed on a client-chosen field.

    Two ways in, both closed by drawing the ordinal at the policy rather than
    from the wire or on arrival:

    (a) ``arrival_index`` travels inside the encrypted request, so a tenant
        sets it freely, and the sliding window drops its history relative to
        it.  Claiming large indices flushes the tenant's own window.
    (b) Counting every packet that reaches ``handle`` instead would let
        *unauthenticated* junk advance the ordinal and age out the same
        window -- the identical bypass, bought with unparseable packets.

    Both measured 40 admissions against a configured limit of 2.
    """
    from csa6g.policies import ProposedNSACQuota

    def drive(indices, junk_between=0):
        r0 = benign_stream(0, n_tenants=1, n_requests=4)[0]
        eap = EdgeAdmissionProxy(make_pool(64), policy=ProposedNSACQuota())
        sk = eap.register_session(run_anchor_bootstrap(rng), r0.slice_type,
                                  r0.service_id, r0.tenant_id)
        now, admitted = time.time(), 0
        for counter, claimed in enumerate(indices, start=1):
            for _ in range(junk_between):       # no key, does not even parse
                eap.handle(os.urandom(ENVELOPE_WIRE_BYTES), now=now,
                           tenant_id=0)
            r = AdmissionRequest(
                tenant_id=r0.tenant_id, pseudonym=r0.pseudonym,
                slice_type=r0.slice_type, service_id=r0.service_id,
                sla_latency_ms=r0.sla_latency_ms,
                # Small enough that the quota is not the binding constraint --
                # the rate gate is what this test is about.
                demand=r0.demand.scaled(0.05),
                arrival_index=claimed, counter=counter)
            out = eap.handle(crypto.seal(sk, r, timestamp=now).raw, now=now,
                             tenant_id=r0.tenant_id)
            if out.decision.admitted:
                admitted += 1
                eap.release_slice(out.session_id)
        return admitted

    n = 40
    honest = drive(list(range(n)))
    assert honest <= DEFAULT_POLICY_PARAMS.p_rate_max + 1, \
        f"the rate gate did not bind even for honest indices ({honest})"
    assert drive([i * 10_000 for i in range(n)]) == honest, \
        "a client bypassed the rate gate by inflating arrival_index"
    assert drive(list(range(n)), junk_between=100) == honest, \
        "unauthenticated junk advanced the window and loosened the rate gate"


def test_a_negative_demand_cannot_mint_capacity_or_quota(rng):
    """A registered tenant must not be able to invert the accounting.

    The adversary of experiment 4 is a *legitimate* tenant, so holding valid
    keys is the attack precondition, not a defence.  A negative demand vector
    used to be admitted: ``fits`` accepted it, ``allocate`` subtracted it from
    the node, and ``demand_norm`` returned a negative cost the quota read as
    credit.  One request drove the pool to 33x its real free capacity and left
    the tenant's consumed share negative -- both structures the DoS result
    rests on, inverted from inside a valid session.
    """
    from csa6g.policies import ProposedNSACQuota
    # Sized so one benign request sits comfortably inside q_base: the point
    # here is the sign of the accounting, and a pool small enough for the
    # quota to bind on its own would confound that with a genuine denial.
    pool = make_pool(64)
    eap = EdgeAdmissionProxy(pool, policy=ProposedNSACQuota())
    req = benign_stream(0, n_tenants=1, n_requests=4)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = time.time()
    before_used = [list(n.used) for n in pool.nodes]

    def send(demand, counter):
        r = AdmissionRequest(
            tenant_id=req.tenant_id, pseudonym=req.pseudonym,
            slice_type=req.slice_type, service_id=req.service_id,
            sla_latency_ms=req.sla_latency_ms, demand=demand,
            arrival_index=counter, counter=counter)
        return eap.handle(crypto.seal(sk, r, timestamp=now).raw, now=now,
                          tenant_id=req.tenant_id)

    out = send(Demand(-100000.0, -100000.0, -100000.0), 1)
    assert not out.decision.admitted, "a negative demand vector was admitted"
    assert out.internal_reason == Reason.MALFORMED_REQUEST
    assert out.external_reason == Reason.NOT_AUTHORIZED

    # Nothing moved: no capacity invented, no quota credited, and the request
    # died before any freshness or counter state was committed.
    assert [list(n.used) for n in pool.nodes] == before_used
    assert pool.dominant_utilisation() >= 0.0
    assert pool.dominant_free_fraction() <= 1.0
    assert eap.policy.nsac.state(req.tenant_id).allocated == 0.0
    assert not len(eap.replay) and not eap.counters._last

    # Non-finite components are rejected by rule, not by accident.
    for bad in (Demand(float("nan"), 1.0, 0.01),
                Demand(float("inf"), 1.0, 0.01),
                Demand(1.0, 1.0, float("-inf"))):
        o = send(bad, 2)
        assert not o.decision.admitted
        assert o.internal_reason == Reason.MALFORMED_REQUEST, bad

    # A normal request from the same session still works.
    good = send(req.demand, 3)
    assert good.decision.admitted, good.internal_reason


def test_unauthenticated_input_cannot_grow_proxy_state_without_bound(rng):
    """The module's own rule, applied to the denial log.

    Unparseable envelopes and unresolvable handles are denied before the HMAC
    runs, so anything ``_deny`` appends to is reachable by a sender holding no
    key at all.  The nonce cache, counter table, sequence table, token store
    and key store all correctly refuse to grow; the denial log was an
    unbounded list, i.e. a remote memory sink at ~65 bytes a packet.
    """
    from csa6g.params import DENIAL_LOG_ENTRIES
    eap = EdgeAdmissionProxy(make_pool(4))
    n = DENIAL_LOG_ENTRIES * 3
    for i in range(n):
        eap.handle(os.urandom(ENVELOPE_WIRE_BYTES), tenant_id=i)  # junk
    assert len(eap.denials) <= DENIAL_LOG_ENTRIES, \
        "an unauthenticated sender grew the denial log without bound"
    assert not len(eap.replay) and not eap.counters._last \
        and not eap.sequences._last and not eap.tokens._issued, \
        "unauthenticated input reached authenticated-only state"
    assert eap._arrivals == 0, \
        ("unauthenticated input advanced the policy's arrival ordinal, which "
         "ages out every tenant's rate-gate window")
    # The aggregate survives in full, keyed on the fixed reason vocabulary, so
    # bounding the log costs no telemetry and hands the attacker no key space.
    assert sum(eap.denial_counts.values()) == n
    assert set(eap.denial_counts) <= {Reason.INTEGRITY, Reason.AUTH_FAIL}


def test_audit_trail_is_bounded_and_never_loses_records_silently(rng):
    """A registered tenant must not be able to grow the trail without bound.

    Retention is deliberate -- released records stay verifiable under
    K_commit after Step 6 wipes the session keys, which is the property the
    operator-side commit key exists to provide. But an unbounded in-memory
    trail is a memory-exhaustion vector an authenticated tenant drives for
    free: one admit/release cycle adds one permanent record and nothing caps
    the cycles (3000 cycles by one tenant measured 3000 retained records).

    The bound is therefore explicit, and the loss it can cause is *counted*,
    because an audit trail that quietly forgets is worse than one that admits
    a limit.
    """
    from csa6g.datastore import AllocationRecord, AuthenticatedStore
    cap = 32
    store = AuthenticatedStore(audit_capacity=cap)
    written = []
    for i in range(cap * 4):
        rec = store.write(AllocationRecord(
            os.urandom(16), 1, int(SliceType.URLLC), 0,
            Demand(1.0, 1.0, 0.1), time.time(), request_counter=i))
        written.append(rec)
        store.mark_released(rec.session_id)

    assert len(store._released) == cap, "the trail is unbounded"
    assert store.evicted_records == cap * 3, store.evicted_records

    # The newest `cap` records are retained AND still verifiable with no
    # session key material anywhere in sight.
    for rec in written[-cap:]:
        assert store.get_released(rec.session_id) is not None
        assert store.verify(rec.session_id), "retained record stopped verifying"
    # The evicted ones are gone, and the counter says exactly how many.
    for rec in written[:-cap]:
        assert store.get_released(rec.session_id) is None

    # Draining is the escape hatch that makes the bound a policy rather than
    # data loss: what comes out carries its MAC and verifies off-box.
    store2 = AuthenticatedStore(audit_capacity=cap)
    recs = []
    for i in range(10):
        r = store2.write(AllocationRecord(
            os.urandom(16), 2, int(SliceType.EMBB), 1,
            Demand(2.0, 2.0, 0.2), time.time(), request_counter=i))
        recs.append(r)
        store2.mark_released(r.session_id)
    drained = store2.drain_released()
    assert len(drained) == 10 and not len(store2._released)
    assert store2.evicted_records == 0, "draining must not count as eviction"
    assert [d[0].session_id for d in drained] == [r.session_id for r in recs], \
        "drain must yield oldest-first"
    import hmac as _h
    from cryptography.hazmat.primitives import hashes as _hs, hmac as _hm
    for record, mac in drained:
        h = _hm.HMAC(store2._commit_keys[record.key_epoch], _hs.SHA256())
        h.update(record.serialise())
        assert _h.compare_digest(mac, h.finalize()), \
            "a drained record cannot be verified off-box"


def test_key_handle_collision_fails_loudly(rng):
    """Two live sessions on one handle must not silently displace each other."""
    from csa6g.keys import KeyHandleCollision, KeyStore
    ks = KeyStore()
    anchor = run_anchor_bootstrap(rng)
    a = SessionKeys.create(anchor, SliceType.URLLC, 7, key_id=b"AAAAAAAA",
                           tenant_id=1)
    b = SessionKeys.create(anchor, SliceType.EMBB, 9, key_id=b"AAAAAAAA",
                           tenant_id=2)
    ks.register(a)
    with pytest.raises(KeyHandleCollision):
        ks.register(b)
    assert ks.get(b"AAAAAAAA").tenant_id == 1, "the live session was displaced"
    ks.register(a)               # idempotent for the same object
    assert len(ks) == 1


def test_release_does_not_scan_the_whole_tracker(rng):
    """Reclaiming one handle's state must not cost O(all live sessions).

    The first version of ``rotate_all`` scanned ``_last`` for matching keys,
    which turns a fix for a memory leak into a DoS surface: work driven by
    *other tenants'* sessions, O(n^2) to drain a busy proxy, measured at
    ~510 us per release with 16k live sessions against ~35 us with 1k.
    Asserted structurally rather than by clock, because a timing threshold in
    a test suite is a flake waiting to happen: the table is instrumented to
    fail if it is iterated at all.
    """
    from csa6g.freshness import CounterTracker

    class NoScanDict(dict):
        scanned = 0

        def __iter__(self):
            type(self).scanned += 1
            return super().__iter__()

    t = CounterTracker()
    t._last = NoScanDict()
    victim = os.urandom(8)
    for _ in range(5000):                       # unrelated live sessions
        t.record(os.urandom(8), 0, 1)
    for epoch in range(3):                      # the handle being released
        t.record(victim, epoch, 1)

    NoScanDict.scanned = 0
    dropped = t.rotate_all(victim)
    assert dropped == 3, dropped
    assert NoScanDict.scanned == 0, \
        "rotate_all iterated the whole tracker instead of using the index"
    assert not [k for k in dict(t._last) if k[0] == victim]
    assert len(t._last) == 5000, "it must not disturb anyone else's state"


def test_release_authenticates_the_token_before_touching_the_issued_store(rng):
    """The proxy's own rule -- authenticate before probing mutable state --
    applied to release.  The issued/revoked token store must not be consulted
    until the token has been cryptographically authenticated; otherwise a
    registered tenant, who can submit arbitrary token strings over its valid
    channel, gets a membership oracle on the issued set.  So a forged or
    unknown-sid release must be rejected by the token AEAD open, before the
    store is touched at all.  Structural, not timed.

    This replaces the earlier revocation-first ordering: the session-table
    membership oracle (the AUC-0.972 finding) is now closed by the decoy
    context, which makes the unknown-sid path do the *same* AEAD work as a live
    one with no early return -- so the store can safely move to last, where the
    authenticate-before-probe rule requires it.
    """
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=2)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted

    class CountingStore:
        """Counts every read/mutate of the issued/revoked set."""
        def __init__(self, inner):
            self.inner = inner
            self.touches = 0

        def consume(self, *a, **kw):
            self.touches += 1
            return self.inner.consume(*a, **kw)

        def validate(self, *a, **kw):
            self.touches += 1
            return self.inner.validate(*a, **kw)

        def revoke(self, *a, **kw):
            self.touches += 1
            return self.inner.revoke(*a, **kw)

        def issue(self, *a, **kw):
            return self.inner.issue(*a, **kw)

    eap.tokens = CountingStore(eap.tokens)

    # A forged token (right length, wrong tag) must fail the AEAD open, before
    # the issued store is ever consulted.
    eap.tokens.touches = 0
    forged = eap.release(out.session_id, os.urandom(len(out.token)))
    assert not forged.ok
    assert forged.internal_reason == Reason.AUTH_FAIL
    assert eap.tokens.touches == 0, "the issued store was probed before auth"

    # An unknown sid with a genuine token still does the (decoy) AEAD work and
    # still never reaches the store -- it cannot open under the decoy key.
    eap.tokens.touches = 0
    unknown = eap.release(os.urandom(16), out.token)
    assert unknown.internal_reason == Reason.AUTH_FAIL
    assert unknown.external_reason == Reason.NOT_AUTHORIZED
    assert eap.tokens.touches == 0, "an unknown sid reached the store pre-auth"

    # The genuine release authenticates first, then consumes the token exactly
    # once (one store touch), and succeeds.
    eap.tokens.touches = 0
    good = eap.release(out.session_id, out.token)
    assert good.ok, good.internal_reason
    assert eap.tokens.touches >= 1


def test_release_decoy_path_derives_the_key_like_a_live_path(rng, monkeypatch):
    """The unknown-sid reject must run the SAME token-key derivation a live-sid
    reject runs. If the decoy branch used a precomputed key it would skip one
    HKDF-Expand and return sooner, so response time would tell a holder of a
    16-byte sid whether it names a live allocation -- the release-path analogue
    of the admission decoy MAC. Counted, not timed."""
    import csa6g.eap as eap_mod
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted

    calls = {"n": 0}
    real = eap_mod.derive_token_key

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(eap_mod, "derive_token_key", counting)

    # Live sid, forged token: the live branch derives once, then the AEAD open
    # fails.
    calls["n"] = 0
    live = eap.release(out.session_id, os.urandom(len(out.token)))
    assert live.internal_reason == Reason.AUTH_FAIL
    live_derivs = calls["n"]

    # Unknown sid: the decoy branch must derive the same number of times.
    calls["n"] = 0
    unknown = eap.release(os.urandom(16), out.token)
    assert unknown.internal_reason == Reason.AUTH_FAIL
    assert calls["n"] == live_derivs == 1, \
        "decoy release path must derive the token key like the live path"


def test_concurrent_release_of_one_token_is_exactly_once(rng):
    """Two submissions of the same valid token must not both deallocate.

    The active->revoked transition is a single atomic check-and-remove
    (:meth:`TokenStore.consume`), so only the first release of a token frees
    the node, returns the quota, retires the record and (if it was the last
    allocation) tears the context down.  The second sees the token already
    consumed and fails ``auth_fail`` without touching pool, policy or keys.
    The prototype is single-threaded, so the two attempts are issued in
    sequence; the property under test is that the *second* is a no-op, which is
    exactly what a compare-and-swap guarantees under real concurrency.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(3, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted
    assert pool.used_tup != (0.0, 0.0, 0.0)
    rollbacks_before = eap.store.rollbacks

    first = eap.release(out.session_id, out.token)
    assert first.ok
    used_after_first = pool.used_tup
    assert used_after_first == (0.0, 0.0, 0.0), "first release must deallocate"
    assert sk.wiped, "sole allocation released -> context torn down once"

    second = eap.release(out.session_id, out.token)
    assert not second.ok
    assert second.internal_reason == Reason.AUTH_FAIL
    # No second deallocation, no second rollback, no double teardown.
    assert pool.used_tup == used_after_first
    assert eap.store.rollbacks == rollbacks_before
    assert req.tenant_id not in eap.live_sessions


def test_admission_is_atomic_no_orphan_on_token_map_failure(rng):
    """A failure after the commit record but before the token map is durable
    must leave no committed allocation, no consumed quota, no active token and
    no orphan record -- the whole admission rolls back.

    ``_fail_token_map`` injects exactly that failure.  Without the transaction,
    the record would be committed and the node held while the requester holds a
    token the release path cannot resolve.
    """
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(5, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)

    eap._fail_token_map = True
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)

    assert not out.decision.admitted
    assert out.external_reason == Reason.TRANSIENT, "must be a retryable code"
    # Nothing left behind: pool free, no live session, no allocation index,
    # no token map, no committed record, no active token.
    assert pool.used_tup == (0.0, 0.0, 0.0), "reservation not rolled back"
    assert len(eap.live_sessions) == 0, "quota/session count not rolled back"
    assert not eap._alloc_by_key, "allocation index not rolled back"
    assert not eap._token_by_session, "token map not rolled back"
    assert len(eap.store) == 0, "committed record left orphaned"
    assert eap.store.rollbacks >= 1, "the commit was not rolled back"

    # The proxy still works for the next request (the flag is one-shot). The
    # retry carries a fresh counter: the failed attempt legitimately consumed a
    # nonce/counter (freshness state is not rolled back -- a real retry re-seals
    # a new envelope), exactly as any commit-failure retry would.
    ok = _admit_under(eap, sk, req, req.counter + 1)
    assert eap.release(ok.session_id, ok.token).ok


def test_release_reclaims_sequence_state_from_every_key_epoch(rng):
    """Rotating the key epoch must not leak one tracker entry per rotation."""
    eap = EdgeAdmissionProxy(make_pool(8))
    anchor = run_anchor_bootstrap(rng)
    req = benign_stream(0, n_tenants=1, n_requests=2)[0]
    sk = eap.register_session(anchor, req.slice_type, req.service_id,
                              req.tenant_id)

    def once(counter):
        r = AdmissionRequest(
            tenant_id=req.tenant_id, pseudonym=req.pseudonym,
            slice_type=req.slice_type, service_id=req.service_id,
            sla_latency_ms=req.sla_latency_ms, demand=req.demand,
            arrival_index=counter, counter=counter)
        out = eap.handle(crypto.seal(sk, r).raw, tenant_id=req.tenant_id)
        assert out.decision.admitted, out.internal_reason
        return out

    for c in range(1, 5):
        out = once(c)
        eap.release_slice(out.session_id)
        sk.rotate_key_epoch(anchor)
    assert len({e for _, e in eap.sequences._last}) == 4, "precondition"

    final = once(5)
    eap.release(final.session_id, final.token)
    assert not [k for k in eap.sequences._last if k[0] == sk.key_id], \
        "sequence state survived secure release for a rotated key handle"
    assert not [k for k in eap.counters._last if k[0] == sk.key_id]


def _admit_under(eap, sk, req, counter):
    """Admit one request under an existing session key, at a given counter.

    The nonce (and thus the encryption sequence) advances via
    :meth:`SessionKeys.next_nonce` inside ``seal``; the request counter must be
    advanced by the caller, so several allocations can be stood up on one key.
    """
    r = AdmissionRequest(
        tenant_id=req.tenant_id, pseudonym=req.pseudonym,
        slice_type=req.slice_type, service_id=req.service_id,
        sla_latency_ms=req.sla_latency_ms, demand=req.demand,
        arrival_index=counter, counter=counter)
    out = eap.handle(crypto.seal(sk, r).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted, out.internal_reason
    return out


def test_releasing_one_allocation_leaves_siblings_under_the_same_key_intact(rng):
    """A security context may back several allocations at once; releasing one
    must not wipe the shared K_slice out from under the others.

    One registration is one context (K_slice/K_mac, the encryption sequence,
    the key epoch); each admitted request is one allocation (a node, a commit
    record, an admit token).  ``release`` closes *that* allocation and revokes
    *its* token; the context and every sibling stay live until the last
    allocation is released, at which point the context is torn down and
    K_slice is wiped.  Before the split, releasing one of two allocations
    wiped the shared key and silently killed the other.
    """
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=2)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)

    a = _admit_under(eap, sk, req, 1)
    b = _admit_under(eap, sk, req, 2)
    assert a.session_id != b.session_id
    assert eap._alloc_by_key[sk.key_id] == {a.session_id, b.session_id}
    assert len(eap.keys) == 1, "two allocations, one security context"
    used_both = pool.used_tup

    # Release A: its node and quota come back, the context lives, B is untouched.
    assert eap.release(a.session_id, a.token).ok
    assert not sk.wiped, "releasing one of two allocations wiped the context"
    assert eap._alloc_by_key[sk.key_id] == {b.session_id}
    assert pool.used_tup != (0.0, 0.0, 0.0), "B's allocation was freed too"
    assert pool.used_tup != used_both, "A's allocation was not freed"

    # A's token is revoked; a replayed release of A fails, not the sibling.
    assert eap.release(a.session_id, a.token).internal_reason == Reason.AUTH_FAIL

    # B is still fully releasable under the same, still-live key.
    assert eap.release(b.session_id, b.token).ok
    # Its last allocation gone, the context is now torn down.
    assert sk.wiped, "context not torn down after its last allocation released"
    assert pool.used_tup == (0.0, 0.0, 0.0)
    assert sk.key_id not in eap._alloc_by_key
    assert not [k for k in eap.sequences._last if k[0] == sk.key_id]
    assert not [k for k in eap.counters._last if k[0] == sk.key_id]


def test_close_session_force_closes_every_allocation_and_revokes_tokens(rng):
    """Operator teardown retires a context and all its live allocations at
    once, with no per-allocation token, and revokes each outstanding token."""
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(1, n_tenants=1, n_requests=3)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)

    a = _admit_under(eap, sk, req, 1)
    b = _admit_under(eap, sk, req, 2)
    assert eap._alloc_by_key[sk.key_id] == {a.session_id, b.session_id}
    assert pool.used_tup != (0.0, 0.0, 0.0)

    assert eap.close_session(sk.key_id).ok
    assert sk.wiped, "close_session did not wipe the context"
    assert pool.used_tup == (0.0, 0.0, 0.0), "not every allocation was freed"
    assert sk.key_id not in eap._alloc_by_key
    assert set(eap.live_sessions) == set(), "live-session count not cleared"
    assert not [k for k in eap.sequences._last if k[0] == sk.key_id]
    assert not [k for k in eap.counters._last if k[0] == sk.key_id]

    # Every force-closed allocation's token is revoked.
    for out in (a, b):
        assert eap.release(out.session_id, out.token).internal_reason \
            == Reason.AUTH_FAIL

    # Idempotent on an already-closed (or never-known) handle.
    assert eap.close_session(sk.key_id).internal_reason == Reason.AUTH_FAIL


def test_the_store_assigns_transaction_ids_not_the_caller(request_):
    """Two records must never share a transaction identity."""
    from csa6g.datastore import AllocationRecord, AuthenticatedStore
    store = AuthenticatedStore()
    pinned = b"CALLER-CHOSEN-XX"
    a = store.write(AllocationRecord(os.urandom(16), 1, int(SliceType.URLLC),
                                     0, request_.demand, time.time(),
                                     transaction_id=pinned))
    b = store.write(AllocationRecord(os.urandom(16), 2, int(SliceType.EMBB),
                                     1, request_.demand, time.time(),
                                     transaction_id=pinned))
    assert a.transaction_id != pinned, "caller dictated the transaction id"
    assert a.transaction_id != b.transaction_id
    assert store.verify(a.session_id) and store.verify(b.session_id)


def test_reordered_delivery_is_rejected_pins_the_assumption(rng):
    """Both monotonicity trackers assume in-order per-session delivery.

    Pinned, not celebrated: two genuine envelopes delivered out of order cost
    the earlier one.  This is the behaviour an RFC 4303-style sliding window
    would change, and it must be changed for the counter tracker and the
    sequence tracker together -- so this test asserts *both* rejections, and a
    future window implementation has to update it deliberately rather than
    discover the coupling afterwards.
    """
    pool = make_pool(8)
    eap = EdgeAdmissionProxy(pool)
    req = benign_stream(0, n_tenants=1, n_requests=4)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = time.time()

    def at(counter):
        return AdmissionRequest(
            tenant_id=req.tenant_id, pseudonym=req.pseudonym,
            slice_type=req.slice_type, service_id=req.service_id,
            sla_latency_ms=req.sla_latency_ms, demand=req.demand,
            arrival_index=req.arrival_index, counter=counter)

    first = crypto.seal(sk, at(1), timestamp=now)     # sequence 0
    second = crypto.seal(sk, at(2), timestamp=now)    # sequence 1

    # Delivered second-then-first. The sequence tracker sees the rollback.
    assert eap.handle(second.raw, now=now,
                      tenant_id=req.tenant_id).decision.admitted
    out = eap.handle(first.raw, now=now, tenant_id=req.tenant_id)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.SEQUENCE_ROLLBACK

    # And with the sequence tracker neutralised, the pre-existing counter
    # tracker rejects the very same envelope one stage later: reordering was
    # already inadmissible before the structured token existed, which is why
    # a sliding window has to replace both marks or neither.
    eap2 = EdgeAdmissionProxy(make_pool(8))
    sk2 = eap2.register_session(run_anchor_bootstrap(rng), req.slice_type,
                                req.service_id, req.tenant_id)
    e1 = crypto.seal(sk2, at(1), timestamp=now)
    e2 = crypto.seal(sk2, at(2), timestamp=now)
    eap2.sequences.check = lambda *a, **kw: Reason.OK
    assert eap2.handle(e2.raw, now=now,
                       tenant_id=req.tenant_id).decision.admitted
    out2 = eap2.handle(e1.raw, now=now, tenant_id=req.tenant_id)
    assert not out2.decision.admitted
    assert out2.internal_reason == Reason.COUNTER_ROLLBACK


# --------------------------------------------- operator-side commit key -----


def _committed_record(store, request_):
    from csa6g.datastore import AllocationRecord
    rec = AllocationRecord(os.urandom(16), 1, int(SliceType.URLLC), 0,
                           request_.demand, time.time(),
                           request_counter=3,
                           expires_at=time.time() + 300.0)
    return store.write(rec)


def test_commit_records_survive_session_release(rng):
    """The regression this finding is about: the datastore's records must
    remain verifiable after the session whose allocation they describe has
    been released and its keys wiped.  The commit MAC is under the
    operator-side K_commit, so no session key is involved at all."""
    pool = make_pool(4)
    eap = EdgeAdmissionProxy(pool)
    stream = benign_stream(0, n_tenants=1, n_requests=1)
    req = stream[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    out = eap.handle(crypto.seal(sk, req).raw, tenant_id=req.tenant_id)
    assert out.decision.admitted
    sid = out.session_id
    assert eap.store.verify(sid), "record must verify while the session lives"

    rel = eap.release(sid, out.token)
    assert rel.ok
    assert sk.wiped and set(bytes(sk.kmac)) == {0}, \
        "session keys must be wiped at Step 6"
    assert eap.store.verify(sid), \
        "record must remain verifiable after release wipes the session keys"
    assert eap.store.get_released(sid) is not None


def test_every_protected_record_field_is_covered_by_the_mac(request_):
    """Mutating any single field of the audit schema breaks verification."""
    from dataclasses import replace as _replace
    from csa6g.datastore import AuthenticatedStore
    store = AuthenticatedStore()
    rec = _committed_record(store, request_)
    mutations = dict(
        version=rec.version + 1,
        transaction_id=os.urandom(16),
        tenant_id=rec.tenant_id + 1,
        slice_type=int(SliceType.MMTC),
        node_id=rec.node_id + 1,
        granted=Demand(rec.granted.cpu + 1.0, rec.granted.mem,
                       rec.granted.storage),
        request_counter=rec.request_counter + 1,
        policy_version=rec.policy_version + 1,
        committed_at=rec.committed_at + 1.0,
        expires_at=rec.expires_at + 1.0,
    )
    for fieldname, bad in mutations.items():
        forged = _replace(rec, **{fieldname: bad})
        store._records[rec.session_id] = (forged,
                                          store._records[rec.session_id][1])
        assert not store.verify(rec.session_id), \
            f"mutation of {fieldname} went undetected"
    # restore and confirm the original still verifies
    mac = store._records[rec.session_id][1]
    store._records[rec.session_id] = (rec, mac)
    assert store.verify(rec.session_id)


def test_a_record_cannot_be_substituted_across_sessions(request_):
    """Record/MAC pairs are bound to their session: swapping session A's
    entry under session B's id fails, because session_id is inside the MAC."""
    from dataclasses import replace as _replace
    from csa6g.datastore import AuthenticatedStore
    store = AuthenticatedStore()
    rec_a = _committed_record(store, request_)
    rec_b = _committed_record(store, request_)
    entry_a = store._records[rec_a.session_id]
    # graft A's record+MAC under B's key, relabelled with B's session id
    store._records[rec_b.session_id] = (
        _replace(entry_a[0], session_id=rec_b.session_id), entry_a[1])
    assert not store.verify(rec_b.session_id), \
        "session A's record was accepted as session B's"


def test_commit_key_epoch_rotation_keeps_history_verifiable(request_):
    """New records commit under the new epoch; old epochs stay verifiable;
    an unknown epoch never verifies."""
    from dataclasses import replace as _replace
    from csa6g.datastore import AuthenticatedStore
    store = AuthenticatedStore()
    old = _committed_record(store, request_)
    store.rotate_key_epoch()
    new = _committed_record(store, request_)
    assert old.key_epoch == 0 and new.key_epoch == 1
    assert store.verify(old.session_id), "historical epoch must verify"
    assert store.verify(new.session_id)
    # A record claiming an epoch this store has never provisioned:
    entry = store._records[new.session_id]
    store._records[new.session_id] = (_replace(entry[0], key_epoch=99),
                                      entry[1])
    assert not store.verify(new.session_id), "unknown epoch must not verify"


def test_commit_key_is_independent_of_all_session_material(rng, request_):
    """K_commit is derived from the operator root key alone: wiping every
    session key changes nothing about record verification, and two stores
    with different root keys never accept each other's MACs."""
    from csa6g.datastore import AuthenticatedStore
    sk = SessionKeys.create(run_anchor_bootstrap(rng), SliceType.URLLC, 7)
    store = AuthenticatedStore()
    rec = _committed_record(store, request_)
    sk.wipe()
    assert store.verify(rec.session_id)

    other = AuthenticatedStore()          # different random K_RM
    other._records[rec.session_id] = store._records[rec.session_id]
    assert not other.verify(rec.session_id), \
        "a MAC minted under one operator root key verified under another"


def test_rollback_removes_the_provisional_record(request_):
    from csa6g.datastore import AuthenticatedStore
    store = AuthenticatedStore()
    rec = _committed_record(store, request_)
    assert store.get(rec.session_id) is not None
    store.rollback(rec.session_id)
    assert store.get(rec.session_id) is None
    assert store.get_released(rec.session_id) is None
    assert not store.verify(rec.session_id)
    assert store.rollbacks == 1


# Submission audit regressions, added after the archived measurements.
@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"),
                                        float("-inf")],
                         ids=["nan", "positive_infinity", "negative_infinity"])
def test_nonfinite_timestamp_is_rejected_before_freshness_mutation(rng,
                                                                 timestamp):
    eap = EdgeAdmissionProxy(make_pool(8))
    req = benign_stream(0, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = 1700000000.0
    bad = crypto.seal(sk, req, timestamp=timestamp)
    with crypto.count_aead_operations() as ops:
        out = eap.handle(bad.raw, now=now)
    assert not out.decision.admitted
    assert out.internal_reason == Reason.MALFORMED_REQUEST
    assert out.external_reason == Reason.NOT_AUTHORIZED
    assert ops["decrypt"] == 0
    assert len(eap.replay) == 0
    assert not eap.counters._last and not eap.sequences._last
    assert not eap.sessions and eap._arrivals == 0
    assert eap.pool.used_tup == (0.0, 0.0, 0.0)
    # Rejection must not consume this request's inner counter.
    good = eap.handle(crypto.seal(sk, req, timestamp=now).raw, now=now)
    assert good.decision.admitted
    assert eap.release(good.session_id, good.token, now=now).ok


def test_live_allocations_release_with_their_issuance_epoch_after_rotation(rng):
    eap = EdgeAdmissionProxy(make_pool(8))
    req = benign_stream(0, n_tenants=1, n_requests=1)[0]
    anchor = run_anchor_bootstrap(rng)
    sk = eap.register_session(anchor, req.slice_type, req.service_id,
                              req.tenant_id)
    old = _admit_under(eap, sk, req, 1)
    assert eap._token_epoch_by_session[old.session_id] == 0
    sk.rotate_key_epoch(anchor)
    new = _admit_under(eap, sk, req, 2)
    assert eap._token_epoch_by_session[new.session_id] == 1

    assert eap.release(old.session_id, old.token).ok
    assert old.session_id not in eap._token_epoch_by_session
    assert not sk.wiped and new.session_id in eap.sessions
    assert not eap.release(old.session_id, old.token).ok
    assert eap.live_sessions[req.tenant_id] == 1
    assert eap.release(new.session_id, new.token).ok
    assert sk.wiped and not eap.sessions
    assert not eap._token_epoch_by_session
    assert not eap.live_sessions
    assert eap.pool.used_tup == (0.0, 0.0, 0.0)


# ------------------------------------------- demand validation regressions ---

def test_all_zero_demand_is_malformed_and_cannot_hold_isolation_slots(rng):
    """Review finding (2026-09-19): an all-zero demand costs nothing against
    the dominant-share quota but a placement still consumes one of a node's
    ``beta`` anti-co-location slots, so a registered tenant could hold every
    same-slice slot in the pool for free.  It is now denied as malformed
    before any freshness or allocation state is touched."""
    from csa6g.policies import ProposedNSACQuota
    n_nodes = 6
    pool = make_pool(n_nodes)
    eap = EdgeAdmissionProxy(pool, policy=ProposedNSACQuota(), beta=1)
    attacker, victim = benign_stream(0, n_tenants=2, n_requests=2)[:2]
    victim = dataclasses.replace(victim, slice_type=attacker.slice_type,
                                 tenant_id=attacker.tenant_id + 1)
    sk_a = eap.register_session(run_anchor_bootstrap(rng), attacker.slice_type,
                                attacker.service_id, attacker.tenant_id)
    sk_v = eap.register_session(run_anchor_bootstrap(rng), victim.slice_type,
                                victim.service_id, victim.tenant_id)
    now = time.time()

    def send(sk, req, demand, counter):
        r = dataclasses.replace(req, demand=demand, arrival_index=counter,
                                counter=counter)
        return eap.handle(crypto.seal(sk, r, timestamp=now).raw, now=now,
                          tenant_id=req.tenant_id)

    for i in range(1, n_nodes + 1):
        out = send(sk_a, attacker, Demand(0.0, 0.0, 0.0), i)
        assert not out.decision.admitted
        assert out.internal_reason == Reason.MALFORMED_REQUEST
        assert out.external_reason == Reason.NOT_AUTHORIZED
    assert all(len(n.hosted) == 0 for n in pool.nodes), \
        "a zero-demand request occupied a placement slot"
    assert eap.policy.nsac.state(attacker.tenant_id).allocated == 0.0
    assert not len(eap.replay) and not eap.counters._last

    # The same slice is still placeable for a real request afterwards.
    out = send(sk_v, victim, victim.demand, 1)
    assert out.decision.admitted, out.internal_reason

    # Epsilon demand is admissible; the slot accounting prices it at one slot share.
    assert Demand(1e-9, 0.0, 0.0).is_admissible()
    assert not Demand(0.0, 0.0, 0.0).is_admissible()


def test_seal_never_pairs_a_sequence_with_keys_from_another_epoch(session):
    """Review finding (2026-09-19): reserving the sequence under the lock but
    reading the epoch/keys afterwards let a concurrent ``rotate_key_epoch``
    interleave, producing two seals with the same (epoch, sequence) IV under
    one rotated K_slice.  ``reserve`` now returns nonce and keys from one
    critical section; this races sealing against rotation and checks that
    every emitted (epoch, sequence) is unique and that the nonce's epoch
    always matches the key bytes it was sealed with."""
    import threading
    from csa6g.keys import derive_slice_keys, parse_nonce
    anchor = os.urandom(32)
    sk = SessionKeys.create(anchor, session.slice_type, session.service_id)
    expected = {}                       # epoch -> K_slice for that epoch
    for e in range(0, 200):
        expected[e] = derive_slice_keys(anchor, sk.slice_type, sk.service_id,
                                        key_epoch=e, key_id=sk.key_id)[0]
    stop = threading.Event()
    seen: list[tuple[bytes, bytes]] = []
    lock = threading.Lock()

    def sealer():
        local = []
        while not stop.is_set():
            nonce, ks, _ = sk.reserve()
            local.append((nonce, ks))
        with lock:
            seen.extend(local)

    def rotator():
        for _ in range(150):
            sk.rotate_key_epoch(anchor)
            time.sleep(0.0005)
        stop.set()

    threads = [threading.Thread(target=sealer) for _ in range(4)]
    threads.append(threading.Thread(target=rotator))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) > 1000
    ivs = [parse_nonce(n)[:2] for n, _ in seen]
    assert len(set(ivs)) == len(ivs), "duplicate (epoch, sequence) IV emitted"
    for nonce, ks in seen:
        epoch = parse_nonce(nonce)[0]
        assert ks == expected[epoch], \
            "nonce epoch does not match the K_slice it was reserved with"


# ------------------------------------------------ slot accounting regressions ---

@pytest.mark.parametrize("variant", ["cap", "share"])
def test_tiny_positive_demands_cannot_exhaust_same_slice_slots(rng, variant):
    """With only all-zero demand rejected, one tenant could still admit 72 epsilon-demand allocations in
    the paper's 24-node / beta = 3 deployment, holding every same-slice slot
    at ~1e-12 of the resource pool.  Both slot-accounting variants bound it:
    ``cap`` (shipped) to ``p_slot_cap`` active allocations per slice, and
    ``share`` (PS) by pricing the slot in the dominant-share quota."""
    from csa6g.model import slot_share
    from csa6g.policies import ProposedNSACQuota, ProposedSlotPricedQuota
    n_nodes, beta = 24, 3
    pool = make_pool(n_nodes)
    policy = (ProposedNSACQuota() if variant == "cap"
              else ProposedSlotPricedQuota())
    eap = EdgeAdmissionProxy(pool, policy=policy, beta=beta)
    attacker, victim = benign_stream(0, n_tenants=2, n_requests=2)[:2]
    victim = dataclasses.replace(victim, slice_type=attacker.slice_type,
                                 tenant_id=attacker.tenant_id + 1)
    sk_a = eap.register_session(run_anchor_bootstrap(rng), attacker.slice_type,
                                attacker.service_id, attacker.tenant_id)
    sk_v = eap.register_session(run_anchor_bootstrap(rng), victim.slice_type,
                                victim.service_id, victim.tenant_id)
    now = time.time()
    tiny = Demand(1e-9, 1e-9, 1e-9)

    def send(sk, req, demand, counter):
        r = dataclasses.replace(req, demand=demand, arrival_index=counter,
                                counter=counter)
        return eap.handle(crypto.seal(sk, r, timestamp=now).raw, now=now,
                          tenant_id=req.tenant_id)

    # Space requests past the rate window so the slot control, not the rate
    # gate, is what is under test.
    window = DEFAULT_POLICY_PARAMS.p_rate_window + 1
    slots = beta * n_nodes
    held, reasons = 0, set()
    for i in range(1, slots + 1):
        out = send(sk_a, attacker, tiny, i * window)
        held += out.decision.admitted
        reasons.add(out.internal_reason)
    assert held < slots, "one tenant occupied every same-slice slot"
    if variant == "cap":
        assert held == DEFAULT_POLICY_PARAMS.p_slot_cap
        assert Reason.SLOT_CAP in reasons
        assert eap.policy.nsac.slots_held(attacker.tenant_id,
                                          attacker.slice_type) == held
    else:
        q = DEFAULT_POLICY_PARAMS.p_base_quota_frac
        assert held <= max(1, int(q * slots)) + 1
        assert eap.policy.nsac.state(attacker.tenant_id).allocated == \
            pytest.approx(held * slot_share(n_nodes, beta))

    # A legitimate same-slice tenant is still placeable.
    out = send(sk_v, victim, victim.demand, 1)
    assert out.decision.admitted, out.internal_reason

    # Release refunds slot and quota, so the accounting stays balanced.
    held_v = eap.policy.nsac.slots_held(victim.tenant_id, victim.slice_type)
    before = eap.policy.nsac.state(victim.tenant_id).allocated
    assert held_v == 1
    assert eap.release(out.session_id, out.token).ok
    assert eap.policy.nsac.slots_held(victim.tenant_id, victim.slice_type) == 0
    assert eap.policy.nsac.state(victim.tenant_id).allocated < before


def test_slot_cap_is_per_slice_and_frees_on_release(rng):
    """The shipped cap counts active allocations per (tenant, slice type):
    reaching it in one slice does not block another, and a release frees a
    slot for the same tenant."""
    from csa6g.policies import ProposedNSACQuota
    pool = make_pool(24)
    eap = EdgeAdmissionProxy(pool, policy=ProposedNSACQuota(
        dataclasses.replace(DEFAULT_POLICY_PARAMS, p_rate_max=10_000)),
        beta=3)
    cap = DEFAULT_POLICY_PARAMS.p_slot_cap
    req = benign_stream(1, n_tenants=1, n_requests=1)[0]
    tiny = Demand(1e-6, 1e-6, 1e-6)
    sk = {}
    for st in (SliceType.URLLC, SliceType.MMTC):
        sk[st] = eap.register_session(run_anchor_bootstrap(rng), st,
                                      req.service_id, req.tenant_id)
    now = time.time()
    outs = []
    for i in range(cap + 1):
        r = dataclasses.replace(req, slice_type=SliceType.URLLC, demand=tiny,
                                arrival_index=i, counter=i + 1)
        outs.append(eap.handle(crypto.seal(sk[SliceType.URLLC], r,
                                           timestamp=now).raw,
                               now=now, tenant_id=req.tenant_id))
    assert [o.decision.admitted for o in outs] == [True] * cap + [False]
    assert outs[-1].internal_reason == Reason.SLOT_CAP
    # another slice type is unaffected
    r = dataclasses.replace(req, slice_type=SliceType.MMTC, demand=tiny,
                            arrival_index=50, counter=1)
    o = eap.handle(crypto.seal(sk[SliceType.MMTC], r, timestamp=now).raw,
                   now=now, tenant_id=req.tenant_id)
    assert o.decision.admitted, o.internal_reason
    # release one URLLC allocation -> one more URLLC admission possible
    assert eap.release(outs[0].session_id, outs[0].token).ok
    r = dataclasses.replace(req, slice_type=SliceType.URLLC, demand=tiny,
                            arrival_index=60, counter=cap + 2)
    o = eap.handle(crypto.seal(sk[SliceType.URLLC], r, timestamp=now).raw,
                   now=now, tenant_id=req.tenant_id)
    assert o.decision.admitted, o.internal_reason


def test_every_tenant_can_hold_one_allocation_in_a_small_deployment(rng):
    """The slot-priced variant must not make small deployments unusable:
    with six nodes and beta = 1 one slot is 1/6 of the pool, far above
    q_base, so its cap is floored at one slot's share."""
    from csa6g.policies import ProposedSlotPricedQuota
    pool = make_pool(6)
    eap = EdgeAdmissionProxy(pool, policy=ProposedSlotPricedQuota(), beta=1)
    reqs = benign_stream(3, n_tenants=5, n_requests=5)
    now = time.time()
    seen = set()
    for i, req in enumerate(reqs):
        if req.tenant_id in seen:
            continue
        seen.add(req.tenant_id)
        sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                                  req.service_id, req.tenant_id)
        r = dataclasses.replace(req, arrival_index=i, counter=1)
        out = eap.handle(crypto.seal(sk, r, timestamp=now).raw, now=now,
                         tenant_id=req.tenant_id)
        assert out.decision.admitted, (req.tenant_id, out.internal_reason)
        assert eap.release(out.session_id, out.token).ok


def test_wipe_cannot_interleave_with_reservation_or_rotation(session):
    """Review finding (2026-09-19, independent): ``wipe()`` did not take the
    lifecycle lock and ``reserve``/``rotate_key_epoch`` checked liveness
    before acquiring it, so an operation past its check could continue after
    a wipe (sealing with zeroed keys; rotation repopulating a wiped object).
    All three now serialise on one lock with the liveness check inside.

    The interleaving is forced deterministically: the liveness check is
    wrapped so that, once it passes, a wipe is started on another thread and
    given 50 ms.  With the fix the wipe blocks on the lock until the
    operation completes; without it the wipe lands in the middle."""
    import threading
    anchor = os.urandom(32)

    def forced(sk):
        real = sk.assert_live
        wiper = threading.Thread(target=sk.wipe)

        def hook():
            real()
            if not wiper.is_alive() and wiper.ident is None:
                wiper.start()
                wiper.join(timeout=0.05)
        sk.assert_live = hook
        return wiper

    # (a) reservation: must never return zeroed key material
    sk = SessionKeys.create(anchor, session.slice_type, session.service_id)
    w = forced(sk)
    nonce, ks, km = sk.reserve()
    w.join()
    assert ks != bytes(32) and km != bytes(32), \
        "reserve returned zeroed keys after an interleaved wipe"
    assert sk.wiped and bytes(sk.kslice) == bytes(32)

    # (b) rotation: a wiped object must never be repopulated with live keys
    sk = SessionKeys.create(anchor, session.slice_type, session.service_id)
    w = forced(sk)
    sk.rotate_key_epoch(anchor)
    w.join()
    assert sk.wiped and bytes(sk.kslice) == bytes(32) \
        and bytes(sk.kmac) == bytes(32), \
        "rotation repopulated key material on a wiped session"
    with pytest.raises(RuntimeError):
        sk.reserve()


def test_actual_sealing_races_rotation_without_iv_reuse(session, request_):
    """Integration form of the sequence-reservation regression: real ``crypto.seal``
    calls, not just ``reserve``, concurrent with key rotation; every emitted
    envelope must carry a unique (epoch, sequence) IV, and its nonce epoch
    must match the K_slice it was sealed under (verified by decrypting)."""
    import threading
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from csa6g.keys import derive_slice_keys, iv_from_nonce, parse_nonce
    from csa6g.crypto import GCM_TAG_BYTES
    anchor = os.urandom(32)
    sk = SessionKeys.create(anchor, session.slice_type, session.service_id)
    keys = {e: derive_slice_keys(anchor, sk.slice_type, sk.service_id,
                                 key_epoch=e, key_id=sk.key_id)[0]
            for e in range(0, 120)}
    stop = threading.Event(); out: list = []; lock = threading.Lock()

    def sealer():
        local = []
        while not stop.is_set():
            local.append(crypto.seal(sk, request_))
        with lock:
            out.extend(local)

    def rotator():
        for _ in range(100):
            sk.rotate_key_epoch(anchor); time.sleep(0.0005)
        stop.set()

    ts = [threading.Thread(target=sealer) for _ in range(3)]
    ts.append(threading.Thread(target=rotator))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(out) > 500
    ivs = [parse_nonce(e.nonce)[:2] for e in out]
    assert len(set(ivs)) == len(ivs), "duplicate (epoch, sequence) IV"
    for e in out:
        epoch = parse_nonce(e.nonce)[0]
        AESGCM(keys[epoch]).decrypt(iv_from_nonce(e.nonce),
                                    e.ciphertext + e.gcm_tag, b"")
