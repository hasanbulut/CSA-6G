"""The Edge Admission Proxy (paper Steps 3-6).

    A(u, s, c, d_u) = 1  <=>  AuthOK  &&  Fresh  &&  IntOK  &&  QuotaOK
                              &&  Feasible  &&  Isolate

The conjunction is evaluated in an order fixed by one security rule:

    **Unauthenticated input must not reach state an observer can probe, and
    must not buy work an attacker can amplify.**

Both halves of that rule bind.  Nothing derived from the envelope is allowed to
touch mutable state -- the nonce cache, the counter table -- until the outer
HMAC has proved the envelope was minted by the session it names; and no
expensive primitive runs on input that a cheaper authenticator has already
shown to be forged.  The second half happens to coincide with increasing cost
order, which is why the pipeline mostly looks like a cost-sorted one.  Cost
order is not the principle, and treating it as one produces a real
vulnerability -- see below.

    1. parse            (fixed-size structural check)
    2. key lookup       (dictionary)
    3. outer HMAC       (one HMAC-SHA256 over 337 bytes)      <- IntOK, cheap path
    4. freshness_check  (timestamp arithmetic + nonce set membership)
    5. AEAD open        (AES-256-GCM decrypt)                 <- IntOK, full path
    6. context_check    (decrypted slice/service must match the key context;
                         demand vector must be finite and non-negative)
    7. counter_check    (monotonicity)
    8. local revalidate (optional keyed challenge-response cost surrogate)
    9. freshness_write  (commit the nonce to the replay window)
   10. counter_write    (commit the counter value)
   11. quota            (proxy-assigned arrival ordinal, then policy)
   12. placement        (placement rule)
   13. commit + token   (Step 5)

Where the two criteria disagree, security wins.  Integrity precedes freshness
even though the HMAC costs several times the freshness check, so a replayed
envelope now pays for an HMAC it used to skip.  Sorting these two by cost is
what leaked: the replay window is a cache keyed on an attacker-supplied nonce,
so consulting it before authenticating the envelope answers "is this nonce in
your cache?" for any unauthenticated party willing to time the response.  Both
outcomes return the same reason code, so Step 5's minimal-disclosure property
holds on the channel it governs and the oracle sits entirely outside it.
Experiment 7 measures the channel in both orderings: under the cost-sorted
order a single-probe observer separates the two outcomes almost perfectly --
AUC above 0.99 on every run recorded here -- and at chance once integrity comes
first.  The price of the fix is one HMAC on the replay path.  (Read the figure
from ``exp7_timing_oracle_summary.csv`` rather than from this docstring; it is
a measurement and it moves with the machine.)

The general lesson, and the reason this docstring leads with the rule rather
than with the cost table: an ordering justified by cost is only accidentally
secure, and the accident does not survive the first check whose cost and whose
sensitivity to attacker-supplied input happen to disagree.

``integrity_first=False`` restores the leaky order.  It exists so experiment 7
can measure what the fix bought rather than assert it, and should not be used
otherwise.

Protocol assumption: in-order delivery per session
--------------------------------------------------
Both monotonicity trackers -- the encrypted per-request counter (stage 7) and
the cleartext encryption sequence (stage 4b) -- hold a single high-water mark
per session and reject anything at or below it.  Two genuine envelopes
delivered out of order therefore cost the earlier one: it is rejected with
``counter_rollback`` or ``sequence_rollback`` despite being authentic and
never seen before.

This is a real limitation and it is a *protocol assumption*, not a defect
introduced with the structured freshness token: the counter tracker has had
these semantics throughout, so reordered traffic was already rejected before
the sequence tracker existed.  It sits awkwardly beside
:meth:`csa6g.keys.SessionKeys.next_nonce`, which takes a lock precisely so
concurrent seals are safe -- a sender may seal concurrently, but whatever it
seals must reach the E-AP in sequence order.  For the single-session,
in-order workloads the experiments drive, the assumption holds by
construction.

The upgrade path is the standard one: replace each high-water mark with an
RFC 4303-style sliding window -- accept a value within W of the mark if its
bit is unset, then set it -- which tolerates reordering up to W while still
rejecting replays.  Both trackers must be converted together.  Converting
only the sequence tracker buys nothing, because the counter check downstream
would reject the same reordered envelope one stage later; converting only the
counter tracker buys nothing for the same reason in reverse.
``test_reordered_delivery_is_rejected_pins_the_assumption`` pins the current
behaviour so that a future window implementation has to update it
deliberately.

Every stage is timed separately, which is what makes the component breakdown in
the latency experiment reproducible rather than asserted.

Freshness and counter each get a *check* and a *write* stage rather than one
bucket apiece.  The two axes are independent and neither decomposition recovers
the other: merging by concern (freshness, counter) buries how much of the cost
is the read-only check that every rejected request pays versus the write only
an authentic one reaches; merging by phase (check, record) buries which of the
two mechanisms the cost belongs to.  Splitting both ways costs one extra
``perf_counter_ns`` per request and loses nothing.

Stages 8 and 9 are the same decision that :mod:`csa6g.harness` runs without the
cryptography for the policy comparison of the fairness experiment; ``test_eap_and_harness_
agree`` pins the two together.  See that module's docstring for why there are
two loops at all.
"""
from __future__ import annotations

import hmac as _hmac
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, replace as _replace

from cryptography.hazmat.primitives import hashes, hmac

from . import crypto
from .crypto import EnvelopeError
from .datastore import (AllocationRecord, AuthenticatedStore, CommitError,
                        issue_admit_token)
from .freshness import CounterTracker, ReplayWindow, TokenStore
from .keys import KeyStore, SessionKeys, derive_token_key, parse_nonce
from .model import Decision, Reason, ResourcePool, SliceType, demand_norm
from .params import (ADMIT_TOKEN_TTL_S, COLOCATION_BUDGET_BETA,
                     DENIAL_LOG_ENTRIES)
from .placement import select_node
from .policies import AllocationPolicy, UncontrolledFCFS

STAGES = ("parse", "keylookup", "hmac", "freshness_check", "aead",
          "context_check", "counter_check", "auth", "freshness_write", "counter_write",
          "quota", "placement", "commit", "token")


def replay_key(key_id: bytes, nonce: bytes) -> bytes:
    """Cache key for the shared replay window: ``key_id || n_u``.

    One definition, used by the check and the record site alike, so the two
    can never drift apart -- a window written under one key and read under
    another degrades silently to "never a replay" rather than failing.  The
    session handle is part of the key because ``n_u`` is only unique *per
    session*: its first twelve bytes are epoch || sequence, identical across
    sessions at the same point in their sequence, leaving the 4-byte random
    tail as the only cross-session entropy.  See :mod:`csa6g.freshness`.
    """
    return key_id + nonce


class _Timer:
    """Nanosecond stage timer.  Kept dead simple so its own overhead (one
    perf_counter_ns call per stage boundary) is uniform across scenarios."""

    __slots__ = ("t", "last")

    def __init__(self):
        self.t = {s: 0.0 for s in STAGES}
        self.last = time.perf_counter_ns()

    def mark(self, stage: str) -> None:
        now = time.perf_counter_ns()
        self.t[stage] += (now - self.last) / 1e9
        self.last = now

    def skip(self) -> None:
        self.last = time.perf_counter_ns()

    @property
    def total(self) -> float:
        return sum(self.t.values())


def secondary_authentication(sk: SessionKeys, rng=None) -> bool:
    """Local keyed revalidation cost surrogate.

    Two real HMAC-SHA256 operations are executed instead of charging a fixed
    constant, so scenario S5 measures an implemented local computation path.
    Both operations run in-process against the same key; the comparison cannot
    fail under the default stub.  This is therefore neither an implementation
    nor an end-to-end latency estimate of 3GPP NSSAA, which is EAP based and can
    involve several network functions and network round trips.  The measured
    S5 overhead must be interpreted only as a lower-bound compute component of
    any per-request revalidation design.

    The E-AP takes its authenticator as a parameter precisely so this stub is
    not the only thing the AUTH_FAIL branch can ever see; see
    ``EdgeAdmissionProxy(authenticator=...)``.
    """
    sk.assert_live()
    challenge = os.urandom(32)
    h = hmac.HMAC(bytes(sk.kmac), hashes.SHA256())
    h.update(b"nssaa" + challenge)
    response = h.finalize()
    v = hmac.HMAC(bytes(sk.kmac), hashes.SHA256())
    v.update(b"nssaa" + challenge)
    return _hmac.compare_digest(response, v.finalize())


@dataclass
class AdmissionOutcome:
    decision: Decision
    session_id: bytes | None = None
    token: bytes | None = None
    internal_reason: str = Reason.OK
    external_reason: str = Reason.OK


@dataclass
class ReleaseOutcome:
    """Step 6 result.  ``external_reason`` is the only field a requester may
    ever see; the fine-grained ``internal_reason`` stays inside the network,
    exactly as on the admission path.  ``release`` used to return the internal
    code bare, which left the Step 5 minimal-disclosure invariant enforced on
    admission and merely hoped-for on release: ``token_expired`` versus
    ``auth_fail`` told a requester whether a guessed pair had once been valid.
    """
    internal_reason: str
    external_reason: str

    @property
    def ok(self) -> bool:
        return self.internal_reason == Reason.OK


class EdgeAdmissionProxy:
    """E-AP: terminates the encrypted envelope and runs the admission decision.

    The prototype is deployment-neutral.  It consumes an application-layer
    envelope and emits a decision, but it does not claim that the E-AP is an
    existing 3GPP network function or that the current NSACF/AF interfaces can
    carry this protocol unchanged.  Standards integration is a deployment
    requirement, not an implemented result.
    """

    def __init__(self, pool: ResourcePool,
                 policy: AllocationPolicy | None = None,
                 beta: int = COLOCATION_BUDGET_BETA,
                 replay_window: ReplayWindow | None = None,
                 authenticator=secondary_authentication,
                 integrity_first: bool = True,
                 envelope_mode: str = "dual"):
        # ``envelope_mode`` selects the envelope format.  "dual" is the
        # 337-byte dual-tag envelope (outer HMAC under K_mac, then AES-GCM).
        # "aad" is the conventional 305-byte AES-GCM envelope with the
        # cleartext header bound as associated data and no outer HMAC; it
        # exists for the side-by-side comparison of experiment 8.  In "aad"
        # mode the integrity check *is* the AEAD open, so the "hmac" stage
        # stays at zero and integrity cost is billed to "aead".
        if envelope_mode not in ("dual", "aad"):
            raise ValueError(f"unknown envelope_mode {envelope_mode!r}")
        self.envelope_mode = envelope_mode
        # ``authenticator`` is injectable so the AUTH_FAIL branch of stage 8 is
        # reachable.  The default stub recomputes an HMAC against the key it
        # just used, so it cannot return False and that branch would otherwise
        # be unreachable dead code that no test could cover.
        self.authenticator = authenticator
        # See the module docstring.  False restores the pre-fix order in which
        # the replay window was consulted before the envelope was authenticated;
        # experiment 7 uses it to measure the timing oracle that created.
        self.integrity_first = integrity_first
        self.pool = pool
        self.policy = policy or UncontrolledFCFS()
        self.policy.reset(pool, beta)
        self.beta = beta
        self.keys = KeyStore()
        # Decoy MAC key for the unknown-key-handle path: a handle that does
        # not resolve must cost the same HMAC a bad tag costs, or the response
        # time answers "is this session still live?" for anyone replaying a
        # captured envelope.  Random per proxy; its output is never compared.
        self._decoy_kmac = os.urandom(32)
        # Operator-side admit-token root, held only by the E-AP.  The
        # per-session token key is derived from this (never from a UE-held
        # key), so a tenant holding K_slice/K_mac cannot mint a token the
        # release path accepts.  Random per proxy; a deployment provisions it
        # out of band alongside K_RM.
        self._k_eap = os.urandom(32)
        # Decoy key *handle* for the release path's unknown-``session_id``
        # branch.  The unknown-sid branch derives its token key from this via
        # the same HKDF-Expand the live branch runs (see :meth:`release`), so
        # the two perform identical work.  A precomputed decoy *key* -- what
        # this used to be -- skipped the derivation, so an unknown-sid reject
        # returned one HKDF sooner than a live-sid reject and leaked, by timing,
        # whether the id named a live allocation (a session-table membership
        # oracle).  The admission path equalises the analogous unknown-key
        # handle case the same way, via :func:`csa6g.crypto.burn_outer_mac`.
        self._decoy_key_id = os.urandom(8)
        # ``is None``, not ``or``: a freshly constructed ReplayWindow is empty
        # and therefore falsy (``__len__`` is 0), so ``replay_window or
        # ReplayWindow()`` silently discarded every injected window and tests
        # that thought they were shrinking the cache were exercising the
        # default one.
        self.replay = ReplayWindow() if replay_window is None else replay_window
        self.counters = CounterTracker()
        # Strictly-monotone tracker over the cleartext (key_epoch, sequence)
        # prefix of n_u.  This is the receiver half of the structured-IV
        # design: a repeated or rolled-back encryption sequence is rejected
        # even after the bounded nonce cache has evicted the nonce, and an
        # envelope minted under a stale key epoch never reaches the AEAD.
        self.sequences = CounterTracker(reason=Reason.SEQUENCE_ROLLBACK)
        self.tokens = TokenStore()
        self.store = AuthenticatedStore()
        self.sessions: dict[bytes, tuple[SessionKeys, AllocationRecord]] = {}
        # Live allocations grouped by the security context that backs them.
        # One registration (one ``SessionKeys`` / ``key_id``) is a security
        # context; one admitted request (one ``session_id``) is an allocation,
        # and a context may back several allocations at once.  This index maps
        # ``key_id -> {session_id, ...}`` so per-allocation release can tell
        # whether the allocation it just closed was the *last* one on its
        # context -- and therefore whether the context should be torn down --
        # without wiping the shared K_slice out from under sibling allocations.
        self._alloc_by_key: dict[bytes, set[bytes]] = {}
        # The admit token issued for each live allocation, kept so an operator
        # teardown (:meth:`close_session`) can revoke the tokens of allocations
        # it force-closes.  Dropped as each allocation is released.
        self._token_by_session: dict[bytes, bytes] = {}
        # Preserve the issuance epoch separately from mutable SessionKeys:
        # rotating envelope keys must not invalidate a live allocation's
        # release token, which was sealed under the earlier token key.
        self._token_epoch_by_session: dict[bytes, int] = {}
        # Fault injection for the admission-transaction rollback test: when set,
        # the token->allocation map insertion raises, exercising the path that
        # must undo the commit record, the issued token, the reservation and the
        # quota rather than leave an allocation the requester can never release.
        self._fail_token_map = False
        # Live-session count per tenant, not a set of everyone ever seen: this
        # is the denominator of the pressure-adaptive cap in Step 4, so a tenant
        # has to leave it again when its last session is released.
        self.live_sessions: Counter[int] = Counter()
        # Bounded.  ``_deny`` is reached by input that has proved nothing --
        # an unparseable envelope or an unresolvable handle is denied before
        # the HMAC runs -- so an unbounded denial log is the one piece of proxy
        # state an attacker with no key material can grow without limit, and a
        # DoS-resilience claim should not ship one.  The recent log is a
        # ``deque`` with a fixed cap; the aggregate that telemetry actually
        # wants is ``denial_counts``, keyed on the internal reason code, whose
        # vocabulary is the fixed set in :class:`Reason` and so has a size an
        # attacker cannot influence.  The tenant label is deliberately *not*
        # part of that key: before a handle resolves it is caller-supplied and
        # unauthenticated, so counting per tenant would hand the key space
        # straight back to the attacker.
        self.denials: deque[tuple[int, str]] = deque(maxlen=DENIAL_LOG_ENTRIES)
        self.denial_counts: Counter[str] = Counter()
        # Arrival ordinal for the policy, assigned by the proxy rather than
        # read off the request.  Advanced only by requests that reach
        # admission control; see the quota stage of :meth:`handle`.
        self._arrivals = 0

    @property
    def active_tenants(self) -> set[int]:
        """Tenants holding at least one live session."""
        return set(self.live_sessions)

    # ------------------------------------------------------------- session
    def register_session(self, k_anchor: bytes, slice_type: SliceType,
                         service_id: int, tenant_id: int) -> SessionKeys:
        """Step 2: derive and register slice-scoped keys for one session.

        ``tenant_id`` is required: registration happens at the trust level of
        primary authentication, so this is the one point where tenant
        attribution can be bound to key material.  Quota and fairness state is
        keyed on the tenant, and :meth:`handle` reads the tenant from the
        session -- a per-call label would let an adversary evade per-tenant
        caps, or drain a victim's, by claiming an arbitrary tenant.
        """
        sk = SessionKeys.create(k_anchor, slice_type, service_id,
                                tenant_id=tenant_id)
        self.keys.register(sk)
        return sk

    # ------------------------------------------------------------- handle
    def handle(self, raw: bytes, *, now: float | None = None,
               tenant_id: int = -1, run_secondary_auth: bool = False,
               ) -> AdmissionOutcome:
        """Run the admission decision on one raw envelope.

        ``tenant_id`` here is *untrusted telemetry*: it labels denials that
        happen before a session is resolved (unparseable envelope, unknown
        handle), where no authenticated attribution exists.  The moment the
        key handle resolves, the session's bound tenant replaces it -- quota
        accounting never runs on a caller-supplied label.
        """
        now = time.time() if now is None else now
        tm = _Timer()

        # 1 -- parse -------------------------------------------------------
        aad_mode = self.envelope_mode == "aad"
        try:
            env = crypto.parse_aad(raw) if aad_mode else crypto.parse(raw)
        except EnvelopeError:
            tm.mark("parse")
            return self._deny(Reason.INTEGRITY, tenant_id, tm)
        tm.mark("parse")

        # 2 -- key lookup --------------------------------------------------
        sk = self.keys.get(env.key_id)
        tm.mark("keylookup")
        if sk is None or sk.wiped:
            # Same work as the bad-MAC rejection below, under a decoy key:
            # without it, response time reveals whether the handle names a
            # live session (see crypto.burn_outer_mac).
            if aad_mode:
                # Decoy AEAD open: the same work a bad-tag rejection costs.
                try:
                    crypto.open_aad_padded(self._decoy_kmac, env)
                except EnvelopeError:
                    pass
                tm.mark("aead")
            else:
                crypto.burn_outer_mac(self._decoy_kmac, env)
                tm.mark("hmac")
            return self._deny(Reason.AUTH_FAIL, tenant_id, tm)
        if sk.tenant_id >= 0:
            # Authenticated attribution: the tenant bound at registration
            # overrides whatever the caller claimed.
            tenant_id = sk.tenant_id

        # 3, 4 -- IntOK (outer HMAC, no AEAD) then Fresh(n_u, tau_u) ------
        # Integrity first: the replay window is keyed on an attacker-supplied
        # nonce, so probing it before authentication is a timing oracle on
        # cache membership.  ``integrity_first=False`` is the leaky order and
        # exists only so experiment 7 can measure the difference.
        rkey = replay_key(env.key_id, env.nonce)
        padded = None
        if aad_mode:
            # Conventional envelope: the AEAD open authenticates header and
            # ciphertext and decrypts in one call.  Same ordering rule:
            # integrity before the replay window unless the leaky order is
            # requested for measurement.
            if not self.integrity_first:
                reason = self.replay.check(rkey, env.timestamp, now)
                tm.mark("freshness_check")
                if reason != Reason.OK:
                    return self._deny(reason, tenant_id, tm)
            try:
                padded = crypto.open_aad_padded(bytes(sk.kslice), env)
            except EnvelopeError as e:
                tm.mark("aead")
                return self._deny(e.reason, tenant_id, tm)
            tm.mark("aead")
            if self.integrity_first:
                reason = self.replay.check(rkey, env.timestamp, now)
                tm.mark("freshness_check")
                if reason != Reason.OK:
                    return self._deny(reason, tenant_id, tm)
        elif self.integrity_first:
            ok = crypto.verify_outer(sk, env)
            tm.mark("hmac")
            if not ok:
                return self._deny(Reason.INTEGRITY, tenant_id, tm)
            reason = self.replay.check(rkey, env.timestamp, now)
            tm.mark("freshness_check")
            if reason != Reason.OK:
                return self._deny(reason, tenant_id, tm)
        else:
            reason = self.replay.check(rkey, env.timestamp, now)
            tm.mark("freshness_check")
            if reason != Reason.OK:
                return self._deny(reason, tenant_id, tm)
            ok = crypto.verify_outer(sk, env)
            tm.mark("hmac")
            if not ok:
                return self._deny(Reason.INTEGRITY, tenant_id, tm)

        # 4b -- structured-nonce checks (authenticated header, pre-AEAD) ---
        # The HMAC has vouched for the header by this point in either order.
        # The key epoch must match the session's live epoch, and the
        # encryption sequence must be strictly monotone per (key, epoch):
        # this rejects rolled-back sequence numbers -- including an exact
        # replay whose nonce the bounded cache has already evicted -- before
        # any AEAD work, and is what makes IV uniqueness enforceable at the
        # receiver rather than a sender promise.
        n_epoch, n_seq = parse_nonce(env.nonce)
        if n_epoch != sk.key_epoch:
            tm.mark("freshness_check")
            return self._deny(Reason.EPOCH_MISMATCH, tenant_id, tm)
        reason = self.sequences.check(env.key_id, n_epoch, n_seq)
        tm.mark("freshness_check")
        if reason != Reason.OK:
            return self._deny(reason, tenant_id, tm)

        # 5 -- IntOK, full path: AEAD open --------------------------------
        try:
            if aad_mode:
                req = crypto.decode_padded(padded, tenant_id)
            else:
                req = crypto.open_envelope(sk, env, tenant_id=tenant_id)
        except EnvelopeError as e:
            tm.mark("aead")
            return self._deny(e.reason, tenant_id, tm)
        tm.mark("aead")

        # 6 -- ContextOK: the authenticated plaintext must remain inside the
        # slice/service namespace that selected the session key.  Key
        # derivation alone does not enforce this binding: a holder of one
        # valid key can encrypt arbitrary slice/service identifiers under it.
        # Reject before any mutable freshness/counter state is committed and
        # before quota or placement acts on attacker-selected metadata.
        if req.slice_type != sk.slice_type or req.service_id != sk.service_id:
            tm.mark("context_check")
            return self._deny(Reason.CONTEXT_MISMATCH, tenant_id, tm)
        # ...and the demand vector must be a quantity of resource. Authenticity
        # is not admissibility: a negative component satisfies every downstream
        # bound by being below it, so ``fits`` accepts it, ``allocate``
        # subtracts it from the node, and ``demand_norm`` hands the quota check
        # a negative cost to treat as credit. Rejected here, in the same stage
        # and for the same reason as a context mismatch -- the authenticated
        # plaintext is not a request this session may make -- and, like it,
        # before any freshness, counter, quota or placement state moves.
        if not req.demand.is_admissible():
            tm.mark("context_check")
            return self._deny(Reason.MALFORMED_REQUEST, tenant_id, tm)
        tm.mark("context_check")

        # 7 -- counter monotonicity, read-only ----------------------------
        reason = self.counters.check(env.key_id, sk.epoch, req.counter)
        tm.mark("counter_check")
        if reason != Reason.OK:
            return self._deny(reason, tenant_id, tm)

        # 8 -- AuthOK: optional NSSAA / CSA re-authentication --------------
        if run_secondary_auth:
            if not self.authenticator(sk):
                tm.mark("auth")
                return self._deny(Reason.AUTH_FAIL, tenant_id, tm)
            tm.mark("auth")
        else:
            tm.skip()

        # 9, 10 -- commit freshness and counter state ----------------------
        # Only once the request has proved authentic; otherwise a forged
        # envelope could burn a nonce.  Each write is timed as its own stage:
        # billing them to whatever stage happened to be marked next put the
        # nonce-cache insert inside the quota figure of the the latency experiment
        # breakdown, and folding them back into the check stages hid how much
        # of that cost only an authentic request pays.
        self.replay.record(rkey, env.timestamp, now)
        self.sequences.record(env.key_id, n_epoch, n_seq)
        tm.mark("freshness_write")
        self.counters.record(env.key_id, sk.epoch, req.counter)
        tm.mark("counter_write")

        # 11 -- QuotaOK -----------------------------------------------------
        # The proxy's arrival ordinal replaces the client's ``arrival_index``,
        # and it is drawn *here* -- at the one point a request actually reaches
        # admission control -- for two reasons.
        #
        # It must not come from the request: ``arrival_index`` rides inside the
        # ciphertext, so the sender picks it, and the sliding-window rate gate
        # discards a tenant's admission history relative to it. Claiming a
        # large index flushes the tenant's own window and the gate stops
        # binding (40 admitted against a limit of 2).
        #
        # It must not be drawn on arrival either. Counting every packet that
        # reaches ``handle`` lets *unauthenticated* junk advance the ordinal,
        # which ages out every tenant's window just as effectively -- the same
        # 40-of-40 bypass, bought with 100 unparseable packets per request
        # instead of a lie. That is precisely what this module's own rule
        # forbids: unauthenticated input must not buy an attacker leverage.
        # Drawing it after authentication, validation and the counter check
        # means only requests that reach the policy advance it, which is also
        # exactly what ``harness.run_policy`` models -- so the semantics
        # experiment 4 measures are unchanged and the two paths still agree.
        req = _replace(req, arrival_index=self._arrivals)
        self._arrivals += 1
        cost = demand_norm(req.demand, self.pool.total_tup)
        free_frac = self.pool.dominant_free_fraction()
        reason = self.policy.admit(req, cost, free_frac,
                                   max(1, len(self.live_sessions)))
        tm.mark("quota")
        if reason != Reason.OK:
            self.policy.on_deny(req, reason)
            return self._deny(reason, tenant_id, tm)

        # 12 -- Feasible(R, d_u) and Isolate(x_u, P) ------------------------
        node, reason = select_node(self.pool.nodes, req.demand,
                                   req.slice_type, self.beta)
        tm.mark("placement")
        if node is None:
            self.policy.on_deny(req, reason)
            return self._deny(reason, tenant_id, tm)

        # 13 -- authenticated commit, then token --------------------------
        session_id = os.urandom(16)
        node.allocate(req.demand, req.slice_type)
        # Widened audit schema: the commit MAC (under the operator-side
        # K_commit, not any session key) covers transaction identity, request
        # counter, policy version, lifetime, and key epoch alongside the
        # allocation tuple.  ``write`` assigns transaction_id and key_epoch.
        record = AllocationRecord(session_id, req.tenant_id,
                                  int(req.slice_type), node.node_id,
                                  req.demand, now,
                                  request_counter=req.counter,
                                  expires_at=now + ADMIT_TOKEN_TTL_S)
        try:
            record = self.store.write(record)
            if not self.store.verify(session_id):
                raise CommitError()
        except CommitError:
            # rollback: delete the record, release the resources, deny with a
            # transient code -- no state leakage.
            self.store.rollback(session_id)
            node.free(req.demand, req.slice_type)
            tm.mark("commit")
            return self._deny(Reason.COMMIT_FAIL, tenant_id, tm)
        tm.mark("commit")

        # One clock. ``now`` already stamps the revocation-list entry and the
        # record's ``expires_at``; the sealed token's own expiry has to come
        # from the same one or a simulated clock splits the three apart.
        token_key = derive_token_key(self._k_eap, sk.key_id, sk.key_epoch)
        token = issue_admit_token(token_key, session_id, now=now)
        self.tokens.issue(token, now)
        tm.mark("token")

        # Token issuance and the token->allocation map insertion are part of the
        # same commit transaction.  An admission is externally successful only
        # once the reservation, the commit record, the issued token, *and* the
        # map that lets the token be released all exist together; a failure to
        # make the map durable must leave none of them behind, or the tenant
        # holds an allocation (and a live token) that the release path cannot
        # resolve -- an unreleasable, quota-consuming orphan.  On failure the
        # token is revoked, the record rolled back, the node freed, and the
        # request denied with the same transient code any commit failure uses.
        try:
            if self._fail_token_map:
                self._fail_token_map = False
                raise CommitError()
            self.sessions[session_id] = (sk, record)
            self._alloc_by_key.setdefault(sk.key_id, set()).add(session_id)
            self._token_by_session[session_id] = token
            self._token_epoch_by_session[session_id] = sk.key_epoch
        except CommitError:
            self.tokens.revoke(token)
            self.sessions.pop(session_id, None)
            live = self._alloc_by_key.get(sk.key_id)
            if live is not None:
                live.discard(session_id)
                if not live:
                    self._alloc_by_key.pop(sk.key_id, None)
            self._token_by_session.pop(session_id, None)
            self._token_epoch_by_session.pop(session_id, None)
            self.store.rollback(session_id)
            node.free(req.demand, req.slice_type)
            return self._deny(Reason.COMMIT_FAIL, tenant_id, tm)

        # Only now, with the whole transaction durable, does the admission
        # become externally successful: charge the policy and count the session.
        self.policy.on_admit(req, cost)
        self.live_sessions[req.tenant_id] += 1

        decision = Decision(True, Reason.OK, node.node_id, req.demand,
                            dict(tm.t))
        return AdmissionOutcome(decision, session_id, token, Reason.OK,
                                Reason.OK)

    # ------------------------------------------------------------- release
    def release_slice(self, session_id: bytes) -> str:
        """Give the resources back without tearing down the security context.

        Internal bookkeeping helper for the experiments -- it takes no token
        and its return value is never requester-facing, so it returns the
        internal reason bare.  The requester-facing path is :meth:`release`.

        Step 6 proper is :meth:`release`, which additionally checks the admit
        token, revokes it, rotates the counter epoch and wipes K_slice.
        Experiments 1 and 5 need the pool empty between timed requests while
        the tenant *keeps* its session -- wiping the keys would re-run AKA on
        every request and the S2/S4 "no AKA" scenarios could not exist.  They
        used to do it by reaching into ``sessions`` and freeing the node by
        hand, which left the commit record, the admit token and the
        live-session count behind; this does the bookkeeping.
        """
        entry = self.sessions.pop(session_id, None)
        if entry is None:
            return Reason.AUTH_FAIL
        sk_entry, record = entry
        # Drop the allocation from its context's live set (and forget its
        # token).  This does *not* touch the security context itself: the key
        # stays registered and usable, which is exactly why the latency and
        # field-hiding experiments call this helper -- they free the node
        # between timed requests while the tenant keeps its session.
        live = self._alloc_by_key.get(sk_entry.key_id)
        if live is not None:
            live.discard(session_id)
            if not live:
                del self._alloc_by_key[sk_entry.key_id]
        self._token_by_session.pop(session_id, None)
        self._token_epoch_by_session.pop(session_id, None)
        node = self.pool.nodes[record.node_id]
        node.free(record.granted, SliceType(record.slice_type))
        # Hand the budget back to the policy as well as to the pool. Without
        # this the tenant's consumed quota only ever grew, so a tenant that
        # admitted and released repeatedly would eventually starve itself while
        # the resources sat idle.
        self.policy.on_release(record.tenant_id, record.granted,
                               demand_norm(record.granted,
                                           self.pool.total_tup),
                               slice_type=SliceType(record.slice_type))
        # Release, not rollback: the commit happened, so the record moves to
        # the store's audit trail where it remains verifiable under K_commit
        # -- including after Step 6 wipes every session key.
        self.store.mark_released(session_id)
        # Discard-on-zero via subtraction: ``Counter`` returns 0 for a missing
        # key without inserting it, so a bare ``del`` on a 0/absent key would
        # raise.  ``-=`` then dropping non-positive entries is safe even if the
        # count is ever already 0.
        self.live_sessions[record.tenant_id] -= 1
        if self.live_sessions[record.tenant_id] <= 0:
            del self.live_sessions[record.tenant_id]
        return Reason.OK

    def release(self, session_id: bytes, token: bytes,
                now: float | None = None) -> ReleaseOutcome:
        """Step 6: validate the release token and close *one allocation*.

        An allocation and the security context that backs it are distinct and
        are torn down separately.  One registration derives one ``SessionKeys``
        (a *context*: K_slice/K_mac, the encryption sequence, the key epoch);
        each admitted request draws a fresh ``session_id`` (an *allocation*: a
        node reservation, a commit record, an admit token), and a single
        context may back several concurrent allocations.  This method closes
        the one allocation the token names -- it frees that allocation's node,
        returns its quota to the policy, retires its commit record to the audit
        trail, and revokes its admit token -- and does **not** disturb the
        context or any sibling allocation on it.

        The context is torn down (K_slice/K_mac wiped, the counter and sequence
        trackers rotated for the handle, the replay epoch bumped) only when the
        allocation being released is the *last* live one on its context.  That
        preserves the single-allocation lifecycle exactly -- a session with one
        allocation still wipes its keys the moment that allocation is released
        -- while fixing the case a multi-allocation session exposed: releasing
        one allocation used to wipe the shared K_slice, silently killing every
        sibling allocation still holding a valid token under the same handle.
        An operator that wants to tear a whole context down regardless of live
        allocations calls :meth:`close_session`.

        Per-allocation replay-after-release does not depend on the context
        teardown: :meth:`csa6g.freshness.TokenStore.consume` drops the released
        token, and while the context stays live the (still un-rotated) sequence
        and counter trackers reject any replay of the released allocation's own
        admission envelope.  Only the last release, which does rotate them,
        additionally retires that state.

        Returns a :class:`ReleaseOutcome`; only its ``external_reason`` may be
        forwarded to the requester.  The externalisation happens here, inside
        the proxy, for the same reason ``_deny`` does it on the admission
        path: an invariant enforced by every caller separately is not an
        invariant.

        ``now`` is injectable so the expired-token class of experiment 3 can be
        driven through this method rather than through the token helper
        directly -- an attack class that never touches the code path it is
        supposed to attack proves nothing about it.

        Order: authenticate the token *before* any mutable state is probed.
        --------------------------------------------------------------------
        The proxy's own rule -- unauthenticated input must not reach state an
        observer can probe -- governs release too, and it binds against a
        *registered* tenant, who can submit arbitrary token strings over its
        authenticated channel, not only against an off-path attacker the
        transport already excludes.  So the issued/revoked store is consulted
        **last**, after the token has been cryptographically authenticated;
        consulting it first (a dict probe on the raw token) would answer "is
        this token in your issued set?" for any such tenant -- an admittance
        oracle the high token entropy makes impractical to exploit but that the
        rule forbids on principle.  The steps:

        1. resolve ``session_id`` to its allocation and registration context,
           using a *decoy* context for an unknown id so the work that follows
           is identical whether or not the id is live -- this is what keeps the
           session table from becoming its own membership oracle (the earlier
           AUC-0.972 finding), now achieved without an early return;
        2. derive that context's operator-side token key and open the token
           AEAD -- one AEAD operation, run for every input including the decoy;
        3. require the decrypted ``sid`` to equal the presented one, and read
           the token's expiry from the *authenticated* payload;
        4. only now consult the issued/revoked store, with
           :meth:`csa6g.freshness.TokenStore.consume` -- a single atomic
           check-and-revoke that returns OK for exactly one caller, so two
           concurrent releases of one token yield exactly one deallocation and
           one quota return;
        5. free the allocation, and tear the context down iff it was the last.
           Steps 4-5 are one critical section: the exactly-once ``consume``
           gates the teardown, so two simultaneous last-allocation releases
           cannot both reach :meth:`_teardown_context`.

        The cost of authenticating first is one token AEAD open on the reject
        path (an expired or forged token now pays for the open it used to
        skip), the release-path analogue of the one HMAC the admission path
        spends to close its own timing oracle.
        """
        now = time.time() if now is None else now

        # 1 -- resolve the allocation's context; decoy for an unknown sid so
        #      the AEAD work below is identical whether or not it is live.
        entry = self.sessions.get(session_id)
        if entry is not None:
            sk = entry[0]
            issuance_epoch = self._token_epoch_by_session[session_id]
            token_key = derive_token_key(self._k_eap, sk.key_id, issuance_epoch)
        else:
            sk = None
            # Same HKDF-Expand as the live branch (one derivation), under a
            # fixed decoy handle, so the unknown-sid path does identical work
            # and does not leak liveness by timing.  Its output never matches a
            # real token, so the AEAD open below fails as it must.
            token_key = derive_token_key(self._k_eap, self._decoy_key_id, 0)

        # 2, 3 -- authenticate: open the token AEAD and match the sid.  Run for
        #         every input (including the decoy) and reject an unknown sid,
        #         a bad tag, or a sid mismatch with the single ``auth_fail``
        #         code -- before the issued store is touched at all.  Expiry is
        #         read from the authenticated payload, so it is trustworthy
        #         here without having consulted the store.
        from .datastore import open_admit_token
        sid, reason = open_admit_token(token_key, token, now=now)
        if entry is None or reason == Reason.AUTH_FAIL or sid != session_id:
            return self._release_outcome(Reason.AUTH_FAIL)
        if reason == Reason.TOKEN_EXPIRED:
            return self._release_outcome(Reason.TOKEN_EXPIRED)

        # 4 -- authenticated: the store is the compare-and-swap that makes
        #      release exactly-once.  A token already consumed by a prior
        #      release (or never issued) fails here.
        reason = self.tokens.consume(token, now)
        if reason != Reason.OK:
            return self._release_outcome(reason)

        # 5 -- one critical section: free this allocation, and tear the context
        #      down iff it was the last.  ``consume`` gates entry, so two
        #      simultaneous last-allocation releases cannot both get here.
        self.release_slice(session_id)        # pool, policy budget, store,
                                              # and this allocation's index entry
        if not self._alloc_by_key.get(sk.key_id):
            self._teardown_context(sk.key_id)
        return self._release_outcome(Reason.OK)

    def close_session(self, key_id: bytes,
                      now: float | None = None) -> ReleaseOutcome:
        """Operator-initiated teardown of a whole security context.

        Force-closes every allocation still live on ``key_id`` -- freeing each
        node, returning each tenant's quota, retiring each commit record, and
        revoking each admit token -- and then tears the context down (wipes
        K_slice/K_mac, rotates the counter and sequence trackers, bumps the
        replay epoch).  Unlike :meth:`release` it takes no admit token: it is an
        operator/RM action, not a requester-facing one, the counterpart of
        :meth:`release_slice` at the context granularity.  Use it to retire a
        context whose tenant is being deprovisioned or whose key is suspected
        compromised, without waiting for each outstanding allocation to be
        released individually.

        Idempotent on an unknown handle: a context that is already gone returns
        ``auth_fail`` rather than raising.
        """
        if self.keys.get(key_id) is None:
            return self._release_outcome(Reason.AUTH_FAIL)
        # Snapshot: release_slice mutates ``_alloc_by_key`` as it goes.
        for sid in list(self._alloc_by_key.get(key_id, ())):
            token = self._token_by_session.get(sid)
            self.release_slice(sid)
            if token is not None:
                self.tokens.revoke(token)
        self._teardown_context(key_id)
        return self._release_outcome(Reason.OK)

    def _teardown_context(self, key_id: bytes) -> None:
        """Retire a security context once no allocation depends on it.

        Rotating the trackers covers *every* epoch the handle held, not just
        the live one: the handle is dead in all of them after this, and a
        session that rotated its key epoch during its life left one tracker
        entry behind per rotation that retiring only the current epoch would
        never reclaim.  ``keys.release`` wipes K_slice/K_mac and bumps the
        replay epoch.
        """
        self._alloc_by_key.pop(key_id, None)
        self.counters.rotate_all(key_id)
        self.sequences.rotate_all(key_id)
        self.keys.release(key_id)             # wipes K_slice, bumps epoch

    @staticmethod
    def _release_outcome(internal: str) -> ReleaseOutcome:
        return ReleaseOutcome(internal, Reason.externalise(internal))

    # -------------------------------------------------------------- helper
    def _deny(self, internal: str, tenant_id: int, tm: _Timer
              ) -> AdmissionOutcome:
        self.denials.append((tenant_id, internal))
        self.denial_counts[internal] += 1
        ext = Reason.externalise(internal)
        return AdmissionOutcome(Decision(False, ext, None, None, dict(tm.t)),
                                None, None, internal, ext)
