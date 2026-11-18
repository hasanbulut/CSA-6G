"""Freshness and replay detection (paper: Threat Model).

A request is fresh iff::

    |tau_recv - tau_u| <= Delta_tau        with Delta_tau = 30 s
    (kid, n_u) not in the replay window    with |W| = 2^16 entries, LRU-evicted

plus a per-session monotone counter, which closes the counter-rollback class
that a pure nonce cache does not: an adversary who replays an *old but still
in-tolerance* envelope after the nonce has been LRU-evicted would otherwise
succeed.  This counter is the mechanism behind the counter-rollback test
class.

Why the cache key carries the session handle
--------------------------------------------
The window is shared by every session the E-AP serves, so its key has to be
unique across sessions, and ``n_u`` alone is not.  Since the freshness token
became structured -- ``key_epoch || encryption_sequence || random(4)`` -- the
first twelve bytes are *identical* for any two sessions at the same point in
their sequence, and every session starts at epoch 0, sequence 0.  Two distinct
sessions therefore share a bare-``n_u`` key whenever their 4-byte random tails
collide: 2^-32 per pair, which over a full 2^16-entry window is a coin flip,
and it denies a legitimate request with ``nonce_replay``.  Keying on
``key_id || n_u`` restores independence between sessions without weakening
replay detection at all -- a genuine replay carries the same handle *and* the
same token.  This is the receiver-side counterpart of the sender-side
uniqueness argument in :mod:`csa6g.keys`: per-key structural uniqueness is
what the IV needs, and it is not the same property as global uniqueness,
which is what a shared cache key needs.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict

from .model import Reason
from .params import (ADMIT_TOKEN_TTL_S, CLOCK_TOLERANCE_S,
                     REPLAY_WINDOW_ENTRIES)


class ReplayWindow:
    """Bounded LRU freshness cache with timestamp-based pruning.

    The parameter is called ``token``, not ``nonce``, deliberately: the caller
    owns the uniqueness of what it hands over, and the value must be unique
    across *every session this window serves*, not merely within one.  The
    E-AP supplies ``key_id || n_u`` (see the module docstring for what went
    wrong when it supplied ``n_u``).
    """

    def __init__(self, capacity: int = REPLAY_WINDOW_ENTRIES,
                 tolerance_s: float = CLOCK_TOLERANCE_S):
        self.capacity = capacity
        self.tolerance = tolerance_s
        self._seen: OrderedDict[bytes, float] = OrderedDict()
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._seen)

    def _prune(self, now: float) -> None:
        # Entries older than the tolerance can never be accepted again on the
        # timestamp check, so they are safe to drop.
        while self._seen:
            token, ts = next(iter(self._seen.items()))
            if now - ts > self.tolerance:
                self._seen.popitem(last=False)
            else:
                break

    def check(self, token: bytes, ts: float, now: float | None = None) -> str:
        """Return :data:`Reason.OK` or the internal denial reason. Read-only."""
        now = time.time() if now is None else now
        # An authenticated sender still controls its timestamp. In particular,
        # both comparisons below are false for NaN, which would otherwise
        # bypass the clock-tolerance rule for a newly sealed envelope.
        if not math.isfinite(ts):
            return Reason.MALFORMED_REQUEST
        delta = now - ts
        if delta > self.tolerance:
            return Reason.STALE
        if delta < -self.tolerance:
            return Reason.FUTURE
        if token in self._seen:
            return Reason.NONCE_REPLAY
        return Reason.OK

    def record(self, token: bytes, ts: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._prune(now)
        self._seen[token] = ts
        self._seen.move_to_end(token)
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
            self.evictions += 1

    def check_and_record(self, token: bytes, ts: float,
                         now: float | None = None) -> str:
        now = time.time() if now is None else now
        reason = self.check(token, ts, now)
        if reason == Reason.OK:
            self.record(token, ts, now)
        return reason


class CounterTracker:
    """Per-session strictly-monotone counter (rollback protection).

    Counters are keyed by ``(key_id, epoch)`` so that the epoch rotation
    performed at secure release (Step 6) invalidates every counter value that
    was valid before release.

    Deliberately unbounded.  An entry lives exactly as long as its session
    (``rotate`` drops it at release), and that retention *is* the mechanism:
    evicting counter state under memory pressure would re-open the very
    replay-after-eviction window this tracker exists to close -- the nonce
    cache is the bounded structure, and this is its backstop.  The cost is
    one dict entry per live session, which is the same order as the key store
    itself.
    """

    def __init__(self, reason: str = Reason.COUNTER_ROLLBACK):
        # ``reason`` is what a rollback reports.  The E-AP instantiates this
        # class twice: once over the encrypted per-request counter
        # (COUNTER_ROLLBACK) and once over the cleartext encryption-sequence
        # prefix of n_u (SEQUENCE_ROLLBACK), which rejects repeated or
        # rolled-back sequence numbers even after the nonce cache has evicted.
        self._last: dict[tuple[bytes, int], int] = {}
        # key_id -> the epochs it has state under.  Exists so that release is
        # O(epochs for this handle) rather than O(all live sessions); see
        # :meth:`rotate_all`.
        self._epochs: dict[bytes, set[int]] = {}
        self.reason = reason

    def check(self, key_id: bytes, epoch: int, counter: int) -> str:
        last = self._last.get((key_id, epoch))
        if last is not None and counter <= last:
            return self.reason
        return Reason.OK

    def record(self, key_id: bytes, epoch: int, counter: int) -> None:
        self._last[(key_id, epoch)] = counter
        self._epochs.setdefault(key_id, set()).add(epoch)

    def rotate_all(self, key_id: bytes) -> int:
        """Drop every epoch's state for one key handle.  Returns the count.

        Retiring only the *current* epoch leaks one entry per rotation.  This
        tracker is also driven over the key epoch, which
        :meth:`csa6g.keys.SessionKeys.rotate_key_epoch` may advance any number
        of times during a session's life, and each advance opens a new
        ``(key_id, epoch)`` slot that nothing afterwards reclaims.  The class
        is deliberately unbounded because retention is what closes the
        replay-after-eviction window -- which is exactly what makes it the
        wrong structure to leave orphans in once the session that justified
        them is gone.  At secure release the handle is dead in every epoch, so
        all of it goes.

        There is deliberately no single-epoch ``rotate``.  One existed, secure
        release called it, and that was the whole of the leak; leaving it
        available beside this method would invite a future caller to
        reintroduce it.

        Cost is O(epochs held by *this* handle), via the ``_epochs`` index.
        The obvious implementation -- scan ``_last`` for matching keys -- is
        O(all live sessions) per release and therefore O(n^2) to drain a busy
        proxy, which converts a fix for a memory leak into a denial-of-service
        surface: measured at 16k live sessions a single release took ~510 us
        against ~35 us at 1k, and the work is driven by *other tenants'*
        sessions rather than by anything the releasing tenant owns.
        """
        epochs = self._epochs.pop(key_id, None)
        if not epochs:
            return 0
        dropped = 0
        for epoch in epochs:
            if self._last.pop((key_id, epoch), None) is not None:
                dropped += 1
        return dropped


class TokenStore:
    """Admit tokens issued at Step 5: the E-AP's revocation list.

    Tokens are opaque handles here; the AEAD-sealed form is minted by
    :func:`csa6g.datastore.issue_admit_token` and carries its own expiry.  The
    two checks are not redundant.  The sealed token proves *authenticity* and
    that the stated expiry has not passed, but a token stays cryptographically
    valid after the session it names has been released -- nothing inside it
    changes.  This store is what makes release stick: :meth:`revoke` drops the
    handle, so a replayed post-release token fails here before any AEAD work.
    """

    # Amortised sweep cadence for expired entries.  Without it the store only
    # ever shrank via explicit revocation, so tokens whose sessions simply let
    # them expire accumulated forever.  Sweeping every N issues keeps the
    # amortised cost O(1) per issue and, unlike sweeping on every call, stays
    # off the timed token stage of experiment 1.
    PRUNE_EVERY = 4096

    def __init__(self, ttl_s: float = ADMIT_TOKEN_TTL_S):
        self.ttl = ttl_s
        self._issued: dict[bytes, float] = {}
        self._issue_count = 0

    def issue(self, token: bytes, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._issued[token] = now + self.ttl
        self._issue_count += 1
        if self._issue_count % self.PRUNE_EVERY == 0:
            self._issued = {t: exp for t, exp in self._issued.items()
                            if exp >= now}

    def validate(self, token: bytes, now: float | None = None) -> str:
        """Read-only membership/expiry check.  Does not mutate the store.

        Kept for callers that need to look without consuming; the release path
        uses :meth:`consume` instead, so that the check and the state change are
        one atomic step (see there).
        """
        now = time.time() if now is None else now
        exp = self._issued.get(token)
        if exp is None:
            return Reason.AUTH_FAIL
        if now > exp:
            return Reason.TOKEN_EXPIRED
        return Reason.OK

    def consume(self, token: bytes, now: float | None = None) -> str:
        """Atomically check the token is active and, on success, revoke it.

        Returns :data:`Reason.OK` for a token that is currently issued and
        unexpired -- and removes it in the same step, so a *second* call with
        the same token returns ``AUTH_FAIL``.  This single check-and-remove is
        the compare-and-swap that makes secure release exactly-once: two
        concurrent releases of one token cannot both proceed, because only the
        first ``consume`` observes the token as active.  An unknown or
        already-consumed token returns ``AUTH_FAIL``; an expired one returns
        ``TOKEN_EXPIRED`` and is also dropped (it can never succeed again).

        In this single-threaded prototype the atomicity is structural -- one
        dict ``pop`` under the GIL; a threaded or distributed deployment
        implements the same contract with a row-level compare-and-swap on the
        token's ``active`` flag.
        """
        now = time.time() if now is None else now
        exp = self._issued.get(token)
        if exp is None:
            return Reason.AUTH_FAIL
        if now > exp:
            self._issued.pop(token, None)
            return Reason.TOKEN_EXPIRED
        del self._issued[token]
        return Reason.OK

    def revoke(self, token: bytes) -> None:
        self._issued.pop(token, None)
