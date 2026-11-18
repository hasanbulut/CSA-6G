"""Authenticated state commit and secure release (paper Steps 5 and 6).

Two-phase write:

  (a) the Resource Manager computes a commit MAC over the *widened* allocation
      record

        M_c = HMAC-SHA256_{K_commit^(e)}(version || transaction_id ||
              session_id || tenant || slice_type || node_id || granted ||
              request_counter || policy_version || committed_at ||
              expires_at || key_epoch)

      and writes ``(record, M_c)`` atomically.  ``K_commit^(e)`` is an
      operator-side key derived from a Resource Manager root key ``K_RM``
      (see :func:`csa6g.keys.derive_commit_key`); it is independent of every
      session key and is never provisioned to the UE.

      MACing the record under the *session* key ``K_mac`` would make every
      committed record unverifiable once Step 6 wipes that key at release,
      which is exactly when audit, accounting or dispute resolution needs it,
      and would reuse one key across two message families in two trust
      domains.  The operator-side key keeps records verifiable after the
      session keys are wiped (the store retains released records as an audit
      trail).

  (b) on a verified write the E-AP issues the admit token
        AEAD_{K_token}(session_id || tau_exp)
      under the operator-side key K_token (see :func:`csa6g.keys.derive_token_key`),
      not K_slice.  The dedicated key exists because the envelope AEAD and the
      token AEAD generate their IVs differently (structured epoch||sequence
      versus drawn at random), and two IV regimes under one AES-GCM key is an
      avoidable cross-context collision risk -- see :mod:`csa6g.keys`.

A failed or unverified write triggers rollback: the record is deleted,
resources are released, and the requester sees only a transient-error code.
``fail_next`` injects that failure so the rollback path is exercised by the
test suite rather than merely asserted in prose.
"""
from __future__ import annotations

import hmac as _hmac
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, replace

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import SessionKeys, derive_commit_key
from .model import Demand, Reason
from .params import ADMIT_TOKEN_TTL_S, AUDIT_TRAIL_ENTRIES, POLICY_VERSION

RECORD_VERSION = 1
_ZERO_TXN = b"\x00" * 16


@dataclass(frozen=True)
class AllocationRecord:
    """Durable allocation record (Step 5).

    The first six fields are the original allocation tuple; the remainder is
    the audit schema the commit MAC must cover so that a record cannot be
    replayed under a different transaction, session, tenant, placement,
    policy revision, or lifetime.  ``transaction_id`` and ``key_epoch`` are
    assigned by :meth:`AuthenticatedStore.write`.
    """
    session_id: bytes
    tenant_id: int
    slice_type: int
    node_id: int
    granted: Demand
    committed_at: float
    version: int = RECORD_VERSION
    transaction_id: bytes = _ZERO_TXN
    request_counter: int = 0
    policy_version: int = POLICY_VERSION
    expires_at: float = 0.0
    key_epoch: int = 0

    def serialise(self) -> bytes:
        # Doubles for the demand vector, matching AdmissionRequest.STRUCT.
        # Every field of the audit schema is serialised, so the commit MAC
        # covers all of them; mutating any single field breaks verification.
        return struct.pack(
            "<B16s16siIIdddQIddI",
            self.version, self.transaction_id, self.session_id,
            self.tenant_id, self.slice_type, self.node_id,
            self.granted.cpu, self.granted.mem, self.granted.storage,
            self.request_counter, self.policy_version,
            self.committed_at, self.expires_at, self.key_epoch)


class CommitError(Exception):
    def __init__(self, reason=Reason.COMMIT_FAIL):
        super().__init__(reason)
        self.reason = reason


class AuthenticatedStore:
    """In-memory datastore whose every record carries a commit MAC.

    MACs are computed under the operator-side ``K_commit^(e)``, so
    verification does not depend on any session key: records written during a
    session remain verifiable after that session is released and its keys are
    wiped.  ``rotate_key_epoch`` starts MACing new records under the next
    epoch while historical epochs stay verifiable; a record naming an unknown
    epoch fails verification.
    """

    def __init__(self, k_rm: bytes | None = None, rm_id: bytes = b"RM-0",
                 key_epoch: int = 0,
                 audit_capacity: int = AUDIT_TRAIL_ENTRIES):
        # ``k_rm`` models the operator-provisioned Resource Manager root key.
        # The prototype draws it at random per store; a deployment provisions
        # it out of band.  It is never handed to any session or UE.
        self._k_rm = k_rm if k_rm is not None else os.urandom(32)
        self.rm_id = rm_id
        self.key_epoch = key_epoch
        self._commit_keys: dict[int, bytes] = {
            key_epoch: derive_commit_key(self._k_rm, rm_id, key_epoch)}
        self._records: dict[bytes, tuple[AllocationRecord, bytes]] = {}
        # Ordered by release time so the bound evicts oldest-first.
        self._released: "OrderedDict[bytes, tuple[AllocationRecord, bytes]]" \
            = OrderedDict()
        self.audit_capacity = audit_capacity
        #: Records dropped from the in-memory trail because the bound was hit.
        #: Non-zero means audit history was lost and the deployment is not
        #: draining fast enough -- it is exposed rather than logged-and-
        #: forgotten so the loss is answerable.
        self.evicted_records = 0
        self.fail_next = False          # fault injection for rollback tests
        self.rollbacks = 0
        self.commits = 0

    # ------------------------------------------------------------ key mgmt
    def rotate_key_epoch(self) -> int:
        """Advance the commit-key epoch; historical epochs stay verifiable."""
        self.key_epoch += 1
        self._commit_keys[self.key_epoch] = derive_commit_key(
            self._k_rm, self.rm_id, self.key_epoch)
        return self.key_epoch

    # ----------------------------------------------------------- phase (a)
    def commit_mac(self, record: AllocationRecord) -> bytes:
        key = self._commit_keys.get(record.key_epoch)
        if key is None:
            raise CommitError()
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(record.serialise())
        return h.finalize()

    def write(self, record: AllocationRecord) -> AllocationRecord:
        """Complete, MAC, and store a record; return the completed record.

        ``transaction_id`` (fresh, random) and ``key_epoch`` (the store's
        current epoch) are filled in here: the transaction identity and the
        key that vouches for the record are the store's to assign, not the
        caller's.

        Both are assigned *unconditionally*.  Honouring a caller-supplied
        transaction id -- as this used to, whenever the field was not the zero
        sentinel -- contradicted that sentence and let two distinct records be
        committed under one transaction identity, each with a perfectly valid
        MAC, which is exactly the confusion a transaction id exists to prevent
        in an audit trail.  Nothing needs the override: the E-AP always
        constructs its record with the sentinel, and the tests that pin a
        transaction id do so by writing into ``_records`` directly, because
        what they check is MAC coverage rather than assignment.
        """
        record = replace(record,
                         transaction_id=os.urandom(16),
                         key_epoch=self.key_epoch)
        mac = self.commit_mac(record)
        if self.fail_next:
            self.fail_next = False
            raise CommitError()
        self._records[record.session_id] = (record, mac)
        self.commits += 1
        return record

    def verify(self, session_id: bytes) -> bool:
        """Verify a record's commit MAC.  Works for live *and* released
        records, and requires no session key material whatsoever."""
        entry = self._records.get(session_id) or self._released.get(session_id)
        if entry is None:
            return False
        record, mac = entry
        key = self._commit_keys.get(record.key_epoch)
        if key is None:
            return False
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(record.serialise())
        return _hmac.compare_digest(mac, h.finalize())

    def mark_released(self, session_id: bytes) -> AllocationRecord | None:
        """Step 6: retire a record to the audit trail.

        Release is not rollback: the allocation happened, so the record is
        retained (still MACed, still verifiable under ``K_commit``) rather
        than deleted.  This is the property the commit-key separation buys --
        the session keys can be wiped without orphaning the history.

        The trail is bounded.  Retention is the whole point of it, but an
        unbounded in-memory trail is a memory-exhaustion vector an
        *authenticated* tenant drives for free: one admit/release cycle adds
        one permanent record, and nothing caps the cycles.  Past
        ``audit_capacity`` the oldest entries are evicted and counted in
        :attr:`evicted_records`; a deployment that cannot afford to lose them
        drains to durable storage with :meth:`drain_released` instead of
        letting the bound bite.  Evicted records stop verifying, which is the
        honest cost of a bound and the reason the counter is public.
        """
        entry = self._records.pop(session_id, None)
        if entry is None:
            return None
        self._released[session_id] = entry
        self._released.move_to_end(session_id)
        while len(self._released) > self.audit_capacity:
            self._released.popitem(last=False)
            self.evicted_records += 1
        return entry[0]

    def drain_released(self, limit: int | None = None
                       ) -> list[tuple[AllocationRecord, bytes]]:
        """Remove and return the oldest retained ``(record, mac)`` pairs.

        The escape hatch that makes the bound above a policy rather than a
        data-loss bug: an operator persists what it drains, so the in-memory
        trail stays small without the history being destroyed.  Returned
        pairs carry their commit MAC, so they remain verifiable off-box under
        the same ``K_commit`` epoch -- verification never needed a session key
        and does not need this store either.
        """
        n = len(self._released) if limit is None else min(limit,
                                                          len(self._released))
        return [self._released.popitem(last=False)[1] for _ in range(n)]

    def rollback(self, session_id: bytes) -> AllocationRecord | None:
        """Undo a *failed* commit.

        The counter is incremented on every invocation, not only when a record
        is found.  A write that failed before reaching the store still leaves
        an allocated node to revert, and that reversion is the rollback the
        Step 5 property is about; counting only the record deletions would
        report zero rollbacks in exactly the case the property matters most.
        """
        self.rollbacks += 1
        entry = self._records.pop(session_id, None)
        return entry[0] if entry is not None else None

    def get(self, session_id: bytes) -> AllocationRecord | None:
        entry = self._records.get(session_id)
        return entry[0] if entry else None

    def get_released(self, session_id: bytes) -> AllocationRecord | None:
        entry = self._released.get(session_id)
        return entry[0] if entry else None

    def __len__(self) -> int:
        return len(self._records)


# ----------------------------------------------------------- phase (b) -----

def issue_admit_token(token_key: bytes, session_id: bytes,
                      ttl_s: float = ADMIT_TOKEN_TTL_S,
                      now: float | None = None) -> bytes:
    """AEAD_{K_token}(session_id || tau_exp).

    ``token_key`` is the operator-side admit-token key
    (:func:`csa6g.keys.derive_token_key`), derived by the E-AP from its
    K_EAP root rather than from any UE-held key, so the UE cannot mint a
    token the release path accepts.

    ``now`` is the issuing clock, and it is a parameter for the same reason
    :meth:`csa6g.eap.EdgeAdmissionProxy.release` takes one: the E-AP already
    stamps the revocation-list entry and the allocation record's ``expires_at``
    from an injectable clock, and reading the wall clock here instead put the
    sealed expiry on a *different* timeline from the two checks that gate it.
    A proxy driven at a simulated ``now`` -- a discrete-event run, a frozen or
    offset clock in a test -- then minted tokens that ``open_admit_token``
    rejected as ``token_expired`` the instant they were issued, while the
    ``TokenStore`` that shares the E-AP's clock still considered them live.
    Defaults to the wall clock, so callers that do not simulate time are
    unaffected.
    """
    now = time.time() if now is None else now
    iv = os.urandom(12)
    payload = session_id + struct.pack("<d", now + ttl_s)
    return iv + AESGCM(bytes(token_key)).encrypt(iv, payload, b"admit")


def open_admit_token(token_key: bytes, token: bytes,
                     now: float | None = None) -> tuple[bytes, str]:
    """Return ``(session_id, reason)``. ``reason`` is OK, TOKEN_EXPIRED or
    AUTH_FAIL.  ``token_key`` is the operator-side key of
    :func:`csa6g.keys.derive_token_key`."""
    from cryptography.exceptions import InvalidTag
    now = time.time() if now is None else now
    try:
        payload = AESGCM(bytes(token_key)).decrypt(token[:12], token[12:],
                                                   b"admit")
    except (InvalidTag, RuntimeError, ValueError):
        # ValueError covers a truncated token, whose 12-byte nonce slice is
        # short enough that the AEAD rejects the *nonce* before the tag.
        return b"", Reason.AUTH_FAIL
    if len(payload) < 24:
        return b"", Reason.AUTH_FAIL
    session_id, tau_b = payload[:16], payload[16:24]
    (tau_exp,) = struct.unpack("<d", tau_b)
    if now > tau_exp:
        return session_id, Reason.TOKEN_EXPIRED
    return session_id, Reason.OK
