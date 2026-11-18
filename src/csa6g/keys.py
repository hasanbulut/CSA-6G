"""Key hierarchy and lifecycle.

The implementation starts from a 256-bit application/edge anchor ``K_A`` that
is assumed to be established *after* primary 5G authentication inside a trusted
operator boundary.  It must not be read as exporting raw ``K_SEAF`` to an
application function: 3GPP TS 33.501 requires ``K_SEAF`` to remain inside the
SEAF.  A deployment therefore needs either co-location with the relevant
trusted function or a standards-sanctioned child-key interface.

Slice- and service-scoped keys are derived with HKDF-Expand (RFC 5869) under
domain-separated info strings::

    K_slice = HKDF-Expand(K_A, "slice" || S-NSSAI || svc_id, 32)
    K_mac   = HKDF-Expand(K_A, "mac"   || S-NSSAI || svc_id, 32)

Three implementation choices are worth making explicit:

1.  ``K_A`` is modelled as uniformly random.  ``extract_first=True`` is exposed
    so the extract-then-expand variant can also be benchmarked when the input
    keying material is not already a suitable pseudorandom key.
2.  AES-GCM needs a 96-bit IV, and NIST SP 800-38D requires the IV-generation
    mechanism to *guarantee* uniqueness per key -- a guarantee an IV derived
    from a freely chosen random nonce cannot give (a faulty or reseeded sender
    RNG repeats n_u, and with it the IV, under an unchanged K_slice).  The
    freshness token is therefore structured, not random::

        n_u  = key_epoch (32) || encryption_sequence (64) || random (32)
        IV_u = key_epoch (32) || encryption_sequence (64)

    The sequence is allocated atomically per session key and never resets
    while the same K_slice is live; the key epoch is bound into the key
    derivation itself, so an epoch change rotates K_slice and the (epoch,
    sequence) pair -- and hence the IV -- is structurally unique per key
    rather than probabilistically unique.  n_u stays 16 bytes on the wire, the
    receiver reads the IV from the cleartext header before any decryption,
    and the outer HMAC covers the whole field.  The trailing 32 random bits
    make the token unpredictable to an off-path party but carry no IV weight.

    What those 32 bits are *not* is a cross-session identifier.  Uniqueness
    here is per key: two sessions at the same point in their sequence share
    the whole epoch || sequence prefix, so n_u values from different sessions
    differ only in the random tail.  That is sufficient for the IV -- equal
    IVs under distinct K_slice are harmless -- and insufficient for anything
    keyed globally.  The E-AP's shared replay window is exactly such a thing,
    so it keys on key_id || n_u; see :func:`csa6g.eap.replay_key`.
3.  The admit token of Step 5 is an E-AP-issued release capability, so its
    key must not be one the UE holds.  It is derived from an E-AP-only root
    K_EAP as K_token = HKDF-Expand(K_EAP, "CSA6G/token" || kid || e, 32)
    (see :func:`derive_token_key`), never from K_slice.  This keeps the token
    key out of the UE's reach -- a tenant holding K_slice/K_mac cannot mint a
    token the release path accepts -- and keeps the envelope and token AEADs
    in separate key spaces so their IV regimes cannot interact.
"""
from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand

from .model import SliceType

KEY_BYTES = 32

# Structured freshness token n_u (16 bytes on the wire, unchanged):
#   key_epoch (4) || encryption_sequence (8) || random (4)
# The AEAD IV is the first 12 bytes -- epoch || sequence -- so IV uniqueness
# under one K_slice is structural (atomic sequence allocation), not
# probabilistic (random nonce collision).
NONCE_EPOCH_BYTES = 4
NONCE_SEQ_BYTES = 8
NONCE_RANDOM_BYTES = 4
IV_BYTES = NONCE_EPOCH_BYTES + NONCE_SEQ_BYTES          # 12
_NONCE_PREFIX = struct.Struct(">IQ")                     # epoch, sequence
MAX_ENCRYPTION_SEQUENCE = (1 << (8 * NONCE_SEQ_BYTES)) - 1


class KeyHandleCollision(RuntimeError):
    """Two live sessions drew the same opaque key handle.

    Astronomically unlikely at 8 random bytes and not attacker-steerable, but
    silently resolving it in favour of the newcomer breaks the displaced
    session with no diagnostic anywhere.
    """


class SequenceExhausted(RuntimeError):
    """The 64-bit encryption sequence is exhausted for this key epoch.

    The sequence must never restart while the same ``K_slice`` is live, so the
    only way forward is a key-epoch rotation (which rederives ``K_slice``).
    Unreachable in practice at 2^64 seals; the guard exists so the invariant
    is enforced rather than assumed.
    """


def _expand(prk: bytes, info: bytes, length: int) -> bytes:
    return HKDFExpand(algorithm=hashes.SHA256(), length=length,
                      info=info).derive(prk)


def _extract_expand(ikm: bytes, info: bytes, length: int,
                    salt: bytes | None = None) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt,
                info=info).derive(ikm)


def snssai_bytes(slice_type: SliceType, sd: int = 0) -> bytes:
    """4-byte S-NSSAI: 1-byte SST + 3-byte Slice Differentiator."""
    return struct.pack("<B", int(slice_type)) + struct.pack("<I", sd)[:3]


def derive_slice_keys(k_anchor: bytes, slice_type: SliceType, service_id: int,
                      sd: int = 0, extract_first: bool = False,
                      key_epoch: int = 0, key_id: bytes = b""
                      ) -> tuple[bytes, bytes]:
    """Return ``(K_slice, K_mac)`` from the post-authentication anchor.

    ``key_epoch`` is bound into the derivation context: rotating the epoch
    rotates ``K_slice`` and ``K_mac``.  That is what lets the encryption
    sequence restart at zero after a rotation without ever producing a
    repeated (epoch, sequence) pair -- and hence a repeated AEAD IV -- under
    one key.

    ``key_id`` is the random per-session key handle.  Binding it into the
    context makes the derived keys unique per *registration*, not merely per
    anchor: a client that restarts and re-registers -- even one that somehow
    presents the same anchor -- obtains a fresh ``K_slice``, so its
    encryption sequence restarting at zero can never repeat an IV under a key
    that has sealed traffic before.  This is the "rotate the key rather than
    resume the sequence" arm of the SP 800-38D uniqueness requirement,
    enforced structurally.
    """
    ctx = (snssai_bytes(slice_type, sd) + struct.pack("<I", service_id)
           + struct.pack("<I", key_epoch) + key_id)
    f = _extract_expand if extract_first else _expand
    return (f(k_anchor, b"slice" + ctx, KEY_BYTES),
            f(k_anchor, b"mac" + ctx, KEY_BYTES))


def derive_token_key(k_eap: bytes, key_id: bytes, key_epoch: int) -> bytes:
    """Operator-side admit-token key ``K_token^(kid,e)``.

    ``K_token = HKDF-Expand(K_EAP, "CSA6G/token" || kid || e, 32)``

    Derived from an E-AP-only root ``K_EAP`` and scoped to the session key
    handle and epoch.  Crucially it is *not* derived from ``K_slice``: the UE
    holds ``K_slice``/``K_mac`` to build and authenticate the envelope, but the
    admit token is an E-AP-issued release capability, so its key must live in a
    key space the UE cannot reach.  A tenant therefore cannot mint a token that
    the release path will accept, even for its own session.
    """
    return _expand(k_eap, b"CSA6G/token" + key_id + struct.pack("<I", key_epoch),
                   KEY_BYTES)


def derive_commit_key(k_rm: bytes, rm_id: bytes, epoch: int) -> bytes:
    """Operator-side datastore commit key ``K_commit^(e)``.

    ``K_commit^(e) = HKDF-Expand(K_RM, "CSA6G/commit" || RM_ID || e, 32)``

    ``K_RM`` is an operator-provisioned Resource Manager root key.  The commit
    key is independent of every session key (``K_slice``, ``K_mac``,
    ``K_token``), is never provisioned to the UE, and outlives any session:
    committed allocation records therefore stay verifiable for audit,
    accounting, and dispute resolution after the session keys are wiped at
    secure release.  ``epoch`` is the datastore key epoch used for rotation.
    """
    return _expand(k_rm, b"CSA6G/commit" + rm_id + struct.pack("<I", epoch),
                   KEY_BYTES)


def build_nonce(key_epoch: int, sequence: int,
                random_tail: bytes | None = None) -> bytes:
    """Assemble the structured 16-byte freshness token n_u."""
    tail = os.urandom(NONCE_RANDOM_BYTES) if random_tail is None else random_tail
    return _NONCE_PREFIX.pack(key_epoch, sequence) + tail


def parse_nonce(nonce: bytes) -> tuple[int, int]:
    """Return ``(key_epoch, encryption_sequence)`` from a freshness token."""
    return _NONCE_PREFIX.unpack(nonce[:IV_BYTES])


def iv_from_nonce(nonce: bytes) -> bytes:
    """96-bit AEAD IV: the epoch || sequence prefix of the freshness token.

    Deterministic from the cleartext header field, so the receiver derives the
    IV before any decryption -- and structurally unique per key, because the
    sender allocates the sequence atomically and an epoch change rotates the
    key.
    """
    return nonce[:IV_BYTES]


@dataclass
class SessionKeys:
    """Slice-scoped key material for one admitted session.

    ``tenant_id`` is the authenticated tenant attribution: it is bound here,
    at registration time (i.e. at the trust level of primary authentication),
    and the E-AP reads it from the session rather than trusting a label the
    requester supplies per call.  Quota and fairness state is keyed on it, so
    an unbound attribution would let an adversary evade per-tenant caps by
    claiming an arbitrary tenant.  ``-1`` means unbound and only occurs when
    key material is built directly for crypto-level tests.

    Keys are held in ``bytearray`` so that :meth:`wipe` zeroises the
    long-lived copies in place (Step 6, secure release).  That is best-effort
    in CPython, not a guarantee: every call into the ``cryptography`` API
    makes an immutable ``bytes`` copy that cannot be wiped, and the derivation
    output the buffers were filled from is equally beyond reach.  The property
    the tests enforce is the functional one -- after :meth:`wipe` the session
    is dead (``assert_live`` raises), the epoch has rotated, and no envelope
    or token sealed under the old keys is accepted again.

    ``epoch`` is the counter that is rotated on release to prevent
    replay-after-release.

    ``key_epoch`` is the 32-bit IV/key-derivation epoch carried in the first
    four bytes of every freshness token; it is bound into the HKDF context, so
    rotating it rederives ``K_slice``/``K_mac`` and lets the 64-bit encryption
    sequence restart without ever repeating an (epoch, sequence) IV under one
    key.  It is distinct from ``epoch``, which scopes replay-counter state and
    only ever advances at secure release.

    ``reserve`` allocates the encryption sequence and snapshots the key
    material for it in one critical section, so concurrent seals can never
    draw the same sequence number and a seal can never pair a sequence with
    keys from a different epoch; :meth:`rotate_key_epoch` takes the same
    lock.  The sequence never resets while this key is live: the only reset
    path is :meth:`rotate_key_epoch`, which rotates the key material itself.  A
    process restart cannot silently reset it either -- key material is held
    only in memory, so a restart loses the key and forces re-registration,
    which is the "rotate the key before restarting the sequence" arm of the
    SP 800-38D requirement.
    """
    key_id: bytes
    kslice: bytearray
    kmac: bytearray
    slice_type: SliceType
    service_id: int
    tenant_id: int = -1
    epoch: int = 0
    key_epoch: int = 0
    wiped: bool = False
    _seq_next: int = 0
    _seq_lock: threading.Lock = field(default_factory=threading.Lock,
                                      repr=False, compare=False)

    @classmethod
    def create(cls, k_anchor: bytes, slice_type: SliceType, service_id: int,
               key_id: bytes | None = None, tenant_id: int = -1,
               key_epoch: int = 0) -> "SessionKeys":
        key_id = key_id or os.urandom(8)
        ks, km = derive_slice_keys(k_anchor, slice_type, service_id,
                                   key_epoch=key_epoch, key_id=key_id)
        return cls(key_id=key_id,
                   kslice=bytearray(ks), kmac=bytearray(km),
                   slice_type=slice_type, service_id=service_id,
                   tenant_id=tenant_id, key_epoch=key_epoch)

    def next_nonce(self) -> bytes:
        """Atomically allocate the next structured freshness token n_u.

        Raises :class:`SequenceExhausted` once the 64-bit space is spent for
        this key epoch; the sequence is never wrapped or reset under a live
        key.
        """
        return self.reserve()[0]

    def reserve(self) -> tuple[bytes, bytes, bytes]:
        """Atomically allocate a nonce *and* snapshot the keys it belongs to.

        Returns ``(nonce, k_slice, k_mac)`` taken under one critical section.
        Reading ``key_epoch`` or the key bytes after releasing the lock would
        let a concurrent :meth:`rotate_key_epoch` interleave, so that a seal
        could carry the new epoch with a sequence reserved before the reset
        -- and a later genuine seal would then repeat that (epoch, sequence)
        IV under the same rotated ``K_slice``.  Sealing must use exactly the
        material returned here.
        """
        with self._seq_lock:
            self.assert_live()
            seq = self._seq_next
            if seq > MAX_ENCRYPTION_SEQUENCE:
                raise SequenceExhausted(
                    "encryption sequence exhausted; rotate the key epoch")
            self._seq_next = seq + 1
            return (build_nonce(self.key_epoch, seq),
                    bytes(self.kslice), bytes(self.kmac))

    def rotate_key_epoch(self, k_anchor: bytes) -> None:
        """Advance the key epoch: rederive the key material, reset the sequence.

        The sequence may restart at zero *only because* the keys it seals
        under have changed -- envelope IVs from the old epoch were minted
        under the old ``K_slice`` and can never collide with the new ones.
        Envelopes sealed under the previous epoch fail both the receiver's
        epoch check and the rotated MACs.
        """
        with self._seq_lock:
            self.assert_live()
            self.key_epoch += 1
            ks, km = derive_slice_keys(k_anchor, self.slice_type,
                                       self.service_id,
                                       key_epoch=self.key_epoch,
                                       key_id=self.key_id)
            self.kslice[:] = ks
            self.kmac[:] = km
            self._seq_next = 0

    def wipe(self) -> None:
        """Zeroise key material in place and mark the session dead.

        Takes the same lifecycle lock as :meth:`reserve` and
        :meth:`rotate_key_epoch`, and those check liveness *inside* their
        critical sections, so no reservation or rotation can
        be half-way through when the buffers are zeroed: a reserve that
        wins the lock first returns an immutable snapshot of live keys and
        that seal may complete; one that loses raises.  Snapshots already
        handed out are Python ``bytes`` and cannot be erased retroactively;
        this is the documented best-effort scope of wiping.
        """
        with self._seq_lock:
            for buf in (self.kslice, self.kmac):
                for i in range(len(buf)):
                    buf[i] = 0
            self.epoch += 1
            self.wiped = True

    def assert_live(self) -> None:
        if self.wiped:
            raise RuntimeError("session keys have been wiped (post-release use)")


class KeyStore:
    """Maps opaque key handles to session key material.

    The handle travels in cleartext in the envelope header so that the E-AP can
    select a decryption key without learning S-NSSAI.  Handles are random per
    session, so they leak session linkage but not slice type -- a residual
    channel that experiment 6 quantifies rather than hides.
    """

    def __init__(self):
        self._by_id: dict[bytes, SessionKeys] = {}

    def register(self, sk: SessionKeys) -> bytes:
        """Register a session under its handle.

        Raises :class:`KeyHandleCollision` rather than overwriting.  A handle
        is 8 bytes drawn server-side, so a collision needs on the order of
        2^32 concurrent sessions and is not something a requester can steer --
        but the silent-overwrite behaviour this replaces displaced the earlier
        session while leaving it believing it was live, so its traffic failed
        to resolve for reasons nothing reported.  A condition that rare and
        that confusing should fail loudly at the point it happens rather than
        surface later as one tenant's inexplicable outage.
        """
        existing = self._by_id.get(sk.key_id)
        if existing is not None and existing is not sk:
            raise KeyHandleCollision(
                f"key handle {sk.key_id.hex()} is already registered; "
                f"re-draw the handle rather than displacing the live session")
        self._by_id[sk.key_id] = sk
        return sk.key_id

    def get(self, key_id: bytes) -> SessionKeys | None:
        return self._by_id.get(key_id)

    def release(self, key_id: bytes) -> bool:
        sk = self._by_id.pop(key_id, None)
        if sk is None:
            return False
        sk.wipe()
        return True

    def __len__(self) -> int:
        return len(self._by_id)


def run_anchor_bootstrap(rng) -> bytes:
    """Return a random 256-bit ``K_A`` for the simulation.

    This function does not implement 5G-AKA, EAP-AKA', AKMA, or a 3GPP key
    export.  It only supplies the downstream prototype with a fresh
    post-authentication anchor.  The latency experiment accounts separately
    for this local bootstrap surrogate.
    """
    return bytes(rng.integers(0, 256, size=KEY_BYTES, dtype="uint8"))


# Backward-compatible name retained for older experiment scripts.  New code
# should use ``run_anchor_bootstrap`` to avoid implying that AKA is implemented.
run_primary_authentication = run_anchor_bootstrap
