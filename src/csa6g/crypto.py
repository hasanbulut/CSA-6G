"""The encrypted admission envelope (paper Section 4.1).

Wire format -- constant 337 bytes regardless of request content::

    +---------+----------+-----------+---------------+
    | ver (1) | kid (8)  | n_u (16)  | tau_u (8)     |   header, cleartext
    +---------+----------+-----------+---------------+
      n_u = key_epoch (4) || encryption_sequence (8) || random (4);
      the AEAD IV is its first 12 bytes (see csa6g.keys).
    | C_u (256)                                      |   AES-256-GCM ciphertext
    +------------------------------------------------+
    | T_gcm (16)                                     |   AEAD tag
    +------------------------------------------------+
    | T_u (32)                                       |   outer HMAC-SHA256
    +------------------------------------------------+

Why both tags.  ``T_gcm`` authenticates the ciphertext under K_slice.  The
outer ``T_u`` authenticates *the whole envelope including the AEAD tag and the
freshness fields* under the independently derived K_mac.  That lets the E-AP
verify session binding and reject stale or rebound envelopes with one HMAC and
no AEAD decryption -- which is what makes the cheap-rejection path in Section
6.3 cheap.  Experiment 3 asserts this ordering empirically.

Notation: the paper's T^gcm_u and T^mac_u are ``gcm_tag`` and ``outer_tag``
here.
"""
from __future__ import annotations

import contextlib
import hmac as _hmac
import os
import struct
import time
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import SessionKeys, iv_from_nonce
from .model import AdmissionRequest, Reason
from .params import (ENVELOPE_PLAINTEXT_BYTES, ENVELOPE_WIRE_BYTES,
                     GCM_TAG_BYTES, HMAC_BYTES, KEY_ID_BYTES, NONCE_BYTES)

VERSION = 1
_HEADER = struct.Struct("<B8s16sd")   # ver, key_id, nonce, timestamp
assert _HEADER.size == 1 + KEY_ID_BYTES + NONCE_BYTES + 8


class EnvelopeError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------- instrumentation --

@contextlib.contextmanager
def count_aead_operations():
    """Count *envelope* AEAD operations performed inside the block.

    The cheap-rejection ordering claim of the replay experiment -- an envelope failing
    the outer HMAC is rejected without an AEAD decryption -- has to be checked
    by counting the primitive, not by reading a stage timer.  ``_Timer``
    pre-initialises every stage to zero, so ``timings["aead"] == 0.0`` is also
    what you get when the stage ran but never reached its ``mark``; the
    assertion cannot fail for the reason it claims to guard.

    Instrumentation lives here rather than on the admission path so the E-AP
    pays nothing for it outside the ``with`` block.  Only this module's AEAD is
    wrapped -- the admit-token AEAD in :mod:`csa6g.datastore` is a separate
    binding and is deliberately not counted.
    """
    global AESGCM
    real, counts = AESGCM, {"encrypt": 0, "decrypt": 0}

    class _Counting:
        __slots__ = ("_inner",)

        def __init__(self, key):
            self._inner = real(key)

        def encrypt(self, *a, **kw):
            counts["encrypt"] += 1
            return self._inner.encrypt(*a, **kw)

        def decrypt(self, *a, **kw):
            counts["decrypt"] += 1
            return self._inner.decrypt(*a, **kw)

    AESGCM = _Counting
    try:
        yield counts
    finally:
        AESGCM = real


# ------------------------------------------------------------- padding ------

def pad_fixed(msg: bytes, size: int = ENVELOPE_PLAINTEXT_BYTES) -> bytes:
    """PKCS#7-style padding to a single fixed-size block.

    Unlike block-wise PKCS#7 this pads to an absolute length, so the plaintext
    -- and hence the ciphertext -- is the same size for every slice type.  That
    is the property experiment 5 tests.

    The length field itself occupies two bytes, so the largest representable
    message is ``size - 2``: at ``size - 1`` there is no room left to record a
    pad length of one, and the naive bound ``size - 1`` silently emits
    ``size + 1`` bytes instead of raising.
    """
    if len(msg) > size - 2:
        raise EnvelopeError("plaintext exceeds fixed envelope size")
    pad_len = size - len(msg)
    # Store the pad length in two bytes so pad_len up to 65535 is expressible.
    return msg + b"\x00" * (pad_len - 2) + struct.pack("<H", pad_len)


def unpad_fixed(blob: bytes) -> bytes:
    if len(blob) < 2:
        raise EnvelopeError(Reason.INTEGRITY)
    (pad_len,) = struct.unpack("<H", blob[-2:])
    if not (2 <= pad_len <= len(blob)):
        raise EnvelopeError(Reason.INTEGRITY)
    return blob[: len(blob) - pad_len]


# ------------------------------------------------------------ envelope ------

@dataclass(frozen=True)
class Envelope:
    """Parsed view of E_u.  ``raw`` is the exact bytes seen on the wire."""
    version: int
    key_id: bytes
    nonce: bytes
    timestamp: float
    ciphertext: bytes
    gcm_tag: bytes
    outer_tag: bytes
    raw: bytes

    @property
    def size(self) -> int:
        return len(self.raw)


def _outer_mac(kmac: bytes, header: bytes, ct: bytes, tag: bytes) -> bytes:
    h = hmac.HMAC(bytes(kmac), hashes.SHA256())
    h.update(header)
    h.update(ct)
    h.update(tag)
    return h.finalize()


def seal(sk: SessionKeys, req: AdmissionRequest, *,
         nonce: bytes | None = None, timestamp: float | None = None,
         aad: bytes = b"") -> Envelope:
    """Build E_u = C_u || T_gcm || T_u || n_u || tau_u  (envelope definition).

    The freshness token is allocated by :meth:`SessionKeys.next_nonce` --
    key_epoch || encryption_sequence || random -- so the AEAD IV (its first
    12 bytes) is structurally unique per key.  The ``nonce`` override exists
    only so the adversarial experiments and tests can construct replayed or
    malformed tokens; production sealing must never pass it.
    """
    sk.assert_live()
    if nonce is None:
        # Nonce and the keys it belongs to come from one critical section, so
        # a concurrent key rotation cannot split them (see SessionKeys.reserve).
        nonce, k_slice, k_mac = sk.reserve()
    else:
        k_slice, k_mac = bytes(sk.kslice), bytes(sk.kmac)
    if len(nonce) != NONCE_BYTES:
        raise EnvelopeError("freshness token must be 16 bytes")
    ts = time.time() if timestamp is None else timestamp

    plaintext = pad_fixed(req.serialise())
    iv = iv_from_nonce(nonce)
    blob = AESGCM(k_slice).encrypt(iv, plaintext, aad)
    ct, gcm_tag = blob[:-GCM_TAG_BYTES], blob[-GCM_TAG_BYTES:]

    header = _HEADER.pack(VERSION, sk.key_id, nonce, ts)
    outer = _outer_mac(k_mac, header, ct, gcm_tag)
    raw = header + ct + gcm_tag + outer
    assert len(raw) == ENVELOPE_WIRE_BYTES, len(raw)
    return Envelope(VERSION, sk.key_id, nonce, ts, ct, gcm_tag, outer, raw)


def parse(raw: bytes) -> Envelope:
    if len(raw) != ENVELOPE_WIRE_BYTES:
        raise EnvelopeError(Reason.INTEGRITY)
    ver, key_id, nonce, ts = _HEADER.unpack(raw[: _HEADER.size])
    if ver != VERSION:
        raise EnvelopeError(Reason.INTEGRITY)
    off = _HEADER.size
    ct = raw[off: off + ENVELOPE_PLAINTEXT_BYTES]
    off += ENVELOPE_PLAINTEXT_BYTES
    gcm_tag = raw[off: off + GCM_TAG_BYTES]
    off += GCM_TAG_BYTES
    outer = raw[off: off + HMAC_BYTES]
    return Envelope(ver, key_id, nonce, ts, ct, gcm_tag, outer, raw)


def verify_outer(sk: SessionKeys, env: Envelope) -> bool:
    """Constant-time check of the outer HMAC.  No AEAD work is done here."""
    header = env.raw[: _HEADER.size]
    expect = _outer_mac(bytes(sk.kmac), header, env.ciphertext, env.gcm_tag)
    return _hmac.compare_digest(expect, env.outer_tag)


def burn_outer_mac(kmac: bytes, env: Envelope) -> None:
    """Compute and discard an outer MAC over ``env``.

    Timing equalisation for the unknown-key-handle path.  Without it a
    rejection for an unresolvable ``key_id`` returns one HMAC earlier than a
    rejection for a bad MAC, and since the two share an external reason code
    the response time is the only thing separating them: an observer holding a
    captured envelope could probe whether its session is still live -- the
    same class of oracle, on the key table instead of the nonce cache, that
    the integrity-first ordering exists to close.  The E-AP burns this MAC
    under a decoy key so both rejections cost the same work.

    The real check (:func:`verify_outer`) also runs a constant-time
    ``compare_digest``, so this mirrors it against the envelope's own tag --
    the comparison can never succeed under the decoy key and its result is
    discarded; it exists only so the decoy path and the real path spend the
    same work down to the comparison.
    """
    burned = _outer_mac(kmac, env.raw[: _HEADER.size], env.ciphertext,
                        env.gcm_tag)
    _hmac.compare_digest(burned, env.outer_tag)


def open_envelope(sk: SessionKeys, env: Envelope, aad: bytes = b"",
                  tenant_id: int = -1) -> AdmissionRequest:
    """Decrypt and authenticate.  Raises :class:`EnvelopeError` on failure.

    Authenticity does not imply well-formedness.  The AEAD proves the
    plaintext was sealed by a holder of ``K_slice``; it says nothing about
    whether that holder built a *valid* request.  A buggy or hostile client
    with legitimate keys can seal a plaintext whose pad length consumes the
    whole block, or whose slice-type octet is not a TS 23.501 codepoint, and
    the decode then raises ``struct.error`` or ``ValueError`` -- neither of
    which is an :class:`EnvelopeError`.  Left unconverted those propagated out
    of ``EdgeAdmissionProxy.handle`` as an unhandled exception: no decision, no
    denial recorded, no reason code, and one dead request handler per attempt
    for any tenant that had completed registration.  Every decode failure is a
    failure to produce an admissible request, which is what
    :data:`Reason.INTEGRITY` means, so it is reported as one.
    """
    sk.assert_live()
    iv = iv_from_nonce(env.nonce)
    try:
        padded = AESGCM(bytes(sk.kslice)).decrypt(
            iv, env.ciphertext + env.gcm_tag, aad)
    except InvalidTag:
        raise EnvelopeError(Reason.INTEGRITY)
    try:
        return AdmissionRequest.deserialise(unpad_fixed(padded), tenant_id)
    except (struct.error, ValueError) as exc:
        raise EnvelopeError(Reason.INTEGRITY) from exc


# ------------------------------------------- conventional AEAD-with-AAD ------
# Comparison variant for the side-by-side envelope experiment (exp8).  The
# header h_u = ver || kid || n_u || tau_u is carried in the clear and bound as
# AES-GCM associated data; there is no outer HMAC and no K_mac.  Same padded
# 256-byte payload, same structured IV, so the only difference from the
# dual-tag envelope is the integrity mechanism:
#
#     E_u = h_u || C_u || T_gcm        33 + 256 + 16 = 305 bytes
#
# Integrity is checked by the AEAD open itself, which also decrypts, so the
# receiver's integrity stage and decryption stage are one operation.

AAD_ENVELOPE_WIRE_BYTES = (_HEADER.size + ENVELOPE_PLAINTEXT_BYTES
                           + GCM_TAG_BYTES)


def seal_aad(sk: SessionKeys, req: AdmissionRequest, *,
             nonce: bytes | None = None,
             timestamp: float | None = None) -> Envelope:
    """Build the conventional envelope h_u || AEAD(K_slice, IV, m, aad=h_u)."""
    sk.assert_live()
    if nonce is None:
        nonce, k_slice, _ = sk.reserve()
    else:
        k_slice = bytes(sk.kslice)
    if len(nonce) != NONCE_BYTES:
        raise EnvelopeError("freshness token must be 16 bytes")
    ts = time.time() if timestamp is None else timestamp
    header = _HEADER.pack(VERSION, sk.key_id, nonce, ts)
    blob = AESGCM(k_slice).encrypt(iv_from_nonce(nonce),
                                   pad_fixed(req.serialise()), header)
    ct, gcm_tag = blob[:-GCM_TAG_BYTES], blob[-GCM_TAG_BYTES:]
    raw = header + ct + gcm_tag
    assert len(raw) == AAD_ENVELOPE_WIRE_BYTES, len(raw)
    return Envelope(VERSION, sk.key_id, nonce, ts, ct, gcm_tag, b"", raw)


def parse_aad(raw: bytes) -> Envelope:
    if len(raw) != AAD_ENVELOPE_WIRE_BYTES:
        raise EnvelopeError(Reason.INTEGRITY)
    ver, key_id, nonce, ts = _HEADER.unpack(raw[: _HEADER.size])
    if ver != VERSION:
        raise EnvelopeError(Reason.INTEGRITY)
    off = _HEADER.size
    ct = raw[off: off + ENVELOPE_PLAINTEXT_BYTES]
    gcm_tag = raw[off + ENVELOPE_PLAINTEXT_BYTES:]
    return Envelope(ver, key_id, nonce, ts, ct, gcm_tag, b"", raw)


def open_aad_padded(k_slice: bytes, env: Envelope) -> bytes:
    """Authenticate header+ciphertext and decrypt in one AEAD call.

    Returns the padded plaintext; raises EnvelopeError(INTEGRITY) on a bad
    tag.  Used both for live sessions and, under a decoy key, to equalise the
    unknown-handle path (whose result is discarded).
    """
    try:
        return AESGCM(bytes(k_slice)).decrypt(
            iv_from_nonce(env.nonce), env.ciphertext + env.gcm_tag,
            env.raw[: _HEADER.size])
    except InvalidTag:
        raise EnvelopeError(Reason.INTEGRITY)


def decode_padded(padded: bytes, tenant_id: int = -1) -> AdmissionRequest:
    try:
        return AdmissionRequest.deserialise(unpad_fixed(padded), tenant_id)
    except (struct.error, ValueError) as exc:
        raise EnvelopeError(Reason.INTEGRITY) from exc


# ------------------------------------------------- adversarial mutations ----
# Used by experiment 3 to construct the six attack classes.

def tamper_ciphertext(env: Envelope, byte_index: int = 0) -> Envelope:
    raw = bytearray(env.raw)
    off = _HEADER.size + byte_index
    raw[off] ^= 0x01
    return parse(bytes(raw))


def tamper_outer_tag(env: Envelope) -> Envelope:
    raw = bytearray(env.raw)
    raw[-1] ^= 0x01
    return parse(bytes(raw))


def retime(env: Envelope, new_ts: float) -> bytes:
    """Rewrite tau_u without re-MACing -- the naive replay-with-fresh-timestamp
    attack.  Returns raw bytes because the envelope is now inconsistent."""
    raw = bytearray(env.raw)
    struct.pack_into("<d", raw, 1 + KEY_ID_BYTES + NONCE_BYTES, new_ts)
    return bytes(raw)
