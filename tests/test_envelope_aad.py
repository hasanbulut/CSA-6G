"""The conventional AEAD-with-AAD envelope used by experiment 8.

These pin that the comparison variant is a real, working alternative: it
admits valid requests, rejects tampering of every header field,
replays and context mismatches, keeps the integrity-before-freshness order,
and spends the same work on the unknown-handle path as on a bad tag.
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csa6g import crypto
from csa6g.eap import EdgeAdmissionProxy
from csa6g.harness import make_pool
from csa6g.keys import run_anchor_bootstrap
from csa6g.model import Reason
from csa6g.workload import benign_stream


def _setup(mode="aad", integrity_first=True, seed=0):
    rng = np.random.default_rng(seed)
    eap = EdgeAdmissionProxy(make_pool(24), integrity_first=integrity_first,
                             envelope_mode=mode)
    reqs = benign_stream(seed, n_tenants=1, n_requests=3)
    sk = eap.register_session(run_anchor_bootstrap(rng), reqs[0].slice_type,
                              reqs[0].service_id, reqs[0].tenant_id)
    return eap, sk, reqs


def test_aad_envelope_is_305_bytes_and_dual_is_337(rng=None):
    eap, sk, reqs = _setup()
    assert len(crypto.seal_aad(sk, reqs[0]).raw) == 305
    assert crypto.AAD_ENVELOPE_WIRE_BYTES == 305
    assert len(crypto.seal(sk, reqs[1]).raw) == 337


def test_aad_mode_admits_a_valid_request():
    eap, sk, reqs = _setup()
    now = time.time()
    out = eap.handle(crypto.seal_aad(sk, reqs[0], timestamp=now).raw, now=now)
    assert out.decision.admitted, out.internal_reason


@pytest.mark.parametrize("offset", [0, 1, 9, 25, 40, -1])
def test_aad_mode_rejects_tampering_anywhere(offset):
    """Header (version, handle, nonce, timestamp), ciphertext and tag are all
    covered: flipping any byte is rejected.  A flipped version byte or key
    handle is rejected before the AEAD; everything else by the AEAD."""
    eap, sk, reqs = _setup()
    now = time.time()
    raw = bytearray(crypto.seal_aad(sk, reqs[0], timestamp=now).raw)
    raw[offset] ^= 0x01
    out = eap.handle(bytes(raw), now=now)
    assert not out.decision.admitted
    assert out.external_reason == Reason.externalise(Reason.INTEGRITY)


def test_aad_mode_rejects_replay_and_rollback():
    eap, sk, reqs = _setup()
    now = time.time()
    env = crypto.seal_aad(sk, reqs[0], timestamp=now).raw
    assert eap.handle(env, now=now).decision.admitted
    again = eap.handle(env, now=now)
    assert not again.decision.admitted
    assert again.internal_reason in (Reason.NONCE_REPLAY,
                                     Reason.SEQUENCE_ROLLBACK)


def test_aad_mode_rejects_context_mismatch():
    import dataclasses
    from csa6g.model import SliceType
    eap, sk, reqs = _setup()
    now = time.time()
    other = next(t for t in SliceType if t != reqs[0].slice_type)
    bad = dataclasses.replace(reqs[0], slice_type=other)
    out = eap.handle(crypto.seal_aad(sk, bad, timestamp=now).raw, now=now)
    assert out.internal_reason == Reason.CONTEXT_MISMATCH


def test_aad_mode_bad_tag_never_touches_the_replay_window():
    """Integrity-first holds in AAD mode: a forged envelope carrying a cached
    nonce is rejected as integrity_fail, not nonce_replay."""
    eap, sk, reqs = _setup()
    now = time.time()
    env = crypto.seal_aad(sk, reqs[0], timestamp=now).raw
    assert eap.handle(env, now=now).decision.admitted
    forged = bytearray(env)
    forged[-1] ^= 0xFF
    out = eap.handle(bytes(forged), now=now)
    assert out.internal_reason == Reason.INTEGRITY


def test_aad_mode_freshness_first_consults_the_cache_before_the_aead():
    eap, sk, reqs = _setup(integrity_first=False)
    now = time.time()
    env = crypto.seal_aad(sk, reqs[0], timestamp=now).raw
    assert eap.handle(env, now=now).decision.admitted
    forged = bytearray(env)
    forged[-1] ^= 0xFF
    with crypto.count_aead_operations() as n:
        out = eap.handle(bytes(forged), now=now)
    assert out.internal_reason == Reason.NONCE_REPLAY
    assert n["decrypt"] == 0


def test_aad_mode_unknown_handle_does_one_decoy_aead():
    eap, sk, reqs = _setup()
    now = time.time()
    raw = bytearray(crypto.seal_aad(sk, reqs[0], timestamp=now).raw)
    raw[1:9] = b"\x00" * 8
    with crypto.count_aead_operations() as n:
        out = eap.handle(bytes(raw), now=now)
    assert out.internal_reason == Reason.AUTH_FAIL
    assert n["decrypt"] == 1


def test_dual_mode_is_unchanged_default():
    eap, sk, reqs = _setup(mode="dual")
    now = time.time()
    assert eap.handle(crypto.seal(sk, reqs[0], timestamp=now).raw,
                      now=now).decision.admitted
    with pytest.raises(ValueError):
        EdgeAdmissionProxy(make_pool(2), envelope_mode="nope")
