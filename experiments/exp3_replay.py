"""Experiment 3 -- replay, freshness and tampering (the replay experiment).

Eight deterministic adversarial classes, each repeated n times against a live
E-AP.  For every trial we record (a) whether the outcome was correct, (b) the
internal reason code, (c) the external reason the requester would see, and
(d) how far down the pipeline the request got before rejection.

Three things are checked:

1.  **Correctness.** Every adversarial request is rejected with the right
    internal code; every valid control is admitted.  0 false positives, 0
    false negatives.
2.  **Minimal disclosure (Step 5).** Every denial externalises to the single
    coarse category ``not_authorized``.  An impostor and a quota-exhausted
    legitimate tenant are indistinguishable from outside the network.
3.  **Cheap-rejection ordering.** Envelopes that fail the outer HMAC are
    rejected *without* an AEAD decryption, which is the stated reason for
    carrying two tags.  This is checked by counting AES-GCM invocations, not by
    reading the stage timer: ``_Timer`` pre-initialises every stage to zero, so
    ``timings["aead"] == 0.0`` is equally what you get from a stage that ran
    but never reached its ``mark``, and the assertion could not fail for the
    reason it claimed to guard.  ``valid_control`` carries the positive control
    -- exactly one decryption -- so the counter is shown to work.

This is functional verification of the tested classes.  A Dolev-Yao /
ProVerif model is future work.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g import crypto
from csa6g.eap import EdgeAdmissionProxy, replay_key
from csa6g.harness import make_pool
from csa6g.keys import run_anchor_bootstrap
from csa6g.model import AdmissionRequest, Reason, SliceType
from csa6g.params import CLOCK_TOLERANCE_S, DEFAULT_EXPERIMENT, QUICK_EXPERIMENT
from csa6g.workload import benign_stream

CLASSES = {
    "expired_token":    Reason.TOKEN_EXPIRED,
    "nonce_replay":     Reason.NONCE_REPLAY,
    "evicted_replay":   Reason.SEQUENCE_ROLLBACK,
    "timestamp_replay": Reason.STALE,
    "counter_rollback": Reason.COUNTER_ROLLBACK,
    "tampered_hmac":    Reason.INTEGRITY,
    "context_mismatch": Reason.CONTEXT_MISMATCH,
    "valid_control":    Reason.OK,
}


def _fresh_eap(seed: int):
    rng = np.random.default_rng(seed)
    stream = benign_stream(seed, n_tenants=1, n_requests=4)
    pool = make_pool()
    eap = EdgeAdmissionProxy(pool)
    req = stream[0]
    sk = eap.register_session(run_anchor_bootstrap(rng),
                              req.slice_type, req.service_id, req.tenant_id)
    return eap, sk, req, pool


def _mk(req: AdmissionRequest, counter: int) -> AdmissionRequest:
    return AdmissionRequest(
        tenant_id=req.tenant_id, pseudonym=req.pseudonym,
        slice_type=req.slice_type, service_id=req.service_id,
        sla_latency_ms=req.sla_latency_ms, demand=req.demand,
        arrival_index=req.arrival_index, counter=counter)


def trial(cls: str, seed: int) -> dict:
    eap, sk, req, pool = _fresh_eap(seed)
    now = time.time()

    # Every class needs one legitimate request first, so that session state
    # (nonce cache, counter, token) exists to attack.
    good = crypto.seal(sk, _mk(req, 1), timestamp=now)
    first = eap.handle(good.raw, now=now, tenant_id=req.tenant_id,
                       run_secondary_auth=True)
    assert first.decision.admitted, first.internal_reason

    if cls == "valid_control":
        env = crypto.seal(sk, _mk(req, 2), timestamp=now)
        with crypto.count_aead_operations() as ops:
            out = eap.handle(env.raw, now=now, tenant_id=req.tenant_id,
                             run_secondary_auth=True)
        return dict(cls=cls, correct=out.decision.admitted,
                    internal=out.internal_reason, external=out.external_reason,
                    aead_cost=out.decision.timings["aead"],
                    aead_opens=ops["decrypt"])

    if cls == "expired_token":
        # The admit token from the first request, presented to the real Step 6
        # release path after its TTL.  Calling ``open_admit_token`` directly --
        # as this class used to -- exercises the helper but never the method an
        # attacker would actually reach, and made the trial depend on the
        # TokenStore TTL and the sealed-token TTL being separately configured
        # to the same value.
        expired_at = now + eap.tokens.ttl + 1.0
        with crypto.count_aead_operations() as ops:
            rel = eap.release(first.session_id, first.token, now=expired_at)
        # ``external`` is now read off the proxy's own outcome rather than
        # recomputed here: the minimal-disclosure mapping on the release path
        # is enforced by the E-AP, and this experiment measures it as shipped.
        return dict(cls=cls, correct=(rel.internal_reason
                                      == Reason.TOKEN_EXPIRED),
                    internal=rel.internal_reason,
                    external=rel.external_reason, aead_cost=0.0,
                    aead_opens=ops["decrypt"])

    if cls == "evicted_replay":
        # The bounded nonce cache has evicted the first request's nonce, so
        # the pure replay-window check would pass an exact replay whose
        # timestamp is still inside the tolerance.  This is the attack the
        # structured freshness token closes: the replayed envelope carries an
        # encryption sequence at or below the session's high-water mark, so
        # the sequence tracker rejects it (and the IV can never be reused
        # against the AEAD, because the request dies before decryption).
        from csa6g.freshness import ReplayWindow
        eap2, sk2, req2, _ = _fresh_eap(seed)
        eap2.replay = ReplayWindow(capacity=1)
        first_env = crypto.seal(sk2, _mk(req2, 1), timestamp=now)
        out1 = eap2.handle(first_env.raw, now=now, tenant_id=req2.tenant_id)
        assert out1.decision.admitted, out1.internal_reason
        eap2.release_slice(out1.session_id)
        second = crypto.seal(sk2, _mk(req2, 2), timestamp=now + 0.5)
        out2 = eap2.handle(second.raw, now=now + 0.5,
                           tenant_id=req2.tenant_id)
        assert out2.decision.admitted, out2.internal_reason
        # The composite key the E-AP writes (key_id || n_u), not the bare
        # nonce: a bare nonce is never a cache key, so asserting on it would
        # hold vacuously and stop testing the eviction this class depends on.
        evicted = replay_key(sk2.key_id, first_env.nonce)
        assert evicted not in eap2.replay._seen, \
            "eviction precondition not met"
        with crypto.count_aead_operations() as ops:
            out = eap2.handle(first_env.raw, now=now + 1.0,
                              tenant_id=req2.tenant_id)
        return dict(cls=cls,
                    correct=(not out.decision.admitted
                             and out.internal_reason
                             == Reason.SEQUENCE_ROLLBACK),
                    internal=out.internal_reason,
                    external=out.external_reason,
                    aead_cost=out.decision.timings["aead"],
                    aead_opens=ops["decrypt"])

    if cls == "nonce_replay":
        attack = (good.raw, now + 1.0)

    elif cls == "timestamp_replay":
        env = crypto.seal(sk, _mk(req, 2),
                          timestamp=now - CLOCK_TOLERANCE_S - 5.0)
        attack = (env.raw, now)

    elif cls == "counter_rollback":
        # Fresh nonce, in-tolerance timestamp, but a counter already consumed.
        env = crypto.seal(sk, _mk(req, 1), timestamp=now + 0.5)
        attack = (env.raw, now + 0.5)

    elif cls == "tampered_hmac":
        env = crypto.tamper_outer_tag(crypto.seal(sk, _mk(req, 2),
                                                  timestamp=now))
        attack = (env.raw, now)

    elif cls == "context_mismatch":
        other_slice = (SliceType.EMBB if req.slice_type != SliceType.EMBB
                       else SliceType.URLLC)
        mismatched = AdmissionRequest(
            tenant_id=req.tenant_id, pseudonym=req.pseudonym,
            slice_type=other_slice, service_id=req.service_id + 1,
            sla_latency_ms=req.sla_latency_ms, demand=req.demand,
            arrival_index=req.arrival_index, counter=2)
        env = crypto.seal(sk, mismatched, timestamp=now)
        attack = (env.raw, now)

    else:
        raise ValueError(cls)

    raw, at = attack
    with crypto.count_aead_operations() as ops:
        out = eap.handle(raw, now=at, tenant_id=req.tenant_id)

    expected = CLASSES[cls]
    return dict(cls=cls,
                correct=(not out.decision.admitted
                         and out.internal_reason == expected),
                internal=out.internal_reason, external=out.external_reason,
                aead_cost=out.decision.timings["aead"],
                aead_opens=ops["decrypt"])


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    banner(f"Experiment 3: replay / freshness / tampering, "
           f"{P.replay_trials} trials x {len(CLASSES)} classes")

    records = [trial(cls, seed)
               for cls in CLASSES
               for seed in range(P.replay_trials)]
    df = pd.DataFrame(records)

    summary = (df.groupby("cls")
                 .agg(trials=("correct", "size"),
                      correct=("correct", "sum"),
                      accuracy=("correct", "mean"),
                      expected=("cls", lambda s: CLASSES[s.iloc[0]]),
                      observed=("internal", lambda s: s.mode().iloc[0]),
                      external=("external", lambda s: s.mode().iloc[0]),
                      aead_opens=("aead_opens", "sum"))
                 .reset_index())
    save_table(df, "exp3_replay_raw")
    save_table(summary, "exp3_replay_summary")
    print(summary.to_string(index=False))

    # -- assertion 2: minimal disclosure ----------------------------------
    denials = df[df["cls"] != "valid_control"]
    leaks = sorted(set(denials["external"]) - {Reason.NOT_AUTHORIZED,
                                               Reason.TRANSIENT})
    print(f"\nExternal reason codes on denial: {sorted(set(denials['external']))}")
    assert not leaks, f"internal reason leaked to requester: {leaks}"

    # -- assertion 3: cheap-rejection ordering ----------------------------
    # Counted AES-GCM invocations, not the stage timer. The positive control
    # comes first: if valid_control does not show exactly one decryption the
    # counter is not wired up and the zero below would mean nothing.
    control = df[df["cls"] == "valid_control"]
    assert (control["aead_opens"] == 1).all(), \
        ("the AEAD counter is not observing the admission path; the "
         "zero-decryption assertion below would be vacuous")
    hmac_fail = df[df["cls"] == "tampered_hmac"]
    assert (hmac_fail["aead_opens"] == 0).all(), \
        "AEAD ran on an envelope that failed the outer HMAC"
    print(f"Outer-HMAC failures: {int(hmac_fail['aead_opens'].sum())} AEAD "
          f"decryptions over {len(hmac_fail)} trials "
          f"(admitted control: {int(control['aead_opens'].sum())}/"
          f"{len(control)}) -- cheap-rejection ordering confirmed")

    overall = float(df["correct"].mean())
    print(f"\nOverall correct-outcome rate: {overall:.4f} "
          f"({int(df['correct'].sum())}/{len(df)})")
    assert overall == 1.0, "at least one class did not behave as specified"

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.bar(summary["cls"], summary["accuracy"] * 100, color="#3b6ea5",
           width=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Correct outcome (%)")
    ax.set_title("Adversarial class handling (n = "
                 f"{P.replay_trials} per class)")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "fig5_replay_classes")

    save_json({"summary": summary.to_dict("records"),
               "overall_accuracy": overall,
               "minimal_disclosure_holds": True,
               "cheap_rejection_holds": True,
               "cheap_rejection_evidence": {
                   "measure": "counted AES-GCM decrypt invocations on the "
                              "envelope key, not the stage timer",
                   "aead_opens_on_hmac_failure": int(
                       hmac_fail["aead_opens"].sum()),
                   "aead_opens_on_admitted_control": int(
                       control["aead_opens"].sum()),
                   "trials_per_class": int(P.replay_trials)},
               "caveat": "functional verification of the tested classes; "
                         "not a formal protocol proof"}, "exp3_replay")


if __name__ == "__main__":
    main()
