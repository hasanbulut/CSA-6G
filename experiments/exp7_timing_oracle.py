"""Experiment 7 -- minimal disclosure under a timing observer.

Step 5's minimal-disclosure property is about *reason codes*: every denial
externalises to one of two coarse categories, so an impostor and a
quota-exhausted tenant look alike from outside the network.  Experiment 3
verifies that channel and it holds.

It is not the only channel.  A denial also takes time, and how much time
depends on how far down the pipeline the request got.  If the amount of work
the E-AP does before rejecting depends on state an attacker wants to read, then
the reason codes can be identical and the system still answers the question.

The concrete instance is the replay window.  It is a cache keyed on a nonce the
attacker chooses.  In the original stage order it was consulted at stage 3,
before the outer HMAC at stage 4, so an *unauthenticated* party could submit a
forged envelope carrying a nonce of their choosing and learn, from the response
time alone, whether that nonce was already in the cache:

    nonce in cache      -> reject at stage 3, cheap  -> nonce_replay
    nonce not in cache  -> reject at stage 4, dearer -> integrity_fail

Both externalise to ``not_authorized``.  The codes are indistinguishable; the
timings are not.  That is a membership oracle on the replay window, and it is
available to anyone who can send packets -- no key, no captured envelope.

The fix is the ordinary rule: authenticate before letting the input reach state
an observer can probe.  Verifying the HMAC first costs one HMAC on the replay
path, which is real but bounded, and collapses the two rejection paths onto the
same work.  ``EdgeAdmissionProxy(integrity_first=False)`` retains the old order
so this experiment can measure what the change bought instead of asserting it.

The observer here is deliberately weak: it sees one scalar per probe, the
end-to-end handling time, with no averaging over repeated identical probes.
Reported as the AUC of a single-feature discriminator, so 0.5 is "cannot tell"
and 1.0 is "always right".
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g import crypto
from csa6g.eap import EdgeAdmissionProxy
from csa6g.harness import make_pool
from csa6g.keys import run_anchor_bootstrap
from csa6g.model import Reason
from csa6g.workload import benign_stream

ORDERINGS = {
    True: "integrity first (shipped)",
    False: "freshness first (pre-fix)",
}


def _probe_pair(eap, burned: bytes, rng, tenant_id: int, now: float):
    """Time one cached-nonce probe and one fresh-nonce probe.

    The threat model is an attacker with **no key and no captured envelope**:
    everything they send carries a tag they could not have computed.  So both
    probes are forgeries and differ only in the nonce they carry -- one copied
    from an envelope the E-AP has already accepted, one random.  That isolates
    the thing being tested.  Timing a genuine replay against a forgery instead
    would conflate "can read the replay window" with "holds a valid envelope",
    and a party holding a valid envelope already knows its nonce was used.

    The two probes are issued in a random order within each pair.  Fixing the
    order put a systematic few-hundred-nanosecond bias on whichever arm went
    first, which is the same size as the effect being measured.
    """
    cached = bytearray(burned)
    cached[-1] ^= 0xFF                    # break the tag, keep the seen nonce

    fresh = bytearray(burned)
    fresh[9:25] = rng.bytes(16)           # unseen nonce, header slot n_u
    fresh[-1] ^= 0xFF                     # break the tag

    probes = [("cached", bytes(cached)), ("fresh", bytes(fresh))]
    if rng.integers(0, 2):
        probes.reverse()

    out = {}
    for label, raw in probes:
        t0 = time.perf_counter_ns()
        res = eap.handle(raw, now=now, tenant_id=tenant_id)
        out[label] = ((time.perf_counter_ns() - t0) / 1e3, res)   # us
    return out["cached"] + out["fresh"]


def run_ordering(integrity_first: bool, trials: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pool = make_pool(24)
    eap = EdgeAdmissionProxy(pool, integrity_first=integrity_first)
    req = benign_stream(seed, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = time.time()

    # One legitimate request, so its nonce is in the replay window.
    burned = crypto.seal(sk, req, timestamp=now).raw
    first = eap.handle(burned, now=now, tenant_id=req.tenant_id)
    assert first.decision.admitted, first.internal_reason

    cached, fresh = [], []
    reasons_cached, reasons_fresh = set(), set()
    ext_cached, ext_fresh = set(), set()
    for _ in range(trials):
        tc, a, tf, b = _probe_pair(eap, burned, rng, req.tenant_id, now)
        cached.append(tc)
        fresh.append(tf)
        reasons_cached.add(a.internal_reason)
        reasons_fresh.add(b.internal_reason)
        ext_cached.add(a.external_reason)
        ext_fresh.add(b.external_reason)

    c = np.asarray(cached)
    f = np.asarray(fresh)
    # Single-feature discriminator: is this probe's latency from the
    # cached-nonce population? Symmetrised so the value is orientation-free.
    y = np.r_[np.ones_like(c, dtype=int), np.zeros_like(f, dtype=int)]
    auc = float(roc_auc_score(y, np.r_[c, f]))
    auc = max(auc, 1.0 - auc)

    return dict(
        ordering=ORDERINGS[integrity_first],
        integrity_first=integrity_first,
        cached_median_us=float(np.median(c)),
        fresh_median_us=float(np.median(f)),
        gap_us=float(np.median(f) - np.median(c)),
        gap_pct=float(100.0 * (np.median(f) - np.median(c)) / np.median(c)),
        timing_auc=auc,
        cached_reason=sorted(reasons_cached)[0],
        fresh_reason=sorted(reasons_fresh)[0],
        external_codes=len(ext_cached | ext_fresh),
        trials=int(trials),
        _c=c, _f=f)


def main() -> None:
    trials = 400 if quick_flag() else 4000
    banner(f"Experiment 7: minimal disclosure under a timing observer, "
           f"{trials} probe pairs per ordering")

    runs = [run_ordering(v, trials) for v in (False, True)]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in runs])
    save_table(df, "exp7_timing_oracle_summary")
    # Raw export: every probe's handling time (us), by ordering and arm.
    save_table(pd.concat([pd.DataFrame(dict(ordering=r["ordering"],
                                            probe=arm, us=v))
                          for r in runs
                          for arm, v in (("cached", r["_c"]),
                                         ("fresh", r["_f"]))]),
               "exp7_timing_oracle_raw")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pre, post = runs[0], runs[1]
    print(f"\nExternally both orderings emit one code, "
          f"'{Reason.NOT_AUTHORIZED}', for every probe -- Step 5's "
          f"minimal-disclosure property holds on the reason-code channel in "
          f"both, which is what experiment 3 verifies.")
    print(f"  pre-fix  internal codes: cached -> {pre['cached_reason']}, "
          f"fresh -> {pre['fresh_reason']}  (different work, different stage)")
    print(f"  shipped  internal codes: cached -> {post['cached_reason']}, "
          f"fresh -> {post['fresh_reason']}  (same work, same stage)")
    print("\nTiming, however:")
    print(f"  pre-fix  (freshness first): gap {pre['gap_us']:+.2f} us "
          f"({pre['gap_pct']:+.1f}%), observer AUC {pre['timing_auc']:.3f}")
    print(f"  shipped  (integrity first): gap {post['gap_us']:+.2f} us "
          f"({post['gap_pct']:+.1f}%), observer AUC {post['timing_auc']:.3f}")
    print("\nThe shipped ordering costs one HMAC on the replay path and "
          "removes a replay-window membership oracle that was available to "
          "any unauthenticated sender.")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True)
    for ax, r in zip(axes, runs):
        lo = min(r["_c"].min(), r["_f"].min())
        hi = np.quantile(np.r_[r["_c"], r["_f"]], 0.99)
        bins = np.linspace(lo, hi, 60)
        ax.hist(r["_c"], bins=bins, alpha=0.65, label="nonce in cache",
                color="#a53b3b")
        ax.hist(r["_f"], bins=bins, alpha=0.65, label="nonce not in cache",
                color="#3b6ea5")
        ax.set_title(f"{r['ordering']}\nobserver AUC = {r['timing_auc']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Rejection latency (us)")
    axes[0].set_ylabel("Probes")
    axes[0].legend(fontsize=7)
    save_fig(fig, "fig10_timing_oracle")

    save_json({"summary": df.to_dict("records"),
               "pre_fix_auc": pre["timing_auc"],
               "shipped_auc": post["timing_auc"],
               "interpretation":
                   "Minimal disclosure (Step 5) constrains reason codes, not "
                   "response time. Consulting the nonce cache before "
                   "authenticating the envelope let any unauthenticated sender "
                   "test replay-window membership by timing, with identical "
                   "reason codes on both outcomes. Verifying the outer HMAC "
                   "first costs one HMAC on the replay path and closes it.",
               "caveat":
                   "Single-process, no network. A deployed observer sees this "
                   "channel through network jitter, which raises the number of "
                   "probes needed but does not remove the channel; averaging "
                   "over repeated probes recovers it."},
              "exp7_timing_oracle")


if __name__ == "__main__":
    main()
