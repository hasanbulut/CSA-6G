"""Experiment 8 -- dual-tag envelope vs. conventional AEAD-with-AAD.

The CSA-6G envelope carries two tags: AES-256-GCM over the padded
payload and an outer HMAC-SHA256 over header || ciphertext || GCM tag.  The
standard construction binds the cleartext header as AES-GCM associated data and
has no second tag.  Both authenticate the header.  This experiment measures,
on the *same* pipeline, workload, padding and structured IV, what the second
tag costs and buys:

  A. valid admissions  -- client seal cost, server-side pipeline cost, and the
     integrity+decryption stages, per seed (paired across modes);
  B. rejection classes -- end-to-end handling time and envelope AEAD opens for
     an outsider forgery (bad tag), an exact replay of an accepted envelope,
     and an unknown key handle;
  C. the replay-window timing oracle (exp7 design) for each envelope under both
     validation orders;
  D. wire bytes.

Modes alternate order per seed so slow drift on the host cannot favour one.
All raw per-probe observations are exported (``exp8_*_raw.csv``).
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
from csa6g.model import AdmissionRequest, Reason
from csa6g.stats import mean_ci
from csa6g.workload import benign_stream

MODES = ("dual", "aad")
SEAL = {"dual": crypto.seal, "aad": crypto.seal_aad}
SIZE = {"dual": 337, "aad": crypto.AAD_ENVELOPE_WIRE_BYTES}


def _contexts(stream):
    """One slice/service context per tenant, as in exp1."""
    ctx, out = {}, []
    for r in stream:
        ctx.setdefault(r.tenant_id, (r.slice_type, r.service_id))
        st, sv = ctx[r.tenant_id]
        out.append(AdmissionRequest(
            tenant_id=r.tenant_id, pseudonym=r.pseudonym, slice_type=st,
            service_id=sv, sla_latency_ms=r.sla_latency_ms, demand=r.demand,
            arrival_index=r.arrival_index, counter=0, malicious=False))
    return out


def run_seed(mode: str, seed: int, n_requests: int, n_tenants: int,
             n_probes: int):
    rng = np.random.default_rng(seed)
    rng.integers(0, 256, size=32, dtype="uint8")
    stream = _contexts(benign_stream(seed, n_tenants=n_tenants,
                                     n_requests=n_requests))
    eap = EdgeAdmissionProxy(make_pool(), envelope_mode=mode)
    seal = SEAL[mode]
    sessions, counters, accepted = {}, {}, []
    valid_rows = []
    for req in stream:
        t = req.tenant_id
        if t not in sessions:
            sessions[t] = eap.register_session(run_anchor_bootstrap(rng),
                                               req.slice_type, req.service_id,
                                               t)
            counters[t] = 0
        counters[t] += 1
        sreq = AdmissionRequest(
            tenant_id=t, pseudonym=req.pseudonym, slice_type=req.slice_type,
            service_id=req.service_id, sla_latency_ms=req.sla_latency_ms,
            demand=req.demand, arrival_index=req.arrival_index,
            counter=counters[t], malicious=False)
        c0 = time.perf_counter_ns()
        env = seal(sessions[t], sreq)
        seal_us = (time.perf_counter_ns() - c0) / 1e3
        out = eap.handle(env.raw, tenant_id=t)
        if not out.decision.admitted:
            raise RuntimeError(f"{mode}: unexpected denial "
                               f"{out.internal_reason}")
        assert eap.release_slice(out.session_id) == Reason.OK
        tim = out.decision.timings
        valid_rows.append(dict(
            mode=mode, seed=seed, seal_us=seal_us,
            server_us=1e6 * sum(tim.values()),
            integrity_decrypt_us=1e6 * (tim["hmac"] + tim["aead"])))
        accepted.append(env.raw)

    # B -- rejection classes, timed end to end on this proxy.
    now = time.time()
    t_any = stream[0].tenant_id
    probe_rows = []
    classes = ("forged_tag", "exact_replay", "unknown_handle")
    for i in range(n_probes):
        base = accepted[int(rng.integers(0, len(accepted)))]
        forged = bytearray(base)
        forged[9:25] = rng.bytes(16)          # unseen nonce
        forged[-1] ^= 0xFF                    # break the last tag
        unknown = bytearray(base)
        unknown[1:9] = rng.bytes(8)
        payloads = {"forged_tag": bytes(forged), "exact_replay": base,
                    "unknown_handle": bytes(unknown)}
        order = list(classes)
        rng.shuffle(order)
        for cls in order:
            with crypto.count_aead_operations() as n:
                t0 = time.perf_counter_ns()
                res = eap.handle(payloads[cls], now=now, tenant_id=t_any)
                dt = (time.perf_counter_ns() - t0) / 1e3
            assert not res.decision.admitted
            probe_rows.append(dict(mode=mode, seed=seed, cls=cls, us=dt,
                                   aead_opens=n["decrypt"],
                                   internal=res.internal_reason,
                                   external=res.external_reason))
    return valid_rows, probe_rows


def oracle(mode: str, integrity_first: bool, trials: int, seed: int = 0):
    """exp7's cached-vs-fresh nonce probe, for either envelope."""
    rng = np.random.default_rng(seed)
    eap = EdgeAdmissionProxy(make_pool(24), integrity_first=integrity_first,
                             envelope_mode=mode)
    req = benign_stream(seed, n_tenants=1, n_requests=1)[0]
    sk = eap.register_session(run_anchor_bootstrap(rng), req.slice_type,
                              req.service_id, req.tenant_id)
    now = time.time()
    burned = SEAL[mode](sk, req, timestamp=now).raw
    assert eap.handle(burned, now=now).decision.admitted
    rows = []
    for _ in range(trials):
        cached = bytearray(burned)
        cached[-1] ^= 0xFF
        fresh = bytearray(burned)
        fresh[9:25] = rng.bytes(16)
        fresh[-1] ^= 0xFF
        pair = [("cached", bytes(cached)), ("fresh", bytes(fresh))]
        if rng.integers(0, 2):
            pair.reverse()
        for label, raw in pair:
            t0 = time.perf_counter_ns()
            r = eap.handle(raw, now=now, tenant_id=req.tenant_id)
            rows.append(dict(mode=mode, integrity_first=integrity_first,
                             probe=label,
                             us=(time.perf_counter_ns() - t0) / 1e3,
                             internal=r.internal_reason))
    df = pd.DataFrame(rows)
    y = (df.probe == "cached").astype(int)
    auc = float(roc_auc_score(y, df.us))
    return max(auc, 1 - auc), df


def main() -> None:
    quick = quick_flag()
    seeds = 10 if quick else 100
    n_req, n_ten = (50, 5) if quick else (100, 5)
    n_probes = 20 if quick else 100
    trials = 400 if quick else 4000
    banner(f"Experiment 8: dual-tag vs AEAD-with-AAD envelope, {seeds} seeds")

    valid, probes = [], []
    for seed in range(seeds):
        order = MODES if seed % 2 == 0 else MODES[::-1]
        for mode in order:
            v, p = run_seed(mode, seed, n_req, n_ten, n_probes)
            valid += v
            probes += p
    vdf, pdf = pd.DataFrame(valid), pd.DataFrame(probes)
    save_table(vdf, "exp8_valid_raw")
    save_table(pdf, "exp8_rejection_raw")

    # Per-seed means, then paired AAD - dual differences.
    vseed = vdf.groupby(["mode", "seed"]).mean(numeric_only=True)
    pseed = pdf.groupby(["mode", "cls", "seed"]).us.mean()
    summary = []
    for metric in ("seal_us", "server_us", "integrity_decrypt_us"):
        d = vseed.loc["dual", metric].values
        a = vseed.loc["aad", metric].values
        diff = mean_ci(a - d)
        summary.append(dict(
            quantity=f"valid:{metric}", dual_mean=mean_ci(d)[0],
            aad_mean=mean_ci(a)[0], diff_aad_minus_dual=diff[0],
            diff_lo=diff[1], diff_hi=diff[2],
            diff_pct=100 * diff[0] / mean_ci(d)[0]))
    for cls in ("forged_tag", "exact_replay", "unknown_handle"):
        d = pseed.loc["dual", cls].values
        a = pseed.loc["aad", cls].values
        diff = mean_ci(a - d)
        opens = pdf[pdf.cls == cls].groupby("mode").aead_opens.agg(
            ["min", "max"])
        summary.append(dict(
            quantity=f"reject:{cls}", dual_mean=mean_ci(d)[0],
            aad_mean=mean_ci(a)[0], diff_aad_minus_dual=diff[0],
            diff_lo=diff[1], diff_hi=diff[2],
            diff_pct=100 * diff[0] / mean_ci(d)[0],
            dual_aead_opens=int(opens.loc["dual", "max"]),
            aad_aead_opens=int(opens.loc["aad", "max"])))
    ext = pdf.external.nunique()

    orows, oraw = [], []
    for mode in MODES:
        for ifirst in (False, True):
            auc, df = oracle(mode, ifirst, trials)
            oraw.append(df)
            g = df.groupby("probe").us.median()
            orows.append(dict(mode=mode, integrity_first=ifirst,
                              cached_median_us=g["cached"],
                              fresh_median_us=g["fresh"], timing_auc=auc))
    odf = pd.DataFrame(orows)
    save_table(pd.concat(oraw), "exp8_oracle_raw")
    sdf = pd.DataFrame(summary)
    sdf.insert(1, "dual_bytes", SIZE["dual"])
    sdf.insert(2, "aad_bytes", SIZE["aad"])
    save_table(sdf, "exp8_envelope_summary")
    save_table(odf, "exp8_oracle_summary")
    print(sdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(odf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"external reason codes across all rejections: {ext}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = ["valid\n(server)", "forged\ntag", "exact\nreplay",
              "unknown\nhandle"]
    keys = ["valid:server_us", "reject:forged_tag", "reject:exact_replay",
            "reject:unknown_handle"]
    x = np.arange(len(keys))
    s = sdf.set_index("quantity")
    ax.bar(x - 0.18, [s.loc[k, "dual_mean"] for k in keys], 0.36,
           label="dual-tag (337 B)", color="#3b6ea5")
    ax.bar(x + 0.18, [s.loc[k, "aad_mean"] for k in keys], 0.36,
           label="AEAD with AAD (305 B)", color="#c08a2b")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean per-request cost (us)")
    ax.legend(fontsize=7)
    save_fig(fig, "fig_envelope_comparison")
    save_json({"summary": sdf.to_dict("records"),
               "oracle": odf.to_dict("records"),
               "external_codes": int(ext), "seeds": seeds,
               "requests_per_seed": n_req, "probes_per_class": n_probes},
              "exp8_envelope_comparison")


if __name__ == "__main__":
    main()
