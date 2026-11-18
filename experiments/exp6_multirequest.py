"""Experiment 6 -- multi-request correlation leakage.

A single-request observer is a weak model.  This experiment runs a stronger,
session-level observer to measure where the envelope's protection ends.

The observer.  It cannot decrypt anything.  It sees, per session: the
cleartext key handle in the envelope header (which is what lets it *group*
requests into sessions at all), arrival timestamps, the constant envelope
length, and the admission decision latency.  From a whole session it builds
request count, session duration and inter-arrival statistics, then tries to
infer the session's slice type.  Total observed bytes is deliberately *not* a
feature: with a fixed-size envelope it is exactly the request count times 337,
so it carries nothing the count does not.

Two workloads:

    rate-independent   request rate and session length are independent of
                       slice type.  The envelope should still hide everything.
    rate-correlated    mMTC sessions are many small frequent requests, URLLC
                       few and sparse, eMBB in between -- which is how these
                       service classes actually behave.

The expected result is that the fixed-size envelope survives the first
workload and leaks under the second.  What leaks is not the field
content but the *behaviour*, and no amount of padding a single request fixes
that; it needs cover traffic, batching, or a rotating key handle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g.model import ALL_SLICE_TYPES
from csa6g.params import DEFAULT_EXPERIMENT, QUICK_EXPERIMENT

# (mean requests per session, mean inter-arrival seconds) per slice type
BEHAVIOUR = {
    "rate_independent": {
        "URLLC": (8.0, 2.0), "eMBB": (8.0, 2.0), "mMTC": (8.0, 2.0),
    },
    "rate_correlated": {
        "URLLC": (3.0, 12.0), "eMBB": (9.0, 4.0), "mMTC": (40.0, 0.6),
    },
}


def sessions(workload: str, n_sessions: int, seed: int = 0):
    """Build per-session observable feature vectors and slice-type labels."""
    rng = np.random.default_rng(seed)
    beh = BEHAVIOUR[workload]
    X, y = [], []
    for _ in range(n_sessions):
        k = int(rng.integers(0, 3))
        stype = ALL_SLICE_TYPES[k]
        mean_n, mean_gap = beh[stype.label]

        n_req = max(1, int(rng.poisson(mean_n)))
        # n_req - 1 gaps, so a single-request session has none.  Drawing
        # max(1, n_req - 1) invented an inter-arrival for sessions that have no
        # interval at all, and because the gap distribution is slice-conditional
        # under rate_correlated that fabricated draw handed the observer real
        # slice-type information no observer could have.
        gaps = rng.exponential(mean_gap, size=n_req - 1)
        duration = float(gaps.sum())
        # Admission-decision latency is essentially constant per request; the
        # observer sees it with measurement noise only.
        lat = rng.normal(0.08, 0.01, size=n_req)

        X.append([
            float(n_req),
            duration,
            float(gaps.mean()) if gaps.size else 0.0,
            float(gaps.std()) if gaps.size > 1 else 0.0,
            float(n_req / duration) if duration > 0 else 0.0,   # request rate
            float(lat.mean()), float(lat.std()),
        ])
        y.append(k)

    # One feature row per session by construction: each loop iteration is an
    # independent session, and its requests are aggregated into that single
    # row.  Session groups are therefore singletons, so the stratified
    # request-level split used below coincides with a session-grouped split
    # (holding out complete sessions) and no session can appear in both the
    # training and the test fold.
    assert len(X) == len(y) == n_sessions, \
        "one feature row per session is a precondition of the CV protocol"
    # No ``total_bytes`` column.  Observed volume is exactly
    # n_requests * ENVELOPE_WIRE_BYTES -- a deterministic multiple of a feature
    # already present, precisely *because* the envelope is fixed-size.  Carrying
    # it added no information the observer did not have and split the mutual
    # information of the request count across two collinear rows of the table.
    names = ["n_requests", "session_duration_s", "mean_interarrival_s",
             "std_interarrival_s", "request_rate",
             "mean_latency_ms", "std_latency_ms"]
    return np.asarray(X), np.asarray(y), names


def macro_auc(X, y, model: str, folds: int = 5, seed: int = 0) -> float:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 3), dtype=float)
    for tr, te in skf.split(X, y):
        m = (make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, random_state=seed))
             if model == "logreg"
             else RandomForestClassifier(n_estimators=200, random_state=seed,
                                         n_jobs=-1))
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    n = P.privacy_samples
    banner(f"Experiment 6: multi-request session observer, {n} sessions")

    rows, raw = [], []
    for workload in BEHAVIOUR:
        X, y, names = sessions(workload, n)
        r = pd.DataFrame(X, columns=names)
        r.insert(0, "workload", workload)
        r["label"] = y
        raw.append(r)                      # raw export
        for model in ("logreg", "randomforest"):
            auc = macro_auc(X, y, model, P.privacy_folds)
            rows.append(dict(workload=workload, classifier=model,
                             macro_auc=auc))
            print(f"  {workload:18s} {model:12s} macro AUC = {auc:.4f}")
        # Which behavioural feature carries the leak?
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X, y, random_state=0)
        top = pd.DataFrame(dict(workload=workload, feature=names,
                                mutual_information=mi))
        rows_mi = top.sort_values("mutual_information", ascending=False)
        print(rows_mi.head(4).to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}"))

    df = pd.DataFrame(rows)
    save_table(df, "exp6_multirequest_summary")
    save_table(pd.concat(raw), "exp6_sessions_raw")

    indep = df[df.workload == "rate_independent"]["macro_auc"].max()
    corr = df[df.workload == "rate_correlated"]["macro_auc"].max()
    print(f"\nBest attacker: rate-independent {indep:.4f}, "
          f"rate-correlated {corr:.4f}")
    print("The fixed-size envelope hides field content, not session behaviour.")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    pivot = df.pivot(index="workload", columns="classifier",
                     values="macro_auc")
    pivot.plot.bar(ax=ax, width=0.7, color=["#3b6ea5", "#a53b3b"])
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.text(1.45, 0.51, "random guess", fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Macro AUC (session-level observer)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Residual leakage to a multi-request observer", fontsize=9)
    save_fig(fig, "fig9_multirequest_leakage")

    save_json({"results": df.to_dict("records"),
               "rate_independent_auc": float(indep),
               "rate_correlated_auc": float(corr),
               "interpretation": "the fixed-size envelope removes single-"
                                 "request field leakage but does not remove "
                                 "session-behaviour leakage; mitigations are "
                                 "cover traffic, batching, or rotating the "
                                 "cleartext key handle"},
              "exp6_multirequest")


if __name__ == "__main__":
    main()
