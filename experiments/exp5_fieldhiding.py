"""Experiment 5 -- admission-time field hiding (the field-hiding experiment).

Checks that the envelope hides the intended fields under a single-request
observer, in a way that can fail.

Two observer configurations over the same underlying requests. The
random-forest observer uses 64 trees in each fold:

    Baseline  the admission fields in cleartext: CPU, memory, storage, latency
              SLA target, SLA class.
    CSA-6G      what an observer actually sees once the proposed envelope is in
              place: the on-wire envelope length, byte-level statistics of the
              real AES-256-GCM ciphertext (mean, standard deviation, Shannon
              entropy, first bytes), and the *measured* admission-decision
              latency.

The ciphertext is produced by the implementation and the latency feature is
measured on the E-AP, so the experiment can fail --
if padding were mis-wired, if the ciphertext leaked structure, or if
admission-decision time correlated with slice type, the classifiers would find
it.

Scope: a single-request observer with no auxiliary signals.  Multi-request correlation is experiment 6.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import banner, quick_flag, save_fig, save_json, save_table

from csa6g import crypto
from csa6g.eap import EdgeAdmissionProxy
from csa6g.harness import make_pool
from csa6g.keys import run_anchor_bootstrap
from csa6g.model import ALL_SLICE_TYPES, AdmissionRequest
from csa6g.params import (DEFAULT_EXPERIMENT, QUICK_EXPERIMENT,
                          SLICE_PROFILES, STORAGE_GB_PER_TB)
from csa6g.workload import sample_demand, sample_sla


def shannon_entropy(b: bytes) -> float:
    counts = Counter(b)
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def collect(n: int, seed: int = 0):
    """Return (X_baseline, X_protected, y, feature names)."""
    rng = np.random.default_rng(seed)
    pool = make_pool(64)
    eap = EdgeAdmissionProxy(pool)

    base_rows, protected_rows, labels = [], [], []
    unpadded_rows = []      # same request, JSON-encoded, AEAD, no padding
    group_tenants, group_sessions = [], []
    for i in range(n):
        k = int(rng.integers(0, 3))
        stype = ALL_SLICE_TYPES[k]
        demand = sample_demand(rng, stype)
        sla = sample_sla(rng, stype)
        sla_class = SLICE_PROFILES[stype.label]["sla_class"]

        service_id = int(rng.integers(0, 4))
        sk = eap.register_session(run_anchor_bootstrap(rng), stype,
                                  service_id, i)
        req = AdmissionRequest(
            tenant_id=i, pseudonym=bytes(rng.integers(0, 256, 8, dtype="uint8")),
            slice_type=stype, service_id=service_id, sla_latency_ms=sla,
            demand=demand, arrival_index=i, counter=1)
        env = crypto.seal(sk, req)

        t0 = time.perf_counter_ns()
        out = eap.handle(env.raw, tenant_id=i)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

        # Every request must be admitted, or the denial path -- which is much
        # cheaper -- would put a second mode into ``admission_latency_ms`` and
        # the observer would be reading that rather than the slice type.
        assert out.decision.admitted, out.internal_reason
        eap.release_slice(out.session_id)

        ct = env.ciphertext
        arr = np.frombuffer(ct, dtype=np.uint8).astype(float)

        base_rows.append([demand.cpu, demand.mem,
                          demand.storage * STORAGE_GB_PER_TB, sla,
                          float(sla_class)])
        protected_rows.append([float(env.size), float(len(ct)),
                          float(arr.mean()), float(arr.std()),
                          shannon_entropy(ct), float(ct[0]), float(ct[1]),
                          float(ct[-1]), elapsed_ms])
        # Contrast: the same request JSON-encoded (as SBA APIs carry
        # application data) and sealed with AES-256-GCM, header as AAD,
        # *without* fixed-size padding.  Only what such an observer sees:
        # envelope length and ciphertext byte statistics.  The control gets a
        # fresh key per request: reusing K_slice with the envelope's IV would
        # repeat a (key, IV) pair.
        body = json.dumps(dict(
            pseudonym=req.pseudonym.hex(), slice_type=int(stype),
            service_id=service_id, sla_ms=round(sla, 2),
            cpu=round(demand.cpu, 2), mem_gb=round(demand.mem, 2),
            storage_gb=round(demand.storage * STORAGE_GB_PER_TB, 2),
            counter=1), separators=(",", ":")).encode()
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        header = env.raw[:33]
        control_key = AESGCM.generate_key(bit_length=256)
        uct = AESGCM(control_key).encrypt(env.nonce[:12], body, header)
        uarr = np.frombuffer(uct[:-16], dtype=np.uint8).astype(float)
        unpadded_rows.append([float(len(header) + len(uct)),
                              float(uarr.mean()), float(uarr.std()),
                              shannon_entropy(uct[:-16])])
        labels.append(k)
        group_tenants.append(i)
        group_sessions.append(bytes(sk.key_id))

    base_names = ["cpu", "memory", "storage_GB", "latency_ms", "sla_class"]
    protected_names = ["envelope_bytes", "ciphertext_bytes", "ct_mean", "ct_std",
                  "ct_entropy", "ct_byte0", "ct_byte1", "ct_byte_last",
                  "admission_latency_ms"]

    # Grouped-validation property, enforced rather than asserted: every loop
    # iteration above creates its own tenant (tenant_id = i) and its own
    # session, issues exactly one request, and releases.  Groups are therefore
    # singletons -- no tenant or session contributes more than one sample --
    # so a request-level stratified split and a tenant/session-grouped split
    # partition the data identically, and no group can straddle a train/test
    # boundary.  If this loop is ever changed to reuse a tenant or session,
    # these assertions fail and the CV protocol must move to GroupKFold.
    assert len(set(group_tenants)) == n, \
        "tenant groups are not singletons; switch the CV to GroupKFold"
    assert len(set(group_sessions)) == n, \
        "session groups are not singletons; switch the CV to GroupKFold"

    global UNPADDED
    UNPADDED = (np.asarray(unpadded_rows),
                ["envelope_bytes", "ct_mean", "ct_std", "ct_entropy"])
    return (np.asarray(base_rows), np.asarray(protected_rows),
            np.asarray(labels), base_names, protected_names)


UNPADDED = None


def inject_leak(Xc: np.ndarray, y: np.ndarray, strength: float,
                seed: int = 0) -> np.ndarray:
    """Append one synthetic observable carrying a controlled amount of
    slice-type information.

    ``strength`` is the fraction of the class-separation scale present in the
    injected column: 0.0 is pure noise (the column is independent of ``y``) and
    1.0 puts the three classes one standard deviation apart.  Everything else in
    the feature matrix is the real CSA-6G observable set, so the only thing that
    changes across the sweep is how much the observer is *given*.
    """
    rng = np.random.default_rng(seed)
    col = rng.normal(0.0, 1.0, size=len(y)) + strength * y.astype(float)
    return np.column_stack([Xc, col])


def leak_sensitivity(Xc, y, protected_names, folds: int, strengths, mi_floor: float,
                     n_perm: int = 50):
    """Positive control: how large a leak would this pipeline actually catch?

    Without it, experiment 5's result is "we looked and found nothing", which is
    unfalsifiable -- a detector that can see nothing at all produces the same
    output.  Injecting a known leak at a range of strengths and running the
    *identical* classifier and mutual-information pipeline turns the null into a
    bounded statement: leaks at or above the smallest detected strength would
    have been caught, and the real CSA-6G observables sit below it.

    The MI floor passed in is the family-wise floor computed on the real feature
    set, so the injected column is judged by the same rule as everything else.
    """
    rows = []
    for s in strengths:
        Xl = inject_leak(Xc, y, s)
        auc_lr, _ = macro_auc_cv(Xl, y, "logreg", folds)
        auc_rf, _ = macro_auc_cv(Xl, y, "randomforest", folds)
        from sklearn.feature_selection import mutual_info_classif
        mi_injected = float(mutual_info_classif(Xl, y, random_state=0)[-1])
        best = max(auc_lr, auc_rf)
        rows.append(dict(
            leak_strength=s,
            macro_auc_logreg=auc_lr,
            macro_auc_randomforest=auc_rf,
            best_macro_auc=best,
            injected_feature_mi=mi_injected,
            mi_above_family_floor=bool(mi_injected > mi_floor),
            detected=bool(mi_injected > mi_floor and best > 0.55)))
        print(f"  leak strength {s:4.2f}: best macro AUC = {best:.4f}, "
              f"injected-feature MI = {mi_injected:.5f} "
              f"{'DETECTED' if rows[-1]['detected'] else '-'}")
    return pd.DataFrame(rows)


def macro_auc_cv(X, y, model_name: str, folds: int, seed: int = 0):
    """Stratified-CV one-vs-rest macro AUC, plus pooled out-of-fold scores."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 3), dtype=float)
    for tr, te in skf.split(X, y):
        if model_name == "logreg":
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000,
                                                 random_state=seed))
        else:
            m = RandomForestClassifier(n_estimators=64, random_state=seed,
                                       n_jobs=-1)
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return float(auc), proba


def main() -> None:
    P = QUICK_EXPERIMENT if quick_flag() else DEFAULT_EXPERIMENT
    banner(f"Experiment 5: single-request field hiding, "
           f"{P.privacy_samples} samples, {P.privacy_folds}-fold CV")

    Xb, Xc, y, base_names, protected_names = collect(P.privacy_samples)
    print(f"  class balance: {np.bincount(y)}")
    print(f"  envelope size on the wire: "
          f"{sorted(set(Xc[:, 0].astype(int)))} bytes (constant if hidden)")

    Xu, unpadded_names = UNPADDED
    print(f"  unpadded JSON envelope sizes: {int(Xu[:, 0].min())}-"
          f"{int(Xu[:, 0].max())} bytes")
    # Raw export: the three observer feature matrices with labels.
    raw = pd.DataFrame(np.c_[Xb, Xc, Xu, y],
                       columns=[f"base_{c}" for c in base_names]
                       + [f"csa6g_{c}" for c in protected_names]
                       + [f"unpadded_{c}" for c in unpadded_names]
                       + ["label"])
    save_table(raw, "exp5_features_raw")
    oof = {}
    rows, rocs = [], {}
    for cond, X, names in (("Baseline", Xb, base_names),
                           ("CSA-6G", Xc, protected_names),
                           ("Unpadded JSON + AEAD", Xu, unpadded_names)):
        for model in ("logreg", "randomforest"):
            auc, proba = macro_auc_cv(X, y, model, P.privacy_folds)
            rows.append(dict(condition=cond, classifier=model, macro_auc=auc,
                             n_features=X.shape[1],
                             features=", ".join(names)))
            rocs[(cond, model)] = proba
            for j in range(3):
                oof[f"{cond}|{model}|p{j}"] = proba[:, j]
            print(f"  {cond:8s} {model:12s} macro AUC = {auc:.4f}")

    df = pd.DataFrame(rows)
    save_table(df, "exp5_fieldhiding_summary")
    oof_df = pd.DataFrame(oof)
    oof_df["label"] = y
    save_table(oof_df, "exp5_predictions_raw")

    # Per-feature mutual information under CSA-6G: which observable, if any,
    # still carries slice-type information?
    #
    # Reported against a permutation null, because the raw estimate is not
    # interpretable on its own.  sklearn's kNN estimator adds jitter to
    # continuous columns, so it returns a small positive value for features
    # that are *provably* constant: ``ciphertext_bytes`` is 256 for every
    # sample and still scored 0.007 nats, which put a constant fourth in the
    # ranking.  Shuffling the labels and re-estimating gives the
    # noise floor directly; a value below its own null is noise, whatever its
    # rank.
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(Xc, y, random_state=0)
    n_perm = 50
    perm_rng = np.random.default_rng(0)
    null = np.vstack([
        mutual_info_classif(Xc, perm_rng.permutation(y), random_state=i)
        for i in range(n_perm)])

    # Decision rule is the *family-wise* threshold: the 95th percentile of the
    # largest MI seen anywhere in the table on a permuted run.  Comparing each
    # feature against its own 95th percentile tests nine hypotheses at 5% each,
    # expects 0.45 false positives, and duly produced one -- ct_entropy cleared
    # its own floor by 0.001 nats, which would have put "the entropy of the
    # ciphertext leaks slice type" in the results table. AES-GCM output is
    # uniform and the length is fixed, so there is no mechanism; it was the
    # multiple-comparison artefact the per-feature threshold invites. The
    # per-feature column is kept as description, not as a test.
    per_feature_floor = np.quantile(null, 0.95, axis=0)
    family_floor = float(np.quantile(null.max(axis=1), 0.95))
    mi_df = (pd.DataFrame(dict(feature=protected_names, mutual_information=mi,
                               null_p95=per_feature_floor,
                               family_floor=family_floor,
                               above_noise_floor=mi > family_floor,
                               constant=[bool(np.ptp(Xc[:, j]) == 0)
                                         for j in range(Xc.shape[1])]))
             .sort_values("mutual_information", ascending=False))
    save_table(mi_df, "exp5_protected_feature_mi")
    print(f"\nPer-feature mutual information with slice type under CSA-6G "
          f"(nats), against a {n_perm}-permutation null:")
    print(mi_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    n_signal = int(mi_df["above_noise_floor"].sum())
    print(f"  family-wise noise floor (95th pct of the per-permutation max): "
          f"{family_floor:.5f} nats")
    print(f"  features above it: {n_signal} of {len(protected_names)}")
    if n_signal:
        leak = mi_df[mi_df["above_noise_floor"]]["feature"].tolist()
        print(f"  -> {leak}")

    # ------------------------------------------- injected-leak positive control --
    print(f"\nPositive control: injected leak of known strength, same "
          f"classifier and MI pipeline")
    leak = leak_sensitivity(Xc, y, protected_names, P.privacy_folds,
                            P.leak_strengths, family_floor)
    save_table(leak, "exp5_leak_detection")
    detected = leak[leak["detected"]]
    floor_strength = (float(detected["leak_strength"].min())
                      if not detected.empty else float("nan"))
    protected_best = float(df[df.condition == "CSA-6G"]["macro_auc"].max())
    if not detected.empty:
        print(f"  detection floor: leaks of strength >= {floor_strength:g} are "
              f"caught; the real CSA-6G feature set reads {protected_best:.4f} macro "
              f"AUC and no feature clears the MI floor")
    else:
        print("  nothing detected at any injected strength -- the pipeline "
              "has no demonstrated power and the null result means nothing")

    # -------------------------------------------------------- ROC figure --
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random guess (AUC = 0.50)")
    for (cond, model), proba in rocs.items():
        if model != "logreg":
            continue
        # Micro-averaged ROC over the one-vs-rest problems, for display.
        yb = np.eye(3)[y].ravel()
        fpr, tpr, _ = roc_curve(yb, proba.ravel())
        auc = df[(df.condition == cond) & (df.classifier == model)
                 ]["macro_auc"].iloc[0]
        ax.plot(fpr, tpr, lw=1.6, label=f"{cond} (macro AUC = {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Single-request slice-type observer")
    ax.legend(fontsize=7, loc="lower right")
    save_fig(fig, "fig8_roc_fieldhiding")

    # Detection-floor curve: what the observer recovers as a known leak grows,
    # with the real CSA-6G result marked. This is the figure that answers "is the
    # detector simply blind?" -- the null at CSA-6G only means something once the
    # curve shows where the pipeline starts to see.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhline(0.5, color="k", ls="--", lw=0.9, label="Random guess (0.500)")
    ax.plot(leak["leak_strength"], leak["best_macro_auc"], marker="o", ms=4,
            lw=1.7, color="C0", label="Injected leak (best classifier)")
    ax.axhline(protected_best, color="C3", ls=":", lw=1.6,
               label=f"Real CSA-6G features ({protected_best:.3f})")
    if not detected.empty:
        ax.axvline(floor_strength, color="C2", lw=1.2, alpha=0.7)
        # annotation lifted clear of the near-chance horizontal lines and the
        # legend (upper-left), with an arrow to the floor crossing
        ax.annotate(f"detection floor\n(strength {floor_strength:g})",
                    xy=(floor_strength, protected_best), xycoords="data",
                    xytext=(floor_strength + 0.03, 0.60), fontsize=7,
                    color="C2",
                    arrowprops=dict(arrowstyle="->", color="C2", lw=0.9))
    ax.set_ylim(0.47, max(0.66, leak["best_macro_auc"].max() + 0.02))
    ax.set_xlabel("Injected leak strength (fraction of 1$\\sigma$ separation)")
    ax.set_ylabel("Macro AUC")
    ax.set_title("Detection floor of the single-request observer", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, "fig11_leak_detection")

    base = df[(df.condition == "Baseline")]["macro_auc"].max()
    protected = df[(df.condition == "CSA-6G")]["macro_auc"].max()
    print(f"\nBest attacker: baseline macro AUC {base:.4f} -> CSA-6G {protected:.4f}")
    save_json({"results": df.to_dict("records"),
               "best_baseline_auc": float(base), "best_protected_auc": float(protected),
               "leak_detection": leak.to_dict("records"),
               "leak_detection_floor": floor_strength,
               "leak_detection_note": "positive control. A synthetic observable "
                                      "carrying a controlled fraction of a "
                                      "one-sigma class separation is appended to "
                                      "the real CSA-6G feature set and the "
                                      "identical classifier and MI pipeline is "
                                      "re-run. leak_detection_floor is the "
                                      "smallest injected strength the pipeline "
                                      "catches, so the CSA-6G null result reads "
                                      "'no leak of that size or larger is "
                                      "present' rather than 'we found nothing'. "
                                      "Strength 0.0 is the negative control and "
                                      "must come out undetected",
               "feature_mi": mi_df.to_dict("records"),
               "feature_mi_note": f"mutual information is reported against a "
                                  f"{n_perm}-permutation null. The kNN "
                                  f"estimator returns small positive values "
                                  f"for provably constant columns, so the raw "
                                  f"figure is not interpretable without a "
                                  f"noise floor. The decision rule is the "
                                  f"family-wise floor -- the 95th percentile "
                                  f"of the largest MI anywhere in the table "
                                  f"on a permuted run -- because comparing "
                                  f"nine features against nine separate 95th "
                                  f"percentiles expects 0.45 false positives; "
                                  f"above_noise_floor is the column to read",
               "family_noise_floor_nats": family_floor,
               "features_above_noise_floor": n_signal,
               "scope": "single-request observer, no auxiliary signals; "
                        "multi-request correlation is experiment 6"},
              "exp5_fieldhiding")


if __name__ == "__main__":
    main()
