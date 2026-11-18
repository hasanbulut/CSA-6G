"""Statistical procedures used in the evaluation.

Each function returns the full set of quantities needed to check a claim,
not just a p-value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- CIs -------

def mean_ci(x, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using the Student-t interval."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = float(x.std(ddof=1) / np.sqrt(n))
    h = float(sps.t.ppf(0.5 + conf / 2, n - 1) * se)
    return m, m - h, m + h


# --------------------------------------------------------------- TOST -------

@dataclass(frozen=True)
class TOSTResult:
    mean_a: float
    mean_b: float
    diff: float
    margin: float
    ci_lo: float
    ci_hi: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in
                ("mean_a", "mean_b", "diff", "margin", "ci_lo", "ci_hi",
                 "p_tost", "equivalent")}


def tost(a, b, margin_frac: float, paired: bool = True,
         alpha: float = 0.05) -> TOSTResult:
    """Two one-sided tests for equivalence of means.

    ``margin_frac`` is the equivalence margin as a fraction of the *reference*
    mean ``b`` -- so ``margin_frac=0.03`` gives a +/-3% margin.  Equivalence is declared iff the (1-2*alpha) confidence interval of
    the difference lies entirely inside +/- margin, which is the interval form
    of TOST.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    margin = margin_frac * float(b.mean())

    if paired:
        d = a - b
        n = d.size
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n))
        df = n - 1
    else:
        # Welch, not pooled.  The standard error below is already the Welch
        # form (separate variances), so pairing it with the pooled df
        # n1 + n2 - 2 mixed two different tests and overstated the degrees of
        # freedom whenever the variances differed.  Welch-Satterthwaite is the
        # df that goes with this SE.
        n1, n2 = a.size, b.size
        v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
        m = float(a.mean() - b.mean())
        se = float(np.sqrt(v1 / n1 + v2 / n2))
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = float(num / den) if den > 0 else float(n1 + n2 - 2)

    if se == 0:
        se = 1e-18
    t_lower = (m + margin) / se
    t_upper = (m - margin) / se
    p_lower = float(sps.t.sf(t_lower, df))      # H0: diff <= -margin
    p_upper = float(sps.t.cdf(t_upper, df))     # H0: diff >= +margin
    p = max(p_lower, p_upper)

    crit = float(sps.t.ppf(1 - alpha, df))      # (1-2a) two-sided interval
    lo, hi = m - crit * se, m + crit * se
    return TOSTResult(float(a.mean()), float(b.mean()), m, margin, lo, hi,
                      p_lower, p_upper, p, bool(lo > -margin and hi < margin))


# --------------------------------------------- paired tests + corrections ---

def cliffs_delta(a, b) -> tuple[float, str]:
    """Cliff's delta effect size and its conventional magnitude label."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    d = float((gt - lt) / (a.size * b.size))
    ad = abs(d)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return d, mag


def holm_bonferroni(pvals, alpha: float = 0.05):
    """Return (adjusted_p, reject) arrays in the original order.

    Rejection is ``adj <= alpha``, not ``<``: Holm's step-down procedure
    rejects a hypothesis whose adjusted p-value *equals* the level.
    """
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


def paired_comparison(a, b, label: str = "") -> dict:
    """Paired Wilcoxon signed-rank test plus Cliff's delta and the median
    paired difference. Returns a dict ready for a results table."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, 0):
        stat, p = 0.0, 1.0
    else:
        stat, p = sps.wilcoxon(a, b, zero_method="wilcox",
                               alternative="two-sided")
    delta, mag = cliffs_delta(a, b)
    m, lo, hi = mean_ci(d)
    return dict(comparison=label, mean_diff=m, ci_lo=lo, ci_hi=hi,
                median_diff=float(np.median(d)), wilcoxon_stat=float(stat),
                p_raw=float(p), cliffs_delta=delta, magnitude=mag, n=int(d.size))
