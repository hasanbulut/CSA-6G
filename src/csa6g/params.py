"""Central configuration.

Every magic number used anywhere in the package lives here so the full
parameterisation fits on one screen.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- crypto ----

ENVELOPE_PLAINTEXT_BYTES = 256   # fixed-size padded plaintext          [4.1]
NONCE_BYTES = 16                 # n_u, freshness token                 [3 Threat Model]
KEY_ID_BYTES = 8                 # opaque per-session key handle
GCM_TAG_BYTES = 16
HMAC_BYTES = 32                  # outer HMAC-SHA256 tag T_u
TIMESTAMP_BYTES = 8
VERSION_BYTES = 1

# Total on-wire envelope size is therefore constant and independent of content:
ENVELOPE_WIRE_BYTES = (
    VERSION_BYTES + KEY_ID_BYTES + NONCE_BYTES + TIMESTAMP_BYTES
    + ENVELOPE_PLAINTEXT_BYTES + GCM_TAG_BYTES + HMAC_BYTES
)  # = 337

# Version of the allocation-policy configuration recorded in every committed
# allocation record (audit schema, Step 5).  Bump when PolicyParams changes in
# a way that affects admission decisions.
POLICY_VERSION = 1

# ------------------------------------------------------------- freshness ----

CLOCK_TOLERANCE_S = 30.0         # Delta-tau                            [3 Threat Model]
REPLAY_WINDOW_ENTRIES = 1 << 16  # W = 2^16                             [3 Threat Model]

# In-memory audit-trail capacity: released allocation records retained by
# ``AuthenticatedStore`` so they stay verifiable under K_commit after Step 6
# wipes the session keys.  Retention is the point (see the commit-key
# rationale in ``datastore``), but an unbounded in-memory trail is a
# memory-exhaustion vector an *authenticated* tenant can drive on its own: one
# tenant cycling admit/release adds one permanent record per cycle at no cost
# to itself.  The cap is deliberately large -- eviction should be something a
# deployment has arranged not to need, by draining to durable storage with
# ``drain_released`` -- and eviction is counted, never silent, because an audit
# trail that quietly forgets is worse than one that admits a bound.
AUDIT_TRAIL_ENTRIES = 1 << 16

# Bounded recent-denial log held by the E-AP.  Every denial is appended,
# including the ones an *unauthenticated* sender can trigger (unparseable
# envelope, unresolvable key handle), so an unbounded list is a remote memory
# sink: it is the one piece of proxy state a party with no key can grow, at
# roughly 65 bytes per junk packet and with no upper limit.  The aggregate
# counts live in ``EdgeAdmissionProxy.denial_counts``, whose key space is the
# fixed reason-code vocabulary and therefore cannot be grown by an attacker at
# all; this log keeps only the most recent entries for inspection.
DENIAL_LOG_ENTRIES = 4096

# Admit-token lifetime (Step 5).  Single source of truth: both the AEAD-sealed
# token minted by ``datastore.issue_admit_token`` and the E-AP's revocation
# list ``freshness.TokenStore`` read it from here.  They used to carry
# independent 300.0 defaults that agreed only by coincidence, so experiment 3's
# expired-token class would have gone on passing had either one been retuned.
ADMIT_TOKEN_TTL_S = 300.0

# ------------------------------------------------------------ topology ------

@dataclass(frozen=True)
class ResourceSpec:
    """Per-node capacity. Units: CPU cores, memory GB, storage TB."""
    cpu: float
    mem: float
    storage: float


DEFAULT_NODE = ResourceSpec(cpu=200.0, mem=512.0, storage=10.0)
DEFAULT_NUM_NODES = 6
COLOCATION_BUDGET_BETA = 1       # beta in the placement rule (anti-co-location)

# -------------------------------------------------------------- workload ----

# Class-conditional admission-field distributions (paper: synthetic field distributions table).
# Each entry: (cpu, mem, storage) as (mu, sigma) of a Normal, plus a
# log-normal latency-SLA target (mu, sigma) in log-space.
SLICE_PROFILES = {
    "URLLC": dict(cpu=(20.0, 5.0), mem=(16.0, 4.0), storage=(50.0, 10.0),
                  latency=(0.0, 0.2), sla_class=0),
    "eMBB":  dict(cpu=(80.0, 15.0), mem=(128.0, 20.0), storage=(500.0, 80.0),
                  latency=(2.3, 0.3), sla_class=1),
    "mMTC":  dict(cpu=(5.0, 2.0), mem=(4.0, 1.0), storage=(10.0, 3.0),
                  latency=(4.6, 0.4), sla_class=2),
}
# Storage in the field distributions table is expressed in GB; the resource pool uses TB.
STORAGE_GB_PER_TB = 1000.0

# ------------------------------------------------------------- policies -----

@dataclass(frozen=True)
class PolicyParams:
    """Tunables for the allocation policy ladder (the fairness experiment)."""
    # B1 -- static equal cap: q = R / n_max
    b1_max_tenants: int = 60
    # B2 -- token bucket: rate (normalised units per arrival) and burst
    b2_rate: float = 0.00002
    b2_burst: float = 0.02
    # P  -- NSAC quota + AIMD dynamic cap
    p_base_quota_frac: float = 0.025    # q_base as a fraction of R
    # cap <= kappa * R_free / active.
    #
    # Fixed engineering operating point used throughout the reported
    # experiments.  It is not claimed to be globally optimal.  The full
    # (q_base, kappa) sensitivity surface is exported by exp4 because different
    # objectives (benign fairness, benign acceptance, attacker capacity share,
    # and utilisation) select different points.  Change this constant only if
    # the complete result tree is regenerated and the trade-off is re-reported.
    p_pressure_kappa: float = 4.0
    p_rate_window: int = 64             # sliding window (arrivals) for rate gate
    p_rate_max: int = 2                 # max admitted requests per tenant / window
    p_md_factor: float = 0.5            # multiplicative decrease on denial
    p_ai_step: float = 0.15             # additive increase on success
    p_cap_floor: float = 0.1            # cap never shrinks below floor * q_base
    # Placement-slot accounting.  Every admitted allocation
    # occupies one of a node's beta same-slice anti-co-location slots (placement rule)
    # whatever its resource demand, so an unpriced slot lets a tenant hold
    # every same-slice slot at negligible resource charge.  Two variants:
    #   "cap"   -- shipped: the resource quota is unchanged and a tenant may
    #              hold at most p_slot_cap active allocations per slice type;
    #   "share" -- the slot is a fourth resource in the dominant-share cost
    #              (share 1 / (beta * nodes)), with a one-slot entitlement.
    # Both are evaluated in experiment 4; "share" is the PS ladder entry.
    p_slot_mode: str = "cap"
    p_slot_cap: int = 4
    # Recovery and penalty rules.  If the AIMD scale recovered only on
    # admission, a tenant pushed to the floor by legitimate congestion whose
    # smallest request exceeded floor * q_base could never be admitted again.
    # Two parameters prevent this:
    #   p_recover_step   -- the scale recovers by this amount per
    #                       p_rate_window arrivals elapsed since the tenant's
    #                       last *penalised* denial (linear in arrivals; a
    #                       tenant that keeps being denied keeps resetting
    #                       its own clock and stays throttled);
    #   p_penalize_placement -- whether a placement failure (no node has
    #                       capacity / isolation room: congestion, not the
    #                       tenant's own behaviour) shrinks the scale.
    # p_recover_step = 0.0 with p_penalize_placement = True gives the
    # admission-only recovery rule (pinned by a regression test).
    p_recover_step: float = 0.15
    p_penalize_placement: bool = False
    # Ablation switches (experiment 4b).  All True is the shipped policy.
    p_use_aimd: bool = True             # False: scale fixed at 1
    # Pressure-adaptive term of Eq. (cap).  Off by default: the
    # experiment-4b ablation showed it never binds at the operating point
    # (identical decisions with and without it), and equal tuning chose the
    # largest kappa.  Kept as a switch so the ablation can show that.
    p_use_pressure: bool = False
    p_use_rate_gate: bool = True        # False: no sliding-window gate
    # Slice-profile conformance.  A request whose demand on any
    # dimension exceeds its slice type's declared profile (mean + z * sd,
    # SLICE_PROFILES, i.e. a GST/NEST-style resource envelope) is denied and
    # counts as the tenant's own denial.  0 disables the check.
    p_profile_z: float = 3.0


DEFAULT_POLICY_PARAMS = PolicyParams()

# Reference policy without the recovery and profile rules, used by the
# recovery experiment: no time-based recovery, placement denials
# penalised, pressure term on, no profile check.
ADMISSION_ONLY_RECOVERY_PARAMS = PolicyParams(p_recover_step=0.0, p_penalize_placement=True,
                                p_use_pressure=True, p_profile_z=0.0)

# ---------------------------------------------------------- experiments -----

@dataclass(frozen=True)
class ExperimentParams:
    seeds: int = 100
    requests_per_seed: int = 100
    tenants: int = 5
    burst_sizes: tuple = (200, 5000)
    replay_trials: int = 100
    fairness_seeds: int = 100
    fairness_benign_tenants: int = 50
    fairness_attacker_tenants: int = 10
    fairness_benign_requests: int = 250
    fairness_nodes: int = 24
    fairness_beta: int = 3
    isolation_budgets: tuple = (1, 2, 3, 6, 999)
    malicious_ratios: tuple = (1.0, 2.0, 5.0, 10.0, 20.0)
    headline_ratio: float = 10.0
    tost_margins: tuple = (0.03, 0.01)
    privacy_samples: int = 6000
    privacy_folds: int = 5
    # Positive control for experiment 5: fractions of a one-sigma class
    # separation injected as a synthetic observable, to establish what size of
    # leak the detector would actually catch. 0.0 is the negative control (the
    # injected column is pure noise) and must come out undetected.
    # Spaced finely between 0.2 and 0.5 because that is where the family-wise MI
    # floor is crossed; a coarser grid reports a detection floor of 0.5 when the
    # true one is lower, which understates the instrument and so overstates
    # nothing -- but a loose bound is a weaker claim than the data supports.
    leak_strengths: tuple = (0.0, 0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.35,
                             0.40, 0.50)


DEFAULT_EXPERIMENT = ExperimentParams()

QUICK_EXPERIMENT = ExperimentParams(
    seeds=10, requests_per_seed=50, fairness_seeds=8,
    fairness_benign_tenants=50, fairness_attacker_tenants=10,
    fairness_benign_requests=150,
    replay_trials=20, privacy_samples=1500, burst_sizes=(200, 1000),
    leak_strengths=(0.0, 0.10, 0.50),
)
