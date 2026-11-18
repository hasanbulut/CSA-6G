"""csa6g -- secure, privacy-aware slice admission for beyond-5G/6G studies.

Reference implementation of the framework described in the paper
"CSA-6G: Confidentiality-Preserving Slice Admission and Defense-in-Depth
Allocation Toward 6G".

Module map (paper section in brackets):

    params            experiment configuration constants
    model             slice types, demand vectors, resource pool, requests  [4.1]
    keys              post-auth anchor, HKDF context derivation, key wipe    [Step 2, Step 6]
    crypto            admission envelope: AES-256-GCM + outer HMAC + padding [4.1 envelope]
    freshness         nonce/timestamp replay window, monotone counters      [3 Threat Model]
    placement         isolation-aware first-fit placement                   [4 Step 4 placement]
    nsac              NSAC-style quota + AIMD dynamic caps                  [Step 4]
    policies          allocation policy ladder B0..B4 and proposed P        [6 fairness]
    datastore         authenticated two-phase state commit + rollback       [Step 5]
    eap               Edge Admission Proxy: the admission decision A(.)     [4.1 decision]
    workload          seeded request-stream generator (benign + adversarial)
    metrics           Jain's index, utilisation, acceptance rates
    stats             TOST, paired Wilcoxon + Holm, Cliff's delta, t-CIs
"""

__version__ = "1.0.0"
