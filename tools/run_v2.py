"""[EXPLORATORY] Run the D-068 selectors (OVL, NN) and their evaluation on the given datasets.

usage: python tools/run_v2.py fiqa cqa_android hiaq      (D2 needs HA_UNSEAL=heldout_A)
"""

import json
import sys

from ha.select.variants import run_eval, run_select

for ds in sys.argv[1:]:
    for m in run_select(ds):
        print(ds, json.dumps({k: m[k] for k in ("split_seed", "N", "R_overlap_mean",
                                                 "OVL_overlap_mean", "NN_overlap_mean",
                                                 "band_quota_at_N", "band_available",
                                                 "OVL_band_counts_draw0", "NN_size")}))
    res = run_eval(ds)
    b = res["bootstrap"]
    print(ds, res["run_id"])
    print(ds, "tau:", {m: round(v["tau_b"], 3) for m, v in b["tau_seed_mean"].items()})
    print(ds, "delta:", {c: (round(v["delta"], 3), round(v["ci_low"], 3), round(v["ci_high"], 3),
                             v["p_one_sided"]) for c, v in b["comparisons"].items()})
