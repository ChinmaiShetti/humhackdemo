"""EXPLORATORY: authentic-query noise ceiling for tau_b (team-approved 2026-09-28, D-040).

Context only, for interpreting tau_b. It does NOT modify H2, the configuration pool, prompts,
selection or the freeze, and it is not a hypothesis. It uses only D1 (FiQA, development data)
and reuses the E0 per-query nDCG@10 tables (no retrieval is re-run). Held-out D2/D4 are never
touched.

  (a) random halves: 1,000 random 324/324 partitions of the 648 FiQA queries, exploratory seed
      20260928 (documented; not pre-registered); tau_b(half 1, half 2) over C1-C7.
  (b) frozen split seeds 0-4: tau_b(R_s, A_s) over C1-C7 (R 260 / A 388 queries).
  (c) per-pair flip rate across the random halves (which pairs make tau_b < 1).
  (d) sensitivity: two INDEPENDENT resamples (with replacement, 324 each) instead of
      complementary halves. Complementary halves are anti-correlated on every pair difference
      (their means sum to the overall mean), so near-tie pairs flip almost always and (a)
      understates agreement between independent query sets; (d) is the matching reference for
      an independent synthetic set vs A. Exploratory seed 20260929 (documented).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from ha import registry
from ha.data.public import processed_dir
from ha.eval.agreement import pair_disagreements, set_scores, tau_b
from ha.paths import get_paths

EXPLORATORY_SEED = 20260928
INDEPENDENT_SEED = 20260929
N_PARTITIONS = 1000
METRIC = "ndcg@10"
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def load_perquery(ds: str = "fiqa") -> tuple[list[str], dict[str, np.ndarray]]:
    frames = {}
    for c in CONFIGS:
        p = get_paths().runs / ds / c / "auth.perquery.parquet"
        frames[c] = pd.read_parquet(p).set_index("qid")[METRIC]
    df = pd.DataFrame(frames).sort_index()
    return df.index.tolist(), {c: df[c].to_numpy() for c in CONFIGS}


def compute(ds: str = "fiqa") -> dict:
    qids, pq = load_perquery(ds)
    n = len(qids)
    rng = np.random.default_rng(EXPLORATORY_SEED)
    taus, flips = [], Counter()
    for _ in range(N_PARTITIONS):
        perm = rng.permutation(n)
        a, b = set_scores(pq, perm[: n // 2]), set_scores(pq, perm[n // 2:])
        taus.append(tau_b(a, b))
        for pair, differs in pair_disagreements(a, b).items():
            flips[pair] += int(differs)
    taus = np.array(taus)

    rng_ind = np.random.default_rng(INDEPENDENT_SEED)
    taus_ind, flips_ind = [], Counter()
    for _ in range(N_PARTITIONS):
        a = set_scores(pq, rng_ind.integers(0, n, n // 2))
        b = set_scores(pq, rng_ind.integers(0, n, n // 2))
        taus_ind.append(tau_b(a, b))
        for pair, differs in pair_disagreements(a, b).items():
            flips_ind[pair] += int(differs)
    taus_ind = np.array(taus_ind)

    splits = pd.read_parquet(processed_dir(ds) / "splits.parquet").set_index("qid").loc[qids]
    frozen = {}
    for s in range(5):
        is_r = (splits[f"split_s{s}"] == "R").to_numpy()
        frozen[s] = tau_b(set_scores(pq, np.where(is_r)[0]), set_scores(pq, np.where(~is_r)[0]))

    return {
        "label": "EXPLORATORY (not a hypothesis; D-040)",
        "dataset": ds,
        "metric": METRIC,
        "n_queries": n,
        "random_halves": {
            "n_partitions": N_PARTITIONS, "seed": EXPLORATORY_SEED, "half_size": n // 2,
            "tau_b_mean": float(taus.mean()), "tau_b_median": float(np.median(taus)),
            "tau_b_p2.5": float(np.percentile(taus, 2.5)),
            "tau_b_p97.5": float(np.percentile(taus, 97.5)),
            "share_tau_b_equals_1": float((taus >= 1 - 1e-12).mean()),
        },
        "independent_resamples": {
            "n_pairs_of_samples": N_PARTITIONS, "seed": INDEPENDENT_SEED, "sample_size": n // 2,
            "tau_b_mean": float(taus_ind.mean()), "tau_b_median": float(np.median(taus_ind)),
            "tau_b_p2.5": float(np.percentile(taus_ind, 2.5)),
            "tau_b_p97.5": float(np.percentile(taus_ind, 97.5)),
            "share_tau_b_equals_1": float((taus_ind >= 1 - 1e-12).mean()),
            "pair_flip_rate_nonzero": {f"{p[0]}-{p[1]}": c / N_PARTITIONS for p, c in
                                       sorted(flips_ind.items(), key=lambda kv: -kv[1]) if c},
        },
        "frozen_split_seeds": {str(s): float(t) for s, t in frozen.items()},
        "pair_flip_rate": {f"{p[0]}-{p[1]}": c / N_PARTITIONS
                           for p, c in sorted(flips.items(), key=lambda kv: -kv[1])},
    }


def run(ds: str = "fiqa") -> dict:
    res = compute(ds)
    run_id = registry.new_run_id("EXP-noise-ceiling")
    rh = res["random_halves"]
    rows = [{"run_id": run_id, "exp": "EXP-noise-ceiling", "dataset": "D1", "split_seed": None,
             "method": "authentic_random_half_vs_half", "system_set": "C1-C7",
             "metric": "tau_b_ndcg@10_mean", "value": rh["tau_b_mean"],
             "ci_low": rh["tau_b_p2.5"], "ci_high": rh["tau_b_p97.5"],
             "n_queries": rh["half_size"], "exploratory": True}]
    ir = res["independent_resamples"]
    rows.append({"run_id": run_id, "exp": "EXP-noise-ceiling", "dataset": "D1",
                 "split_seed": None, "method": "authentic_independent_resamples",
                 "system_set": "C1-C7", "metric": "tau_b_ndcg@10_mean",
                 "value": ir["tau_b_mean"], "ci_low": ir["tau_b_p2.5"],
                 "ci_high": ir["tau_b_p97.5"], "n_queries": ir["sample_size"],
                 "exploratory": True})
    for s, t in res["frozen_split_seeds"].items():
        rows.append({"run_id": run_id, "exp": "EXP-noise-ceiling", "dataset": "D1",
                     "split_seed": int(s), "method": "authentic_R_vs_A", "system_set": "C1-C7",
                     "metric": "tau_b_ndcg@10", "value": t, "ci_low": None, "ci_high": None,
                     "n_queries": res["n_queries"], "exploratory": True})
    registry.record(rows, config={"analysis": "noise_ceiling", "seed": EXPLORATORY_SEED,
                                  "independent_seed": INDEPENDENT_SEED,
                                  "n_partitions": N_PARTITIONS, "metric": METRIC})
    res["run_id"] = run_id
    out = get_paths().results / "EXP_noise_ceiling.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res
