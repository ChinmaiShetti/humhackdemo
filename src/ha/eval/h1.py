"""H1: raw synthetic queries vs authentic A (prereg §2 as amended by A1; details D-064).

Per dataset d and split seed s, "raw synthetic" = the population B0 draws from: U minus the
seed's near-duplicates. Against A_s:
  length   words per query (whitespace tokens): two-sample KS, plus W1 with a bootstrap CI;
  overlap  share of a query's content-word types (lowercase alphanumeric, English stopwords
           removed) that occur in its gold passage: the source passage for U, the best-matching
           highest-grade qrels passage for A: KS, plus W1 with a bootstrap CI;
  topic    JS divergence between the two distributions over the R_s strata of §5.2 (the same
           k-means), against a 1,000-permutation null of the pooled labels.
A dataset's p for a measure is the largest of its 5 per-seed p-values (significant only if every
seed is: an intersection-union rule). Holm runs over 3 datasets x 3 measures (m = 9); D4's three
tests count as not rejected, p = 1 (A1). The tau_b(B0, A) < 0.8 criterion comes from the
evaluation bootstrap (seed-averaged point estimate with its CI).
Bootstrap and permutations use numpy seed 20260927.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

SEED, B_ITER, N_PERM = 20260927, 1000, 1000
MEASURES = ("length", "overlap", "topic")
ALL_DATASETS = ("fiqa", "cqa_android", "hiaq")       # the Holm family of prereg §2 (m = 9)
_TOKEN = re.compile(r"[a-z0-9]+")


def _stopwords() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


def n_words(text: str) -> int:
    return len(text.split())


def overlap(query: str, passage: str, stop: frozenset[str] | None = None) -> float:
    stop = _stopwords() if stop is None else stop
    q = {t for t in _TOKEN.findall(query.lower()) if t not in stop}
    if not q:
        return 0.0
    return len(q & set(_TOKEN.findall(passage.lower()))) / len(q)


def w1_with_ci(x: np.ndarray, y: np.ndarray, rng, b_iter: int = B_ITER) -> dict[str, float]:
    reps = [wasserstein_distance(rng.choice(x, len(x)), rng.choice(y, len(y)))
            for _ in range(b_iter)]
    return {"w1": float(wasserstein_distance(x, y)),
            "ci_low": float(np.percentile(reps, 2.5)), "ci_high": float(np.percentile(reps, 97.5))}


def js_test(lab_x: np.ndarray, lab_y: np.ndarray, k: int, rng,
            n_perm: int = N_PERM) -> dict[str, float]:
    """JS divergence (base 2) of two label distributions over k strata, permutation p-value."""
    def js(a, b):
        pa, pb = np.bincount(a, minlength=k) / len(a), np.bincount(b, minlength=k) / len(b)
        return float(jensenshannon(pa, pb, base=2) ** 2)

    obs = js(lab_x, lab_y)
    pooled = np.concatenate([lab_x, lab_y])
    null = []
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        null.append(js(p[:len(lab_x)], p[len(lab_x):]))
    return {"js": obs, "p": float((1 + sum(v >= obs for v in null)) / (1 + n_perm))}


def seed_tests(syn: dict[str, np.ndarray], auth: dict[str, np.ndarray], k: int,
               rng) -> dict[str, dict[str, float]]:
    """syn/auth: {'length', 'overlap', 'topic'} arrays for one dataset and seed."""
    out = {}
    for m in ("length", "overlap"):
        ks = ks_2samp(syn[m], auth[m])
        out[m] = {"ks": float(ks.statistic), "p": float(ks.pvalue),
                  "mean_synthetic": float(np.mean(syn[m])),
                  "mean_authentic": float(np.mean(auth[m])),
                  **w1_with_ci(syn[m], auth[m], rng)}
    out["topic"] = js_test(syn["topic"], auth["topic"], k, rng)
    return out


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down; True = rejected."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, stop = len(order), {}, False
    for i, key in enumerate(order):
        if not stop and pvals[key] <= alpha / (m - i):
            out[key] = True
        else:
            stop = True
            out[key] = False
    return out


def decide(per_dataset_seed: dict[str, list[dict]], alpha: float = 0.05) -> dict[str, Any]:
    """Dataset p = max over seeds; datasets without results (D4, A1) get p = 1; Holm m = 9."""
    pvals = {}
    for d in ALL_DATASETS:
        for m in MEASURES:
            seeds = per_dataset_seed.get(d)
            pvals[f"{d}:{m}"] = max(s[m]["p"] for s in seeds) if seeds else 1.0
    return {"p": pvals, "rejected": holm(pvals, alpha), "m": len(pvals)}


def dataset_inputs(ds: str, seed: int) -> tuple[dict, dict, int]:
    """Length, overlap and stratum label arrays for U (minus near-duplicates) and A_s."""
    from ha.data.public import processed_dir
    from ha.data.splits import load_split
    from ha.gen.generate import load_source_corpus
    from ha.neighbours.retrieve import load_U
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.select.sets import fit_strata
    from ha.systems.retrieval import embed_queries

    stop = _stopwords()
    passages = load_source_corpus(ds).set_index("pid")["text"]
    u = load_U(ds)
    nb = pd.read_parquet(nb_path(ds, seed, "U"))
    u = u[~u["qid"].isin(set(nb.loc[nb["near_duplicate"], "qid"]))].reset_index(drop=True)
    a = load_split(ds, seed, "A")                                    # sealed for held-out
    qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
    qrels = qrels[qrels["qid"].isin(set(a["qid"]))]
    top = qrels[qrels["grade"] == qrels.groupby("qid")["grade"].transform("max")]
    gold = top.groupby("qid")["pid"].apply(list)
    km = fit_strata(embed_queries(load_split(ds, seed, "R")["text"].tolist()))
    syn = {"length": np.array([n_words(t) for t in u["text"]]),
           "overlap": np.array([overlap(t, passages.loc[p], stop)
                                for t, p in zip(u["text"], u["source_pid"], strict=True)]),
           "topic": km.predict(embed_queries(u["text"].tolist()))}
    auth = {"length": np.array([n_words(t) for t in a["text"]]),
            "overlap": np.array([max(overlap(t, passages.loc[p], stop) for p in gold.loc[q])
                                 for q, t in zip(a["qid"], a["text"], strict=True)]),
            "topic": km.predict(embed_queries(a["text"].tolist()))}
    return syn, auth, km.n_clusters


def run_h1(datasets: list[str], seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    import json

    from ha import registry
    from ha.data.public import dataset_config
    from ha.paths import get_paths
    from ha.utils import git_sha, utc_now_iso

    if git_sha().endswith("-dirty"):
        raise RuntimeError("commit first: registry numbers need a clean git sha")
    rng = np.random.default_rng(SEED)
    per: dict[str, list[dict]] = {}
    for d in datasets:
        per[d] = []
        for s in seeds:
            syn, auth, k = dataset_inputs(d, s)
            per[d].append({"seed": s, "n_synthetic": len(syn["length"]),
                           "n_authentic": len(auth["length"]), "k": k,
                           **seed_tests(syn, auth, k, rng)})
    decision = decide(per)
    roles = {d: dataset_config(d)["role"] for d in datasets}
    run_id = registry.new_run_id("H1")
    rows = []
    for d in datasets:
        did = dataset_config(d)["id"]
        for r in per[d]:
            for m in MEASURES:
                stat = "js" if m == "topic" else "w1"
                rows.append({"run_id": run_id, "exp": "H1", "dataset": did,
                             "split_seed": r["seed"], "method": "B0-population",
                             "system_set": None, "metric": f"{m}_{stat}", "value": r[m][stat],
                             "ci_low": r[m].get("ci_low"), "ci_high": r[m].get("ci_high"),
                             "n_queries": r["n_synthetic"], "p": r[m]["p"], "role": roles[d]})
    registry.record(rows, {"seed": SEED, "B": B_ITER, "perm": N_PERM, "seeds": seeds})
    result = {"run_id": run_id, "git_sha": git_sha(), "created_at": utc_now_iso(),
              "datasets": datasets, "roles": roles, "per_seed": per, "decision": decision,
              "note": "tau_b(B0, A) < 0.8 is read from the evaluation bootstrap"}
    out = get_paths().results / "eval" / f"H1-{'+'.join(datasets)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result
