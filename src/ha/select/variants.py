"""[EXPLORATORY, after the freeze] Two alternative selectors and their evaluation (D-068).

Both were specified in D-068 before they were run on any dataset, and run unchanged on D1, D2
and D4. Neither uses the LLM judge. Same N, eligibility (U minus the seed's near-duplicates)
and strata machinery as the frozen selection (ha.select.sets, D-063).

  OVL  overlap-matched random: 5 equal-frequency bands of gold-passage overlap
       (ha.eval.h1.overlap) computed on R_s (never A); synthetic queries fill the bands in
       R_s's proportions (largest remainder, shortfall re-allocated), random within a band;
       20 draws, SeedSequence [3, split seed, draw].
  NN   nearest-to-real: score = nn_mean5_cos (mean cosine to the 5 R_s neighbours, P4.1);
       greedy within the k-means strata of R_s with the 0.90 dedupe, ties by a seeded key
       [4, split seed] (the M1 procedure with this score instead of the judge's).

Evaluation: the §7 procedure (per-query nDCG@10, tau_b over C1-C7, paired bootstrap B = 1,000,
seed 20260927), with the frozen B0, B1 and M1 sets as comparators. No hypothesis decision.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ha.select.sets import DEDUPE_COS, N_DRAWS, allocate, fit_strata, stratified_pick

N_BANDS = 5
STREAM_OVL, STREAM_NN = 3, 4
METHODS = ("M1", "B0", "B1", "OVL", "NN")
COMPARISONS = (("OVL", "B0"), ("OVL", "B1"), ("OVL", "M1"),
               ("NN", "B0"), ("NN", "B1"), ("NN", "M1"))


def band_edges(r_overlap: np.ndarray, n: int = N_BANDS) -> np.ndarray:
    """Interior edges of n equal-frequency bands of R's overlap (duplicates removed)."""
    return np.unique(np.quantile(r_overlap, np.linspace(0, 1, n + 1)[1:-1]))


def bands(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=True)


def select_ovl(seed: int, u_overlap: np.ndarray, eligible: list[int], r_overlap: np.ndarray,
               n: int) -> list[list[int]]:
    edges = band_edges(r_overlap)
    k = len(edges) + 1
    pi = np.bincount(bands(r_overlap, edges), minlength=k) / len(r_overlap)
    stratum = bands(u_overlap, edges)
    out = []
    for d in range(N_DRAWS):
        rng = np.random.default_rng([STREAM_OVL, seed, d])
        order = [eligible[j] for j in rng.permutation(len(eligible))]
        out.append(stratified_pick(order, stratum, pi, n))
    return out


def select_nn(seed: int, nn_score: np.ndarray, eligible: list[int], u_emb: np.ndarray,
              r_emb: np.ndarray, n: int) -> list[int]:
    km = fit_strata(r_emb)
    pi = np.bincount(km.labels_, minlength=km.n_clusters) / len(r_emb)
    tie = np.random.default_rng([STREAM_NN, seed]).random(len(nn_score))
    order = sorted(eligible, key=lambda i: (-nn_score[i], tie[i]))
    return stratified_pick(order, km.predict(u_emb), pi, n, u_emb, DEDUPE_COS)


def out_path(ds: str, seed: int):
    from ha.paths import get_paths

    return get_paths().data / "select_v2" / ds / f"s{seed}" / "sets.parquet"


def run_select(ds: str, seeds=(0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    from ha.data.public import processed_dir
    from ha.data.splits import load_split
    from ha.eval.h1 import _stopwords, overlap
    from ha.gen.generate import load_source_corpus
    from ha.neighbours.retrieve import load_U
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.select import sets as S
    from ha.systems.retrieval import embed_queries
    from ha.utils import git_sha, utc_now_iso

    stop = _stopwords()
    passages = load_source_corpus(ds).set_index("pid")["text"]
    u = load_U(ds)
    qids = u["qid"].tolist()
    u_emb = embed_queries(u["text"].tolist())
    u_ov = np.array([overlap(t, passages.loc[p], stop)
                     for t, p in zip(u["text"], u["source_pid"], strict=True)])
    qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
    metas = []
    for s in seeds:
        frozen = pd.read_parquet(S.out_path(ds, s))
        n = int(frozen[frozen["method"] == "B0"].groupby("draw").size().iloc[0])
        nb = pd.read_parquet(nb_path(ds, s, "U")).set_index("qid").loc[qids]
        eligible = [i for i, q in enumerate(qids) if not bool(nb.loc[q, "near_duplicate"])]
        r = load_split(ds, s, "R")                                      # R only, never A
        rq = qrels[qrels["qid"].isin(set(r["qid"]))]
        top = rq[rq["grade"] == rq.groupby("qid")["grade"].transform("max")]
        gold = top.groupby("qid")["pid"].apply(list)
        r_ov = np.array([max(overlap(t, passages.loc[p], stop) for p in gold.loc[q])
                         for q, t in zip(r["qid"], r["text"], strict=True)])
        rows = []
        for d, picked in enumerate(select_ovl(s, u_ov, eligible, r_ov, n)):
            rows += [{"method": "OVL", "draw": d, "qid": qids[i], "rank": j}
                     for j, i in enumerate(picked)]
        picked = select_nn(s, nb["nn_mean5_cos"].to_numpy(), eligible, u_emb,
                           embed_queries(r["text"].tolist()), n)
        rows += [{"method": "NN", "draw": 0, "qid": qids[i], "rank": j}
                 for j, i in enumerate(picked)]
        sets = pd.DataFrame(rows)
        pos = {q: i for i, q in enumerate(qids)}
        edges = band_edges(r_ov)
        kb = len(edges) + 1
        draw0 = sets[(sets["method"] == "OVL") & (sets["draw"] == 0)]["qid"]
        p = out_path(ds, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        sets.to_parquet(p, index=False)
        meta = {"dataset": ds, "split_seed": s, "N": n, "decision": "D-068",
                "band_edges": edges.round(4).tolist(),
                "R_overlap_mean": float(r_ov.mean()),
                "OVL_overlap_mean": float(np.mean([u_ov[[pos[q] for q in g["qid"]]].mean()
                                                   for _, g in sets[sets["method"] == "OVL"]
                                                   .groupby("draw")])),
                "NN_overlap_mean": float(u_ov[picked].mean()), "NN_size": len(picked),
                "band_quota_at_N": allocate(n, np.bincount(bands(r_ov, edges), minlength=kb)
                                            / len(r_ov), np.full(kb, n)).tolist(),
                "band_available": np.bincount(bands(u_ov[eligible], edges), minlength=kb).tolist(),
                "OVL_band_counts_draw0": np.bincount(bands(u_ov[[pos[q] for q in draw0]], edges),
                                                     minlength=kb).tolist(),
                "git_sha": git_sha(), "created_at": utc_now_iso()}
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                               encoding="utf-8")
        metas.append(meta)
    return metas


# ----------------------------------------------------------------------------- evaluation
def method_taus(si: dict, idx_a=None, rng=None) -> dict[str, float]:
    """Mean tau_b per method for one seed; with rng, one bootstrap replicate (A by idx_a,
    each set resampled independently, in the fixed order of si['sets'])."""
    from ha.eval.sets import tau_b_rows

    a = (si["A"] if idx_a is None else si["A"][idx_a]).mean(axis=0)
    names, scores = [], []
    for method, _d, m in si["sets"]:
        rows = m if rng is None else m[rng.integers(0, len(m), len(m))]
        names.append(method)
        scores.append(rows.mean(axis=0))
    taus = tau_b_rows(np.array(scores), a)
    return {meth: float(np.nanmean([t for t, nm in zip(taus, names, strict=True) if nm == meth]))
            for meth in dict.fromkeys(names)}


def bootstrap(inputs: list[dict], b_iter: int = 1000, seed: int = 20260927) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    point = [method_taus(si) for si in inputs]
    reps: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in COMPARISONS}
    tau_reps: dict[str, list[float]] = {m: [] for m in METHODS}
    for _ in range(b_iter):
        ts = []
        for si in inputs:
            idx_a = rng.integers(0, len(si["A"]), len(si["A"]))
            ts.append(method_taus(si, idx_a, rng))
        for a, b in COMPARISONS:
            reps[f"{a}-{b}"].append(float(np.mean([t[a] - t[b] for t in ts])))
        for m in METHODS:
            tau_reps[m].append(float(np.mean([t[m] for t in ts])))
    out: dict[str, Any] = {"B": b_iter, "seed": seed, "comparisons": {}, "tau_seed_mean": {}}
    for a, b in COMPARISONS:
        r = np.array(reps[f"{a}-{b}"])
        out["comparisons"][f"{a}-{b}"] = {
            "delta": float(np.mean([t[a] - t[b] for t in point])),
            "ci_low": float(np.nanpercentile(r, 2.5)), "ci_high": float(np.nanpercentile(r, 97.5)),
            "p_one_sided": float(np.mean(r <= 0))}
    for m in METHODS:
        out["tau_seed_mean"][m] = {"tau_b": float(np.mean([t[m] for t in point])),
                                   "ci_low": float(np.nanpercentile(tau_reps[m], 2.5)),
                                   "ci_high": float(np.nanpercentile(tau_reps[m], 97.5))}
    out["per_seed"] = point
    return out


def run_eval(ds: str, seeds=(0, 1, 2, 3, 4), b_iter: int = 1000) -> dict[str, Any]:
    from ha import registry
    from ha.data.public import dataset_config
    from ha.data.splits import load_split
    from ha.eval.sets import perquery_table
    from ha.paths import get_paths
    from ha.select import sets as S
    from ha.utils import git_sha, utc_now_iso

    if git_sha().endswith("-dirty"):
        raise RuntimeError("commit first: registry numbers need a clean git sha")
    auth, synth = perquery_table(ds, "auth"), perquery_table(ds, "U")
    inputs = []
    for s in seeds:
        a_q = load_split(ds, s, "A")["qid"].tolist()            # sealed for held-out datasets
        both = pd.concat([pd.read_parquet(S.out_path(ds, s)), pd.read_parquet(out_path(ds, s))])
        groups = []
        for method in METHODS:
            for draw, g in both[both["method"] == method].groupby("draw", sort=True):
                groups.append((method, int(draw), synth.loc[g["qid"]].to_numpy()))
        inputs.append({"seed": s, "A": auth.loc[a_q].to_numpy(), "sets": groups})
    boot = bootstrap(inputs, b_iter)
    role = dataset_config(ds)["role"]
    run_id = registry.new_run_id(f"EXP-v2-{ds}")
    n = len(inputs[0]["sets"][0][2])
    rows = [{"run_id": run_id, "exp": "EXP-v2", "dataset": dataset_config(ds)["id"],
             "split_seed": None, "method": m, "system_set": "C1-C7", "metric": "tau_b_seed_mean",
             "value": v["tau_b"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
             "n_queries": n, "role": f"exploratory ({role})"}
            for m, v in boot["tau_seed_mean"].items()]
    rows += [{"run_id": run_id, "exp": "EXP-v2", "dataset": dataset_config(ds)["id"],
              "split_seed": None, "method": c, "system_set": "C1-C7", "metric": "delta_tau_b",
              "value": v["delta"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
              "n_queries": n, "p": v["p_one_sided"], "role": f"exploratory ({role})"}
             for c, v in boot["comparisons"].items()]
    registry.record(rows, {"exp": "EXP-v2", "decision": "D-068", "B": b_iter, "seeds": seeds})
    res = {"run_id": run_id, "dataset": ds, "role": role, "label": "EXPLORATORY (D-068)",
           "git_sha": git_sha(), "created_at": utc_now_iso(), "bootstrap": boot}
    p = get_paths().results / "eval" / f"v2-{ds}.json"
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res
