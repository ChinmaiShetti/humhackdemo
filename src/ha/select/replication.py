"""Replications R1 (docs/replication_prereg.md, D-069) and R2 (docs/replication2_prereg.md,
D-072): B0, B1 and OVL on fresh datasets. `tag` selects the freeze record and output names.

Selection reuses the frozen machinery unchanged: B0 and B1 as ha.select.sets (D-063), OVL as
ha.select.variants (D-068). No LLM judging. Eligible = U minus the seed's near-duplicates;
N = min(|A_s|, |eligible|), where |A_s| comes from the split labels only.

Evaluation follows prereg §7 in form: per-query nDCG@10, tau_b over C1-C7, paired bootstrap
(B = 1,000, seed 20260927; datasets in the given order, seeds ascending, sets in the order OVL,
B0, B1 with draws ascending); Delta-bar_X = mean_d mean_s [tau(OVL, A) - tau(X, A)].
R1 is supported iff p_B0 < 0.05 and p_B1 < 0.05.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ha.select.sets import N_DRAWS, STREAM, fit_strata, stratified_pick
from ha.select.variants import select_ovl

METHODS = ("OVL", "B0", "B1")
COMPARATORS = ("B0", "B1")


def select_seed(seed: int, eligible: list[int], u_emb: np.ndarray, r_emb: np.ndarray,
                u_overlap: np.ndarray, r_overlap: np.ndarray,
                n_a: int) -> tuple[dict[str, list[list[int]]], int]:
    """Pure selection for one split seed: {method: [draw positions in U]} and N."""
    n = min(n_a, len(eligible))
    out: dict[str, list[list[int]]] = {"OVL": select_ovl(seed, u_overlap, eligible, r_overlap, n),
                                       "B0": [], "B1": []}
    km = fit_strata(r_emb)
    pi = np.bincount(km.labels_, minlength=km.n_clusters) / len(r_emb)
    stratum = km.predict(u_emb)
    for d in range(N_DRAWS):
        rng = np.random.default_rng([STREAM["B0"], seed, d])
        out["B0"].append(rng.choice(eligible, size=n, replace=False).tolist())
        rng = np.random.default_rng([STREAM["B1"], seed, d])
        out["B1"].append(stratified_pick([eligible[j] for j in rng.permutation(len(eligible))],
                                         stratum, pi, n))
    return out, n


def out_path(ds: str, seed: int):
    from ha.paths import get_paths

    return get_paths().data / "select_rep" / ds / f"s{seed}" / "sets.parquet"


def run_select(ds: str, seeds=(0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    from ha.data.public import processed_dir
    from ha.data.splits import load_split
    from ha.eval.h1 import _stopwords, overlap
    from ha.gen.generate import load_source_corpus
    from ha.neighbours.retrieve import load_U
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.select.sets import a_size
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
        nb = pd.read_parquet(nb_path(ds, s, "U")).set_index("qid").loc[qids]
        eligible = [i for i, q in enumerate(qids) if not bool(nb.loc[q, "near_duplicate"])]
        r = load_split(ds, s, "R")                                      # R only, never A
        rq = qrels[qrels["qid"].isin(set(r["qid"]))]
        top = rq[rq["grade"] == rq.groupby("qid")["grade"].transform("max")]
        gold = top.groupby("qid")["pid"].apply(list)
        r_ov = np.array([max(overlap(t, passages.loc[p], stop) for p in gold.loc[q])
                         for q, t in zip(r["qid"], r["text"], strict=True)])
        picked, n = select_seed(s, eligible, u_emb, embed_queries(r["text"].tolist()),
                                u_ov, r_ov, a_size(ds, s))
        rows = [{"method": m, "draw": d, "qid": qids[i], "rank": j}
                for m in METHODS for d, draw in enumerate(picked[m]) for j, i in enumerate(draw)]
        p = out_path(ds, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(p, index=False)
        meta = {"dataset": ds, "split_seed": s, "N": n, "n_A": a_size(ds, s),
                "eligible": len(eligible), "near_duplicates": len(qids) - len(eligible),
                "R_overlap_mean": float(r_ov.mean()),
                **{f"{m}_overlap_mean": float(np.mean([u_ov[d].mean() for d in picked[m]]))
                   for m in METHODS},
                "git_sha": git_sha(), "created_at": utc_now_iso()}
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                               encoding="utf-8")
        metas.append(meta)
    return metas


def bootstrap(per_dataset: dict[str, list[dict]], b_iter: int = 1000,
              seed: int = 20260927) -> dict[str, Any]:
    """Pooled paired bootstrap of R1 (protocol §5). per_dataset: {ds: [seed inputs]}, each
    seed input {"A": matrix, "sets": [(method, draw, matrix), ...]} in the fixed set order."""
    from ha.select.variants import method_taus

    rng = np.random.default_rng(seed)
    dsets = list(per_dataset)

    def deltas(taus: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
        out = {}
        for x in COMPARATORS:
            per = {d: float(np.mean([t["OVL"] - t[x] for t in taus[d]])) for d in dsets}
            out[x] = {"pooled": float(np.mean(list(per.values()))), **per}
        return out

    point_t = {d: [method_taus(si) for si in per_dataset[d]] for d in dsets}
    point = deltas(point_t)
    reps = {x: {k: [] for k in ["pooled", *dsets]} for x in COMPARATORS}
    tau_reps = {d: {m: [] for m in METHODS} for d in dsets}
    for _ in range(b_iter):
        taus = {}
        for d in dsets:
            taus[d] = []
            for si in per_dataset[d]:
                idx_a = rng.integers(0, len(si["A"]), len(si["A"]))
                taus[d].append(method_taus(si, idx_a, rng))
        for x, vals in deltas(taus).items():
            for k, v in vals.items():
                reps[x][k].append(v)
        for d in dsets:
            for m in METHODS:
                tau_reps[d][m].append(float(np.mean([t[m] for t in taus[d]])))
    out: dict[str, Any] = {"B": b_iter, "seed": seed, "datasets": dsets, "comparisons": {},
                           "tau_seed_mean": {}, "per_seed": point_t}
    for x in COMPARATORS:
        out["comparisons"][f"OVL-{x}"] = {
            k: {"delta": point[x][k], "ci_low": float(np.nanpercentile(reps[x][k], 2.5)),
                "ci_high": float(np.nanpercentile(reps[x][k], 97.5)),
                "p_one_sided": float(np.mean(np.array(reps[x][k]) <= 0))}
            for k in ["pooled", *dsets]}
    for d in dsets:
        out["tau_seed_mean"][d] = {
            m: {"tau_b": float(np.mean([t[m] for t in point_t[d]])),
                "ci_low": float(np.nanpercentile(tau_reps[d][m], 2.5)),
                "ci_high": float(np.nanpercentile(tau_reps[d][m], 97.5))} for m in METHODS}
    c = out["comparisons"]
    out["R1_supported"] = bool(c["OVL-B0"]["pooled"]["p_one_sided"] < 0.05
                               and c["OVL-B1"]["pooled"]["p_one_sided"] < 0.05)
    return out


FREEZE_NAME = {"R1": "FROZEN_replication.json", "R2": "FROZEN_replication2.json"}
PROTOCOL = {"R1": "docs/replication_prereg.md", "R2": "docs/replication2_prereg.md"}


def write_freeze(datasets=("cqa_webmasters", "nfcorpus"), seeds=(0, 1, 2, 3, 4),
                 tag: str = "R1") -> dict:
    """Protocol §6 step 3: hashes of pools, neighbours, sets and U runs + the code commit."""
    from ha.freeze import sha256_file
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.paths import get_paths
    from ha.systems.run_all import run_path
    from ha.utils import git_sha, utc_now_iso

    sha = git_sha()
    if sha.endswith("-dirty"):
        raise RuntimeError("commit first: the replication freeze needs a clean commit")
    data = get_paths().data
    rec: dict[str, Any] = {"protocol": PROTOCOL[tag], "tag": tag,
                           "git_sha": sha, "created_at": utc_now_iso(), "datasets": {}}
    for ds in datasets:
        e = {"pools": {p.parent.name: sha256_file(p)
                       for p in sorted((data / "synthetic" / ds).glob("*/queries.parquet"))},
             "neighbours": {f"s{s}": sha256_file(nb_path(ds, s, "U")) for s in seeds},
             "sets": {f"s{s}": sha256_file(out_path(ds, s)) for s in seeds},
             "runs_U": {c: sha256_file(run_path(ds, c, "U"))
                        for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")}}
        rec["datasets"][ds] = e
    p = get_paths().results / FREEZE_NAME[tag]
    if p.exists():
        raise RuntimeError(f"{p.name} already exists; a replication freeze is written once")
    p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def run_eval(datasets=("cqa_webmasters", "nfcorpus"), seeds=(0, 1, 2, 3, 4),
             b_iter: int = 1000, tag: str = "R1") -> dict[str, Any]:
    from ha import registry
    from ha.data.public import dataset_config
    from ha.data.splits import load_split
    from ha.eval.sets import perquery_table
    from ha.paths import get_paths
    from ha.utils import git_sha, utc_now_iso

    if git_sha().endswith("-dirty"):
        raise RuntimeError("commit first: registry numbers need a clean git sha")
    if not (get_paths().results / FREEZE_NAME[tag]).exists():
        raise RuntimeError("write and commit the replication freeze first (protocol §6)")
    out_file = get_paths().results / "eval" / f"{tag}-replication.json"
    if out_file.exists():
        raise RuntimeError(f"{out_file.name} exists: the replication is evaluated once")
    per = {}
    for ds in datasets:
        auth, synth = perquery_table(ds, "auth"), perquery_table(ds, "U")
        per[ds] = []
        for s in seeds:
            a_q = load_split(ds, s, "A")["qid"].tolist()
            sets = pd.read_parquet(out_path(ds, s))
            groups = [(m, int(d), synth.loc[g["qid"]].to_numpy())
                      for m in METHODS
                      for d, g in sets[sets["method"] == m].groupby("draw", sort=True)]
            per[ds].append({"seed": s, "A": auth.loc[a_q].to_numpy(), "sets": groups})
    boot = bootstrap(per, b_iter)
    run_id = registry.new_run_id(tag)
    ids = {d: dataset_config(d)["id"] for d in datasets}
    role = f"confirmatory replication ({PROTOCOL[tag]})"
    rows = []
    for comp, by in boot["comparisons"].items():
        for k, v in by.items():
            dataset = "pooled:" + "+".join(ids.values()) if k == "pooled" else ids[k]
            rows.append({"run_id": run_id, "exp": tag, "dataset": dataset, "split_seed": None,
                         "method": comp, "system_set": "C1-C7", "metric": "delta_tau_b",
                         "value": v["delta"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_queries": len(per[datasets[0]][0]["sets"][0][2]),
                         "p": v["p_one_sided"], "role": role})
    for d, by in boot["tau_seed_mean"].items():
        for m, v in by.items():
            rows.append({"run_id": run_id, "exp": tag, "dataset": ids[d], "split_seed": None,
                         "method": m, "system_set": "C1-C7", "metric": "tau_b_seed_mean",
                         "value": v["tau_b"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_queries": len(per[d][0]["sets"][0][2]), "role": role})
    registry.record(rows, {"exp": tag, "protocol": PROTOCOL[tag],
                           "B": b_iter, "seeds": seeds, "datasets": list(datasets)})
    res = {"run_id": run_id, "git_sha": git_sha(), "created_at": utc_now_iso(),
           "protocol": PROTOCOL[tag], "bootstrap": boot}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    return res
