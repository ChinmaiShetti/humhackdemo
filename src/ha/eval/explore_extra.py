"""[EXPLORATORY] E-density, E-metrics and E-dims (D-071; specified before they were run).

All three reuse existing retrieval runs, query sets and judgements. Bootstrap as prereg §7
(B = 1,000, seed 20260927; A resampled once per seed, every set independently, draws averaged
at the tau level), per dataset, no pooling. Outputs: results/eval/explore_<name>.json and
registry rows (exp EXP-<name>, role exploratory). Held-out A needs HA_UNSEAL=heldout_A.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ha.select.variants import method_taus

SYSTEMS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
B_ITER, SEED = 1000, 20260927


# ----------------------------------------------------------------------------- inputs
def sparse_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    """One passage per query: the highest grade, ties by the smallest sha256(pid) (D-071)."""
    q = qrels.copy()
    q["_h"] = [hashlib.sha256(str(p).encode("utf-8")).hexdigest() for p in q["pid"]]
    q = q.sort_values(["qid", "grade", "_h"], ascending=[True, False, True])
    return q.drop_duplicates("qid", keep="first").drop(columns="_h").reset_index(drop=True)


def perquery(ds: str, qset: str, metric: str, qrels: pd.DataFrame | None = None) -> pd.DataFrame:
    """qid x C1..C7 for one metric; qrels default: official ('auth') or source passage ('U')."""
    from ha.data.public import dataset_config, processed_dir
    from ha.eval.metrics import per_query
    from ha.eval.sets import synthetic_qrels
    from ha.seal import require_unsealed
    from ha.systems.run_all import run_path

    if qrels is None:
        if qset == "auth":
            if dataset_config(ds)["role"] == "heldout":
                require_unsealed("heldout_A", what=f"authentic per-query {metric}", dataset=ds)
            qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
        else:
            qrels = synthetic_qrels(ds)
    return pd.DataFrame({c: per_query(pd.read_parquet(run_path(ds, c, qset)), qrels, [metric])
                         .set_index("qid")[metric] for c in SYSTEMS})


def set_table(ds: str, seed: int, sources: tuple[str, ...]) -> pd.DataFrame:
    """Selected sets from 'frozen' (M1, M1-iso, B0, B1), 'v2' (OVL, NN), 'rep' (OVL, B0, B1)."""
    from ha.select import replication, sets, variants

    paths = {"frozen": sets.out_path, "v2": variants.out_path, "rep": replication.out_path}
    return pd.concat([pd.read_parquet(paths[s](ds, seed)) for s in sources], ignore_index=True)


def seed_inputs(a_tab: pd.DataFrame, u_tab: pd.DataFrame, a_qids: list[str],
                sets: pd.DataFrame, methods: tuple[str, ...]) -> dict:
    groups = [(m, int(d), u_tab.loc[g["qid"]].to_numpy())
              for m in methods
              for d, g in sets[sets["method"] == m].groupby("draw", sort=True)]
    return {"A": a_tab.loc[a_qids].to_numpy(), "sets": groups}


def bootstrap(inputs: list[dict], comparisons: list[tuple[str, str]],
              b_iter: int = B_ITER, seed: int = SEED) -> dict[str, Any]:
    """Per-dataset paired bootstrap; methods = whatever the inputs hold."""
    rng = np.random.default_rng(seed)
    point = [method_taus(si) for si in inputs]
    methods = list(point[0])
    reps = {f"{a}-{b}": [] for a, b in comparisons}
    tau_reps = {m: [] for m in methods}
    for _ in range(b_iter):
        ts = []
        for si in inputs:
            idx_a = rng.integers(0, len(si["A"]), len(si["A"]))
            ts.append(method_taus(si, idx_a, rng))
        for a, b in comparisons:
            reps[f"{a}-{b}"].append(float(np.mean([t[a] - t[b] for t in ts])))
        for m in methods:
            tau_reps[m].append(float(np.mean([t[m] for t in ts])))
    return {
        "tau_seed_mean": {m: {"tau_b": float(np.mean([t[m] for t in point])),
                              "ci_low": float(np.nanpercentile(tau_reps[m], 2.5)),
                              "ci_high": float(np.nanpercentile(tau_reps[m], 97.5))}
                          for m in methods},
        "comparisons": {f"{a}-{b}": {"delta": float(np.mean([t[a] - t[b] for t in point])),
                                     "ci_low": float(np.nanpercentile(reps[f"{a}-{b}"], 2.5)),
                                     "ci_high": float(np.nanpercentile(reps[f"{a}-{b}"], 97.5)),
                                     "p_one_sided": float(np.mean(np.array(reps[f"{a}-{b}"])
                                                                  <= 0))}
                        for a, b in comparisons},
        "per_seed": point, "n": int(len(inputs[0]["sets"][0][2]))}


def _record(name: str, results: dict[str, dict], extra_cfg: dict) -> dict[str, Any]:
    from ha import registry
    from ha.data.public import dataset_config
    from ha.paths import get_paths
    from ha.utils import git_sha, utc_now_iso

    run_id = registry.new_run_id(f"EXP-{name}")
    rows = []
    undefined = []          # tau_b undefined: every system tied for the set or for A (no row)
    for key, r in results.items():
        ds, label = key.split("|") if "|" in key else (key, "")
        did = dataset_config(ds)["id"]
        for m, v in r["tau_seed_mean"].items():
            if not np.isfinite(v["tau_b"]):
                undefined.append(f"{key}:{m}")
                continue
            rows.append({"run_id": run_id, "exp": f"EXP-{name}", "dataset": did,
                         "split_seed": None, "method": m, "system_set": "C1-C7",
                         "metric": f"tau_b_seed_mean{('/' + label) if label else ''}",
                         "value": v["tau_b"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_queries": r["n"], "role": "exploratory (D-071)"})
        for c, v in r["comparisons"].items():
            if not all(np.isfinite([v["delta"], v["ci_low"], v["ci_high"]])):
                undefined.append(f"{key}:{c}")
                continue
            rows.append({"run_id": run_id, "exp": f"EXP-{name}", "dataset": did,
                         "split_seed": None, "method": c, "system_set": "C1-C7",
                         "metric": f"delta_tau_b{('/' + label) if label else ''}",
                         "value": v["delta"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_queries": r["n"], "p": v["p_one_sided"],
                         "role": "exploratory (D-071)"})
    registry.record(rows, {"exp": f"EXP-{name}", "decision": "D-071", **extra_cfg})
    out = {"run_id": run_id, "label": "EXPLORATORY (D-071)", "git_sha": git_sha(),
           "created_at": utc_now_iso(), "undefined_tau": undefined, "results": results}
    p = get_paths().results / "eval" / f"explore_{name}.json"
    p.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def _clean():
    from ha.utils import git_sha

    if git_sha().endswith("-dirty"):
        raise RuntimeError("commit first: registry numbers need a clean git sha")


# ----------------------------------------------------------------------------- analyses
def run_density(datasets=("cqa_webmasters", "nfcorpus"), seeds=(0, 1, 2, 3, 4)) -> dict:
    from ha.data.public import dataset_config, processed_dir
    from ha.data.splits import load_split
    from ha.seal import require_unsealed

    _clean()
    results = {}
    for ds in datasets:
        if dataset_config(ds)["role"] == "heldout":
            require_unsealed("heldout_A", what="sparsified authentic qrels", dataset=ds)
        sq = sparse_qrels(pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet"))
        a_tab = perquery(ds, "auth", "ndcg@10", sq)
        u_tab = perquery(ds, "U", "ndcg@10")
        inputs = [seed_inputs(a_tab, u_tab, load_split(ds, s, "A")["qid"].tolist(),
                              set_table(ds, s, ("rep",)), ("OVL", "B0", "B1")) for s in seeds]
        results[ds] = bootstrap(inputs, [("OVL", "B0"), ("OVL", "B1")])
        results[ds]["qrels_per_query"] = 1.0
    return _record("density", results, {"qrels": "sparse: max grade, sha256(pid) tie-break"})


def run_metrics(metrics=("mrr@10", "recall@100", "hit_rate@5"), seeds=(0, 1, 2, 3, 4)) -> dict:
    from ha.data.splits import load_split

    _clean()
    plan = {"fiqa": (("frozen", "v2"), ("M1", "M1-iso", "B0", "B1", "OVL", "NN")),
            "cqa_android": (("frozen", "v2"), ("M1", "M1-iso", "B0", "B1", "OVL", "NN")),
            "hiaq": (("frozen", "v2"), ("M1", "M1-iso", "B0", "B1", "OVL", "NN")),
            "cqa_webmasters": (("rep",), ("OVL", "B0", "B1")),
            "nfcorpus": (("rep",), ("OVL", "B0", "B1"))}
    results = {}
    for ds, (sources, methods) in plan.items():
        comps = [(a, b) for a, b in (("M1", "B0"), ("M1", "B1"), ("OVL", "B0"), ("OVL", "B1"))
                 if a in methods and b in methods]
        for metric in metrics:
            a_tab, u_tab = perquery(ds, "auth", metric), perquery(ds, "U", metric)
            inputs = [seed_inputs(a_tab, u_tab, load_split(ds, s, "A")["qid"].tolist(),
                                  set_table(ds, s, sources), methods) for s in seeds]
            results[f"{ds}|{metric}"] = bootstrap(inputs, comps)
    return _record("metrics", results, {"metrics": list(metrics)})


def dims_selection(seed: int, qids: list[str], u_emb: np.ndarray, r_emb: np.ndarray,
                   m1: pd.DataFrame, dim: str, n: int) -> list[int]:
    """SEL-<dim>: the M1 procedure with one judge dimension as the score (D-071)."""
    from ha.select.sets import DEDUPE_COS, STREAM, fit_strata, stratified_pick

    km = fit_strata(r_emb)
    pi = np.bincount(km.labels_, minlength=km.n_clusters) / len(r_emb)
    stratum = km.predict(u_emb)
    tie = np.random.default_rng([STREAM["tiebreak"], seed]).random(len(qids))
    ok = m1[m1["ok"] & (m1["F"] >= 4)].set_index("qid")[dim]
    pos = {q: i for i, q in enumerate(qids)}
    cand = [pos[q] for q in ok.index]
    order = sorted(cand, key=lambda i: (-float(ok.loc[qids[i]]), tie[i]))
    return stratified_pick(order, stratum, pi, n, u_emb, DEDUPE_COS)


def run_dims(datasets=("fiqa", "cqa_android", "hiaq"), seeds=(0, 1, 2, 3, 4)) -> dict:
    from ha.data.splits import load_split
    from ha.judge.bulk import stage_log
    from ha.neighbours.retrieve import load_U
    from ha.select import sets as S
    from ha.systems.retrieval import embed_queries

    _clean()
    dims = ("F", "H", "S", "P")
    results = {}
    for ds in datasets:
        u = load_U(ds)
        qids = u["qid"].tolist()
        u_emb = embed_queries(u["text"].tolist())
        a_tab, u_tab = perquery(ds, "auth", "ndcg@10"), perquery(ds, "U", "ndcg@10")
        inputs = []
        for s in seeds:
            frozen = pd.read_parquet(S.out_path(ds, s))
            n = int(frozen[frozen["method"] == "B0"].groupby("draw").size().iloc[0])
            m1 = pd.read_parquet(stage_log(ds, s).with_suffix(".parquet"))
            r_emb = embed_queries(load_split(ds, s, "R")["text"].tolist())
            rows = [frozen[frozen["method"].isin(["M1", "B0"])]]
            for d in dims:
                picked = dims_selection(s, qids, u_emb, r_emb, m1, d, n)
                rows.append(pd.DataFrame({"method": f"SEL-{d}", "draw": 0,
                                          "qid": [qids[i] for i in picked]}))
            sets = pd.concat(rows, ignore_index=True)
            inputs.append(seed_inputs(a_tab, u_tab, load_split(ds, s, "A")["qid"].tolist(),
                                      sets, ("M1", "B0", *[f"SEL-{d}" for d in dims])))
        comps = [(f"SEL-{d}", ref) for d in dims for ref in ("B0", "M1")]
        results[ds] = bootstrap(inputs, comps)
    return _record("dims", results, {"dims": list(dims)})
