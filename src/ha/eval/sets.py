"""Agreement of the query sets with authentic A, P5.3 / P6 (prereg §7 as amended by A1; A3).

Per dataset d and split seed s, each query set X (B0 and B1: 20 draws each; M1-iso; M1) scores
every configuration C1-C7 by its mean per-query nDCG@10, and A_s does the same with the official
qrels. Agreement = Kendall tau-b between the two 7-system score vectors; B0 and B1 report the mean
over their draws. Synthetic queries are scored against their source passage, grade 1 (§5.4).

Bootstrap (§7): B = 1,000, numpy seed 20260927. Each iteration resamples A_{d,s} once (shared by
every method) and every set independently at its own size; B0 and B1 average tau-b over their
draws. Delta-bar = mean_d mean_s [tau(M1, A) - tau(X, A)] for X in {B0, B1} (H2) and M1-iso (H3);
p = share of iterations with Delta-bar <= 0; 95% percentile CI.
Also reported (§7.5): tau_AP, winner agreement, top-3 overlap, MAE and signed bias.
Held-out A is reached only through the sealed loader. D1 rows are development.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

CONFIGS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
METRIC = "ndcg@10"
B_ITER, BOOT_SEED = 1000, 20260927
COMPARATORS = ("B0", "B1", "M1-iso")              # Delta = tau(M1) - tau(comparator)
_PAIRS = list(combinations(range(len(CONFIGS)), 2))
_I, _J = np.array([p[0] for p in _PAIRS]), np.array([p[1] for p in _PAIRS])


# ----------------------------------------------------------------------------- agreement
def tau_b_rows(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Kendall tau-b of each row of x (n_sets x n_sys) against a (n_sys); nan if undefined.
    Equals scipy.stats.kendalltau(variant='b') (tested)."""
    x = np.atleast_2d(x)
    sx, sa = np.sign(x[:, _I] - x[:, _J]), np.sign(a[_I] - a[_J])
    n0 = len(_PAIRS)
    den = np.sqrt((n0 - (sx == 0).sum(axis=1)) * (n0 - (sa == 0).sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sx * sa).sum(axis=1) / den, np.nan)


def tau_ap(x: np.ndarray, a: np.ndarray) -> float:
    """AP correlation (Yilmaz et al., SIGIR 2008) of ranking x against the reference a."""
    order = np.argsort(-x, kind="stable")
    n = len(order)
    total = 0.0
    for i in range(1, n):
        item = order[i]
        total += sum(a[order[j]] > a[item] for j in range(i)) / i
    return 2 * total / (n - 1) - 1


def describe(x: np.ndarray, a: np.ndarray) -> dict[str, float]:
    top3 = len(set(np.argsort(-x)[:3]) & set(np.argsort(-a)[:3])) / 3
    return {"tau_b": float(tau_b_rows(x, a)[0]), "tau_ap": tau_ap(x, a),
            "winner_agrees": float(np.argmax(x) == np.argmax(a)), "top3_overlap": top3,
            "mae": float(np.abs(x - a).mean()), "signed_bias": float((x - a).mean())}


# ----------------------------------------------------------------------------- inputs
def synthetic_qrels(ds: str) -> pd.DataFrame:
    from ha.neighbours.retrieve import load_U

    u = load_U(ds)
    return pd.DataFrame({"qid": u["qid"], "pid": u["source_pid"], "grade": 1})


def perquery_table(ds: str, qset: str) -> pd.DataFrame:
    """qid x C1..C7 nDCG@10. 'auth' uses the official qrels (held-out: sealed loader first)."""
    from ha.data.public import dataset_config, processed_dir
    from ha.eval.metrics import per_query
    from ha.seal import require_unsealed
    from ha.systems.run_all import run_path

    if qset == "auth":
        if dataset_config(ds)["role"] == "heldout":
            require_unsealed("heldout_A", what="authentic per-query metrics", dataset=ds)
        qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
    elif qset == "U":
        qrels = synthetic_qrels(ds)
    else:
        raise ValueError(qset)
    cols = {}
    for cid in CONFIGS:
        p = run_path(ds, cid, qset)
        if not p.exists():
            raise FileNotFoundError(f"{ds}/{cid}/{qset}: run missing "
                                    f"(ha systems run {ds} --qset {qset})")
        cols[cid] = per_query(pd.read_parquet(p), qrels, [METRIC]).set_index("qid")[METRIC]
    return pd.DataFrame(cols)


def load_inputs(ds: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Per seed: the A matrix and every set's matrix (rows = queries, cols = C1..C7)."""
    from ha.data.splits import load_split
    from ha.select.sets import out_path

    auth, synth = perquery_table(ds, "auth"), perquery_table(ds, "U")
    out = []
    for s in seeds:
        a_q = load_split(ds, s, "A")["qid"].tolist()            # sealed for held-out datasets
        sets = pd.read_parquet(out_path(ds, s))
        groups = []
        for method in ("M1", "M1-iso", "B0", "B1"):
            for draw, g in sets[sets["method"] == method].groupby("draw", sort=True):
                groups.append((method, int(draw), synth.loc[g["qid"]].to_numpy()))
        out.append({"seed": s, "A": auth.loc[a_q].to_numpy(), "sets": groups})
    return out


# ----------------------------------------------------------------------------- analysis
def _method_taus(seed_in: dict, idx_a=None, rng=None) -> dict[str, float]:
    """tau_b per method for one seed: B0/B1 averaged over draws. With rng: one bootstrap
    replicate (A resampled by idx_a, each set resampled independently, in fixed order)."""
    a_mat = seed_in["A"]
    a = (a_mat if idx_a is None else a_mat[idx_a]).mean(axis=0)
    scores, names = [], []
    for method, _draw, m in seed_in["sets"]:
        rows = m if rng is None else m[rng.integers(0, len(m), len(m))]
        scores.append(rows.mean(axis=0))
        names.append(method)
    taus = tau_b_rows(np.array(scores), a)
    out = {}
    for method in ("M1", "M1-iso", "B0", "B1"):
        out[method] = float(np.nanmean([t for t, n in zip(taus, names, strict=True)
                                        if n == method]))
    return out


def point_estimates(inputs: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Per seed and method: tau_b and the §7.5 measures (B0/B1 averaged over draws), plus the
    7-system score vectors for the record."""
    rows, vectors = [], []
    for si in inputs:
        a = si["A"].mean(axis=0)
        vectors.append({"seed": si["seed"], "set": "A", "draw": None,
                        **dict(zip(CONFIGS, a.round(5).tolist(), strict=True))})
        per: dict[str, list[dict]] = {}
        for method, draw, m in si["sets"]:
            x = m.mean(axis=0)
            per.setdefault(method, []).append(describe(x, a))
            vectors.append({"seed": si["seed"], "set": method, "draw": draw,
                            **dict(zip(CONFIGS, x.round(5).tolist(), strict=True))})
        for method, ds_ in per.items():
            rows.append({"seed": si["seed"], "method": method, "n_draws": len(ds_),
                         "n_queries": len(si["sets"][0][2]),
                         **{k: float(np.nanmean([d[k] for d in ds_])) for k in ds_[0]}})
    return pd.DataFrame(rows), vectors


def bootstrap(per_dataset: dict[str, list[dict]], b_iter: int = B_ITER,
              seed: int = BOOT_SEED) -> dict[str, Any]:
    """§7 paired bootstrap. Returns, per comparator, the point Delta-bar (pooled over datasets
    and seeds), its 95% percentile CI, the one-sided p (share of Delta-bar <= 0), and the same
    per dataset (secondary breakdowns)."""
    rng = np.random.default_rng(seed)
    dsets = list(per_dataset)

    def deltas(taus_by_ds: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
        out = {}
        for comp in COMPARATORS:
            per_ds = {d: float(np.mean([t["M1"] - t[comp] for t in taus_by_ds[d]]))
                      for d in dsets}
            out[comp] = {"pooled": float(np.mean(list(per_ds.values()))), **per_ds}
        return out

    methods = ("M1", "M1-iso", "B0", "B1")

    def seed_mean(taus_by_ds: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
        return {d: {m: float(np.mean([t[m] for t in taus_by_ds[d]])) for m in methods}
                for d in dsets}

    point_taus = {d: [_method_taus(si) for si in per_dataset[d]] for d in dsets}
    point, point_tau = deltas(point_taus), seed_mean(point_taus)
    reps: dict[str, dict[str, list[float]]] = {c: {k: [] for k in ["pooled", *dsets]}
                                               for c in COMPARATORS}
    tau_reps: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in dsets}
    for _ in range(b_iter):
        taus_by_ds = {}
        for d in dsets:
            taus_by_ds[d] = []
            for si in per_dataset[d]:
                idx_a = rng.integers(0, len(si["A"]), len(si["A"]))
                taus_by_ds[d].append(_method_taus(si, idx_a, rng))
        for comp, vals in deltas(taus_by_ds).items():
            for k, v in vals.items():
                reps[comp][k].append(v)
        for d, by_m in seed_mean(taus_by_ds).items():
            for m, v in by_m.items():
                tau_reps[d][m].append(v)
    out: dict[str, Any] = {"B": b_iter, "seed": seed, "datasets": dsets, "comparisons": {}}
    out["tau_seed_mean"] = {d: {m: {"tau_b": point_tau[d][m],
                                    "ci_low": float(np.nanpercentile(tau_reps[d][m], 2.5)),
                                    "ci_high": float(np.nanpercentile(tau_reps[d][m], 97.5))}
                                for m in methods} for d in dsets}
    for comp in COMPARATORS:
        out["comparisons"][f"M1-{comp}"] = {
            k: {"delta": point[comp][k],
                "ci_low": float(np.nanpercentile(reps[comp][k], 2.5)),
                "ci_high": float(np.nanpercentile(reps[comp][k], 97.5)),
                "p_one_sided": float(np.mean(np.array(reps[comp][k]) <= 0))}
            for k in ["pooled", *dsets]}
    h2 = out["comparisons"]
    out["H2_supported"] = bool(h2["M1-B0"]["pooled"]["p_one_sided"] < 0.05
                               and h2["M1-B1"]["pooled"]["p_one_sided"] < 0.05)
    out["H3_supported"] = bool(h2["M1-M1-iso"]["pooled"]["p_one_sided"] < 0.05)
    return out


def run_eval(datasets: list[str], seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
             b_iter: int = B_ITER) -> dict[str, Any]:
    """Point estimates per dataset and the pooled bootstrap; results/eval/<label>.json and the
    registry. A development dataset is labelled 'dev'; it never enters a held-out pool."""
    from ha import registry
    from ha.data.public import dataset_config
    from ha.paths import get_paths
    from ha.utils import git_sha, utc_now_iso

    if git_sha().endswith("-dirty"):
        raise RuntimeError("commit first: registry numbers need a clean git sha")
    roles = {d: dataset_config(d)["role"] for d in datasets}
    if len(set(roles.values())) > 1:
        raise ValueError(f"do not pool development and held-out datasets: {roles}")
    label = next(iter(roles.values())) + "-" + "+".join(datasets)   # dev | heldout | exploratory
    inputs = {d: load_inputs(d, seeds) for d in datasets}
    points, vectors = {}, {}
    for d in datasets:
        points[d], vectors[d] = point_estimates(inputs[d])
    boot = bootstrap(inputs, b_iter)
    run_id = registry.new_run_id(f"P5-eval-{label}")
    rows = []
    for d in datasets:
        did = dataset_config(d)["id"]
        for r in points[d].to_dict("records"):
            for m in ("tau_b", "tau_ap", "winner_agrees", "top3_overlap", "mae", "signed_bias"):
                rows.append({"run_id": run_id, "exp": f"P5-eval-{label}", "dataset": did,
                             "split_seed": int(r["seed"]), "method": r["method"],
                             "system_set": "C1-C7", "metric": m, "value": float(r[m]),
                             "ci_low": None, "ci_high": None, "n_queries": int(r["n_queries"]),
                             "role": roles[d]})
    n_set = {d: len(inputs[d][0]["sets"][0][2]) for d in datasets}       # N of split seed 0
    for comp, by in boot["comparisons"].items():
        for k, v in by.items():
            did = "pooled:" + "+".join(dataset_config(d)["id"] for d in datasets) \
                if k == "pooled" else dataset_config(k)["id"]
            n_q = sum(n_set.values()) if k == "pooled" else n_set[k]
            for metric, val, lo, hi in (("delta_tau_b", v["delta"], v["ci_low"], v["ci_high"]),
                                        ("p_one_sided", v["p_one_sided"], None, None)):
                rows.append({"run_id": run_id, "exp": f"P5-eval-{label}", "dataset": did,
                             "split_seed": None, "method": comp, "system_set": "C1-C7",
                             "metric": metric, "value": float(val), "ci_low": lo, "ci_high": hi,
                             "n_queries": n_q, "role": "/".join(sorted(set(roles.values())))})
    registry.record(rows, {"metric": METRIC, "B": b_iter, "seed": BOOT_SEED, "seeds": seeds,
                           "datasets": datasets})
    measures = ("tau_b", "tau_ap", "winner_agrees", "top3_overlap", "mae", "signed_bias")
    summary = {d: {m: {k: {"mean": round(float(g[k].mean()), 4),
                           "sd": round(float(g[k].std(ddof=1)), 4) if len(g) > 1 else None}
                       for k in measures}
                   for m, g in points[d].groupby("method")} for d in datasets}
    result = {"run_id": run_id, "label": label, "git_sha": git_sha(), "created_at": utc_now_iso(),
              "metric": METRIC, "seeds": list(seeds), "roles": roles,
              "per_seed": {d: points[d].to_dict("records") for d in datasets},
              "summary": summary, "bootstrap": boot, "score_vectors": vectors}
    out = get_paths().results / "eval" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result
