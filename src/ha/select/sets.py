"""Query-set selection, P5.2 (prereg §5.2 as amended by A3; tie-break and rounding: D-063).

For a dataset and split seed s, every method gets the same size
  N = min(|A_s|, M1 gate passes, M1-iso gate passes),
counted over U minus the seed's near-duplicates (nn_max_cos > 0.97), which are excluded from
every set. |A_s| is read from the split labels alone; A's queries are never loaded.

  B0      a uniform random draw of N from U (20 draws);
  B1      stratified random: R_s-derived strata and quotas, random within a stratum (20 draws);
  M1-iso  the gate F >= 4 on the no-neighbour judgement, then stratified greedy selection;
  M1      the gate F >= 4 on the seed-s neighbour-grounded judgement, then the same.

Strata: k-means (k = min(10, |R_s| // 15), seed 0, n_init 10) on the bge-m3 embeddings of R_s;
each U query joins its nearest centroid. Quotas n_k = N * pi_k (pi_k = share of R_s in stratum
k), rounded by largest remainder so they sum to N; a stratum that runs short passes its
shortfall to the others in proportion to pi. Greedy = descending equal-weight score (A3), ties
broken by a seeded random key; a candidate whose cosine with an already selected query exceeds
0.90 is skipped. A judge failure is never selectable. B0 and B1 have no dedupe (prereg §5.2
lists it under the judge-based selection only).
Output: HA_DATA/select/<ds>/s<seed>/sets.parquet (method, draw, qid, stratum, rank) + .meta.json
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from ha.judge.aggregate import EQUAL_WEIGHTS, score

DEDUPE_COS = 0.90
N_DRAWS = 20
MAX_K, R_PER_K = 10, 15
KMEANS_SEED, KMEANS_N_INIT = 0, 10
STREAM = {"B0": 0, "B1": 1, "tiebreak": 2}      # numpy SeedSequence entropy: [stream, seed, draw]
METHODS = ("B0", "B1", "M1-iso", "M1")


def n_strata(n_r: int) -> int:
    return max(1, min(MAX_K, n_r // R_PER_K))


def fit_strata(r_emb: np.ndarray):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_strata(len(r_emb)), random_state=KMEANS_SEED,
                  n_init=KMEANS_N_INIT).fit(r_emb)


def allocate(n: int, pi: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Integer quotas summing to min(n, cap.sum()): largest remainder on n * pi, each capped at
    its stratum's capacity; a capped stratum's shortfall goes to the others in proportion to pi.
    Ties in the remainder go to the lower stratum index (deterministic)."""
    pi, cap = np.asarray(pi, float), np.asarray(cap, int)
    quota = np.zeros(len(pi), int)
    left = min(int(n), int(cap.sum()))
    while left > 0:
        room = cap - quota
        w = np.where(room > 0, pi, 0.0)
        if w.sum() == 0:                       # strata with pi = 0 but room: spread evenly
            w = (room > 0).astype(float)
        share = left * w / w.sum()
        add = np.floor(share).astype(int)
        rest = left - int(add.sum())
        if rest:
            frac = np.where(room > 0, share - add, -1.0)
            for i in sorted(range(len(pi)), key=lambda i: (-frac[i], i))[:rest]:
                add[i] += 1
        add = np.minimum(add, room)
        if add.sum() == 0:                     # cannot happen with room > 0; guard anyway
            add[int(np.argmax(room))] = 1
        quota += add
        left -= int(add.sum())
    return quota


def stratified_pick(order: list[int], stratum: np.ndarray, pi: np.ndarray, n: int,
                    emb: np.ndarray | None = None, dedupe: float | None = None) -> list[int]:
    """Walk `order` (candidate positions, best first) and fill each stratum's quota. With
    `dedupe`, skip a candidate whose cosine with any selected one exceeds it. When strata run
    dry, their shortfall is re-allocated (in proportion to pi) and the walk repeats."""
    k = len(pi)
    cap = np.bincount(stratum[order], minlength=k) if order else np.zeros(k, int)
    quota = allocate(n, pi, cap)
    chosen: list[int] = []
    taken = np.zeros(k, int)
    gone: set[int] = set()
    while True:
        for i in order:
            if i in gone:
                continue
            s = int(stratum[i])
            if taken[s] >= quota[s]:
                continue
            gone.add(i)
            if dedupe is not None and chosen and float((emb[chosen] @ emb[i]).max()) > dedupe:
                continue
            chosen.append(i)
            taken[s] += 1
        short = n - len(chosen)
        left = np.bincount(stratum[[i for i in order if i not in gone]], minlength=k) \
            if len(gone) < len(order) else np.zeros(k, int)
        if short <= 0 or left.sum() == 0:
            return chosen
        quota = taken + allocate(short, pi, left)


def judge_scores(df: pd.DataFrame) -> pd.Series:
    """Equal-weight score (A3) per qid; -inf for a failure or a gate fail."""
    out = {}
    for r in df.to_dict("records"):
        dims = {d: (None if r[d] is None or (isinstance(r[d], float) and math.isnan(r[d]))
                    else float(r[d])) for d in "FHSPC"}
        out[r["qid"]] = score(dims, EQUAL_WEIGHTS) if r["ok"] else -math.inf
    return pd.Series(out, dtype=float)


def select_seed(seed: int, u: pd.DataFrame, u_emb: np.ndarray, r_emb: np.ndarray,
                near: set[str], iso: pd.DataFrame, m1: pd.DataFrame,
                n_a: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure selection for one split seed. u: qid in fixed U order (aligned with u_emb)."""
    qids = u["qid"].tolist()
    km = fit_strata(r_emb)
    k = km.n_clusters
    pi = np.bincount(km.labels_, minlength=k) / len(r_emb)
    stratum = km.predict(u_emb)
    eligible = [i for i, q in enumerate(qids) if q not in near]
    s_iso, s_m1 = judge_scores(iso), judge_scores(m1)
    passes = {m: [i for i in eligible if np.isfinite(sc.get(qids[i], -math.inf))]
              for m, sc in (("M1-iso", s_iso), ("M1", s_m1))}
    n = min(n_a, len(passes["M1"]), len(passes["M1-iso"]))
    binding = [name for name, v in (("A", n_a), ("M1", len(passes["M1"])),
                                    ("M1-iso", len(passes["M1-iso"]))) if v == n]
    tie = np.random.default_rng([STREAM["tiebreak"], seed]).random(len(qids))
    rows: list[dict] = []
    meta: dict[str, Any] = {"split_seed": seed, "N": n, "n_A": n_a, "binding": binding,
                            "n_U": len(qids), "near_duplicates": len(near),
                            "M1_passes": len(passes["M1"]),
                            "M1-iso_passes": len(passes["M1-iso"]),
                            "k": k, "pi": [round(float(p), 4) for p in pi], "methods": {}}

    def emit(method: str, draw: int, picked: list[int]) -> None:
        rows.extend({"method": method, "draw": draw, "qid": qids[i], "stratum": int(stratum[i]),
                     "rank": j} for j, i in enumerate(picked))

    for d in range(N_DRAWS):
        rng = np.random.default_rng([STREAM["B0"], seed, d])
        emit("B0", d, rng.choice(eligible, size=n, replace=False).tolist())
        rng = np.random.default_rng([STREAM["B1"], seed, d])
        emit("B1", d, stratified_pick([eligible[j] for j in rng.permutation(len(eligible))],
                                      stratum, pi, n))
    for method, sc in (("M1-iso", s_iso), ("M1", s_m1)):
        cand = passes[method]
        order = sorted(cand, key=lambda i: (-sc[qids[i]], tie[i]))
        picked = stratified_pick(order, stratum, pi, n, u_emb, DEDUPE_COS)
        emit(method, 0, picked)
        chosen_scores = [sc[qids[i]] for i in picked]
        cutoff = min(chosen_scores) if chosen_scores else None
        meta["methods"][method] = {
            "size": len(picked), "shortfall": n - len(picked),
            "min_score": cutoff, "max_score": max(chosen_scores) if chosen_scores else None,
            "candidates_at_cutoff_score": int(sum(sc[qids[i]] == cutoff for i in cand)),
            "selected_at_cutoff_score": int(sum(s == cutoff for s in chosen_scores)),
            "per_stratum": np.bincount(stratum[picked], minlength=k).tolist()}
    for method in ("B0", "B1"):
        meta["methods"][method] = {"size": n, "draws": N_DRAWS}
    meta["quota_from_pi"] = allocate(n, pi, np.full(k, n)).tolist()   # before any shortfall
    return pd.DataFrame(rows), meta


def out_path(ds: str, seed: int):
    from ha.paths import get_paths

    return get_paths().data / "select" / ds / f"s{seed}" / "sets.parquet"


def a_size(ds: str, seed: int) -> int:
    """|A_s| from the split labels only (no query text): N needs the size, never the queries."""
    from ha.data.public import processed_dir

    labels = pd.read_parquet(processed_dir(ds) / "splits.parquet", columns=[f"split_s{seed}"])
    return int((labels[f"split_s{seed}"] == "A").sum())


def run_select(ds: str, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    from ha.data.splits import load_split
    from ha.judge.bulk import stage_log
    from ha.neighbours.retrieve import load_U
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.systems.retrieval import embed_queries
    from ha.utils import git_sha, utc_now_iso

    u = load_U(ds)
    u_emb = embed_queries(u["text"].tolist())
    iso_path = stage_log(ds, None).with_suffix(".parquet")
    iso_meta = json.loads(iso_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert iso_meta["complete"], f"{ds}: M1-iso judging is not complete"
    iso = pd.read_parquet(iso_path)
    metas = []
    for s in seeds:
        m1_path = stage_log(ds, s).with_suffix(".parquet")
        m1_meta = json.loads(m1_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert m1_meta["complete"], f"{ds} s{s}: M1 judging is not complete"
        nb = pd.read_parquet(nb_path(ds, s, "U"))
        near = set(nb.loc[nb["near_duplicate"], "qid"])
        r = load_split(ds, s, "R")                                      # R only, never A
        sets, meta = select_seed(s, u, u_emb, embed_queries(r["text"].tolist()), near, iso,
                                 pd.read_parquet(m1_path), a_size(ds, s))
        meta.update(dataset=ds, amendment="A3", weights=EQUAL_WEIGHTS, dedupe_cos=DEDUPE_COS,
                    draws=N_DRAWS, kmeans={"seed": KMEANS_SEED, "n_init": KMEANS_N_INIT},
                    judge_git_sha={"M1-iso": iso_meta["git_sha"], "M1": m1_meta["git_sha"]},
                    git_sha=git_sha(), created_at=utc_now_iso())
        p = out_path(ds, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        sets.to_parquet(p, index=False)
        p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                               encoding="utf-8")
        metas.append(meta)
    return metas
