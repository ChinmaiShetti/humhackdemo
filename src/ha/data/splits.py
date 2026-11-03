"""R/A splits over 5 seeds with CAL-auth forced into R (spec §6.4; prereg §3.2, §4.1; D-003).

Per dataset: every authentic query gets a language class (ha.data.langid) and a topic cluster
(k-means on bge-m3 embeddings of ALL authentic queries, used only for splitting). For each split
seed, within each (language, cluster) stratum, round(0.4 * n) queries go to R: CAL-auth items
first (forced), then a seeded random draw. CAL-auth is drawn once (seed 17) and is in R for
every seed. A of held-out datasets is only reachable through the sealed loader.
Output: data/processed/<ds>/splits.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ha.data.langid import classify
from ha.data.public import dataset_config, processed_dir
from ha.seal import require_unsealed

SPLIT_SEEDS = (0, 1, 2, 3, 4)
R_FRACTION = 0.4
CAL_AUTH_SEED = 17         # prereg §4.1
CLUSTER_SEED = 0
MIN_QUERIES_PER_CLUSTER = 30


def n_topic_clusters(n: int) -> int:
    """Split-stratification clusters: min(10, n // 30), at least 1 (D-034)."""
    return max(1, min(10, n // MIN_QUERIES_PER_CLUSTER))


def draw_cal_auth(qids: list[str], lang: list[str], n: int, seed: int = CAL_AUTH_SEED) -> set[str]:
    """n queries, proportionally stratified by language class, fixed seed."""
    if n <= 0:
        return set()
    df = pd.DataFrame({"qid": qids, "lang": lang})
    rng = np.random.default_rng(seed)
    shares = df["lang"].value_counts(normalize=True).sort_index()
    quota = (shares * n).round().astype(int)
    while quota.sum() != n:  # fix rounding drift on the largest class
        quota[quota.idxmax()] += 1 if quota.sum() < n else -1
    picked: list[str] = []
    for cls, q in quota.items():
        pool = sorted(df.loc[df["lang"] == cls, "qid"])
        picked += list(rng.choice(pool, size=min(q, len(pool)), replace=False))
    return set(picked)


def assign_splits(strata: pd.Series, forced_r: set[str], seeds=SPLIT_SEEDS) -> pd.DataFrame:
    """strata: index = qid, value = stratum label. Returns one 'R'/'A' column per seed."""
    out = pd.DataFrame(index=strata.index)
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        role = pd.Series("A", index=strata.index)
        for _, members in strata.groupby(strata):
            qids = sorted(members.index)
            target = int(round(R_FRACTION * len(qids)))
            forced = [q for q in qids if q in forced_r]
            rest = [q for q in qids if q not in forced_r]
            n_rest = max(0, target - len(forced))
            chosen = (list(rng.choice(rest, size=min(n_rest, len(rest)), replace=False))
                      if rest else [])
            role[forced + chosen] = "R"
        out[f"split_s{seed}"] = role
    return out


def build_splits(ds: str, n_cal_auth: int = 0, embed_fn=None) -> pd.DataFrame:
    """Annotate and split one dataset's authentic queries; writes splits.parquet."""
    from ha.embed import embed_texts

    embed_fn = embed_fn or embed_texts
    queries = pd.read_parquet(processed_dir(ds) / "queries_auth.parquet")
    lang = [classify(t) for t in queries["text"]]
    queries["lang_class"] = [x["label"] for x in lang]
    queries["cmi"] = [x["cmi"] for x in lang]
    emb = embed_fn(queries["text"].tolist())
    k = n_topic_clusters(len(queries))
    km = KMeans(n_clusters=k, n_init=10, random_state=CLUSTER_SEED)
    queries["topic_cluster"] = km.fit_predict(emb)
    cal = draw_cal_auth(queries["qid"].tolist(), queries["lang_class"].tolist(), n_cal_auth)
    queries["cal_auth"] = queries["qid"].isin(cal)
    strata = pd.Series(
        (queries["lang_class"] + "|" + queries["topic_cluster"].astype(str)).values,
        index=queries["qid"],
    )
    roles = assign_splits(strata, cal)
    out = queries.set_index("qid").join(roles).reset_index()
    out.to_parquet(processed_dir(ds) / "splits.parquet", index=False)
    return out


def load_split(ds: str, seed: int, part: str) -> pd.DataFrame:
    """Queries of one part ('R' or 'A') for a split seed. A of held-out datasets is sealed."""
    if part not in ("R", "A"):
        raise ValueError(part)
    if part == "A" and dataset_config(ds)["role"] == "heldout":
        require_unsealed("heldout_A", what=f"authentic test set A (seed {seed})", dataset=ds)
    df = pd.read_parquet(processed_dir(ds) / "splits.parquet")
    return df[df[f"split_s{seed}"] == part].reset_index(drop=True)
