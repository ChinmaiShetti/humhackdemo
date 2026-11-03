"""[EXPLORATORY] Why curation did not help: lexical overlap with the gold passage (after D-066).

Per dataset and split seed, with the frozen selections and judgements:
  - mean gold-passage overlap (ha.eval.h1.overlap) of each query set (B0/B1: mean over draws)
    and of the authentic test set A;
  - Spearman rho between the M1 judge (equal-weight score, and H alone) and overlap, over the
    U queries that pass the M1 gate;
  - the lexical advantage of each set: mean nDCG@10 of C1 (BM25) minus C3 (dense), against A's.
Needs HA_UNSEAL=heldout_A for D2 (after the signed freeze; every access is logged).
Output: results/eval/explore_overlap.json + registry rows (exp EXP-overlap, role exploratory).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ha import registry
from ha.data.public import dataset_config, processed_dir
from ha.data.splits import load_split
from ha.eval.h1 import _stopwords, overlap
from ha.eval.sets import perquery_table
from ha.gen.generate import load_source_corpus
from ha.judge.bulk import stage_log
from ha.neighbours.retrieve import load_U
from ha.paths import get_paths
from ha.select.sets import judge_scores, out_path
from ha.utils import git_sha, utc_now_iso

DATASETS, SEEDS = ("fiqa", "cqa_android", "hiaq"), (0, 1, 2, 3, 4)
METHODS = ("M1", "M1-iso", "B0", "B1")


def main() -> None:
    if git_sha().endswith("-dirty"):
        raise SystemExit("commit first")
    stop = _stopwords()
    out, rows = {}, []
    run_id = registry.new_run_id("EXP-overlap")
    for ds in DATASETS:
        passages = load_source_corpus(ds).set_index("pid")["text"]
        u = load_U(ds)
        ov_u = pd.Series([overlap(t, passages.loc[p], stop)
                          for t, p in zip(u["text"], u["source_pid"], strict=True)], index=u["qid"])
        qrels = pd.read_parquet(processed_dir(ds) / "qrels_auth.parquet")
        top = qrels[qrels["grade"] == qrels.groupby("qid")["grade"].transform("max")]
        gold = top.groupby("qid")["pid"].apply(list)
        pq_u, pq_a = perquery_table(ds, "U"), perquery_table(ds, "auth")
        iso = judge_scores(pd.read_parquet(stage_log(ds, None).with_suffix(".parquet")))
        per_seed = []
        for s in SEEDS:
            a = load_split(ds, s, "A")
            ov_a = np.mean([max(overlap(t, passages.loc[p], stop) for p in gold.loc[q])
                            for q, t in zip(a["qid"], a["text"], strict=True)])
            m1df = pd.read_parquet(stage_log(ds, s).with_suffix(".parquet")).set_index("qid")
            m1 = judge_scores(m1df.reset_index())
            passed = m1[np.isfinite(m1)].index
            rho_score = spearmanr(m1.loc[passed], ov_u.loc[passed]).statistic
            rho_h = spearmanr(m1df.loc[passed, "H"], ov_u.loc[passed]).statistic
            sets = pd.read_parquet(out_path(ds, s))
            rec = {"seed": s, "A_overlap": float(ov_a), "rho_M1score_overlap": float(rho_score),
                   "rho_H_overlap": float(rho_h),
                   "A_lexical_adv": float(pq_a.loc[a["qid"], "C1"].mean()
                                          - pq_a.loc[a["qid"], "C3"].mean())}
            for m in METHODS:
                g = sets[sets["method"] == m]
                rec[f"{m}_overlap"] = float(np.mean([ov_u.loc[d["qid"]].mean()
                                                     for _, d in g.groupby("draw")]))
                rec[f"{m}_lexical_adv"] = float(np.mean(
                    [pq_u.loc[d["qid"], "C1"].mean() - pq_u.loc[d["qid"], "C3"].mean()
                     for _, d in g.groupby("draw")]))
            per_seed.append(rec)
        df = pd.DataFrame(per_seed)
        summary = df.drop(columns="seed").mean().round(4).to_dict()
        summary["iso_rho_score_overlap"] = float(
            spearmanr(iso[np.isfinite(iso)], ov_u.loc[iso[np.isfinite(iso)].index]).statistic)
        out[ds] = {"role": dataset_config(ds)["role"], "per_seed": per_seed, "seed_mean": summary}
        for k, v in summary.items():
            rows.append({"run_id": run_id, "exp": "EXP-overlap",
                         "dataset": dataset_config(ds)["id"],
                         "split_seed": None, "method": k.split("_")[0], "system_set": "C1,C3",
                         "metric": k, "value": float(v), "ci_low": None, "ci_high": None,
                         "n_queries": len(u), "role": "exploratory"})
    registry.record(rows, {"exp": "EXP-overlap", "seeds": SEEDS})
    res = {"run_id": run_id, "git_sha": git_sha(), "created_at": utc_now_iso(),
           "label": "EXPLORATORY (after the freeze)", "datasets": out}
    p = get_paths().results / "eval" / "explore_overlap.json"
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for ds, v in out.items():
        print(ds, json.dumps(v["seed_mean"]))


if __name__ == "__main__":
    main()
