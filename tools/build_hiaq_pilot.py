"""Build data/processed/hiaq for the EXPLORATORY HiAQ pilot (D-067; A1, D-065).

  corpus.parquet        the 1,572 chunks (pid = chunk_id, empty title, text = chunk text, the
                        same text the synthetic pools were generated from);
  queries_auth.parquet  the collected queries that map to an official chunk (146 of 150);
  qrels_auth.parquet    each query's scenario chunk, grade 2 (spec §6.3; no pooling, A1 item 5);
  splits.parquet        R 40% / A 60% for split seeds 0-4, stratified by language x topic
                        cluster (ha.data.splits.build_splits; no CAL-auth: there is no
                        calibration under A3).
"""

from __future__ import annotations

import json

import pandas as pd

from ha.data.public import processed_dir
from ha.data.splits import build_splits
from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso


def main() -> None:
    data = get_paths().data
    out = processed_dir("hiaq")
    out.mkdir(parents=True, exist_ok=True)
    chunks = pd.read_parquet(data / "hiaq" / "chunks.parquet")
    corpus = pd.DataFrame({"pid": chunks["chunk_id"].astype("string"), "title": "",
                           "text": chunks["text"].astype("string")})
    assert corpus["pid"].is_unique
    col = pd.read_parquet(data / "hiaq" / "collected" / "2026-10-16_group_a.parquet")
    q = col[col["chunk_id"].notna() & ~col["empty"]]
    assert set(q["chunk_id"]) <= set(corpus["pid"])
    queries = pd.DataFrame({"qid": q["qid"].astype("string"), "text": q["text"].astype("string")})
    qrels = pd.DataFrame({"qid": q["qid"].astype("string"), "pid": q["chunk_id"].astype("string"),
                          "grade": 2})
    corpus.to_parquet(out / "corpus.parquet", index=False)
    queries.to_parquet(out / "queries_auth.parquet", index=False)
    qrels.to_parquet(out / "qrels_auth.parquet", index=False)
    splits = build_splits("hiaq", n_cal_auth=0)
    meta = {"dataset": "hiaq", "role": "exploratory", "decision": "D-067", "corpus": len(corpus),
            "queries": len(queries), "excluded_no_chunk": int(col["chunk_id"].isna().sum()),
            "R_per_seed": [int((splits[f"split_s{s}"] == "R").sum()) for s in range(5)],
            "A_per_seed": [int((splits[f"split_s{s}"] == "A").sum()) for s in range(5)],
            "lang_class": splits["lang_class"].value_counts().to_dict(),
            "topic_clusters": int(splits["topic_cluster"].nunique()),
            "git_sha": git_sha(), "created_at": utc_now_iso()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
