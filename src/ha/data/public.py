"""Public dataset download and normalisation (plan P1.1).

Raw files are fetched once from the Hugging Face Hub into `HA_DATA/raw/<ds>/` (with the
revision recorded), then normalised to:
  data/processed/<ds>/corpus.parquet        pid, title, text
  data/processed/<ds>/queries_auth.parquet  qid, text        (evaluation-split queries only)
  data/processed/<ds>/qrels_auth.parquet    qid, pid, grade
  data/processed/<ds>/meta.json             counts, revisions, config hash

`queries_auth` holds R ∪ A before splitting; held-out A is only ever reached through the
sealed loaders in `ha.data.splits`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ha.config import config_hash, load_config
from ha.paths import get_paths
from ha.utils import utc_now_iso

CORPUS_COLS = ["pid", "title", "text"]
QUERY_COLS = ["qid", "text"]
QRELS_COLS = ["qid", "pid", "grade"]


class DatasetCheckError(RuntimeError):
    pass


def dataset_config(ds: str) -> dict[str, Any]:
    return load_config(f"datasets/{ds}.yaml")


def processed_dir(ds: str) -> Path:
    return get_paths().processed / ds


def _download(repo: str, filename: str, raw_dir: Path) -> tuple[Path, str]:
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(repo).sha
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                           local_dir=raw_dir / repo.replace("/", "__"))
    return Path(path), revision


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str)
    raise ValueError(f"unsupported file type {path}")


def _normalise_corpus(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "pid": df["_id"].astype(str),
        "title": df.get("title", pd.Series([""] * len(df))).fillna("").astype(str),
        "text": df["text"].fillna("").astype(str),
    })
    return out.drop_duplicates("pid").reset_index(drop=True)


def _normalise_qrels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "qid": df["query-id"].astype(str),
        "pid": df["corpus-id"].astype(str),
        "grade": pd.to_numeric(df["score"]).astype(int),
    })
    return out[out["grade"] > 0].drop_duplicates(["qid", "pid"]).reset_index(drop=True)


def validate_processed(corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame) -> None:
    """Schema and referential checks shared by every dataset (tested in tests/test_public.py)."""
    for df, cols, name in [(corpus, CORPUS_COLS, "corpus"), (queries, QUERY_COLS, "queries"),
                           (qrels, QRELS_COLS, "qrels")]:
        if list(df.columns) != cols:
            raise DatasetCheckError(f"{name} columns {list(df.columns)} != {cols}")
    if corpus["pid"].duplicated().any() or queries["qid"].duplicated().any():
        raise DatasetCheckError("duplicate ids")
    if (queries["text"].str.strip() == "").any():
        raise DatasetCheckError("empty query text")
    if not set(qrels["qid"]) <= set(queries["qid"]):
        raise DatasetCheckError("qrels reference unknown queries")
    if not set(qrels["pid"]) <= set(corpus["pid"]):
        raise DatasetCheckError("qrels reference unknown passages")
    if not set(queries["qid"]) <= set(qrels["qid"]):
        raise DatasetCheckError("some evaluation queries have no relevant passage")


def build(ds: str, force: bool = False) -> dict[str, Any]:
    """Download (once) and normalise a public dataset. Idempotent."""
    cfg = dataset_config(ds)
    out_dir = processed_dir(ds)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("config_hash") == config_hash(cfg):
            logger.info(f"{ds}: already built ({meta_path})")
            return meta

    raw_dir = get_paths().data / "raw" / ds
    src = cfg["source"]
    revisions: dict[str, str] = {}
    corpus_path, revisions[src["repo"]] = _download(src["repo"], src["files"]["corpus"], raw_dir)
    queries_path, _ = _download(src["repo"], src["files"]["queries"], raw_dir)
    qrels_repo = src.get("qrels_repo", src["repo"])
    qrels_path, revisions[qrels_repo] = _download(
        qrels_repo, src["qrels_files"][cfg["eval_split"]], raw_dir
    )

    corpus = _normalise_corpus(_read_table(corpus_path))
    all_queries = _read_table(queries_path)
    all_queries = pd.DataFrame({"qid": all_queries["_id"].astype(str),
                                "text": all_queries["text"].fillna("").astype(str)})
    qrels = _normalise_qrels(_read_table(qrels_path))
    queries = all_queries[all_queries["qid"].isin(set(qrels["qid"]))].reset_index(drop=True)

    n_eval = len(queries)
    if n_eval > cfg["max_queries"]:
        queries = queries.sample(cfg["max_queries"], random_state=cfg["query_sample_seed"])
        queries = queries.sort_values("qid").reset_index(drop=True)
        qrels = qrels[qrels["qid"].isin(set(queries["qid"]))].reset_index(drop=True)
        logger.info(f"{ds}: sampled {cfg['max_queries']} of {n_eval} queries")

    exp = cfg["expected"]
    checks = {"corpus": (len(corpus), exp["corpus"]), "eval_queries": (n_eval, exp["eval_queries"])}
    for name, (got, want) in checks.items():
        if got != want:
            raise DatasetCheckError(f"{ds}: {name} count {got} != expected {want}")
    validate_processed(corpus, queries, qrels)

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_dir / "corpus.parquet", index=False)
    queries.to_parquet(out_dir / "queries_auth.parquet", index=False)
    qrels.to_parquet(out_dir / "qrels_auth.parquet", index=False)
    meta = {
        "dataset": ds,
        "id": cfg["id"],
        "role": cfg["role"],
        "built_at": utc_now_iso(),
        "config_hash": config_hash(cfg),
        "hf_revisions": revisions,
        "counts": {
            "corpus": len(corpus),
            "eval_queries_available": n_eval,
            "queries_auth": len(queries),
            "qrels_auth": len(qrels),
            "mean_rel_per_query": round(len(qrels) / len(queries), 3),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{ds}: built {meta['counts']}")
    return meta
