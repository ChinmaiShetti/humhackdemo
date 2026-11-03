"""Calibration annotation workflow, human side (P4.3-P4.4; prereg §4; D-050).

Spreadsheet (CSV, UTF-8 with BOM so Excel opens it) instead of Label Studio. Artefacts under
HA_DATA/calibration/, each a separate file that is never merged into another:

  key.csv / practice_key.csv   item_id -> qid, dataset, authentic/synthetic, pool, split, passage
                               pid and rule, human-facing neighbour qids. NOT given to annotators.
  sheets/raw_r<round>_<annotator>.csv
                               one per annotator, blind: no qid, source kind, split or judge output.
                               Only the annotator fills the rating columns. Never overwritten once
                               anything has been entered.
  sheets/practice.csv          the 20 practice items, rated jointly (prereg §4.3).
  adjudication_r<round>.csv    built from two complete raw sheets; the team fills `<dim>_settled`
                               only where `<dim>_discuss` is 1.
  final_labels_r<round>.csv    the human reference: mean of the two ratings, or the settled value
                               where they differ by >= 2 (prereg §4.4). Built only when every
                               flagged disagreement is settled.

Two decisions fixed here (D-050):
  * Authentic items show a real qrels-linked passage (decision 1 = A): among the item's qrels with
    the highest grade whose passage exists and is non-empty, pids sorted ascending, take index
    int(sha256("17|<dataset>|<qid>")[:16], 16) mod n. No length or content criterion (synthetic
    source passages were drawn uniformly, with none either). No qrels -> the item stays pending.
  * Human-facing neighbours (decision 2 = D) use the frozen procedure (bge-m3 top-30 -> reranker
    at 128 -> top-5; authentic items never see themselves or a cos > 0.97 near-duplicate) over
    R_s0 minus all CAL-auth items (prereg-v1: 60; under A2: the 30 FiQA items). This is a
    display rule only: R, the judge's neighbours
    (HA_DATA/neighbours/...) and the judge-side procedure are untouched.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ha.config import domain_description, load_config
from ha.judge.calibrate import (
    CalibrationBlocked,
    amendment_recorded,
    build_items,
    draw_synthetic,
    split_items,
)
from ha.judge.prompt import permute
from ha.neighbours.retrieve import TOP_K, neighbours_for
from ha.paths import get_paths
from ha.utils import git_sha, utc_now_iso

DIMS = ("F", "H", "S", "P", "C", "overall")  # prereg §4.3: F, H, S, P, C and overall realism
REQUIRED = ("F", "H", "S", "P", "overall")   # C is blank for pure-English queries
ANNOTATORS = ("A1", "A2")
DISCUSS_AT = 2                                # prereg §4.4: disagreements of 2 or more
NEIGHBOUR_SPLIT_SEED = 0                      # prereg §4.1: neighbours come from R of seed 0
NOT_APPLICABLE = "NA"                         # settled C for "not code-mixed"
PASSAGE_RULE = "qrels max grade; pids sorted; index sha256('17|dataset|qid')[:16] mod n"
HUMAN_NEIGHBOUR_RULE = "frozen P4.1 procedure over R_s0 minus all CAL-auth (display only)"

NEIGHBOUR_COLS = [f"neighbour_{i}" for i in range(1, TOP_K + 1)]
INFO_COLS = ["item_id", "item_type", "dataset", "domain", "query", "passage", *NEIGHBOUR_COLS]
ENTRY_COLS = ["annotator_id", *DIMS, "flag_unclear", "notes"]
SHEET_COLS = INFO_COLS + ENTRY_COLS
KEY_COLS = ["item_id", "item_type", "dataset", "qid", "kind", "pool", "split", "passage_pid",
            "passage_rule", "human_neighbour_qids", "human_neighbour_rule"]
_RATING = re.compile(r"^[1-5]$")


def _seed() -> int:
    return int(load_config("calibration.yaml")["seed"])


def out_dir() -> Path:
    return get_paths().data / "calibration"


# ---------------------------------------------------------------- decision 1: passages
def select_passage(dataset: str, qid: str, qrels: pd.DataFrame, passages: Mapping[str, str],
                   seed: int = 17) -> str | None:
    """Deterministic, outcome-blind choice of one qrels-linked passage; None = pending."""
    q = qrels[qrels["qid"].astype(str) == str(qid)]
    q = q[[str(passages.get(str(p), "")).strip() != "" for p in q["pid"]]]
    if q.empty:
        return None
    top = sorted(q.loc[q["grade"] == q["grade"].max(), "pid"].astype(str).unique())
    h = int(hashlib.sha256(f"{seed}|{dataset}|{qid}".encode()).hexdigest()[:16], 16)
    return top[h % len(top)]


def attach_passages(items: pd.DataFrame, source_pid: Mapping[str, str],
                    qrels: Mapping[str, pd.DataFrame | None], passages: Mapping[str, Mapping],
                    seed: int = 17) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Synthetic: its own source passage. Authentic: select_passage. Returns (items, pending)."""
    items = items.copy()
    pids, pending = [], []
    for row in items.itertuples():
        if row.kind == "synthetic":
            pid = str(source_pid[row.qid])
        else:
            qr = qrels.get(row.dataset)
            pid = None if qr is None else select_passage(row.dataset, row.qid, qr,
                                                        passages[row.dataset], seed)
            if pid is None:
                pending.append((row.dataset, row.qid))
        pids.append(pid)
    items["passage_pid"] = pids
    items["passage"] = [None if p is None else passages[d][p]
                        for d, p in zip(items["dataset"], pids, strict=True)]
    items["passage_rule"] = np.where(items["kind"] == "synthetic", "source passage", PASSAGE_RULE)
    return items, pending


# ---------------------------------------------------------------- decision 2: neighbours
def human_display_R(r: pd.DataFrame, dataset: str, cal_auth: set[tuple[str, str]]) -> pd.DataFrame:
    """A display-only copy of R_s0 without any CAL-auth item. R itself is not modified."""
    drop = {q for d, q in cal_auth if d == dataset}
    return r[~r["qid"].astype(str).isin(drop)].reset_index(drop=True)


def human_neighbours(items: pd.DataFrame, r_display: pd.DataFrame, q_emb: np.ndarray,
                     r_emb: np.ndarray, rerank_fn, dataset: str,
                     cal_auth: set[tuple[str, str]]) -> pd.DataFrame:
    """Top-5 per item (one dataset) from r_display; authentic items get the D-003 exclusions."""
    parts = []
    for authentic in (True, False):
        m = ((items["kind"] == "authentic") == authentic).to_numpy()
        if m.any():
            q = items.loc[m, ["qid", "text"]].reset_index(drop=True)
            parts.append(neighbours_for(q, r_display, q_emb[m], r_emb, rerank_fn, authentic))
    nb = pd.concat(parts, ignore_index=True)
    shown = {q for lst in nb["neighbour_qids"] for q in lst}
    leaked = shown & {q for d, q in cal_auth if d == dataset}
    if leaked:
        raise AssertionError(f"CAL-auth in a human-facing neighbour block: {sorted(leaked)}")
    text = dict(zip(r_display["qid"].astype(str), r_display["text"], strict=True))
    nb["neighbour_texts"] = [permute([text[q] for q in lst], f"human|{qid}", 0)
                             for qid, lst in zip(nb["qid"], nb["neighbour_qids"], strict=True)]
    return nb[["qid", "neighbour_qids", "neighbour_texts"]]


# ---------------------------------------------------------------- sheets
def assign_item_ids(items: pd.DataFrame, seed: int, prefix: str) -> pd.DataFrame:
    """Random presentation order (prereg §4.3); the ID carries no information about the source."""
    base = items.sort_values(["dataset", "qid"]).reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(base))
    out = base.iloc[order].reset_index(drop=True)
    width = max(3, len(str(len(out))))
    out.insert(0, "item_id", [f"{prefix}{i + 1:0{width}d}" for i in range(len(out))])
    return out


def make_pack(items: pd.DataFrame, item_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """items: item_id, qid, text, dataset, kind, pool, [split], passage_pid, passage,
    passage_rule, neighbour_qids, neighbour_texts. Returns (blank blind sheet, key)."""
    if items["passage"].isna().any():
        raise CalibrationBlocked("an item has no passage (pending qrels); refusing to export")
    sheet = pd.DataFrame({"item_id": items["item_id"], "item_type": item_type,
                          "dataset": items["dataset"],
                          "domain": [domain_description(d) for d in items["dataset"]],
                          "query": items["text"], "passage": items["passage"]})
    for i, col in enumerate(NEIGHBOUR_COLS):
        sheet[col] = [t[i] if i < len(t) else "" for t in items["neighbour_texts"]]
    for col in ENTRY_COLS:
        sheet[col] = ""
    key = pd.DataFrame({"item_id": items["item_id"], "item_type": item_type,
                        "dataset": items["dataset"], "qid": items["qid"], "kind": items["kind"],
                        "pool": items["pool"], "split": items.get("split", ""),
                        "passage_pid": items["passage_pid"], "passage_rule": items["passage_rule"],
                        "human_neighbour_qids": [json.dumps(list(x))
                                                 for x in items["neighbour_qids"]],
                        "human_neighbour_rule": HUMAN_NEIGHBOUR_RULE})
    return sheet[SHEET_COLS], key[KEY_COLS]


def annotator_sheet(blank: pd.DataFrame, annotator_id: str) -> pd.DataFrame:
    s = blank.copy()
    s["annotator_id"] = annotator_id
    return s


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def has_entries(df: pd.DataFrame) -> bool:
    cols = [c for c in (*DIMS, "flag_unclear", "notes") if c in df]
    cols += [c for c in df if c.endswith("_settled") or c == "adjudication_notes"]
    return bool((df[cols].astype(str).apply(lambda s: s.str.strip()) != "").to_numpy().any())


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Refuses to overwrite a file that already holds any human entry."""
    if path.exists() and has_entries(read_csv(path)):
        raise FileExistsError(f"{path} already contains human entries; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- import and adjudication
def validate_sheet(sheet: pd.DataFrame, blank: pd.DataFrame, complete: bool = True) -> list[str]:
    """Problems in a returned raw sheet, checked against the exported blank sheet."""
    probs: list[str] = []
    if list(sheet.columns) != SHEET_COLS:
        return [f"columns differ from the export: {list(sheet.columns)}"]
    if not sheet["item_id"].is_unique or set(sheet["item_id"]) != set(blank["item_id"]):
        return ["item_id set differs from the export"]
    s = sheet.set_index("item_id").loc[blank["item_id"]]
    b = blank.set_index("item_id")
    for col in INFO_COLS[1:]:
        changed = (s[col].astype(str) != b[col].astype(str))
        if changed.any():
            probs.append(f"{col} edited on {list(changed[changed].index[:5])}")
    if s["annotator_id"].nunique() != 1 or s["annotator_id"].iloc[0].strip() == "":
        probs.append("annotator_id must be one non-empty value")
    for d in DIMS:
        v = s[d].astype(str).str.strip()
        bad = v[(v != "") & ~v.str.match(_RATING)]
        if len(bad):
            probs.append(f"{d}: not an integer 1-5 on {list(bad.index[:5])}")
        if complete and d in REQUIRED and (v == "").any():
            probs.append(f"{d}: blank on {list(v[v == ''].index[:5])}")
    return probs


def build_adjudication(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side view of two validated raw sheets. Raw sheets are not modified."""
    ida, idb = a["annotator_id"].iloc[0], b["annotator_id"].iloc[0]
    if ida == idb:
        raise ValueError("the two raw sheets have the same annotator_id")
    a = a.set_index("item_id")
    b = b.set_index("item_id").loc[a.index]
    out = a[INFO_COLS[1:]].reset_index()
    for d in DIMS:
        x = pd.to_numeric(a[d].str.strip().replace("", np.nan)).to_numpy()
        y = pd.to_numeric(b[d].str.strip().replace("", np.nan)).to_numpy()
        both = ~np.isnan(x) & ~np.isnan(y)
        one = np.isnan(x) ^ np.isnan(y)            # e.g. C given by only one annotator
        diff = np.where(both, np.abs(x - y), np.nan)
        out[f"{d}_{ida}"], out[f"{d}_{idb}"] = a[d].to_numpy(), b[d].to_numpy()
        out[f"{d}_mean"] = np.where(both, (x + y) / 2, np.nan)
        out[f"{d}_diff"] = diff
        out[f"{d}_discuss"] = ((both & (np.nan_to_num(diff) >= DISCUSS_AT)) | one).astype(int)
        out[f"{d}_settled"] = ""
    out["notes_" + ida], out["notes_" + idb] = a["notes"].to_numpy(), b["notes"].to_numpy()
    out["adjudication_notes"] = ""
    return out


def finalize(adj: pd.DataFrame) -> pd.DataFrame:
    """Human reference per item and dimension (prereg §4.4). Raises on any unsettled flag."""
    out = adj[["item_id", "dataset"]].copy()
    probs = []
    for d in DIMS:
        settled = adj[f"{d}_settled"].astype(str).str.strip()
        flag = adj[f"{d}_discuss"].astype(int) == 1
        stray = ~flag & (settled != "")
        if stray.any():
            probs.append(f"{d}: settled value on an unflagged item {list(adj.item_id[stray][:5])}")
        missing = flag & (settled == "")
        if missing.any():
            probs.append(f"{d}: unsettled disagreement on {list(adj.item_id[missing][:5])}")
        ok_vals = settled.str.fullmatch(r"[1-4](\.\d+)?|5(\.0+)?") | (settled == "")
        if d == "C":
            ok_vals |= settled == NOT_APPLICABLE
        if (~ok_vals).any():
            probs.append(f"{d}: settled value not in 1-5 on {list(adj.item_id[~ok_vals][:5])}")
        vals = pd.to_numeric(settled.where(flag & (settled != NOT_APPLICABLE), ""),
                             errors="coerce")
        out[d] = np.where(flag, vals, pd.to_numeric(adj[f"{d}_mean"], errors="coerce"))
        out[f"{d}_source"] = np.where(flag, "settled", "mean")
    if probs:
        raise ValueError("; ".join(probs))
    return out


# ---------------------------------------------------------------- orchestration (data, models)
def cal_auth_keys() -> set[tuple[str, str]]:
    from ha.data.public import processed_dir

    keys = set()
    for ds in ("fiqa", "hiaq"):
        p = processed_dir(ds) / "splits.parquet"
        if p.exists():
            sp = pd.read_parquet(p)
            keys |= {(ds, str(q)) for q in sp.loc[sp["cal_auth"], "qid"]}
    return keys


def _auth(ds: str) -> pd.DataFrame | None:
    from ha.data.public import processed_dir

    p = processed_dir(ds) / "splits.parquet"
    if not p.exists():
        return None
    sp = pd.read_parquet(p)
    return sp.loc[sp["cal_auth"], ["qid", "text"]].astype({"qid": str}).reset_index(drop=True)


def _qrels(ds: str) -> pd.DataFrame | None:
    from ha.data.public import processed_dir

    p = processed_dir(ds) / "qrels_auth.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _passages(ds: str) -> dict[str, str]:
    from ha.gen.generate import load_source_corpus  # the exact text the generator was given

    c = load_source_corpus(ds)
    return dict(zip(c["pid"].astype(str), c["text"], strict=True))


def _U(ds: str) -> pd.DataFrame:
    from ha.neighbours.retrieve import load_U

    return load_U(ds)


def _with_human_neighbours(items: pd.DataFrame, cal: set[tuple[str, str]]) -> pd.DataFrame:
    from ha import llm
    from ha.data.splits import load_split
    from ha.neighbours.retrieve import RERANK_MAX_LEN
    from ha.systems.retrieval import Reranker, embed_queries

    embs = {}
    for ds, g in items.groupby("dataset", sort=True):
        r = load_split(ds, NEIGHBOUR_SPLIT_SEED, "R")[["qid", "text", "lang_class"]]  # R only
        rd = human_display_R(r, ds, cal)
        embs[ds] = (g.reset_index(drop=True), rd, embed_queries(g["text"].tolist()),
                    embed_queries(rd["text"].tolist()))
    llm.unload_all()                                   # VRAM rule before torch
    reranker = Reranker(max_length=RERANK_MAX_LEN)
    nbs = [human_neighbours(g, rd, qe, re_, reranker.score_pairs, ds, cal).assign(dataset=ds)
           for ds, (g, rd, qe, re_) in embs.items()]
    return items.merge(pd.concat(nbs, ignore_index=True), on=["dataset", "qid"], how="left",
                       validate="one_to_one")


def practice_items() -> pd.DataFrame:
    """20 extra FiQA synthetic items outside the calibration set (prereg §4.3); seed 17."""
    cfg = load_config("calibration.yaml")
    U = _U("fiqa")
    used = set(draw_synthetic(U, cfg["synthetic"]["fiqa"], cfg["seed"])["qid"])
    rest = U[~U["qid"].isin(used)]
    return draw_synthetic(rest, cfg["practice_items"], cfg["seed"]).assign(
        dataset="fiqa", kind="synthetic")


def prepare_practice() -> dict[str, Any]:
    cal = cal_auth_keys()
    n_fiqa = sum(d == "fiqa" for d, _ in cal)
    if n_fiqa != load_config("calibration.yaml")["cal_auth"]["fiqa"]:
        raise CalibrationBlocked(f"FiQA CAL-auth has {n_fiqa} items, expected 30")
    items = practice_items()
    items, pending = attach_passages(items, dict(zip(items.qid, items.source_pid, strict=True)),
                                     {}, {"fiqa": _passages("fiqa")}, _seed())
    assert not pending
    items = assign_item_ids(_with_human_neighbours(items, cal), _seed(), "P")
    sheet, key = make_pack(items, "practice")
    d = out_dir()
    write_csv(annotator_sheet(sheet, "+".join(ANNOTATORS)), d / "sheets" / "practice.csv")
    write_csv(key, d / "practice_key.csv")
    return _meta("practice", key, d / "practice_meta.json")


def prepare_main(round_: int = 1) -> dict[str, Any]:
    """The calibration pack (size from the config: 100 items under A2). Blocked until every
    configured CAL-auth item and its qrels exist, and until the config's amendment is recorded
    (enforced in build_items)."""
    cfg = load_config("calibration.yaml")
    cal = cal_auth_keys()
    need = sum(cfg["cal_auth"].values())
    if len(cal) != need:
        raise CalibrationBlocked(f"CAL-auth has {len(cal)}/{need} configured items")
    U = {ds: _U(ds) for ds in ("fiqa", "hiaq")}
    items = split_items(build_items(_auth("fiqa"), _auth("hiaq"), U["fiqa"], U["hiaq"]))
    src = {q: p for ds in U for q, p in zip(U[ds].qid, U[ds].source_pid, strict=True)}
    assert len(src) == sum(len(u) for u in U.values()), "synthetic qids collide across datasets"
    items, pending = attach_passages(items, src, {ds: _qrels(ds) for ds in U},
                                     {ds: _passages(ds) for ds in U}, cfg["seed"])
    if pending:
        raise CalibrationBlocked(f"{len(pending)} authentic item(s) have no qrels passage yet "
                                 f"(pending, not substituted): {pending[:5]}")
    items = assign_item_ids(_with_human_neighbours(items, cal), cfg["seed"], "C")
    sheet, key = make_pack(items, "calibration")
    d = out_dir()
    write_csv(key, d / "key.csv")
    for a in ANNOTATORS:
        write_csv(annotator_sheet(sheet, a), d / "sheets" / f"raw_r{round_}_{a}.csv")
    write_csv(sheet, d / "sheets" / "blank.csv")
    return _meta("main", key, d / "main_meta.json")


def _meta(kind: str, key: pd.DataFrame, path: Path) -> dict[str, Any]:
    meta = {"pack": kind, "n_items": len(key),
            "by_dataset_kind": {f"{a}/{b}": int(n)
                                for (a, b), n in key.groupby(["dataset", "kind"]).size().items()},
            "passage_rule": PASSAGE_RULE, "human_neighbour_rule": HUMAN_NEIGHBOUR_RULE,
            "seed": _seed(), "git_sha": git_sha(), "created_at": utc_now_iso()}
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def status() -> dict[str, Any]:
    """Readiness of the calibration pack; reads only R-side data and the qrels of CAL-auth."""
    cfg = load_config("calibration.yaml")
    cal = cal_auth_keys()
    out: dict[str, Any] = {}
    for ds in ("fiqa", "hiaq"):
        auth, qr = _auth(ds), _qrels(ds)
        n = 0 if auth is None else len(auth)
        resolved = 0
        multi = 0
        if auth is not None and qr is not None:
            texts = _passages(ds)
            for q in auth["qid"]:
                resolved += select_passage(ds, q, qr, texts, cfg["seed"]) is not None
                multi += int((qr["qid"].astype(str) == q).sum() > 1)
        out[ds] = {"cal_auth": n, "expected": cfg["cal_auth"][ds],
                   "qrels_available": qr is not None, "passage_resolved": resolved,
                   "with_multiple_qrels_passages": multi}
    out["cal_auth_total"] = len(cal)
    tag = cfg.get("amendment")
    out["amendment"] = tag
    out["amendment_recorded"] = bool(tag) and amendment_recorded(tag)
    out["main_pack_ready"] = (not tag or out["amendment_recorded"]) and all(
        out[ds]["cal_auth"] == out[ds]["expected"] == out[ds]["passage_resolved"]
        for ds in ("fiqa", "hiaq"))
    return out
