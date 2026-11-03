"""The freeze (prereg §6 as amended by A3): results/FROZEN.json, then the lead's signature.

`build()` records what §6 lists, read from the files on disk and the clean git HEAD:
  - the judge: template hashes, prompt version, anchors (none: A3), model digest, decoding,
    permutation rule; the M1 and M1-iso weights (equal, A3) and the gate;
  - the generation config and every pool's hash, per dataset;
  - the neighbour files, judge outputs and selected sets, per dataset and seed (D-063), plus the
    selection, strata and dedupe settings;
  - systems.yaml and the synthetic-query retrieval runs;
  - the analysis code's git sha and the bootstrap settings.
`write()` refuses a dirty tree, an incomplete input, or a record that is already signed.
`sign(name)` adds the signer, the date and the record's sha256, and appends the sign-off to
docs/decisions.md. heldout_A unseals only while that hash still matches (ha.seal). Signing is the
project lead's act (A3 item 6): nothing in the pipeline signs on its own.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DATASETS = ("fiqa", "cqa_android")           # D1 (development) and D2 (confirmatory, A1)
SEEDS = (0, 1, 2, 3, 4)
SIGN_FIELDS = ("signed_by", "signed_at", "record_sha256")


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_sha256(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k not in SIGN_FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def _need(path: Path, what: str) -> str:
    if not path.exists():
        raise FreezeError(f"missing {what}: {path}")
    return sha256_file(path)


def build(datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    from ha import prereg
    from ha.config import load_config
    from ha.judge.aggregate import EQUAL_WEIGHTS
    from ha.judge.bulk import AMENDMENT, stage_log
    from ha.judge.calibrate import amendment_recorded
    from ha.neighbours.retrieve import out_path as nb_path
    from ha.paths import REPO_ROOT, get_paths
    from ha.select import sets as S
    from ha.systems.run_all import run_path
    from ha.utils import git_sha, utc_now_iso

    sha = git_sha()
    if sha.endswith("-dirty") or sha == "nogit":
        raise FreezeError(f"commit first: the analysis code must be a clean commit ({sha})")
    for tag in ("A1", "A2", "A3"):
        if not amendment_recorded(tag):
            raise FreezeError(f"amendment {tag} is not recorded")
    judge = load_config("judge.yaml")
    prompts = REPO_ROOT / "src" / "ha" / "gen" / "prompts"
    rec: dict[str, Any] = {
        "prereg": {"analysis": "prereg-v3 = prereg-v1 + A1 + A2 + A3",
                   "frozen_body_sha256": prereg.frozen_body_sha256(
                       prereg.PREREG_PATH.read_text(encoding="utf-8")),
                   "amendments": ["A1", "A2", "A3"]},
        "judge": {"model": judge["model"]["tag"], "digest": judge["model"]["digest"],
                  "think": judge["model"]["think"], "prompt_version": judge["prompt"]["version"],
                  "templates": {n: _need(prompts / f"{judge['prompt'][k]}.j2", n)
                                for k, n in (("system", "system"), ("user", "user"))},
                  "anchors": [], "decoding": judge["decoding"],
                  "permutations": judge["permutations"], "gate": judge["gate"],
                  "amendment": AMENDMENT, "judge_yaml_sha256": _need(
                      REPO_ROOT / "configs" / "judge.yaml", "judge.yaml")},
        "weights": {"M1": EQUAL_WEIGHTS, "M1-iso": EQUAL_WEIGHTS,
                    "source": "A3: equal weights on (H, S, P, C*), fixed a priori"},
        "generation": {"config_sha256": _need(REPO_ROOT / "configs" / "generation.yaml",
                                              "generation.yaml"), "pools": {}},
        "selection": {"dedupe_cos": S.DEDUPE_COS, "draws": S.N_DRAWS, "max_k": S.MAX_K,
                      "r_per_k": S.R_PER_K, "kmeans_seed": S.KMEANS_SEED,
                      "kmeans_n_init": S.KMEANS_N_INIT, "streams": S.STREAM,
                      "tie_break": "seeded uniform key per U query, [2, split seed] (D-063)",
                      "quotas": "largest remainder; shortfall re-allocated by pi (D-063)",
                      "near_duplicate_cos": 0.97},
        "retrieval": {"systems_yaml_sha256": _need(REPO_ROOT / "configs" / "systems.yaml",
                                                   "systems.yaml"),
                      "metric": "ndcg@10"},
        "analysis": {"git_sha": sha, "bootstrap_B": 1000, "bootstrap_seed": 20260927,
                     "confirmatory_datasets": ["cqa_android"], "development_datasets": ["fiqa"]},
        "datasets": {},
        "created_at": utc_now_iso(),
        **{k: None for k in SIGN_FIELDS},
    }
    data = get_paths().data
    for ds in datasets:
        pools = sorted((data / "synthetic" / ds).glob("*/queries.parquet"))
        rec["generation"]["pools"][ds] = {p.parent.name: sha256_file(p) for p in pools}
        entry: dict[str, Any] = {"judge": {}, "neighbours": {}, "sets": {}, "runs_U": {}}
        iso = stage_log(ds, None).with_suffix(".parquet")
        entry["judge"]["M1-iso"] = _need(iso, f"{ds} M1-iso judgements")
        for s in SEEDS:
            m1 = stage_log(ds, s).with_suffix(".parquet")
            meta = json.loads(_need_text(m1.with_suffix(".meta.json"), f"{ds} s{s} M1 meta"))
            if not meta.get("complete"):
                raise FreezeError(f"{ds} s{s}: M1 judging is not complete")
            entry["judge"][f"M1 s{s}"] = _need(m1, f"{ds} s{s} M1 judgements")
            entry["neighbours"][f"s{s}"] = _need(nb_path(ds, s, "U"), f"{ds} s{s} neighbours")
            entry["sets"][f"s{s}"] = _need(S.out_path(ds, s), f"{ds} s{s} selected sets")
        for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
            entry["runs_U"][cid] = _need(run_path(ds, cid, "U"), f"{ds} {cid} run on U")
        rec["datasets"][ds] = entry
    return rec


def _need_text(path: Path, what: str) -> str:
    if not path.exists():
        raise FreezeError(f"missing {what}: {path}")
    return path.read_text(encoding="utf-8")


def load() -> dict[str, Any] | None:
    from ha.paths import get_paths

    p = get_paths().frozen
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def is_signed() -> bool:
    """True only for a signed record whose content still matches its recorded hash."""
    rec = load()
    return bool(rec and rec.get("signed_by") and rec.get("record_sha256")
                and rec["record_sha256"] == record_sha256(rec))


def write() -> dict[str, Any]:
    from ha.paths import get_paths

    old = load()
    if old and old.get("signed_by"):
        raise FreezeError("FROZEN.json is signed; a later change is a separately reported -v2 "
                          "(prereg §6)")
    rec = build()
    p = get_paths().frozen
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rec


def _next_decision_id(text: str) -> str:
    nums = [int(n) for n in re.findall(r"^### D-(\d{3})", text, flags=re.MULTILINE)]
    return f"D-{max(nums, default=0) + 1:03d}"


def sign(name: str, decisions: Path | None = None) -> dict[str, Any]:
    """The lead's sign-off: signer, date and record hash in FROZEN.json, and a dated entry in
    docs/decisions.md. Refuses an empty name, a missing record or a second signature."""
    from ha.paths import REPO_ROOT, get_paths
    from ha.utils import utc_now_iso

    if not name.strip():
        raise FreezeError("a signer's name is required")
    rec = load()
    if rec is None:
        raise FreezeError("write FROZEN.json first (ha freeze)")
    if rec.get("signed_by"):
        raise FreezeError(f"already signed by {rec['signed_by']}")
    rec["signed_by"], rec["signed_at"] = name.strip(), utc_now_iso()
    rec["record_sha256"] = record_sha256(rec)
    get_paths().frozen.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    dec = decisions or REPO_ROOT / "docs" / "decisions.md"
    raw = dec.read_bytes().decode("utf-8")
    crlf, text = "\r\n" in raw, raw.replace("\r\n", "\n")
    entry = (f"\n### {_next_decision_id(text)} · {rec['signed_at'][:10]} · Freeze signed "
             f"(prereg §6 as amended by A3) · {rec['signed_by']}\n"
             f"- `results/FROZEN.json` record sha256 `{rec['record_sha256']}`; analysis code "
             f"`{rec.get('analysis', {}).get('git_sha')}`.\n"
             "- `heldout_A` may now be unsealed. Any later change to a frozen item is a "
             "separately reported -v2.\n")
    text = text.rstrip("\n") + "\n" + entry
    dec.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return rec
