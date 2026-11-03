"""Calibration set, P4.3 (prereg §4.1-4.2 as amended by A2; configs/calibration.yaml).

prereg-v1: 200 items (CAL-auth 30 FiQA + 30 HiAQ, synthetic 70 FiQA + 70 HiAQ), split 120/80.
Amendment A2 (D-059): no authentic HiAQ data exist, so the HiAQ half is removed. That leaves
100 items: the 30 FiQA CAL-auth (fixed in R for every seed, D-003) and 70 synthetic FiQA items
(35 P-std + 35 P-persona, seed 17, seed-independent U pools), split 60 cal-dev / 40 cal-test,
stratified by dataset x authentic/synthetic, seed 17. cal-test is sealed (scope `cal_test`).

The config names the amendment it implements (`amendment: A2`). build_items refuses to run until
that amendment is recorded under '## Amendments' in docs/preregistration.md, so no calibration
item is built under an unrecorded design. A dataset with zero items in the config is skipped;
any dataset with non-zero items must supply exactly that many CAL-auth items, or the build
stops (no partial sets).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ha import prereg
from ha.config import load_config


class CalibrationBlocked(RuntimeError):
    pass


def _cfg():
    return load_config("calibration.yaml")


def amendment_recorded(tag: str) -> bool:
    """True only if a heading '### <tag> · ...' exists below '## Amendments' in the prereg."""
    text = prereg.PREREG_PATH.read_text(encoding="utf-8")
    if prereg.AMENDMENTS_HEADING not in text:
        return False
    tail = text.split(prereg.AMENDMENTS_HEADING, 1)[1]
    return re.search(rf"^###\s+{re.escape(tag)}\s+·", tail, re.MULTILINE) is not None


def draw_synthetic(U: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n items, half P-std and half P-persona, deterministic (sorted by qid, then seeded)."""
    rng = np.random.default_rng(seed)
    out = []
    for pool, k in (("P-std", n // 2), ("P-persona", n - n // 2)):
        cand = U[U["pool"] == pool].sort_values("qid").reset_index(drop=True)
        out.append(cand.iloc[sorted(rng.choice(len(cand), size=k, replace=False))])
    return pd.concat(out, ignore_index=True)


def build_items(fiqa_auth: pd.DataFrame | None, hiaq_auth: pd.DataFrame | None,
                fiqa_U: pd.DataFrame, hiaq_U: pd.DataFrame | None) -> pd.DataFrame:
    cfg = _cfg()
    tag = cfg.get("amendment")
    if tag and not amendment_recorded(tag):
        raise CalibrationBlocked(f"configs/calibration.yaml implements amendment {tag}, which is "
                                 "not recorded under '## Amendments' in docs/preregistration.md")
    parts = []
    for ds, auth, U in (("fiqa", fiqa_auth, fiqa_U), ("hiaq", hiaq_auth, hiaq_U)):
        n_auth, n_syn = cfg["cal_auth"][ds], cfg["synthetic"][ds]
        if n_auth:
            if auth is None or len(auth) != n_auth:
                got = 0 if auth is None else len(auth)
                raise CalibrationBlocked(f"{ds} CAL-auth must have exactly {n_auth} items, "
                                         f"got {got}")
            parts.append(auth[["qid", "text"]].assign(dataset=ds, kind="authentic",
                                                      pool="authentic"))
        if n_syn:
            s = draw_synthetic(U, n_syn, cfg["seed"])[["qid", "text", "pool"]]
            parts.append(s.assign(dataset=ds, kind="synthetic"))
    items = pd.concat(parts, ignore_index=True)
    total = sum(cfg["cal_auth"].values()) + sum(cfg["synthetic"].values())
    assert len(items) == total and items["qid"].is_unique
    return items


def split_items(items: pd.DataFrame) -> pd.DataFrame:
    """Stratified cal-dev/cal-test by dataset x kind (each stratum split by the config ratio,
    60/40), seed 17. Strata are visited in sorted order with one RNG, so the FiQA strata receive
    the same assignment under A2 as under prereg-v1."""
    cfg = _cfg()
    rng = np.random.default_rng(cfg["seed"])
    frac = cfg["split"]["cal_dev"] / (cfg["split"]["cal_dev"] + cfg["split"]["cal_test"])
    items = items.copy()
    items["split"] = "cal-test"
    for _, g in items.sort_values("qid").groupby(["dataset", "kind"], sort=True):
        dev = rng.choice(g.index.to_numpy(), size=int(round(frac * len(g))), replace=False)
        items.loc[dev, "split"] = "cal-dev"
    counts = items["split"].value_counts().to_dict()
    assert counts == {"cal-dev": cfg["split"]["cal_dev"], "cal-test": cfg["split"]["cal_test"]}
    return items
