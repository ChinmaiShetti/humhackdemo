"""HiAQ survey: form variants and response ingest (spec §6.3 steps 2-4; Appendix D).

Variants: the 150 PASS/WARN-reviewed scenarios are dealt into 25 forms of 6, each scenario
exactly once, sources mixed, seed 17 -> data/hiaq/survey_variants.json (read by
tools/make_forms.gs). Ingest: Google Forms CSV exports (one per variant, form_XX.csv) ->
PII stripped, whitespace normalised (typos kept), language class and CMI attached ->
data/hiaq/survey_queries.parquet. Off-topic judgements are added later by the team, blind to
any method output (prereg §3.1).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ha.data.langid import classify
from ha.paths import get_paths

N_VARIANTS, PER_FORM, SEED = 25, 6, 17

# Appendix D (verbatim). Hinglish is deliberately never mentioned (spec §6.3).
FORM_TITLE = "How do you search for help? (2 minutes)"
FORM_INTRO = (
    "We are two RVCE students studying how people type questions into help pages and chatbots. "
    "You'll see 6 short situations. For each, type exactly what you would type into the app's "
    "help search or a support chatbot: short or long, any language or script you normally use, "
    "typos are fine. There are no right answers. We don't collect your name, email or phone "
    "number. Your answers will be used for research and may be published as an anonymous "
    "dataset. By continuing you agree to this."
)
ITEM_HELP = "What would you type? (short text)"
LANG_QUESTION = "Which languages do you usually type in on your phone?"
LANG_OPTIONS = ["English", "Hindi in English letters", "Hindi in Devanagari", "Kannada", "Other"]
DEVICE_QUESTION = "Did you imagine using a phone or a computer?"
DEVICE_OPTIONS = ["Phone", "Computer"]

PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")),
]


def strip_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


def hiaq_dir() -> Path:
    return get_paths().data / "hiaq"


def build_variants(scenarios: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """scenarios: scenario_id, source, scenario. Deal sources round-robin so forms mix services."""
    if len(scenarios) != N_VARIANTS * PER_FORM:
        raise ValueError(f"need exactly {N_VARIANTS * PER_FORM} scenarios, got {len(scenarios)}")
    rng = np.random.default_rng(seed)
    order = []
    for _, grp in scenarios.groupby("source", sort=True):
        order.append(list(rng.permutation(grp.index)))
    dealt: list[int] = []
    while any(order):
        for lst in order:
            if lst:
                dealt.append(lst.pop())
    forms: list[list[int]] = [[] for _ in range(N_VARIANTS)]
    for i, idx in enumerate(dealt):
        forms[i % N_VARIANTS].append(idx)
    variants = []
    for v, idxs in enumerate(forms, start=1):
        idxs = list(rng.permutation(idxs))
        variants.append({
            "variant": v,
            "items": [{"scenario_id": scenarios.at[i, "scenario_id"],
                       "scenario": scenarios.at[i, "scenario"]} for i in idxs],
        })
    return variants


def load_check_exceptions() -> dict[str, dict]:
    """Team-approved spec §6.3 exceptions (configs/scenario_check_exceptions.yaml; D-052/D-053)."""
    from ha.config import load_config

    return load_config("scenario_check_exceptions.yaml").get("exceptions") or {}


def gate_check(sheet: pd.DataFrame, check: pd.DataFrame,
               exceptions: dict[str, dict]) -> list[str]:
    """Fail closed. PASS/WARN pass. A REJECT passes only if its ID is allowlisted with the same
    checker reason and the same scenario text (sha256). Anything else blocks. Returns the IDs
    of the exceptions applied."""
    if sorted(check["scenario_id"]) != sorted(sheet["scenario_id"]):
        raise RuntimeError("scenario_check.csv does not cover the sheet's scenarios: re-check")
    text = dict(zip(sheet["scenario_id"], sheet["scenario"], strict=True))
    applied, blocked = [], []
    for r in check.itertuples():
        if r.status in ("PASS", "WARN"):
            continue
        e = exceptions.get(r.scenario_id) or {}
        ok = (r.status == "REJECT" and bool(e.get("decision"))
              and r.reasons == e.get("checker_reason")
              and hashlib.sha256(text[r.scenario_id].encode("utf-8")).hexdigest()
              == e.get("scenario_sha256"))
        (applied if ok else blocked).append(r.scenario_id)
    if blocked:
        raise RuntimeError(f"{len(blocked)} scenarios are not PASS/WARN yet and have no approved "
                           f"exception {blocked}: rewrite and re-check")
    return applied


def write_variants() -> list[dict]:
    sheet = pd.read_csv(hiaq_dir() / "scenario_sheet.csv", encoding="utf-8-sig",
                        keep_default_na=False)
    check = pd.read_csv(hiaq_dir() / "scenario_check.csv", encoding="utf-8-sig",
                        keep_default_na=False)
    exceptions = load_check_exceptions()
    applied = gate_check(sheet, check, exceptions)
    variants = build_variants(sheet)
    payload = {"title": FORM_TITLE, "intro": FORM_INTRO, "item_help": ITEM_HELP,
               "lang_question": LANG_QUESTION, "lang_options": LANG_OPTIONS,
               "device_question": DEVICE_QUESTION, "device_options": DEVICE_OPTIONS,
               "variants": variants,
               "check_exceptions": [{"scenario_id": s, "decision": exceptions[s]["decision"],
                                     "checker_reason": exceptions[s]["checker_reason"]}
                                    for s in applied]}
    (hiaq_dir() / "survey_variants.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return variants


# ---- form creation record (tools/make_forms.gs writes one MANIFEST_JSON line per form)
MANIFEST_MARK = "MANIFEST_JSON "
MANIFEST_FIELDS = ("form_number", "form_id", "edit_url", "response_url", "scenario_ids",
                   "accepting_responses", "created_at")


def variants_fingerprint(payload: dict) -> str:
    """sha256 of the variants as JSON.stringify writes them (EXPECTED_VARIANTS_SHA256, .gs)."""
    text = json.dumps(payload["variants"], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_forms_log(log_text: str) -> list[dict]:
    return [json.loads(line[line.index(MANIFEST_MARK) + len(MANIFEST_MARK):])
            for line in log_text.splitlines() if MANIFEST_MARK in line]


def check_forms_manifest(rows: list[dict], payload: dict) -> list[str]:
    """Every form must match its variant exactly: IDs in order, item text verbatim, closed."""
    items = {v["variant"]: v["items"] for v in payload["variants"]}
    probs = []
    if sorted(r["form_number"] for r in rows) != sorted(items):
        probs.append(f"form numbers {sorted(r['form_number'] for r in rows)} != 1..{len(items)}")
    if len({r["form_id"] for r in rows}) != len(rows):
        probs.append("duplicate form_id")
    for r in rows:
        want = items.get(r["form_number"], [])
        n = r["form_number"]
        if r["scenario_ids"] != [it["scenario_id"] for it in want]:
            probs.append(f"form {n}: scenario IDs differ from the variant")
        if r["item_titles"] != [f'"{it["scenario"]}"' for it in want]:
            probs.append(f"form {n}: item text differs from the scenario sheet")
        if not all(str(r[k]).startswith("https://docs.google.com/forms/")
                   for k in ("edit_url", "response_url")):
            probs.append(f"form {n}: unexpected URL")
        if r["accepting_responses"] is not False:
            probs.append(f"form {n}: is accepting responses")
    return probs


def record_forms_manifest(log_path: Path | None = None) -> dict:
    """forms_creation_log.txt -> forms_manifest.json, only if all 25 forms check out."""
    from ha.utils import utc_now_iso

    log_path = log_path or hiaq_dir() / "forms_creation_log.txt"
    raw = (hiaq_dir() / "survey_variants.json").read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    rows = parse_forms_log(log_path.read_text(encoding="utf-8"))
    probs = check_forms_manifest(rows, payload)
    if probs:
        raise ValueError("; ".join(probs))
    manifest = {"survey_variants_sha256": hashlib.sha256(raw).hexdigest(),
                "variants_fingerprint": variants_fingerprint(payload),
                "recorded_at": utc_now_iso(),
                "forms": [{k: r[k] for k in MANIFEST_FIELDS}
                          for r in sorted(rows, key=lambda r: r["form_number"])]}
    (hiaq_dir() / "forms_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def ingest(export_dir: Path | None = None, pilot: bool = False) -> pd.DataFrame:
    """Read form_XX.csv exports (Google Forms -> Sheets -> CSV). Columns by position:
    Timestamp, 6 scenario answers (in variant order), languages, device.
    pilot=True reads pilot_responses/ and writes pilot_queries.parquet, never the dataset file."""
    export_dir = export_dir or hiaq_dir() / ("pilot_responses" if pilot else "responses")
    payload = json.loads((hiaq_dir() / "survey_variants.json").read_text(encoding="utf-8"))
    by_variant = {v["variant"]: v["items"] for v in payload["variants"]}
    sheet = pd.read_csv(hiaq_dir() / "scenario_sheet.csv", encoding="utf-8-sig",
                        keep_default_na=False).set_index("scenario_id")
    rows = []
    for path in sorted(export_dir.glob("form_*.csv")):
        variant = int(re.search(r"form_(\d+)", path.stem).group(1))
        items = by_variant[variant]
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for r_idx, r in df.iterrows():
            respondent = hashlib.sha1(f"{variant}:{r_idx}:{r.iloc[0]}".encode()).hexdigest()[:12]
            langs = r.iloc[1 + PER_FORM] if len(r) > 1 + PER_FORM else ""
            device = r.iloc[2 + PER_FORM] if len(r) > 2 + PER_FORM else ""
            for j, item in enumerate(items):
                raw = str(r.iloc[1 + j])
                text, pii = strip_pii(re.sub(r"\s+", " ", raw).strip())
                rows.append({
                    "qid": f"hiaq-{respondent}-{j}",
                    "respondent": respondent,
                    "variant": variant,
                    "scenario_id": item["scenario_id"],
                    "chunk_id": sheet.at[item["scenario_id"], "chunk_id"],
                    "text": text,
                    "pii_removed": ",".join(pii),
                    "empty": text == "",
                    "languages": langs,
                    "device": device,
                    "submitted_date": str(r.iloc[0])[:10],
                })
    out = pd.DataFrame(rows)
    if len(out):
        lang = [classify(t) if t else {"label": "", "cmi": 0.0} for t in out["text"]]
        out["lang_class"] = [x["label"] for x in lang]
        out["cmi"] = [x["cmi"] for x in lang]
        name = "pilot_queries.parquet" if pilot else "survey_queries.parquet"
        out.to_parquet(hiaq_dir() / name, index=False)
    return out


# ---- collaborative-sheet set (D-056): real people type into one shared sheet. A SEPARATE,
# labelled set: never survey_queries.parquet, never HiAQ's authentic data, splits or main test.
COLLAB_ROUNDS = 4
COLLAB_COLS = ["block", "round", "form", "item", "scenario_id", "situation", "answer",
               "contributor_code", "languages", "device"]
COLLAB_FIXED = COLLAB_COLS[:6]


def collab_dir() -> Path:
    return hiaq_dir() / "collab"


def collab_template(payload: dict, rounds: int = COLLAB_ROUNDS) -> pd.DataFrame:
    """rounds x 25 blocks of one form's 6 situations (600 slots; each scenario `rounds` times).
    Block B001-B025 = round 1 (forms 01-25), B026-B050 = round 2, ..."""
    n = len(payload["variants"])
    rows = [{"block": f"B{(r - 1) * n + v['variant']:03d}", "round": r, "form": v["variant"],
             "item": j, "scenario_id": it["scenario_id"], "situation": it["scenario"]}
            for r in range(1, rounds + 1) for v in payload["variants"]
            for j, it in enumerate(v["items"], start=1)]
    df = pd.DataFrame(rows)
    for c in COLLAB_COLS[len(COLLAB_FIXED):]:
        df[c] = ""
    return df[COLLAB_COLS]


def check_collab(filled: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    """Fixed columns untouched; every answer has a contributor code; one contributor per block;
    nobody answers the same situation twice (i.e. takes two blocks of the same form)."""
    if list(filled.columns) != COLLAB_COLS or len(filled) != len(template):
        return ["columns or row count differ from the template"]
    probs = []
    edited = (filled[COLLAB_FIXED].astype(str) != template[COLLAB_FIXED].astype(str)).any(axis=1)
    if edited.any():
        probs.append(f"template columns edited in sheet rows {list(filled.index[edited][:5] + 2)}")
    ans = filled["answer"].str.strip() != ""
    code = filled["contributor_code"].str.strip()
    if (ans & (code == "")).any():
        probs.append("answers without contributor_code in blocks "
                     f"{sorted(set(filled.loc[ans & (code == ''), 'block']))[:5]}")
    taken = filled[code != ""]
    per_block = taken.groupby("block")["contributor_code"].nunique()
    shared = list(per_block[per_block > 1].index[:5])
    if shared:
        probs.append(f"more than one contributor in blocks {shared}")
    if filled[ans & (code != "")].duplicated(["contributor_code", "scenario_id"]).any():
        probs.append("a contributor answered the same situation twice (two blocks of one form)")
    return probs


def write_collab_template() -> Path:
    payload = json.loads((hiaq_dir() / "survey_variants.json").read_text(encoding="utf-8"))
    path = collab_dir() / "collab_sheet_template.csv"
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it first to regenerate")
    path.parent.mkdir(parents=True, exist_ok=True)
    collab_template(payload).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def ingest_collab(path: Path | None = None) -> pd.DataFrame:
    """Filled sheet (CSV export) -> collab/collab_queries.parquet, source='collab'. Refuses a
    sheet that fails check_collab. Same cleaning as the survey: PII stripped, whitespace only,
    typos kept. Untaken blocks are skipped."""
    path = path or collab_dir() / "collab_sheet_filled.csv"
    payload = json.loads((hiaq_dir() / "survey_variants.json").read_text(encoding="utf-8"))
    filled = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    probs = check_collab(filled, collab_template(payload).astype(str))
    if probs:
        raise ValueError("; ".join(probs))
    sheet = pd.read_csv(hiaq_dir() / "scenario_sheet.csv", encoding="utf-8-sig",
                        keep_default_na=False).set_index("scenario_id")
    rows = []
    for block, g in filled.groupby("block", sort=True):
        code = next((c for c in g["contributor_code"].str.strip() if c), "")
        if not code:
            continue
        who = hashlib.sha1(f"collab:{code}".encode()).hexdigest()[:12]   # code never stored
        langs = next((x.strip() for x in g["languages"] if x.strip()), "")
        device = next((x.strip() for x in g["device"] if x.strip()), "")
        for r in g.itertuples():
            text, pii = strip_pii(re.sub(r"\s+", " ", r.answer).strip())
            rows.append({"qid": f"collab-{block}-{r.item}", "source": "collab",
                         "respondent": who, "block": block, "variant": int(r.form),
                         "scenario_id": r.scenario_id,
                         "chunk_id": sheet.at[r.scenario_id, "chunk_id"], "text": text,
                         "pii_removed": ",".join(pii), "empty": text == "",
                         "languages": langs, "device": device})
    out = pd.DataFrame(rows)
    if len(out):
        lang = [classify(t) if t else {"label": "", "cmi": 0.0} for t in out["text"]]
        out["lang_class"] = [x["label"] for x in lang]
        out["cmi"] = [x["cmi"] for x in lang]
        out.to_parquet(collab_dir() / "collab_queries.parquet", index=False)
    return out
