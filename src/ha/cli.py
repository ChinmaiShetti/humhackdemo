"""`ha` command-line entry point (spec §20.1). Every command is idempotent and resumable."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="HumanAlign-RAG research pipeline.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset building (P1).")
calib_app = typer.Typer(no_args_is_help=True, help="Judge calibration with human labels (P4).")
systems_app = typer.Typer(no_args_is_help=True, help="Retrieval configurations C1-C7 (P2).")
paper_app = typer.Typer(no_args_is_help=True, help="Paper tables and figures (P8).")
models_app = typer.Typer(no_args_is_help=True, help="Verify the locked local model stack (P0.2).")
registry_app = typer.Typer(no_args_is_help=True, help="Results registry tools.")
kaggle_app = typer.Typer(no_args_is_help=True, help="Kaggle packaging (optional offload, D-006).")
for sub, name in [
    (data_app, "data"),
    (calib_app, "calib"),
    (systems_app, "systems"),
    (paper_app, "paper"),
    (models_app, "models"),
    (registry_app, "registry"),
    (kaggle_app, "kaggle"),
]:
    app.add_typer(sub, name=name)


def _todo(wp: str) -> None:
    typer.secho(f"Not implemented yet: work package {wp} (see docs/PLAN.md).", fg="yellow")
    raise typer.Exit(code=2)


@app.command()
def doctor() -> None:
    """Check the environment: CUDA, Ollama, paths, disk, UTF-8, forbidden packages."""
    from ha.doctor import run_doctor

    raise typer.Exit(code=run_doctor())


@models_app.command("verify")
def models_verify() -> None:
    """Smoke-test the locked local stack (never downloads); write env/models.json."""
    from ha.models import verify

    raise typer.Exit(code=0 if verify() else 1)


@models_app.command("repair-reranker")
def models_repair_reranker() -> None:
    """Rebuild the reranker folder from hard links to cached blobs (no download, 0 bytes)."""
    from ha.models import repair_reranker

    typer.echo(str(repair_reranker()))


@data_app.command("build")
def data_build(
    ds: str,
    force: Annotated[bool, typer.Option(help="Rebuild even if up to date")] = False,
) -> None:
    """Download (once) and normalise a public dataset: fiqa | cqa_android."""
    from ha.data.public import build
    from ha.utils import setup_logging

    setup_logging()
    build(ds, force=force)


@data_app.command("hiaq-scrape")
def data_hiaq_scrape(
    only: Annotated[str | None, typer.Option(help="One source from hiaq_sources.yaml")] = None,
) -> None:
    """Politely scrape vetted HiAQ help pages (robots-checked, >=2 s between requests)."""
    from ha.data.hiaq_scrape import scrape
    from ha.utils import setup_logging

    setup_logging()
    scrape(only)


@data_app.command("splits")
def data_splits(ds: str) -> None:
    """Language classes, topic clusters, CAL-auth and R/A splits (5 seeds) for a dataset."""
    from ha.config import load_config
    from ha.data.splits import build_splits
    from ha.utils import setup_logging

    setup_logging()
    n_cal = load_config("calibration.yaml")["cal_auth"].get(ds, 0)
    out = build_splits(ds, n_cal_auth=n_cal)
    typer.echo(f"{ds}: {len(out)} queries; lang {out['lang_class'].value_counts().to_dict()}; "
               f"clusters {out['topic_cluster'].nunique()}; cal_auth {int(out['cal_auth'].sum())}; "
               f"R per seed {[int((out[f'split_s{s}'] == 'R').sum()) for s in range(5)]}")


@data_app.command("hiaq-chunk")
def data_hiaq_chunk() -> None:
    """Chunk scraped HiAQ pages (heading-aware, FAQ-aware, deduplicated)."""
    from ha.data.hiaq_chunk import build_chunks
    from ha.utils import setup_logging

    setup_logging()
    build_chunks()


@data_app.command("hiaq-sheet")
def data_hiaq_sheet() -> None:
    """Write the 150-chunk scenario sheet (data/hiaq/scenario_sheet.csv) for the team."""
    from ha.data.scenarios import write_sheet

    sheet = write_sheet()
    typer.echo(f"{len(sheet)} rows; by source {sheet['source'].value_counts().to_dict()}; "
               f"topics {sheet['topic'].nunique()}")


@data_app.command("hiaq-check")
def data_hiaq_check(
    sheet: Annotated[str | None, typer.Option(help="Sheet to check; default official")] = None,
    out: Annotated[str | None, typer.Option(help="Output CSV; default beside sheet")] = None,
) -> None:
    """Leakage-check scenarios (spec §6.3). Never modifies the sheet. A non-official sheet
    writes `<stem>_check.csv` beside itself, never the official scenario_check.csv."""
    from ha.data.scenarios import check_output_path, check_sheet, hiaq_dir

    sheet_path = Path(sheet) if sheet else hiaq_dir() / "scenario_sheet.csv"
    dest = Path(out) if out else check_output_path(sheet_path)
    res = check_sheet(sheet_path, dest)
    typer.echo(f"{res['status'].value_counts().to_dict()} -> {dest}")


@data_app.command("hiaq-replace")
def data_hiaq_replace(
    ids: Annotated[list[str], typer.Argument(help="Scenario IDs marked skip, e.g. S017 S025")],
    out: Annotated[str, typer.Option(help="Output CSV")] = (
        "data/humanside/scenario_replacements.csv"),
    exclude: Annotated[list[str] | None, typer.Option(
        help="Replacement chunk IDs the team rejected (repeatable)")] = None,
) -> None:
    """Draw deterministic replacement CHUNKS for skipped rows (D-044). Reads the official sheet
    read-only; writes only the replacements file. Scenarios for them are written by the team."""
    import pandas as pd

    from ha.data.scenarios import draw_replacements, hiaq_dir
    from ha.paths import REPO_ROOT

    sheet_df = pd.read_csv(hiaq_dir() / "scenario_sheet.csv", encoding="utf-8-sig",
                           keep_default_na=False)
    chunks = pd.read_parquet(hiaq_dir() / "chunks.parquet")
    rep = draw_replacements(sheet_df, chunks, ids, exclude=exclude)
    path = Path(out) if Path(out).is_absolute() else REPO_ROOT / out
    rep.to_csv(path, index=False, encoding="utf-8-sig")
    typer.echo(rep[["replaces_scenario_id", "new_chunk_id", "topic"]].to_string(index=False))


@data_app.command("hiaq-variants")
def data_hiaq_variants() -> None:
    """Deal the checked scenarios into 25 survey forms -> survey_variants.json."""
    from ha.data.survey import write_variants

    typer.echo(f"{len(write_variants())} variants written")


@data_app.command("hiaq-forms-manifest")
def data_hiaq_forms_manifest(
    log: Annotated[str | None, typer.Option(help="Apps Script log; default "
                                                 "data/hiaq/forms_creation_log.txt")] = None,
) -> None:
    """Check the forms created by tools/make_forms.gs against survey_variants.json and write
    data/hiaq/forms_manifest.json (refuses if any form differs)."""
    from ha.data.survey import record_forms_manifest

    m = record_forms_manifest(Path(log) if log else None)
    typer.echo(f"{len(m['forms'])} forms verified -> forms_manifest.json")


@data_app.command("hiaq-ingest")
def data_hiaq_ingest(
    pilot: Annotated[bool, typer.Option(
        help="Pilot: read data/hiaq/pilot_responses/, write pilot_queries.parquet")] = False,
) -> None:
    """Ingest form_XX.csv exports: strip PII, keep typos, attach language class and CMI."""
    from ha.data.survey import ingest

    out = ingest(pilot=pilot)
    if len(out):
        valid = out[~out["empty"]]
        typer.echo(f"{len(valid)} non-empty answers from {out['respondent'].nunique()} "
                   f"respondents; PII removed in {(out['pii_removed'] != '').sum()}; "
                   f"language classes {valid['lang_class'].value_counts().to_dict()}")
    else:
        folder = "pilot_responses" if pilot else "responses"
        typer.echo(f"no responses found in data/hiaq/{folder}/")


@data_app.command("hiaq-collab-sheet")
def data_hiaq_collab_sheet() -> None:
    """D-056: write the shared-sheet template (100 blocks x 6 = 600 answer rows)."""
    from ha.data.survey import write_collab_template

    typer.echo(f"wrote {write_collab_template()}")


@data_app.command("hiaq-collab-ingest")
def data_hiaq_collab_ingest(
    sheet: Annotated[str | None, typer.Option(
        help="Filled sheet CSV; default data/hiaq/collab/collab_sheet_filled.csv")] = None,
) -> None:
    """D-056: check the filled shared sheet and write collab/collab_queries.parquet (a separate,
    labelled set; never the survey dataset)."""
    from ha.data.survey import ingest_collab

    out = ingest_collab(Path(sheet) if sheet else None)
    valid = out[~out["empty"]] if len(out) else out
    typer.echo(f"{len(valid)} non-empty answers from "
               f"{out['respondent'].nunique() if len(out) else 0} contributors "
               f"({out['block'].nunique() if len(out) else 0}/100 blocks taken)")


@app.command()
def gen(
    ds: str,
    pool: str,
    split_seed: Annotated[int | None, typer.Option(help="P-fewshot: R split seed")] = None,
    limit: Annotated[int | None, typer.Option(help="Pilot: first N passages only")] = None,
    finalize: Annotated[bool, typer.Option(help="Dedupe and write queries/stats")] = False,
    pilot: Annotated[bool, typer.Option(help="Write to data/synthetic_pilot/")] = False,
) -> None:
    """Generate a synthetic pool (P-std | P-persona | P-fewshot), resumable; spec §7."""
    import json

    from ha.config import load_config
    from ha.gen.generate import PoolSpec, finalize_pool, generate_pool
    from ha.utils import setup_logging

    setup_logging()
    spec = PoolSpec(ds, pool, split_seed)
    typer.echo(generate_pool(spec, limit=limit, pilot=pilot))
    if finalize:
        target = limit or load_config("generation.yaml")["source_passages_per_dataset"]
        typer.echo(json.dumps(finalize_pool(spec, target, pilot=pilot), indent=2, default=str))


@app.command()
def neigh(ds: str) -> None:
    """P4.1: authentic neighbours from R_s (seeds 0-4) for U; bge-m3 top-30, rerank top-5."""
    from ha.neighbours.retrieve import build
    from ha.utils import setup_logging

    setup_logging()
    for m in build(ds):
        typer.echo(m)


@app.command()
def judge(
    ds: str,
    seeds: Annotated[str, typer.Option(help="Split seeds for M1, comma-separated")] = "0,1,2,3,4",
    iso: Annotated[bool, typer.Option(help="Also run M1-iso (once per U query)")] = True,
    limit: Annotated[int | None, typer.Option(help="Smoke: first N U queries only")] = None,
) -> None:
    """P5.1 bulk judging over U (prompt v0, A3): M1-iso once, M1 per split seed. Resumable."""
    import json

    from ha.judge.bulk import BulkBlocked, run_bulk
    from ha.utils import setup_logging

    setup_logging()
    try:
        out = run_bulk(ds, tuple(int(s) for s in seeds.split(",")), limit, iso)
    except BulkBlocked as e:
        typer.echo(f"BLOCKED: {e}", err=True)
        raise typer.Exit(3) from e
    typer.echo(json.dumps(out, indent=2))


@calib_app.command("status")
def calib_status() -> None:
    """P4.3: readiness of the 200-item pack (CAL-auth counts, qrels passages)."""
    import json

    from ha.human.calibration import status

    typer.echo(json.dumps(status(), indent=2))


@calib_app.command("export")
def calib_export(practice: bool = typer.Option(False, help="Only the 20 practice items."),
                 round_: int = typer.Option(1, "--round")) -> None:
    """P4.3: write the blind annotator sheets and the separate key (D-050; spreadsheet)."""
    import json

    from ha.human.calibration import prepare_main, prepare_practice
    from ha.judge.calibrate import CalibrationBlocked

    try:
        typer.echo(json.dumps(prepare_practice() if practice else prepare_main(round_), indent=2))
    except CalibrationBlocked as e:
        typer.echo(f"BLOCKED: {e}", err=True)
        raise typer.Exit(3) from e


@calib_app.command("import")
def calib_import(round_: int = typer.Option(1, "--round")) -> None:
    """P4.4: validate both raw sheets and write adjudication_r<round>.csv (raw files untouched)."""
    import json

    from ha.human import calibration as hc

    d = hc.out_dir()
    blank = hc.read_csv(d / "sheets" / "blank.csv")
    raws = [hc.read_csv(d / "sheets" / f"raw_r{round_}_{a}.csv") for a in hc.ANNOTATORS]
    probs = {a: hc.validate_sheet(s, blank) for a, s in zip(hc.ANNOTATORS, raws, strict=True)}
    if any(probs.values()):
        typer.echo(json.dumps(probs, indent=2), err=True)
        raise typer.Exit(1)
    hc.write_csv(hc.build_adjudication(*raws), d / f"adjudication_r{round_}.csv")
    typer.echo(f"wrote {d / f'adjudication_r{round_}.csv'}")


@calib_app.command("finalize")
def calib_finalize(round_: int = typer.Option(1, "--round")) -> None:
    """P4.4: human reference labels from a fully settled adjudication sheet."""
    from ha.human import calibration as hc

    d = hc.out_dir()
    fin = hc.finalize(hc.read_csv(d / f"adjudication_r{round_}.csv"))
    fin.to_csv(d / f"final_labels_r{round_}.csv", index=False, encoding="utf-8-sig")
    typer.echo(f"wrote {d / f'final_labels_r{round_}.csv'} ({len(fin)} items)")


@calib_app.command("report")
def calib_report() -> None:
    _todo("P4.4")


@app.command("select")
def select_(
    ds: str,
    seeds: Annotated[str, typer.Option(help="Split seeds, comma-separated")] = "0,1,2,3,4",
) -> None:
    """P5.2: build B0, B1 (20 draws each), M1-iso and M1 query sets per split seed (§5.2)."""
    import json

    from ha.select.sets import run_select
    from ha.utils import setup_logging

    setup_logging()
    for m in run_select(ds, tuple(int(s) for s in seeds.split(","))):
        typer.echo(json.dumps({k: m[k] for k in ("split_seed", "N", "n_A", "binding",
                                                 "M1_passes", "M1-iso_passes", "k", "methods")}))


@systems_app.command("encode")
def systems_encode(ds: str) -> None:
    """Encode a corpus with Ollama bge-m3 (resumable); logs time and resources."""
    from ha.systems.retrieval import encode_corpus
    from ha.systems.run_all import _log_resources
    from ha.utils import setup_logging

    setup_logging()
    rec = encode_corpus(ds)
    _log_resources(ds, rec)
    typer.echo(rec)


@systems_app.command("index")
def systems_index(ds: str) -> None:
    """Build (and cache) the BM25 indexes of C1/C2 for a corpus."""
    import time

    from ha.systems.retrieval import BM25Index, systems_config
    from ha.systems.run_all import _log_resources

    for cid, c in systems_config()["configs"].items():
        if c["type"] == "bm25":
            t0 = time.perf_counter()
            BM25Index(ds, c["k1"], c["b"])
            _log_resources(ds, {"stage": "bm25_index", "dataset": ds, "config": cid,
                                "seconds": round(time.perf_counter() - t0, 2)})
    typer.echo(f"{ds}: BM25 indexes ready")


@systems_app.command("run")
def systems_run(
    ds: str,
    qset: Annotated[str, typer.Option(help="auth | smoke | U (synthetic pool)")] = "auth",
    systems: Annotated[list[str] | None, typer.Option(help="Subset of config IDs")] = None,
) -> None:
    """Run C1-C7 over a query set (incremental; held-out 'auth' is sealed)."""
    from ha.systems.run_all import run_configs
    from ha.utils import setup_logging

    setup_logging()
    runs = run_configs(ds, qset, systems)
    typer.echo({k: len(v) for k, v in runs.items()})


@app.command("eval")
def eval_(
    datasets: Annotated[list[str], typer.Argument(help="Datasets of one role (dev or held-out)")],
    seeds: Annotated[str, typer.Option(help="Split seeds, comma-separated")] = "0,1,2,3,4",
    b_iter: Annotated[int, typer.Option("--b", help="Bootstrap iterations (prereg: 1000)")] = 1000,
) -> None:
    """P5.3/P6: tau_b of B0, B1, M1-iso, M1 vs authentic A, and the §7 bootstrap (registry)."""
    import json

    from ha.eval.sets import run_eval
    from ha.utils import setup_logging

    setup_logging()
    out = run_eval(datasets, tuple(int(s) for s in seeds.split(",")), b_iter)
    typer.echo(json.dumps({"run_id": out["run_id"], "summary": out["summary"],
                           "bootstrap": out["bootstrap"]}, indent=2))


@app.command("freeze")
def freeze_(
    sign: Annotated[str | None, typer.Option(help="Signer's name: the project lead's sign-off "
                                                  "(A3). Only the lead runs this.")] = None,
) -> None:
    """Prereg §6: write results/FROZEN.json (clean commit, all inputs), or sign it."""
    import json

    from ha import freeze

    try:
        rec = freeze.sign(sign) if sign is not None else freeze.write()
    except freeze.FreezeError as e:
        typer.echo(f"REFUSED: {e}", err=True)
        raise typer.Exit(3) from e
    typer.echo(json.dumps({k: rec.get(k) for k in ("created_at", "signed_by", "signed_at",
                                                    "record_sha256")}
                          | {"analysis_git_sha": rec["analysis"]["git_sha"]}, indent=2))


@app.command("h1")
def h1_(
    datasets: Annotated[list[str], typer.Argument(help="Datasets to test (D4 counts as p = 1)")],
    seeds: Annotated[str, typer.Option(help="Split seeds, comma-separated")] = "0,1,2,3,4",
) -> None:
    """H1 descriptives: length, gold-passage overlap, topic JS; Holm over 3 x 3 (§2, D-064)."""
    import json

    from ha.eval.h1 import run_h1
    from ha.utils import setup_logging

    setup_logging()
    out = run_h1(datasets, tuple(int(s) for s in seeds.split(",")))
    typer.echo(json.dumps({"run_id": out["run_id"], "decision": out["decision"]}, indent=2))


@app.command()
def run(
    exp: str,
    latency: Annotated[bool, typer.Option(help="E0: also measure latency")] = True,
) -> None:
    """Run an experiment end to end and write to the registry (implemented: E0)."""
    if exp != "E0":
        _todo("P5 / P6 / P7")
    import json

    from ha.systems.run_all import run_e0
    from ha.utils import setup_logging

    setup_logging()
    typer.echo(json.dumps(run_e0("fiqa", with_latency=latency), indent=2))


explore_app = typer.Typer(no_args_is_help=True,
                          help="EXPLORATORY analyses (context only; never hypotheses).")
app.add_typer(explore_app, name="explore")


@explore_app.command("noise-ceiling")
def explore_noise_ceiling() -> None:
    """Authentic random-half tau_b ceiling on D1 FiQA (D-040; exploratory)."""
    import json

    from ha.eval.noise_ceiling import run as run_nc

    typer.echo(json.dumps(run_nc("fiqa"), indent=2))


@paper_app.command("build")
def paper_build() -> None:
    _todo("P8")


@app.command()
def dashboard() -> None:
    _todo("P8")


@registry_app.command("check")
def registry_check() -> None:
    """Diff results/expected_runs.yaml against the registry."""
    _todo("P6")


@kaggle_app.command("pack")
def kaggle_pack() -> None:
    """Zip the committed repo at HEAD for upload as a private Kaggle dataset."""
    from ha.kaggle import pack

    typer.echo(str(pack()))


if __name__ == "__main__":
    app()
