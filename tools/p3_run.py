"""Drive P3 over the authorised pools (reports/phase_3_checklist.md).

  python tools/p3_run.py pilot   -> 20 passages per pool into data/synthetic_pilot/
  python tools/p3_run.py full    -> all 800 per pool into data/synthetic/ (resumable)
Writes reports/phase_3_<mode>_stats.json with the per-pool statistics and gate flags.
"""

from __future__ import annotations

import json
import sys
import time

from ha.config import load_config
from ha.gen.generate import PoolSpec, finalize_pool, generate_pool
from ha.paths import REPO_ROOT
from ha.utils import setup_logging, utc_now_iso

POOLS = (
    [PoolSpec(ds, p) for ds in ("fiqa", "cqa_android", "hiaq") for p in ("P-std", "P-persona")]
    + [PoolSpec(ds, "P-fewshot", s) for ds in ("fiqa", "cqa_android") for s in range(5)]
)


def main(mode: str) -> None:
    setup_logging()
    pilot = mode == "pilot"
    limit = 20 if pilot else None
    target = 20 if pilot else load_config("generation.yaml")["source_passages_per_dataset"]
    rows = []
    for spec in POOLS:
        t0 = time.perf_counter()
        g = generate_pool(spec, limit=limit, pilot=pilot)
        st = finalize_pool(spec, target, pilot=pilot)
        st["gen_seconds_this_run"] = g["seconds"]
        st["wall_seconds_incl_dedupe"] = round(time.perf_counter() - t0, 1)
        rows.append(st)
        print(f"{spec.dataset:12s} {spec.dir_name:14s} final {st['final_unique']:4d}/{target} "
              f"failed {st['failed']} dropped {st['dropped_by_filter']} "
              f"dupes {st['exact_duplicates']}+{st['semantic_duplicates_cos_gt_0.95']}",
              flush=True)
    out = REPO_ROOT / "reports" / f"phase_3_{mode}_stats.json"
    out.write_text(json.dumps({"mode": mode, "finished_at": utc_now_iso(), "pools": rows},
                              indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
