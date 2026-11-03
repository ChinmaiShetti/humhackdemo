"""D-028 verification of the core generator + judge (engineering check, not a research result).

- 20 generation calls (10 FiQA passages x {P-std, P-persona}): are outputs real questions, not
  reasoning traces?
- 20 judge calls: 8 generated queries x 2 neighbour permutations (matched passage) + 4 mismatch
  controls (query vs an unrelated passage; F should collapse). Neighbours: BM25 top-5 over FiQA
  *train-split* queries, so the 648 test queries are never touched.
- Latency and VRAM for both.
Output: reports/phase_0_llm_verification.json
"""

from __future__ import annotations

import json
import re
import statistics
import time

import bm25s
import pandas as pd

from ha import llm
from ha.config import load_config
from ha.judge.schema import JudgeOutput
from ha.paths import REPO_ROOT, get_paths
from ha.prompts import render
from ha.utils import set_seed, utc_now_iso

REASONING = re.compile(r"^(hmm|okay|ok,|let me|let's|we are|we need|i need|i will|the user|"
                       r"first,|so,|alright)|<think>|\n", re.IGNORECASE)


def _stats(xs):
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}


def _vram(model: str) -> float | None:
    for m in llm.client().ps().models:
        if m.model == model:
            return round(m.size_vram / 2**30, 2)
    return None


def main() -> None:
    gen_cfg, judge_cfg = load_config("generation.yaml"), load_config("judge.yaml")
    model = gen_cfg["model"]["tag"]
    digest = llm.require_digest(model, gen_cfg["model"]["digest"])
    llm.require_digest(judge_cfg["model"]["tag"], judge_cfg["model"]["digest"])
    set_seed(0)
    c = llm.client()
    llm.unload_all()

    proc = get_paths().processed / "fiqa"
    corpus = pd.read_parquet(proc / "corpus.parquet")
    test_qids = set(pd.read_parquet(proc / "queries_auth.parquet")["qid"])
    raw_q = pd.read_parquet(next((get_paths().data / "raw" / "fiqa").rglob("queries-*.parquet")))
    train_q = raw_q[~raw_q["_id"].astype(str).isin(test_qids)]["text"].tolist()
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(train_q, stopwords="en", show_progress=False),
                    show_progress=False)

    lens = corpus["text"].str.split().str.len()
    sample = corpus[(lens >= 60) & (lens <= 250)].sample(14, random_state=42)["text"].tolist()
    passages, unrelated = sample[:10], sample[10:]
    personas = gen_cfg["pools"]["P-persona"]["personas"]
    dec = gen_cfg["decoding"]

    t0 = time.perf_counter()
    c.chat(model=model, messages=[{"role": "user", "content": "Say ok."}], think=False,
           options={"num_predict": 4})
    out = {"started_at": utc_now_iso(), "model": model, "digest": digest,
           "ollama": llm.server_version(), "cold_first_call_s": round(time.perf_counter() - t0, 2)}

    # ---------------- generation: 20 calls
    gens = []
    for i, p in enumerate(passages):
        for style in ("P-std", "P-persona"):
            persona = personas[i % len(personas)]
            prompt = (render("gen_std_v1", passage=p) if style == "P-std" else
                      render("gen_persona_v1", passage=p, persona_description=persona))
            t = time.perf_counter()
            r = c.chat(model=model, messages=[{"role": "user", "content": prompt}], think=False,
                       options={"temperature": dec["temperature"], "top_p": dec["top_p"],
                                "num_predict": dec["num_predict"],
                                "seed": dec["seed_base"] + len(gens)})
            text = r.message.content.strip()
            gens.append({
                "passage_idx": i, "style": style,
                "persona": persona if style == "P-persona" else None,
                "query": text, "wall_s": time.perf_counter() - t, "output_tokens": r.eval_count,
                "hit_token_cap": r.eval_count >= dec["num_predict"],
                "reasoning_marker": bool(REASONING.search(text)),
                "words": len(text.split()),
            })
    out["vram_gib_after_generation"] = _vram(model)
    out["generation"] = {
        "wall_s": _stats([g["wall_s"] for g in gens]),
        "output_tokens": _stats([g["output_tokens"] for g in gens]),
        "words": _stats([g["words"] for g in gens]),
        "n_reasoning_marker": sum(g["reasoning_marker"] for g in gens),
        "n_hit_token_cap": sum(g["hit_token_cap"] for g in gens),
        "n_ends_with_question_mark": sum(g["query"].rstrip().endswith("?") for g in gens),
        "outputs": [{k: g[k] for k in ("style", "persona", "query")} for g in gens],
    }

    # ---------------- judge: 16 matched + 4 mismatch = 20 calls
    system = render(judge_cfg["prompt"]["system"])
    jdec = judge_cfg["decoding"]
    calls = []
    matched = [g for g in gens if g["style"] == "P-std"][:4] + \
              [g for g in gens if g["style"] == "P-persona"][:4]
    plan = [(g, passages[g["passage_idx"]], perm, "matched") for g in matched for perm in (0, 1)]
    plan += [(g, unrelated[k], 0, "mismatch") for k, g in enumerate(matched[:4])]
    for g, passage, perm, kind in plan:
        ids, _ = retriever.retrieve(bm25s.tokenize([g["query"]], stopwords="en",
                                                   show_progress=False), k=5, show_progress=False)
        neigh = [train_q[j] for j in ids[0]]
        if perm == 1:
            neigh = list(reversed(neigh))
        user = render(judge_cfg["prompt"]["user"], domain_description="personal finance Q&A",
                      passage=passage, neighbours=neigh, query=g["query"], lang_class="EN")
        t = time.perf_counter()
        r = c.chat(model=model, messages=[{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
                   format=JudgeOutput.model_json_schema(), think=False,
                   options={"temperature": jdec["temperature"], "seed": jdec["seed"],
                            "num_ctx": jdec["num_ctx"]})
        wall = time.perf_counter() - t
        try:
            j = JudgeOutput.model_validate_json(r.message.content).model_dump()
        except Exception as e:  # noqa: BLE001
            j = {"invalid": str(e)[:200]}
        calls.append({"kind": kind, "perm": perm, "style": g["style"], "query": g["query"],
                      "judge": j, "wall_s": wall, "prompt_tokens": r.prompt_eval_count,
                      "output_tokens": r.eval_count})
    out["vram_gib_after_judge"] = _vram(model)
    valid = [x for x in calls if "invalid" not in x["judge"]]

    def mean_dim(kind: str, dim: str) -> float | None:
        vals = [x["judge"][dim] for x in valid if x["kind"] == kind]
        return round(statistics.mean(vals), 2) if vals else None

    perm_pairs = [(a, b) for a in valid for b in valid
                  if a["kind"] == b["kind"] == "matched" and a["query"] == b["query"]
                  and a["perm"] == 0 and b["perm"] == 1]
    out["judge"] = {
        "wall_s": _stats([x["wall_s"] for x in calls]),
        "prompt_tokens": _stats([x["prompt_tokens"] for x in calls]),
        "output_tokens": _stats([x["output_tokens"] for x in calls]),
        "json_valid": f"{len(valid)}/{len(calls)}",
        "mean_F_matched": mean_dim("matched", "F"),
        "mean_F_mismatch": mean_dim("mismatch", "F"),
        "mean_H_matched": mean_dim("matched", "H"),
        "mean_S_matched": mean_dim("matched", "S"),
        "mean_P_matched": mean_dim("matched", "P"),
        "C_null_on_english": sum(x["judge"].get("C") is None for x in valid),
        "perm_pairs_any_dim_diff_ge2": sum(
            any(abs(a["judge"][d] - b["judge"][d]) >= 2 for d in "FHSP") for a, b in perm_pairs),
        "n_perm_pairs": len(perm_pairs),
        "calls": [{k: x[k] for k in ("kind", "perm", "style", "query", "judge")} for x in calls],
    }
    out["finished_at"] = utc_now_iso()
    llm.unload_all()
    path = REPO_ROOT / "reports" / "phase_0_llm_verification.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: v for k, v in out.items() if k not in ("generation", "judge")}
    summary["generation"] = {k: v for k, v in out["generation"].items() if k != "outputs"}
    summary["judge"] = {k: v for k, v in out["judge"].items() if k != "calls"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n--- generated queries")
    for g in out["generation"]["outputs"]:
        print(f"[{g['style']:9s}] {g['query']}")
    print("\n--- judge calls")
    for x in out["judge"]["calls"]:
        j = x["judge"]
        s = (f"F{j['F']} H{j['H']} S{j['S']} P{j['P']} C{j['C']} keep={j['keep']} {j['reason']}"
             if "F" in j else j)
        print(f"[{x['kind']:8s} p{x['perm']}] {s} | {x['query'][:80]}")


if __name__ == "__main__":
    main()
