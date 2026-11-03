"""Smoke test for a running Arzi server: one question per language, then a grievance draft.

    .venv\\Scripts\\python.exe app/server/smoke.py [--base http://127.0.0.1:8765]
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

QUESTIONS = [
    "Naukri chhodne ke baad PF kaise nikalein?",
    "पैन को आधार से कैसे लिंक करें?",
    "My employer deducted PF from my salary but has not deposited it. What can I do?",
]


def ask(base: str, q: str) -> tuple[str, list[dict]]:
    tokens, sources, t0 = [], [], time.perf_counter()
    with httpx.stream("POST", f"{base}/api/ask", json={"question": q}, timeout=120) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if event == "token":
                    tokens.append(d["t"])
                elif event == "sources":
                    sources = d["sources"]
                    print(f"  retrieval {d['retrieval_ms']} ms, terms {d['terms']}")
                    for s in sources:
                        print(f"   {s['rank']} bm25={s['bm25_rank']} dense={s['dense_rank']} "
                              f"{s['src']:9} {s['title'][:70]}")
                elif event in ("lang", "done", "error"):
                    print(f"  {event}: {d}")
    print(f"  ({time.perf_counter() - t0:.1f} s)")
    return "".join(tokens), sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    base = ap.parse_args().base
    history = []
    for q in QUESTIONS:
        print(f"=== {q}")
        answer, _ = ask(base, q)
        print(answer, "\n")
        history = [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/api/grievance", timeout=180,
                   json={"messages": history, "lang": "EN",
                         "details": "Deductions since March, nothing in my passbook."})
    print(f"=== grievance ({time.perf_counter() - t0:.1f} s)")
    print(json.dumps(r.json(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
