"""Arzi demo API: multilingual citizen help over the HiAQ corpus (Income Tax + EPFO help pages).

Retrieval is the study's own hybrid (C4): BM25 (C1: k1 0.9, b 0.4) and bge-m3 dense (C3), fused by
reciprocal rank (k 60). Answers and grievance drafts come from the pinned Qwen3-4B-Instruct.
The reranker is left out on purpose: it is a torch model and cannot share the 6 GB GPU with Ollama.

    .venv\\Scripts\\python.exe -m uvicorn app.server.server:app --port 8765

Endpoints
  GET  /api/health
  POST /api/ask        {question, history?}  -> text/event-stream: lang, sources, token*, done
  POST /api/grievance  {messages, lang, details?} -> JSON draft (never submitted anywhere)
  POST /api/xray       {question}            -> per-stage rankings for the lab page
Anything else is served from app/web/dist when it has been built.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd
import regex
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ha.config import load_config
from ha.data.langid import classify
from ha.systems.retrieval import BM25Index, DenseIndex, rrf, systems_config

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "app" / "web" / "dist"
FIELD = ROOT / "app" / "web" / "public" / "field.json"
CHAT = next(m for m in load_config("models.yaml")["ollama"] if m["kind"] == "chat")["tag"]
EMBED = next(m for m in load_config("models.yaml")["ollama"] if m["kind"] == "embed")["tag"]
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEPTH, TOP_K = 20, 5
NUM_CTX = 6144  # chat context; one value everywhere, or Ollama reloads the model between calls

STOPWORDS = set("""a an the and or but if then of to in on at by for with from as is are was were be
been being do does did have has had i me my we our you your he she it they them this that these
those what which who whom how when where why can could should would will shall may might must
not no yes so than too very just about into over after before after up down out off again
any all some such only own same other more most also get got please tell know want need""".split())
TOKEN = regex.compile(r"[\p{L}\p{M}\p{N}][\p{L}\p{M}\p{N}\-]*")  # keeps Hindi vowel signs
LATIN = regex.compile(r"\p{Latin}")

# The style examples are deliberately off-topic: an on-topic example gets copied into answers.
_HINGLISH = ("Hinglish: Hindi sentences written in Roman (English) letters, keeping terms such as "
             "PF, UAN, ITR, PAN, refund or portal in English. Style only, not content: \"Train "
             "subah 7 baje platform 2 se chalti hai.\"")
LANG_NAME = {
    "EN": "English",
    "DEV_HI": ("Hindi in Devanagari script (keep form names such as ITR-1 or UAN as they are). "
               "Style only, not content: \"ट्रेन सुबह 7 बजे प्लेटफ़ॉर्म 2 से चलती है।\""),
    "HINGLISH": _HINGLISH,
    "ROM_HI": _HINGLISH,
}
LANG_SHORT = {"EN": "English", "DEV_HI": "Hindi (Devanagari)",
              "HINGLISH": "Hinglish (Roman script)", "ROM_HI": "Hinglish (Roman script)"}

ANSWER_SYSTEM = """You are Arzi, a help desk for Indian citizens with Income Tax e-filing and EPFO \
(provident fund) questions.
Rules:
- Use ONLY the numbered official passages below. Put the passage number in square brackets right \
after each sentence it supports, like [2] or [1][3].
- Never invent phone numbers, dates, amounts, deadlines, links or form numbers that are not in \
the passages.
- If the passages do not answer the question, say so in one sentence and suggest raising it \
through the official grievance route.
- Every sentence must carry at least one citation. Do not add notes, disclaimers or advice that \
the passages do not contain, and do not repeat the question.
- Be brief: at most 5 sentences, or at most 5 short numbered steps.
- Reply in {lang}."""

GRIEVANCE_SYSTEM = """You turn a citizen's help-desk conversation into a DRAFT grievance they can \
file themselves. Write in formal, plain English, in the first person, as the citizen.
Use only facts the citizen stated. Where a personal detail is needed but unknown (name, PAN, UAN, \
member ID, dates, amounts, acknowledgement numbers), write a placeholder in square brackets such \
as [Your UAN] and list it under missing_info. Never invent such details.
category: the kind of problem in a few words, such as "PF contribution not deposited" or \
"Refund not received" (not the department name).
department: "Income Tax Department" for income tax, PAN, ITR, refunds, TDS or e-filing; "EPFO" \
for provident fund, pension (EPS), EDLI or UAN; otherwise "Other".
summary_local: two sentences summarising the draft in {lang}, so the citizen can check it."""

GRIEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["Income Tax Department", "EPFO", "Other"]},
        "category": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "relief_sought": {"type": "string"},
        "documents": {"type": "array", "items": {"type": "string"}},
        "missing_info": {"type": "array", "items": {"type": "string"}},
        "summary_local": {"type": "string"},
    },
    "required": ["department", "category", "subject", "description", "facts", "relief_sought",
                 "documents", "missing_info", "summary_local"],
}


class Engine:
    def __init__(self) -> None:
        c1 = systems_config()["configs"]["C1"]
        self.bm25 = BM25Index("hiaq", c1["k1"], c1["b"])
        self.dense = DenseIndex("hiaq")
        field = json.loads(FIELD.read_text(encoding="utf-8"))
        self.passages: dict[str, dict[str, Any]] = field["passages"]
        self.xyz = {p["id"]: np.array([p["x"], p["y"], p["z"]]) for p in field["points"]}
        self.rrf_k = systems_config()["configs"]["C4"]["k"]

    def rank(self, question: str, depth: int = DEPTH) -> dict[str, Any]:
        """Hybrid C4 by default. A question with no Latin letters (Devanagari Hindi) cannot share
        words with these English pages, so BM25 only adds noise and the answer uses dense alone;
        the BM25 list is still returned for the X-ray view."""
        t0 = time.perf_counter()
        b = self.bm25.search(["q"], [question], depth)
        d = self.dense_search(question, depth)
        lexical = bool(LATIN.search(question))
        f = rrf([b, d], k=self.rrf_k, depth=depth) if lexical else d
        ms = round((time.perf_counter() - t0) * 1000)
        return {"bm25": b, "dense": d, "fused": f, "ms": ms,
                "mode": "hybrid" if lexical else "dense"}

    def dense_search(self, question: str, depth: int) -> pd.DataFrame:
        """C3 for one question. Same bge-m3 as the corpus vectors, but with a 512-token context
        (questions are capped at 600 characters, so nothing is cut): the study's 8192 would make
        Ollama reserve enough VRAM to push the chat model off the 6 GB GPU on every question."""
        res = ollama.Client(host=HOST).embed(model=EMBED, input=[question], truncate=True,
                                             keep_alive="30m", options={"num_ctx": 512})
        q = np.asarray(res.embeddings, dtype=np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        scores, docs = self.dense.index.search(q, depth)
        return pd.DataFrame({"qid": "q", "pid": [self.dense.pids[int(i)] for i in docs[0]],
                             "rank": np.arange(1, depth + 1), "score": scores[0].astype(float)})

    def place(self, dense) -> list[float]:
        """Put the question among its nearest passages (t-SNE cannot embed new points)."""
        top = dense.head(8)
        w = np.exp((top["score"].to_numpy() - top["score"].max()) * 25.0)
        p = sum(wi * self.xyz[pid] for wi, pid in zip(w, top["pid"], strict=True)) / w.sum()
        return [round(float(v), 3) for v in p]

    def card(self, pid: str, ranks: dict[str, Any], terms: set[str]) -> dict[str, Any]:
        p = self.passages[pid]
        words = {w.lower() for w in TOKEN.findall(p["title"] + " " + p["text"])}
        return {"id": pid, "title": p["title"], "page": p["page"], "url": p["url"],
                "src": p["src"], "text": p["text"],
                "overlap": sorted(t for t in terms if t in words), **ranks}


def content_terms(q: str) -> set[str]:
    return {w.lower() for w in TOKEN.findall(q) if w.lower() not in STOPWORDS and len(w) > 1}


def stage_ranks(r: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {name: dict(zip(r[name]["pid"], r[name]["rank"], strict=True))
            for name in ("bm25", "dense", "fused")}


engine: Engine | None = None
aclient = ollama.AsyncClient(host=HOST)
ANSWERS: dict[tuple[str, str], str] = {}  # first answer to each history-free question


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def health(_: Request) -> JSONResponse:
    try:
        ps = await aclient.ps()
        warm = [m.model for m in ps.models]
        ok = True
    except Exception:  # noqa: BLE001
        warm, ok = [], False
    return JSONResponse({"ok": ok and engine is not None, "ollama": ok, "warm": warm,
                         "chat_model": CHAT, "passages": len(engine.passages) if engine else 0})


async def ask(request: Request) -> StreamingResponse:
    body = await request.json()
    question = str(body.get("question", "")).strip()[:600]
    history = body.get("history", [])[-4:]

    async def stream():
        if not question:
            yield sse("error", {"message": "Type or say a question first."})
            return
        t0 = time.perf_counter()
        lang = classify(question)["label"]
        yield sse("lang", {"label": lang})
        r = await run_in_threadpool(engine.rank, question)
        ranks, terms = stage_ranks(r), content_terms(question)
        top = r["fused"].head(TOP_K)["pid"].tolist()
        cards = [engine.card(pid, {"rank": i + 1, "bm25_rank": ranks["bm25"].get(pid),
                                   "dense_rank": ranks["dense"].get(pid)}, terms)
                 for i, pid in enumerate(top)]
        yield sse("sources", {"sources": cards, "query_xyz": engine.place(r["dense"]),
                              "retrieval_ms": r["ms"], "terms": sorted(terms), "mode": r["mode"]})
        key = (question, lang) if not history else None
        if key in ANSWERS:  # a repeated demo question replays its first answer
            for piece in regex.findall(r"\S+\s*", ANSWERS[key]):
                yield sse("token", {"t": piece})
                await asyncio.sleep(0.018)
            yield sse("done", {"total_ms": round((time.perf_counter() - t0) * 1000),
                               "cached": True})
            return
        context = "\n\n".join(f"[{i + 1}] {c['title']}\n{c['text']}" for i, c in enumerate(cards))
        messages = [{"role": "system", "content": ANSWER_SYSTEM.format(lang=LANG_NAME[lang])}]
        for turn in history:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:1500]})
        messages.append({"role": "user",
                         "content": f"Official passages:\n\n{context}\n\nQuestion: {question}\n\n"
                                    f"(Answer in {LANG_SHORT[lang]}, with [n] citations.)"})
        out = []
        try:
            async for part in await aclient.chat(
                    model=CHAT, messages=messages, stream=True, keep_alive="30m",
                    options={"temperature": 0.2, "num_ctx": NUM_CTX,
                             "num_predict": 900 if lang == "DEV_HI" else 420}):
                if await request.is_disconnected():
                    return
                tok = part.message.content
                if tok:
                    out.append(tok)
                    yield sse("token", {"t": tok})
        except Exception as e:  # noqa: BLE001
            yield sse("error", {"message": f"The language model is not answering ({e}). "
                                           "Check that Ollama is running."})
            return
        if key is not None:
            ANSWERS[key] = "".join(out)
        yield sse("done", {"total_ms": round((time.perf_counter() - t0) * 1000),
                           "cached": False})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def grievance(request: Request) -> JSONResponse:
    body = await request.json()
    lang = body.get("lang") if body.get("lang") in LANG_NAME else "EN"
    convo = "\n".join(f"{m.get('role', 'user').upper()}: {str(m.get('content', ''))[:1500]}"
                      for m in body.get("messages", [])[-8:])
    details = str(body.get("details", "")).strip()[:1500]
    prompt = f"Conversation:\n{convo}\n\nExtra details from the citizen: {details or '(none)'}"
    try:
        res = await aclient.chat(
            model=CHAT, keep_alive="30m", format=GRIEVANCE_SCHEMA,
            messages=[{"role": "system", "content": GRIEVANCE_SYSTEM.format(lang=LANG_NAME[lang])},
                      {"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_ctx": NUM_CTX, "num_predict": 900})
        return JSONResponse(json.loads(res.message.content))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Could not draft the grievance: {e}"}, status_code=502)


async def xray(request: Request) -> JSONResponse:
    body = await request.json()
    question = str(body.get("question", "")).strip()[:600]
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    r = await run_in_threadpool(engine.rank, question, 10)
    ranks, terms = stage_ranks(r), content_terms(question)
    out = {}
    for name in ("bm25", "dense", "fused"):
        out[name] = [engine.card(pid, {"rank": int(rk), "score": round(float(s), 4),
                                       "bm25_rank": ranks["bm25"].get(pid),
                                       "dense_rank": ranks["dense"].get(pid),
                                       "fused_rank": ranks["fused"].get(pid)}, terms)
                     for pid, rk, s in zip(r[name]["pid"], r[name]["rank"], r[name]["score"],
                                           strict=True)]
    return JSONResponse({"lang": classify(question)["label"], "terms": sorted(terms),
                         "ms": r["ms"], **out})


async def spa(request: Request):
    path = DIST / request.path_params.get("path", "")
    if path.is_file() and DIST in path.resolve().parents:
        return FileResponse(path)
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return JSONResponse({"error": "Frontend not built; run `npm run build` in app/web."}, 404)


@contextlib.asynccontextmanager
async def lifespan(_app):
    global engine
    engine = await run_in_threadpool(Engine)
    await run_in_threadpool(engine.rank, "warm up the embedding model", 5)
    try:  # load the chat model now, with the context every call uses, so nothing reloads later
        await aclient.generate(model=CHAT, prompt="", keep_alive="30m",
                               options={"num_ctx": NUM_CTX})
    except Exception:  # noqa: BLE001
        pass
    yield


routes = [
    Route("/api/health", health),
    Route("/api/ask", ask, methods=["POST"]),
    Route("/api/grievance", grievance, methods=["POST"]),
    Route("/api/xray", xray, methods=["POST"]),
]
if (DIST / "assets").exists():
    routes.append(Mount("/assets", StaticFiles(directory=DIST / "assets")))
routes.append(Route("/{path:path}", spa))

app = Starlette(routes=routes, lifespan=lifespan)
