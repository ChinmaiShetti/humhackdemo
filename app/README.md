# Arzi (अर्ज़ी): the demo app

A multilingual citizen help desk for Income Tax and EPFO questions, built on this repository's
retrieval stack and evaluated with its test method. Everything runs locally on the project laptop.

- `/`: the assistant. Ask in English, Hindi or Hinglish, by typing or by voice. The answer cites
  official passages and threads each citation to its passage in a 3D map of all 1,572 HiAQ passages
  and to a source card. "Draft a grievance" turns the conversation into a filled-in draft with blanks
  to complete, and shows where to file it. Arzi never files anything.
- `/lab`: how it is tested. **X-ray** shows one question ranked by word match (BM25), meaning match
  (BGE-M3) and the fused list Arzi uses. **Trust lab** tells the study in five chapters, with every
  number read from `results/eval/*.json`.

## Run

Double-click `run_arzi.bat` in the repository root. It starts Ollama if it isn't running, builds
anything missing, serves the app and opens the browser. To stop Arzi, double-click `stop_arzi.bat` (it also unloads
the models from the GPU) or close the launcher window. Running it
again while Arzi is up just opens the browser. From a terminal:

```powershell
powershell -ExecutionPolicy Bypass -File app\run_demo.ps1          # builds what is missing, opens the browser
powershell -ExecutionPolicy Bypass -File app\run_demo.ps1 -Build   # after changing the frontend
```

Needs Ollama running with the pinned models (`qwen3:4b-instruct-2507-q4_K_M`, `bge-m3`). The page is
served at http://127.0.0.1:8765.

Development, with hot reload:

```powershell
.venv\Scripts\python.exe -m uvicorn app.server.server:app --port 8765   # API
cd app\web; npx vite                                                   # http://localhost:5173, proxies /api
```

Checks: `.venv\Scripts\python.exe app\server\smoke.py` asks one question per language and drafts a
grievance; `node app\web\tools\shoot.mjs <dir> [hero chat-en chat-hi chat-hinglish draft narrow lab]`
saves screenshots, using the installed Chrome.

## How it works

| Piece | What it does |
|---|---|
| `server/build_field.py` | t-SNE of the cached BGE-M3 passage vectors to 3D; 14 k-means topics with hand-checked English and Hindi names (each guarded by a term check). Writes `web/public/field.json` (git-ignored: it holds passage text from `data/`). |
| `server/build_lab.py` | Aggregates the study's results into `web/public/lab.json`: no query text, no per-query rows. |
| `server/server.py` | Starlette API. Retrieval is the study's C4: BM25 (C1) plus dense (C3), fused by RRF. A question in Devanagari uses dense only, because it shares no words with the English pages. Answers and drafts come from the pinned Qwen3-4B. |
| `web/` | Vite + React + Three.js (react-three-fiber, bloom), GSAP, Web Speech API. Fonts: Tiro Devanagari Hindi, Mukta, Martian Mono, bundled locally. |

Design choices worth knowing:

- **No reranker in the app.** It is a torch model and cannot share the 6 GB GPU with the answer model.
  On HiAQ's real questions the reranked systems score highest (C5, nDCG@10 0.551, against 0.476 for
  C4); the lab page says so.
- **One chat context everywhere** (`NUM_CTX`). A different `num_ctx` makes Ollama reload the model.
  Question embeddings use a 512-token context and give rankings identical to the study's 8192
  (checked on three questions in each language).
- Repeated history-free questions replay their first answer (`cached: true` in the `done` event).
- Deep links for recording: `/?q=<question>` asks on load; add `&draft=1` to open the draft once the
  answer is done.

## What to say, and what not to

- Retrieval is the evaluated part. Answer text comes from a 4B model and can be wrong; every sentence is
  meant to carry a citation so a reader can check it.
- The test engine is the study's method: synthetic test questions selected to match how real people
  word things. On HiAQ (146 real questions, a pilot), such sets ranked systems close to how real users
  did (τ_b 0.92 against 0.81 for random). The fix held on forum data and failed on SciFact; the lab
  shows both.
- Voice input uses the browser's speech service (Chrome and Edge send audio to their cloud). Typed
  questions stay on the machine.
- Grievance drafts are drafts. The portals listed are the public ones (e-Filing Grievances, EPFiGMS,
  CPGRAMS); nothing is submitted.

## Video shot lists

**Video 1: the assistant (about 2 minutes).** Use Chrome at 1440×900 or larger. Ask each question
once before recording, so the take replays it smoothly.
1. Open `/`. Let the field assemble while the headline cycles through English, Hindi and Hinglish (12 s).
2. Click the mic (`हिं`) and say "पीएफ का पैसा 20 दिन में नहीं मिला तो शिकायत कहाँ करें?". The camera
   flies to the question's neighbourhood, the threads draw, and the stamp lands.
3. Hover a citation chip: its thread, map point and card light up. Open the official page from a card.
4. Press "Listen" (read aloud in Hindi).
5. Type a Hinglish follow-up, for example "Naukri badalne par PF transfer kaise hota hai?".
6. Press "Draft a grievance": the sheet writes itself, the DRAFT stamp lands; show the blanks, "Where to
   file" and "Download draft".
7. End on "How it's tested" in the top bar (the lead-in to video 2).

**Video 2: X-ray and Trust lab (about 2 minutes).**
1. Open `/lab`. In X-ray, run the Hinglish example and hover a row to follow one passage across the
   three lists.
2. Run the Hindi example: BM25's top 8 share no words with the question, and the fused list switches
   to meaning only.
3. Scroll the Trust lab. In chapter 1 switch the tabs FiQA → Android → HiAQ to show the ranking reshuffle.
   Then cover overlap (chapter 2), the judge (3), the replication forest (4) and the test engine (5).
