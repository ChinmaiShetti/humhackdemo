"""P0.3 diagnostic: can qwen3:4b (2507 build, thinking by default) produce a clean query
without reasoning prose, and does a smaller num_ctx speed up Ollama bge-m3 embeddings?

Variants: (B) "/no_think" soft switch; (D) raw prompt with a pre-filled empty think block,
which is how Qwen3's own non-thinking mode is formed; (T) think=True, answer only.
"""

from __future__ import annotations

import time

import pandas as pd

from ha import llm
from ha.paths import get_paths
from ha.prompts import render

corpus = pd.read_parquet(get_paths().processed / "fiqa" / "corpus.parquet")
lens = corpus["text"].str.split().str.len()
passages = corpus[(lens >= 60) & (lens <= 250)].sample(4, random_state=1)["text"].tolist()
OPTS = {"temperature": 0.7, "top_p": 0.9, "num_predict": 64, "seed": 0}


def raw_prefill(user: str) -> str:
    return (f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


c = llm.client()
for i, p in enumerate(passages):
    prompt = render("gen_std_v1", passage=p)
    t0 = time.perf_counter()
    rb = c.chat(model="qwen3:4b", messages=[{"role": "user", "content": prompt + " /no_think"}],
                think=False, options={**OPTS, "seed": i})
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    rd = c.generate(model="qwen3:4b", prompt=raw_prefill(prompt), raw=True,
                    options={**OPTS, "seed": i, "stop": ["<|im_end|>"]})
    td = time.perf_counter() - t0
    print(f"--- passage {i}")
    print(f"  B /no_think   {tb:4.1f}s {rb.eval_count:3d}tok {rb.message.content.strip()[:160]!r}")
    print(f"  D prefill     {td:4.1f}s {rd.eval_count:3d}tok {rd.response.strip()[:160]!r}")

# T: think=True on one passage, to price the fallback
prompt = render("gen_std_v1", passage=passages[0])
t0 = time.perf_counter()
rt = c.chat(model="qwen3:4b", messages=[{"role": "user", "content": prompt}], think=True,
            options={"temperature": 0.7, "top_p": 0.9, "seed": 0, "num_predict": 2048})
print(f"  T think=True  {time.perf_counter() - t0:4.1f}s {rt.eval_count}tok "
      f"answer={rt.message.content.strip()[:160]!r}")

texts = corpus["text"].sample(160, random_state=0).tolist()
for ctx in (8192, 512):
    t0 = time.perf_counter()
    for b in range(0, len(texts), 32):
        c.embed(model="bge-m3", input=texts[b:b + 32], truncate=True, options={"num_ctx": ctx})
    print(f"== embed 160 passages, num_ctx={ctx}: {len(texts) / (time.perf_counter() - t0):.1f}/s")
print("ps:", [(m.model, round(m.size_vram / 2**30, 2)) for m in c.ps().models])
