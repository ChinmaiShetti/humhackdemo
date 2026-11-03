"""Precompute the 3D "knowledge field" for the Arzi demo: one point per HiAQ passage.

Reads only public material: the HiAQ corpus (official Income Tax / EPFO help pages) and its cached
bge-m3 embeddings. No query text from participants is read or written.
Writes app/web/public/field.json:
  points: [{id, x, y, z, src, c}]   c = topic cluster index
  clusters: [{label, x, y, z, n}]
  passages: {id: {title, url, src, text}}
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "app" / "web" / "public" / "field.json"
N_CLUSTERS = 14
SEED = 20260927

# Hand-checked names for the clusters (from the printout of top terms and headings). Each is
# guarded by one term that must be among the cluster's top TF-IDF terms, so a changed clustering
# falls back to its raw terms instead of carrying a wrong name.
LABELS = {
    0: ("annexure", "Trusts & charity forms", "ट्रस्ट और धर्मार्थ फ़ॉर्म"),
    1: ("claiming deductions", "Deductions in ITR-1", "आईटीआर-1 में कटौतियाँ"),
    2: ("pan active", "PAN & Aadhaar", "पैन और आधार"),
    3: ("eps", "PF, pension & insurance", "पीएफ, पेंशन और बीमा"),
    4: ("regime", "New vs old tax regime", "नई बनाम पुरानी कर व्यवस्था"),
    5: ("registration request", "Registration & access", "पंजीकरण और पहुँच"),
    6: ("payment gateway", "Paying tax online", "ऑनलाइन कर भुगतान"),
    7: ("itr status", "ITR status & verification", "आईटीआर स्थिति और सत्यापन"),
    8: ("defective notice", "Notices & rectification", "नोटिस और सुधार"),
    9: ("mobile app", "App & login", "ऐप और लॉगिन"),
    10: ("audit", "Audit & foreign remittance", "ऑडिट और विदेशी प्रेषण"),
    11: ("authorize", "Representatives & assisted filing", "प्रतिनिधि और सहायता"),
    12: ("presumptive", "Presumptive tax & rebate", "अनुमानित कर और छूट"),
    13: ("investments", "Investments & tax benefits", "निवेश और कर लाभ"),
}
STOP = set("""form forms faq faqs filing file online user manual step guide assessee taxpayer tax
income return returns ay fy the and for of to in on is are be can will an by with as or if it you
your this that from at which under any may have has not do does what how should click page
section details""".split())


QUESTION = re.compile(r"^(?:Q[\s.\-]*\d+[\s.:\-]*)?(?:\d+\.\s*)?(.{12,220}?\?)")


GENERIC = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(step-by-step guide|prerequisites.*|"
                     r"form at a glance|how to access and submit|about the form|user manual|"
                     r"faqs?|additional faqs|overview|introduction|learn with us)$", re.I)


def slug_of(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[-_]+", " ", re.sub(r"\.\w+$", "", slug)).strip()
    slug = re.sub(r"\b(faq|epfo|epf|eps|edli|itr|pan|ais|tds|evc|dsc|uan)\b",
                  lambda m: m.group(1).upper(), slug, flags=re.I)
    return "" if slug.lower() in {"help", "home", "index"} else slug[:1].upper() + slug[1:]


def heading_of(row: pd.Series) -> str:
    """Last heading, else the URL slug: the text the topic clusters are named from."""
    return row.heading_path.split(">")[-1].strip() if row.heading_path.strip() else slug_of(row.url)


def page_of(row: pd.Series) -> str:
    """What a reader sees as the page name: generic headings give way to the URL slug."""
    h = heading_of(row)
    if not h or GENERIC.match(h):
        parts = [x.strip() for x in row.heading_path.split(">") if x.strip()]
        named = next((x for x in reversed(parts) if not GENERIC.match(x)), "") or slug_of(row.url)
        kind = GENERIC.match(h).group(1).capitalize() if h else ""
        h = f"{named} · {kind}" if named and kind else (named or h)
    return h or "Help page"


def title_of(row: pd.Series) -> str:
    """A Q&A chunk is titled by its question; anything else by its page heading."""
    m = QUESTION.match(row.text.strip()) if row.kind == "qa" else None
    return m.group(1).strip() if m else page_of(row)


def main() -> None:
    chunks = pd.read_parquet(ROOT / "data" / "hiaq" / "chunks.parquet")
    corpus = pd.read_parquet(ROOT / "data" / "processed" / "hiaq" / "corpus.parquet")
    assert chunks["chunk_id"].tolist() == corpus["pid"].tolist(), "chunk order != corpus order"
    emb = np.load(ROOT / "cache" / "emb" / "hiaq" / "bge-m3.f16.npy").astype(np.float32)
    assert emb.shape[0] == len(chunks)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    xyz = TSNE(n_components=3, perplexity=30, metric="cosine", init="pca",
               random_state=SEED).fit_transform(emb)
    xyz -= xyz.mean(0)
    xyz /= np.percentile(np.linalg.norm(xyz, axis=1), 95)
    xyz *= 10.0

    km = KMeans(N_CLUSTERS, n_init=10, random_state=SEED).fit(emb)
    labels = km.labels_
    titles = [title_of(r) for r in chunks.itertuples()]
    heads_all = [heading_of(r) for r in chunks.itertuples()]
    docs = [" ".join(t for t, lab in zip(heads_all, labels, strict=True) if lab == k) + " " +
            " ".join(chunks.text[labels == k].str.slice(0, 300)) for k in range(N_CLUSTERS)]
    tf = TfidfVectorizer(stop_words=list(STOP), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
                         ngram_range=(1, 2), max_df=0.5, sublinear_tf=True)
    m = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    clusters = []
    for k in range(N_CLUSTERS):
        top = vocab[np.argsort(-m[k].toarray()[0])[:4]].tolist()
        heads = Counter(h for h, lab in zip(heads_all, labels, strict=True)
                        if lab == k).most_common(3)
        print(k, (labels == k).sum(), top, heads)
        c = xyz[labels == k].mean(0)
        term, en, hi = LABELS.get(k, ("", "", ""))
        if term not in top:
            en, hi = " · ".join(top[:2]), ""
            print(f"  cluster {k}: guard term {term!r} missing, using raw terms")
        clusters.append({"label": en, "hi": hi, "x": round(float(c[0]), 3),
                         "y": round(float(c[1]), 3), "z": round(float(c[2]), 3),
                         "n": int((labels == k).sum())})

    points = [{"id": cid, "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
               "z": round(float(p[2]), 3), "src": s, "c": int(lab)}
              for cid, p, s, lab in zip(chunks.chunk_id, xyz, chunks.source, labels, strict=True)]
    passages = {r.chunk_id: {"title": t, "page": page_of(r), "url": r.url, "src": r.source,
                             "text": r.text}
                for r, t in zip(chunks.itertuples(), titles, strict=True)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"points": points, "clusters": clusters, "passages": passages},
                              ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
