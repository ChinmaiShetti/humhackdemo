"""Polite, resumable scraper for HiAQ-Services help pages (spec §6.2, §0.1).

- robots.txt checked per host (urllib.robotparser; a 401/403 robots file means "disallow all").
- At most one request per `min_delay_s` (2 s), single-threaded, identifying User-Agent.
- Breadth-first within per-source allow/deny regexes, depth and page caps.
- Raw HTML is stored as HA_DATA/hiaq/raw/<source>/<sha1(url)>.html; every fetch (including
  failures) is appended to pages.jsonl with url, final_url, status, fetched_at, sha1, bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

from ha.config import load_config
from ha.paths import get_paths
from ha.utils import utc_now_iso


def raw_dir(source: str) -> Path:
    return get_paths().data / "hiaq" / "raw" / source


def normalise_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/") if urlparse(url).path not in ("", "/") else url


def in_scope(url: str, allow: list[str], deny: list[str]) -> bool:
    return any(re.search(p, url) for p in allow) and not any(re.search(p, url) for p in deny)


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        out.append(normalise_url(urljoin(base_url, href)))
    return out


class PoliteSession:
    def __init__(self, user_agent: str, min_delay_s: float, timeout_s: float) -> None:
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.ua = user_agent
        self.min_delay_s = min_delay_s
        self.timeout_s = timeout_s
        self._last = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_delay_s:
            time.sleep(self.min_delay_s - dt)
        self._last = time.monotonic()

    def allowed(self, url: str) -> bool:
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            self._wait()
            try:
                r = self.s.get(f"{host}/robots.txt", timeout=self.timeout_s)
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.status_code >= 400 or "text/html" in r.headers.get("content-type", ""):
                    rp.allow_all = True  # no robots.txt (or an HTML error page) = no restrictions
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp.disallow_all = True  # cannot verify -> do not crawl
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.ua, url)

    def get(self, url: str) -> requests.Response:
        self._wait()
        return self.s.get(url, timeout=self.timeout_s)


def fix_url(url: str, fixes: list[list[str]]) -> str:
    """Apply per-source [pattern, replacement] regex fixes, then normalise."""
    for pattern, repl in fixes:
        url = re.sub(pattern, repl, url)
    return normalise_url(url)


def sitemap_urls(sess: PoliteSession, cfg: dict[str, Any]) -> list[str]:
    """In-scope URLs listed in the source's sitemap(s), fixed and de-duplicated, in order."""
    out: list[str] = []
    for sm in cfg.get("sitemaps", []):
        if not sess.allowed(sm):
            continue
        r = sess.get(sm)
        if r.status_code != 200:
            logger.warning(f"sitemap {sm}: http {r.status_code}")
            continue
        for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text):
            url = fix_url(loc, cfg.get("url_fixes", []))
            if url not in out and in_scope(url, cfg["allow"], cfg["deny"]):
                out.append(url)
    return out


def _done_urls(index: Path) -> set[str]:
    if not index.exists():
        return set()
    with open(index, encoding="utf-8") as f:
        return {json.loads(line)["url"] for line in f if line.strip()}


def scrape_source(name: str, cfg: dict[str, Any], sess: PoliteSession) -> dict[str, int]:
    out_dir = raw_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "pages.jsonl"
    done = _done_urls(index)
    seeds = [normalise_url(u) for u in cfg["seeds"]]
    seeds += [u for u in sitemap_urls(sess, cfg) if u not in seeds]
    queue: deque[tuple[str, int]] = deque((u, 0) for u in seeds)
    seen = set(seeds)
    stats = {"fetched": 0, "skipped_done": 0, "robots_blocked": 0, "errors": 0}
    consecutive_blocks = 0

    # Re-seed the frontier from pages already on disk so a resumed run continues the BFS.
    if index.exists():
        with open(index, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == 200 and rec["depth"] < cfg["max_depth"]:
                    html = (out_dir / f"{rec['sha1']}.html").read_text(encoding="utf-8")
                    for link in extract_links(html, rec["final_url"]):
                        if link not in seen and in_scope(link, cfg["allow"], cfg["deny"]):
                            seen.add(link)
                            queue.append((link, rec["depth"] + 1))

    while queue and len(done) < cfg["max_pages"]:
        url, depth = queue.popleft()
        if url in done:
            stats["skipped_done"] += 1
            continue
        if not sess.allowed(url):
            stats["robots_blocked"] += 1
            continue
        rec: dict[str, Any] = {"url": url, "depth": depth, "fetched_at": utc_now_iso()}
        try:
            r = sess.get(url)
            rec.update(status=r.status_code, final_url=normalise_url(r.url),
                       content_type=r.headers.get("content-type", ""))
            consecutive_blocks = consecutive_blocks + 1 if r.status_code in (403, 429) else 0
            if r.status_code == 200 and "html" in rec["content_type"]:
                html = r.content.decode(r.encoding or "utf-8", errors="replace")
                sha = hashlib.sha1(url.encode("utf-8")).hexdigest()
                (out_dir / f"{sha}.html").write_text(html, encoding="utf-8")
                rec.update(sha1=sha, bytes=len(r.content))
                if depth < cfg["max_depth"]:
                    for link in extract_links(html, rec["final_url"]):
                        link = fix_url(link, cfg.get("url_fixes", []))
                        if link not in seen and in_scope(link, cfg["allow"], cfg["deny"]):
                            seen.add(link)
                            queue.append((link, depth + 1))
            stats["fetched"] += 1
        except requests.RequestException as e:
            rec.update(status=None, error=f"{type(e).__name__}: {e}"[:200])
            stats["errors"] += 1
        with open(index, "a", encoding="utf-8") as f:  # every request is logged, even the last
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done.add(url)
        if consecutive_blocks >= 3:
            logger.warning(f"{name}: 3 consecutive 403/429 responses; stopping this source")
            break
        if stats["fetched"] % 25 == 0:
            logger.info(f"{name}: {len(done)} pages done, queue {len(queue)}")
    logger.info(f"{name}: {stats}, total on disk {len(done)}")
    return stats


def scrape(only: str | None = None) -> None:
    cfg = load_config("hiaq_sources.yaml")
    sess = PoliteSession(cfg["user_agent"], cfg["min_delay_s"], cfg["timeout_s"])
    for name, scfg in cfg["sources"].items():
        if only and name != only:
            continue
        scrape_source(name, scfg, sess)
