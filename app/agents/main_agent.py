# app/agents/main_agent.py
"""
InfoScout — EnhancedWebAgent (Google-only, DB-integrated, headless by default)

Updated small reranker logging normalization:
- rewrites noisy reranker fallback step text to a clearer single message:
  "[reranker] no model configured — using heuristic scoring"
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

# -------------------------
# db_helpers import (resilient)
# -------------------------
_DB_HELPERS_AVAILABLE = False
try:
    from app.db.db_helpers import (
        init_all_dbs,
        create_search,
        bulk_insert_results,
        mark_search_completed,
        save_memory,
        dedup_key_from_link,
    )
    print("ℹ️ Imported db_helpers from app.db.db_helpers", flush=True)
    _DB_HELPERS_AVAILABLE = True
except Exception as e_primary:
    try:
        from app.db.db_helpers import (
            init_all_dbs,
            create_search,
            bulk_insert_results,
            mark_search_completed,
            save_memory,
            dedup_key_from_link,
        )
        print("ℹ️ Imported db_helpers from db_helpers.py (top-level)", flush=True)
        _DB_HELPERS_AVAILABLE = True
    except Exception as e_fallback:
        print("⚠️ db_helpers import failed (both app.db.db_helpers and db_helpers.py).", flush=True)
        print("   primary import error:", repr(e_primary), flush=True)
        print("   fallback import error:", repr(e_fallback), flush=True)
        _DB_HELPERS_AVAILABLE = False

# -------------------------
# reranker import (resilient)
# -------------------------
_RERANKER_AVAILABLE = False
try:
    from app.llm.reranker import rerank_search
    print("ℹ️ Imported rerank_search from app.llm.reranker", flush=True)
    _RERANKER_AVAILABLE = True
except Exception as e:
    try:
        from llm.reranker import rerank_search
        print("ℹ️ Imported rerank_search from llm.reranker", flush=True)
        _RERANKER_AVAILABLE = True
    except Exception as e2:
        print("⚠️ reranker import failed (app.llm.reranker / llm.reranker).", flush=True)
        print("   errors:", repr(e), repr(e2), flush=True)
        _RERANKER_AVAILABLE = False
        async def rerank_search(*args, **kwargs):
            return {"results": kwargs.get("results", args[0] if args else []),
                    "summary": None,
                    "steps": ["[reranker-missing] no reranker available; heuristic-only used"]}

# -------------------------
# optional Playwright & BeautifulSoup
# -------------------------
try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    async_playwright = None
    _PLAYWRIGHT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    _BS4_AVAILABLE = False

# -------------------------
# simple heuristic helper (copied/inline so main agent can always compute scores)
# -------------------------
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _token_overlap_score(query: str, text: str) -> float:
    qtok = set(_tokens(query))
    ttok = set(_tokens(text))
    if not qtok:
        return 0.0
    return len(qtok & ttok) / len(qtok)


def _heuristic_scores(query: str, title: str, domain: Optional[str], has_image: bool) -> Dict[str, float]:
    rel = _token_overlap_score(query, (title or "")) * 0.95
    authoritative = ("edu", "gov", "ac.", "nature.com", "arxiv.org", "ieee.org", "springer", "ncbi.nlm.nih.gov")
    trust = 0.3
    if domain:
        dom = domain.lower()
        if any(d in dom for d in authoritative):
            trust = 0.75
        elif dom.endswith(".org") or dom.endswith(".com"):
            trust = 0.45
    click = 0.45 + (0.25 if len((title or "")) < 80 else 0.0) + (0.2 if has_image else 0.0)
    click = min(1.0, click)
    rel = max(0.0, min(1.0, rel))
    return {"relevance": round(rel, 4), "clickability": round(click, 4), "trust": round(trust, 4)}

# -------------------------
# dataclasses & memory
# -------------------------
@dataclass
class TaskStep:
    step_id: int
    description: str
    action: str
    parameters: Dict[str, Any]
    completed: bool = False
    result: Any = None


class TaskMemory:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT,
                result TEXT,
                steps_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit(); conn.close()

    def save_task(self, instruction: str, result: Any, steps_taken: List[str] = None):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tasks (instruction, result, steps_taken) VALUES (?, ?, ?)",
                           (instruction, json.dumps(result, ensure_ascii=False), json.dumps(steps_taken or [])))
            conn.commit(); conn.close()
        except Exception:
            print("⚠️ TaskMemory.save_task failed", flush=True)

    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT instruction, result, steps_taken, timestamp FROM tasks ORDER BY id DESC LIMIT ?",
                       (limit,))
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "instruction": row[0],
                "result": json.loads(row[1]) if row[1] else None,
                "steps_taken": json.loads(row[2]) if row[2] else [],
                "timestamp": row[3]
            })
        conn.close(); return tasks


@dataclass
class TaskResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    steps_taken: List[str] = None

    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "steps_taken": self.steps_taken or []
        }

# -------------------------
# Helpers: URL normalization, domain extraction, image fetch
# -------------------------
def extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


def normalize_url(url: Optional[str], base_scheme: str = "https") -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.startswith("//"):
        return f"{base_scheme}:{url}"
    if re.match(r"^[\w-]+\.[\w\.-]+", url) and not url.startswith("http"):
        return "https://" + url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return None


def _fetch_og_or_favicon(url: str, steps_log: List[str]) -> Optional[str]:
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        norm = normalize_url(url)
        if not norm:
            steps_log.append(f"[image-fetch] cannot normalize url: {url}")
            return None
        resp = requests.get(norm, headers=headers, timeout=6)
        html = resp.text or ""
        if _BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html, "html.parser")
                og = soup.find("meta", property="og:image")
                if og and og.get("content"):
                    val = og.get("content")
                    return normalize_url(val) or urljoin(norm, val)
                tw = soup.find("meta", attrs={"name": "twitter:image"})
                if tw and tw.get("content"):
                    val = tw.get("content")
                    return normalize_url(val) or urljoin(norm, val)
                icon = soup.find("link", rel=lambda x: x and "icon" in x.lower())
                if icon and icon.get("href"):
                    href = icon.get("href")
                    return urljoin(norm, href)
            except Exception as e:
                steps_log.append(f"[image-fetch] bs4 parse error: {repr(e)}")
        parsed = urlparse(norm)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
    except Exception as e:
        steps_log.append(f"[image-fetch] request failed: {repr(e)}")
    return None

# -------------------------
# Agent
# -------------------------
class EnhancedWebAgent:
    def __init__(self, headless: bool = True, db_dir: str = "."):
        # canonicalize db_dir and ensure exists
        self.db_dir = Path(db_dir).resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # initialize dbs via helpers (if available)
        if _DB_HELPERS_AVAILABLE:
            try:
                init_all_dbs(str(self.db_dir))
                print(f"ℹ️ DBs initialized in {self.db_dir}", flush=True)
            except Exception as e:
                print("⚠️ init_all_dbs failed:", e, flush=True)

        # absolute paths used across the agent
        self.searches_db = str(self.db_dir / "searches.db")
        self.results_db = str(self.db_dir / "results.db")
        self.memory_db = str(self.db_dir / "memory.db")

        # TaskMemory uses memory_db for consistency
        self.memory = TaskMemory(db_path=self.memory_db)

        # Playwright and state
        self.page = None
        self.context = None
        self._playwright = None
        self.headless = headless

        # model loader placeholder (set self.model_obj externally if desired)
        self.model_obj = None

        # Playwright profile dir
        self.user_data_dir = Path(".playwright_profile").resolve()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

        print(f"🤖 EnhancedWebAgent initialized (headless={self.headless}). DB dir={self.db_dir}", flush=True)

    # lifecycle
    async def start(self):
        if not _PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed; browser automation disabled.", flush=True)
            return
        try:
            self._playwright = await async_playwright().start()
            try:
                self.context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.user_data_dir),
                    channel="chrome",
                    headless=self.headless,
                    viewport={"width": 1366, "height": 768},
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
                )
            except Exception:
                self.context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.user_data_dir),
                    headless=self.headless,
                    viewport={"width": 1366, "height": 768},
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
                )
            pages = self.context.pages
            self.page = pages[0] if pages else await self.context.new_page()
            print("✅ Launched Chrome/Chromium via Playwright.", flush=True)
        except NotImplementedError:
            print("⚠️ Playwright NotImplementedError (loop/subprocess).", flush=True)
        except Exception as e:
            print("⚠️ Playwright start error:", e, flush=True)

    async def stop(self):
        try:
            if self.page:
                await self.page.close()
        except Exception:
            pass
        try:
            if self.context:
                await self.context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    # Playwright extraction for Google SERP
    async def _extract_with_playwright_google(self, url: str, search_term: str, steps_log: List[str]) -> Optional[List[Dict]]:
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(1.0)
            selectors = [
                "a h3",
                "div.g h3",
                "div.yuRUbf > a",
                "div[data-sokoban-container] a h3"
            ]
            results = []
            rank = 1
            found_any = False
            for sel in selectors:
                try:
                    await self.page.wait_for_selector(sel, timeout=2000)
                    found_any = True
                    anchors = await self.page.query_selector_all(sel)
                    for el in anchors:
                        try:
                            tag = await el.get_property("tagName")
                            tagname = (await tag.json_value()).lower() if tag else ""
                            if tagname == "h3":
                                parent = await el.evaluate_handle("node => node.closest('a')")
                                if parent:
                                    href = await parent.get_property("href")
                                    href_val = await href.json_value() if href else None
                                else:
                                    href_val = None
                                title = (await el.inner_text()).strip() if await el.inner_text() else ""
                            else:
                                href_val = await el.get_attribute("href")
                                title_el = await el.query_selector("h3")
                                title = (await title_el.inner_text()).strip() if title_el else (await el.inner_text()).strip()
                            link = normalize_url(href_val) or href_val
                            if title and link and link.startswith("http"):
                                results.append({"rank": rank, "title": title, "link": link, "snippet": "", "source": "google"})
                                rank += 1
                                if rank > 30:
                                    break
                        except Exception:
                            continue
                    if results:
                        steps_log.append(f"[chrome] Extracted {len(results)} results from Google (Playwright).")
                        return results
                except Exception:
                    continue
            if not found_any:
                steps_log.append("[chrome] No common selectors found on Google SERP.")
            else:
                steps_log.append("[chrome] Selector search finished but no valid results extracted.")
            return None
        except Exception as e:
            steps_log.append(f"[playwright] goto/error: {repr(e)}")
            return None

    # HTTP extraction fallbacks (Google only)
    def _http_extract_google_bs4(self, html: str, steps_log: List[str]) -> List[Dict]:
        results = []
        if not _BS4_AVAILABLE:
            return results
        try:
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a h3")
            rank = 1
            seen = set()
            for h3 in anchors:
                a = h3.find_parent("a")
                if not a:
                    continue
                link = a.get("href")
                title = h3.get_text(strip=True)
                link_norm = normalize_url(link) or link
                if link_norm and title and link_norm.startswith("http"):
                    if link_norm in seen: continue
                    seen.add(link_norm)
                    results.append({"rank": rank, "title": title, "link": link_norm, "snippet": "", "source": "google-http"})
                    rank += 1
                    if rank > 30:
                        break
            return results
        except Exception as e:
            steps_log.append(f"[http-bs4] parse error: {repr(e)}")
            return []

    def _http_extract_google_regex(self, html: str, steps_log: List[str]) -> List[Dict]:
        results = []
        try:
            import urllib.parse as up
            anchors = re.findall(r'<a href="(/url\?q=https?://[^"&]+)', html)
            seen = set()
            rank = 1
            for a in anchors:
                if rank > 30: break
                qpart = up.unquote(a)
                if qpart.startswith("/url?q="):
                    qpart = qpart[len("/url?q="):]
                link = qpart.split("&")[0]
                link_norm = normalize_url(link) or link
                if link_norm not in seen:
                    seen.add(link_norm)
                    results.append({"rank": rank, "title": None, "link": link_norm, "snippet": "", "source": "google-http"})
                    rank += 1
            return results
        except Exception as e:
            steps_log.append(f"[http-regex] extraction error: {repr(e)}")
            return []

    async def _http_fetch_and_extract(self, url: str, steps_log: List[str]) -> Optional[List[Dict]]:
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            html = resp.text or ""
            res = []
            if _BS4_AVAILABLE:
                res = self._http_extract_google_bs4(html, steps_log)
            if res:
                steps_log.append(f"[http] Extracted {len(res)} results via HTTP Google (bs4).")
                return res
            res = self._http_extract_google_regex(html, steps_log)
            if res:
                steps_log.append(f"[http] Extracted {len(res)} results via HTTP Google (regex).")
                return res
            return None
        except Exception as e:
            steps_log.append(f"[fallback] HTTP fetch failed: {repr(e)}")
            return None

    # ----------- Reranker integration helper -----------
    async def _run_reranker(self, query: str, results: List[Dict], steps_log: List[str]) -> Tuple[List[Dict], Optional[str]]:
        """
        Call reranker_search and merge returned scores and summary.
        Returns: (results_with_scores, summary_str)
        """
        if not isinstance(results, list):
            return results, None

        out = None
        # try a few call styles to accommodate different reranker signatures
        try:
            if _RERANKER_AVAILABLE:
                # 1) preferred: (model_obj, query, results)
                try:
                    steps_log.append(f"[reranker] calling rerank_search(model_obj, query, results)")
                    out = await rerank_search(self.model_obj, query, results)
                except Exception as e1:
                    steps_log.append(f"[reranker] signature try-1 failed: {repr(e1)}")
                    # 2) try kwargs style: (results, model=..., query=...)
                    try:
                        steps_log.append(f"[reranker] calling rerank_search(results, model=..., query=...)")
                        out = await rerank_search(results, model=self.model_obj, query=query, use_model_for_summary=True)
                    except Exception as e2:
                        steps_log.append(f"[reranker] signature try-2 failed: {repr(e2)}")
                        # 3) try results-only style
                        try:
                            steps_log.append(f"[reranker] calling rerank_search(results) (final fallback)")
                            out = await rerank_search(results)
                        except Exception as e3:
                            steps_log.append(f"[reranker] signature try-3 failed: {repr(e3)}")
                            out = None
            else:
                steps_log.append("[reranker] not available; skipping model call")
                out = await rerank_search(results=results)
        except Exception as e:
            steps_log.append(f"[reranker] final call exception: {repr(e)}")
            out = None

        if not out or not isinstance(out, dict):
            steps_log.append("[reranker] invalid/non-dict output; using heuristic-only")
            return results, None

        # merge returned results
        rr = out.get("results", results)
        summary = out.get("summary")
        rsteps = out.get("steps", [])

        # normalize reranker step messages: rewrite the noisy fallback to a clearer one
        normalized_steps: List[str] = []
        for s in rsteps:
            if not isinstance(s, str):
                s = str(s)
            # normalize legacy phrasing
            if "[reranker-heuristic] no model provided" in s or "no model provided or summary disabled" in s:
                normalized_steps.append("[reranker] no model configured — using heuristic scoring")
            else:
                # strip surrounding whitespace and add
                normalized_steps.append(s.strip())

        # deduplicate consecutive same messages and append to steps_log
        last = None
        for s in normalized_steps:
            if s == last:
                continue
            steps_log.append(f"[reranker] {s.lstrip('[reranker] ').strip()}" if s.startswith("[reranker]") else f"[reranker] {s}")
            last = s

        # ensure numeric fields + compute missing score
        for r in rr:
            r.setdefault("relevance", 0.0)
            r.setdefault("clickability", 0.0)
            r.setdefault("trust", 0.0)
            try:
                r["relevance"] = float(r.get("relevance", 0.0))
            except Exception:
                r["relevance"] = 0.0
            try:
                r["clickability"] = float(r.get("clickability", 0.0))
            except Exception:
                r["clickability"] = 0.0
            try:
                r["trust"] = float(r.get("trust", 0.0))
            except Exception:
                r["trust"] = 0.0
            # compute combined score
            r["score"] = round(0.55 * r["relevance"] + 0.25 * r["clickability"] + 0.2 * r["trust"], 4)
            r.setdefault("extra", r.get("extra", {}))

        return rr, summary

    # ----------- Search execution -----------
    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        search_term = step.parameters.get("search_term", "")
        if not search_term:
            return TaskResult(success=False, data=None, error_message="No search term provided", steps_taken=[])

        steps_log: List[str] = [f"🔎 Start search for: {search_term}"]
        import urllib.parse as up
        q = up.quote_plus(search_term)
        url = f"https://www.google.com/search?q={q}&num=10&hl=en&pws=0"

        # create search record (use absolute path)
        search_id = None
        if _DB_HELPERS_AVAILABLE:
            try:
                search_id = create_search(self.searches_db, search_term, num_results=10, backend="google")
                steps_log.append(f"[db] Created search_id={search_id} (db={self.searches_db})")
                print(f"✅ create_search -> id={search_id} saved to {self.searches_db}", flush=True)
            except Exception as e:
                steps_log.append(f"[db] create_search failed: {repr(e)}")
                print("⚠️ create_search threw:", repr(e), flush=True)
        else:
            steps_log.append("[db] db_helpers not available; skipping DB writes.")
            print("⚠️ db_helpers not available; skipping DB writes.", flush=True)

        results = None
        # Playwright path
        if self.page is not None:
            try:
                results = await self._extract_with_playwright_google(url, search_term, steps_log)
                if results:
                    steps_log.append(f"[flow] Playwright returned {len(results)} results.")
                else:
                    steps_log.append("[chrome] No results extracted from Playwright DOM; falling back to HTTP fetch.")
            except Exception as e:
                steps_log.append(f"[playwright] extraction exception: {repr(e)}")

        # HTTP fallback (Google only)
        if not results:
            http_res = await self._http_fetch_and_extract(url, steps_log)
            if http_res:
                results = http_res
                steps_log.append(f"[flow] HTTP fetch returned {len(results)} results.")

        if not results:
            steps_log.append("[fallback] All attempts (Playwright + HTTP) failed or returned no results")
            try:
                self.memory.save_task(search_term, None, steps_taken=steps_log)
            except Exception:
                pass
            if _DB_HELPERS_AVAILABLE and search_id is not None:
                try:
                    mark_search_completed(self.searches_db, search_id)
                    save_memory(self.memory_db, search_id, search_term, steps_log, summary=None)
                except Exception:
                    pass
            return TaskResult(success=False, data=None, error_message="All attempts (Playwright + HTTP) failed or returned no results", steps_taken=steps_log)

        # normalize and compute heuristics immediately (prevent zeros)
        for r in results:
            link = r.get("link")
            r["link"] = normalize_url(link) or link
            r["domain"] = extract_domain(r["link"])
            try:
                # attempt quick image fetch (may be None)
                img = _fetch_og_or_favicon(r["link"], steps_log) if r.get("link") else None
            except Exception as e:
                steps_log.append(f"[image-fetch] error for {r.get('link')}: {repr(e)}")
                img = None
            r["image"] = r.get("image") or img
            # compute heuristic scores now
            hs = _heuristic_scores(search_term, r.get("title", ""), r.get("domain"), bool(r.get("image")))
            r["relevance"] = hs["relevance"]
            r["clickability"] = hs["clickability"]
            r["trust"] = hs["trust"]
            r["score"] = round(0.55 * r["relevance"] + 0.25 * r["clickability"] + 0.2 * r["trust"], 4)
            r["extra"] = r.get("extra", {})

        # run reranker (merge scores & get summary)
        try:
            merged_results, summary = await self._run_reranker(search_term, results, steps_log)
            if merged_results:
                results = merged_results
                if summary:
                    steps_log.append("[reranker] summary produced")
                else:
                    steps_log.append("[reranker] no summary returned by reranker")
        except Exception as e:
            steps_log.append(f"[reranker] exception: {repr(e)}")
            summary = None

        # persist backward-compatible memory (TaskMemory -> memory_db)
        try:
            self.memory.save_task(search_term, {"search_term": search_term, "results": results}, steps_taken=steps_log)
        except Exception:
            pass

        # persist to results DB (use absolute paths)
        if _DB_HELPERS_AVAILABLE and search_id is not None:
            try:
                bulk_insert_results(self.results_db, search_id, results)
                mark_search_completed(self.searches_db, search_id)
                try:
                    save_memory(self.memory_db, search_id, search_term, steps_log, summary=summary if 'summary' in locals() else None)
                except Exception as e:
                    steps_log.append(f"[db] save_memory failed: {repr(e)}")
            except Exception as e:
                steps_log.append(f"[db] bulk_insert_results/mark_search_completed failed: {repr(e)}")

        payload = {"search_term": search_term, "results": results, "search_id": search_id, "summary": summary if 'summary' in locals() else None}
        return TaskResult(success=True, data=payload, steps_taken=steps_log)

    async def execute_instruction(self, instruction: str) -> TaskResult:
        step = TaskStep(step_id=1, description=f"Search for {instruction}", action="search",
                        parameters={"search_term": instruction})
        result = await self._execute_search_step(step)
        if result.steps_taken is None:
            result.steps_taken = []
        return result

# CLI runner
async def main():
    agent = EnhancedWebAgent(headless=True, db_dir=".")
    # If you want to load your local Llama/other model, set agent.model_obj appropriately before start/after start.
    # e.g. agent.model_obj = llama_model_instance
    await agent.start()
    try:
        while True:
            instr = input("Enter instruction (or 'quit'): ").strip()
            if instr.lower() in ("quit", "exit"):
                break
            res = await agent.execute_instruction(instr)
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
