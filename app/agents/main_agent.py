# app/agents/main_agent.py
"""
InfoScout — EnhancedWebAgent (Google-only, DB-integrated, headless by default)

Enhancements:
- Expanded Google SERP selectors (handles div.tF2Cxc, div.MjjYud, a[jsname="UWckNb"])
- Force Chrome-like headers in Playwright
- Graceful fallback summary if no results
- Cleaner reranker logging
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import os
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
        print("⚠️ db_helpers import failed.", flush=True)
        print("   primary:", repr(e_primary), flush=True)
        print("   fallback:", repr(e_fallback), flush=True)
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
        print("⚠️ reranker import failed.", flush=True)
        print("   errors:", repr(e), repr(e2), flush=True)
        _RERANKER_AVAILABLE = False

        async def rerank_search(*args, **kwargs):
            return {
                "results": kwargs.get("results", args[0] if args else []),
                "summary": None,
                "steps": ["[reranker-missing] no reranker available; heuristic-only used"],
            }

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
# heuristics
# -------------------------
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _token_overlap_score(query: str, text: str) -> float:
    qtok, ttok = set(_tokens(query)), set(_tokens(text))
    return len(qtok & ttok) / len(qtok) if qtok else 0.0


def _heuristic_scores(query: str, title: str, domain: Optional[str], has_image: bool) -> Dict[str, float]:
    rel = _token_overlap_score(query, (title or "")) * 0.95
    trust = 0.3
    if domain:
        dom = domain.lower()
        if any(d in dom for d in ("edu", "gov", "ac.", "nature.com", "arxiv.org", "ieee.org", "springer", "ncbi.nlm.nih.gov")):
            trust = 0.75
        elif dom.endswith(".org") or dom.endswith(".com"):
            trust = 0.45
    click = 0.45 + (0.25 if len((title or "")) < 80 else 0) + (0.2 if has_image else 0)
    click = min(1.0, click)
    return {"relevance": round(max(0.0, min(1.0, rel)), 4), "clickability": round(click, 4), "trust": round(trust, 4)}

# -------------------------
# dataclasses & memory
# -------------------------
@dataclass
class TaskStep:
    step_id: int
    description: str
    action: str
    parameters: Dict[str, Any]


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
            "steps_taken": self.steps_taken or [],
        }


class TaskMemory:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT,
                result TEXT,
                steps_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit(); conn.close()

    def save_task(self, instruction: str, result: Any, steps_taken: List[str] = None):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO tasks (instruction, result, steps_taken) VALUES (?, ?, ?)",
                (instruction, json.dumps(result, ensure_ascii=False), json.dumps(steps_taken or [])),
            )
            conn.commit(); conn.close()
        except Exception:
            print("⚠️ TaskMemory.save_task failed", flush=True)

# -------------------------
# Helpers
# -------------------------
def extract_domain(url: Optional[str]) -> Optional[str]:
    try:
        return urlparse(url).netloc.lower() if url else None
    except Exception:
        return None


def normalize_url(url: Optional[str], base_scheme: str = "https") -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.startswith("//"): return f"{base_scheme}:{url}"
    if re.match(r"^[\w-]+\.[\w\.-]+", url) and not url.startswith("http"):
        return "https://" + url
    if url.startswith(("http://", "https://")): return url
    return None

# -------------------------
# Local model loader
# -------------------------
def load_local_llama(model_path: Optional[str], n_ctx: int = 2048):
    if not model_path or not Path(model_path).exists():
        return None
    try:
        from llama_cpp import Llama
        return Llama(model_path=str(model_path), n_ctx=n_ctx)
    except Exception as e:
        print(f"⚠️ llama_cpp load failed: {repr(e)}", flush=True)
        return None

# -------------------------
# Agent
# -------------------------
class EnhancedWebAgent:
    def __init__(self, headless: bool = True, db_dir: str = "."):
        self.db_dir = Path(db_dir).resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)
        if _DB_HELPERS_AVAILABLE:
            try: init_all_dbs(str(self.db_dir))
            except Exception: pass

        self.searches_db = str(self.db_dir / "searches.db")
        self.results_db = str(self.db_dir / "results.db")
        self.memory_db = str(self.db_dir / "memory.db")
        self.memory = TaskMemory(self.memory_db)

        self.page, self.context, self._playwright = None, None, None
        self.headless = headless
        self.model_obj = load_local_llama(os.getenv("INFOSCOUT_MODEL_PATH"), int(os.getenv("INFOSCOUT_MODEL_CTX", "2048")))
        if self.model_obj:
            print("ℹ️ Model object configured for reranker usage.", flush=True)
        else:
            print("ℹ️ No local model configured (heuristics only).", flush=True)

        self.user_data_dir = Path(".playwright_profile").resolve()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        if not _PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed."); return
        self._playwright = await async_playwright().start()
        try:
            self.context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                channel="chrome",
                headless=self.headless,
                viewport={"width": 1366, "height": 768},
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
            )
        except Exception:
            self.context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                viewport={"width": 1366, "height": 768},
            )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        await self.page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        print("✅ Launched Chrome/Chromium via Playwright.", flush=True)

    async def stop(self):
        if self.page: await self.page.close()
        if self.context: await self.context.close()
        if self._playwright: await self._playwright.stop()

    # Google Playwright extract
    async def _extract_with_playwright_google(self, url: str, search_term: str, steps_log: List[str]) -> Optional[List[Dict]]:
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(1.0)
            selectors = [
                "div.tF2Cxc a h3", "div.MjjYud a h3", "a[jsname='UWckNb']",
                "div.g h3", "a h3", "div.yuRUbf > a", "div[data-sokoban-container] a h3",
            ]
            results, rank = [], 1
            for sel in selectors:
                anchors = await self.page.query_selector_all(sel)
                for el in anchors:
                    href_val = await (await el.evaluate_handle("node => node.closest('a')")).get_property("href") if await el.evaluate_handle("node => node.closest('a')") else None
                    title = (await el.inner_text()).strip()
                    link = normalize_url(href_val) or href_val
                    if title and link and link.startswith("http"):
                        results.append({"rank": rank, "title": title, "link": link, "snippet": "", "source": "google"})
                        rank += 1
                        if rank > 30: break
                if results: break
            if results:
                steps_log.append(f"[chrome] Extracted {len(results)} results from Google (Playwright).")
                return results
            steps_log.append("[chrome] No selectors matched SERP."); return None
        except Exception as e:
            steps_log.append(f"[playwright] goto/error: {repr(e)}"); return None

    # Main search executor
    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        search_term = step.parameters.get("search_term", "")
        steps_log = [f"🔎 Start search for: {search_term}"]
        import urllib.parse as up
        url = f"https://www.google.com/search?q={up.quote_plus(search_term)}&num=10&hl=en&pws=0"

        results = await self._extract_with_playwright_google(url, search_term, steps_log)
        if not results:
            return TaskResult(success=False, data=None, error_message=f"No parsable results found for '{search_term}'. Try refining query.", steps_taken=steps_log)

        # heuristics
        for r in results:
            r["domain"] = extract_domain(r["link"])
            hs = _heuristic_scores(search_term, r.get("title", ""), r.get("domain"), False)
            r.update(hs)
            r["score"] = round(0.55 * r["relevance"] + 0.25 * r["clickability"] + 0.2 * r["trust"], 4)
            r["extra"] = r.get("extra", {})

        merged, summary = await self._run_reranker(search_term, results, steps_log)
        return TaskResult(success=True, data={"search_term": search_term, "results": merged, "summary": summary}, steps_taken=steps_log)

    async def _run_reranker(self, query: str, results: List[Dict], steps_log: List[str]):
        if not _RERANKER_AVAILABLE or not self.model_obj:
            steps_log.append("[reranker] no model configured — using heuristic scoring")
            return results, None
        try:
            out = await rerank_search(self.model_obj, query, results)
            return out.get("results", results), out.get("summary")
        except Exception as e:
            steps_log.append(f"[reranker] exception: {repr(e)}")
            return results, None

    async def execute_instruction(self, instruction: str) -> TaskResult:
        return await self._execute_search_step(TaskStep(1, f"Search for {instruction}", "search", {"search_term": instruction}))

# -------------------------
# CLI runner
# -------------------------
async def main():
    agent = EnhancedWebAgent(headless=True, db_dir=".")
    await agent.start()
    try:
        while True:
            instr = input("Enter instruction (or 'quit'): ").strip()
            if instr.lower() in ("quit", "exit"): break
            res = await agent.execute_instruction(instr)
            print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
