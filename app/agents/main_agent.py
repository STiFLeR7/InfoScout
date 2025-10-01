# updated_main_agent.py
"""
Updated AI Web Agent — Chrome-only with local LLM loader helper.

Improvements:
- More robust Playwright extraction for search results (wait for selectors, multiple fallbacks)
- HTTP fallback prefers BeautifulSoup if available, else regex
- DuckDuckGo fallback if Google fails or blocks
- load_local_mistral_model tries `llama_cpp` if available and otherwise marks file-present
- TaskResult is always serializable (.to_dict)
- Automatic saving to TaskMemory on execute
- Defensive waits and clear steps_taken logging
"""

import asyncio
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    async_playwright = None
    _PLAYWRIGHT_AVAILABLE = False

# try optional deps
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    _BS4_AVAILABLE = False

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
    def __init__(self, db_path: str = "task_memory.db"):
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
            # don't fail the agent if DB write fails
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
# Minimal Mistral parser object
# -------------------------
class SimpleMistralParser:
    def __init__(self):
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.model_obj = None
        self.model_name: Optional[str] = None
        self.loader_repr: Optional[str] = None

# -------------------------
# EnhancedWebAgent
# -------------------------
class EnhancedWebAgent:
    def __init__(self, headless: bool = True):
        self.memory = TaskMemory()
        self.page = None
        self.context = None
        self._playwright = None
        self.headless = headless

        # model attrs
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.mistral_parser = SimpleMistralParser()

        # persistent profile dir for Playwright
        self.user_data_dir = Path(".playwright_profile").resolve()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"🤖 EnhancedWebAgent initialized (headless={self.headless}). Mistral loader present: True", flush=True)

    # -------------------------
    # lifecycle: start / stop
    # -------------------------
    async def start(self):
        if not _PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed; browser automation disabled.", flush=True)
            return
        try:
            self._playwright = await async_playwright().start()
            # prefer system chrome, fallback to chromium
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
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.context.new_page()
            print("✅ Launched Chrome/Chromium via Playwright.", flush=True)
        except NotImplementedError:
            print("⚠️ Playwright NotImplementedError: subprocess support not available in this loop.", flush=True)
        except Exception as e:
            print("⚠️ Playwright start error:", e, flush=True)

    async def stop(self):
        # close page/context/playwright in reverse order; ignore exceptions
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
        # release any model-backed object if it provides close
        try:
            if getattr(self.mistral_parser, "model_obj", None):
                obj = self.mistral_parser.model_obj
                if hasattr(obj, "close"):
                    try: obj.close()
                    except Exception: pass
        except Exception:
            pass

    # -------------------------
    # Local model loader helper
    # -------------------------
    def load_local_mistral_model(self, path: str, force_mark_if_exists: bool = True) -> bool:
        p = Path(path)
        resolved = str(p.resolve()) if p.exists() else None
        self.model_path = resolved
        self.mistral_parser.model_path = resolved

        # try to import llama_cpp
        try:
            from llama_cpp import Llama
            if p.exists():
                try:
                    # instantiate Llama (may be heavy). Adjust kwargs for your machine.
                    llama = Llama(model_path=str(p))
                    self.mistral_parser.model_obj = llama
                    self.mistral_parser.model_loaded = True
                    self.mistral_parser.model_name = getattr(llama, "__repr__", lambda: "llama_model")()
                    self.mistral_parser.loader_repr = repr(llama)[:1000]
                    self.model_loaded = True
                    print(f"✅ Loaded local model via llama_cpp at: {self.model_path}", flush=True)
                    return True
                except Exception as e:
                    print(f"⚠️ llama_cpp failed to load model at {p}: {e}", flush=True)
                    # fall through to file-exists marking
        except Exception:
            # llama_cpp not installed — we'll still mark file presence if allowed
            pass

        if p.exists() and force_mark_if_exists:
            self.mistral_parser.model_loaded = True
            self.mistral_parser.model_name = p.name
            self.mistral_parser.loader_repr = "<file present, loader not bound>"
            self.model_loaded = True
            self.mistral_parser.model_path = str(p.resolve())
            print(f"ℹ️ Model file exists at {self.model_path}. Marked as present (llama_cpp not used).", flush=True)
            return True

        self.mistral_parser.model_loaded = False
        self.model_loaded = False
        print("⚠️ No local model loaded.", flush=True)
        return False

    # -------------------------
    # Search helpers
    # -------------------------
    async def _extract_with_playwright_google(self, url: str, search_term: str, steps_log: List[str]) -> Optional[List[Dict]]:
        """
        Attempt robust extraction from Google SERP using Playwright.
        Returns a list of results or None if extraction yielded nothing.
        """
        try:
            # navigate and wait for relevant selectors
            await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            # small wait for dynamic content / scripts
            await asyncio.sleep(1.0)

            # Try multiple selector strategies to be robust
            selectors = [
                "a h3",                 # generic: anchor contains h3
                "div.g h3",             # google result block h3
                "div.yuRUbf > a",       # older Google structure
                "div[data-sokoban-container] a h3"
            ]
            results = []
            rank = 1
            # Try waiting for any of common selectors
            found_any = False
            for sel in selectors:
                try:
                    # wait a bit (non-fatal)
                    await self.page.wait_for_selector(sel, timeout=2000)
                    found_any = True
                    # select all anchors that contain h3 or are result anchors
                    anchors = await self.page.query_selector_all(sel)
                    for el in anchors:
                        try:
                            # If selector matched h3, climb to parent anchor
                            tag = await el.get_property("tagName")
                            tagname = (await tag.json_value()).lower() if tag else ""
                            if tagname == "h3":
                                # parent anchor likely contains href; climb up
                                parent = await el.evaluate_handle("node => node.closest('a')")
                                if parent:
                                    href = await parent.get_property("href")
                                    href_val = await href.json_value() if href else None
                                else:
                                    href_val = None
                                title = (await el.inner_text()).strip() if await el.inner_text() else ""
                                snippet_el = await (await parent.get_property("parentNode")).as_element() if parent else None
                                snippet = ""
                            else:
                                # element is anchor-like; attempt to extract h3/text and href
                                href_val = await el.get_attribute("href")
                                # text may be h3 inside
                                title_el = await el.query_selector("h3")
                                title = (await title_el.inner_text()).strip() if title_el else (await el.inner_text()).strip()
                                snippet = ""
                            # normalize link
                            link = href_val if href_val and isinstance(href_val, str) else None
                            if title and link and link.startswith("http"):
                                results.append({"rank": rank, "title": title, "link": link, "snippet": snippet, "source": "google"})
                                rank += 1
                                if rank > 10:
                                    break
                        except Exception:
                            continue
                    if results:
                        steps_log.append(f"[chrome] Extracted {len(results)} results from Google (Playwright).")
                        return results
                except Exception:
                    # try next selector
                    continue

            # if we reached here and didn't find results
            if not found_any:
                steps_log.append("[chrome] No common selectors found on Google SERP.")
            else:
                steps_log.append("[chrome] Selector search finished but no valid results extracted.")
            return None
        except Exception as e:
            steps_log.append(f"[playwright] goto/error: {repr(e)}")
            return None

    def _http_extract_google_bs4(self, html: str, steps_log: List[str]) -> List[Dict]:
        """Extract Google results using BeautifulSoup (preferred)."""
        results = []
        if not _BS4_AVAILABLE:
            return results
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Google content frequently has h3 inside anchors
            anchors = soup.select("a h3")
            rank = 1
            seen = set()
            for h3 in anchors:
                a = h3.find_parent("a")
                if not a:
                    continue
                link = a.get("href")
                title = h3.get_text(strip=True)
                if link and title and link.startswith("http"):
                    if link in seen: continue
                    seen.add(link)
                    results.append({"rank": rank, "title": title, "link": link, "snippet": "", "source": "google-http"})
                    rank += 1
                    if rank > 10:
                        break
            return results
        except Exception as e:
            steps_log.append(f"[http-bs4] parse error: {repr(e)}")
            return []

    def _http_extract_google_regex(self, html: str, steps_log: List[str]) -> List[Dict]:
        """Fallback regex extraction on Google SERP (less reliable)."""
        results = []
        try:
            import re, urllib.parse as up
            anchors = re.findall(r'<a href="(/url\?q=https?://[^"&]+)', html)
            seen = set()
            rank = 1
            for a in anchors:
                if rank > 10: break
                qpart = up.unquote(a)
                # remove prefix /url?q=
                if qpart.startswith("/url?q="):
                    qpart = qpart[len("/url?q="):]
                # extract until & or "
                link = qpart.split("&")[0]
                if link not in seen:
                    seen.add(link)
                    results.append({"rank": rank, "title": None, "link": link, "snippet": "", "source": "google-http"})
                    rank += 1
            return results
        except Exception as e:
            steps_log.append(f"[http-regex] extraction error: {repr(e)}")
            return []

    def _http_extract_duckduckgo(self, html: str, steps_log: List[str]) -> List[Dict]:
        """Try DuckDuckGo HTML SERP parsing (if we hit ddg)."""
        results = []
        try:
            if _BS4_AVAILABLE:
                soup = BeautifulSoup(html, "html.parser")
                items = soup.select("a.result__a")
                rank = 1
                for a in items:
                    link = a.get("href")
                    title = a.get_text(strip=True)
                    if link and title:
                        results.append({"rank": rank, "title": title, "link": link, "snippet": "", "source": "duckduckgo"})
                        rank += 1
                        if rank > 10: break
            else:
                # regex fallback for ddg
                import re
                matches = re.findall(r'<a class="result__a" href="([^"]+)"', html)
                rank = 1
                for m in matches:
                    if rank > 10: break
                    results.append({"rank": rank, "title": None, "link": m, "snippet": "", "source": "duckduckgo"})
                    rank += 1
            return results
        except Exception as e:
            steps_log.append(f"[http-ddg] parse error: {repr(e)}")
            return []

    async def _http_fetch_and_extract(self, url: str, steps_log: List[str]) -> Optional[List[Dict]]:
        """
        Make HTTP GET and attempt extraction:
        - prefer bs4 extraction on Google SERP
        - fallback to regex if needed
        - if Google returns nothing, try DuckDuckGo
        """
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            html = resp.text or ""
            # Try bs4 google extractor
            res = []
            if _BS4_AVAILABLE:
                res = self._http_extract_google_bs4(html, steps_log)
            if res:
                steps_log.append(f"[http] Extracted {len(res)} results via HTTP Google (bs4).")
                return res
            # regex fallback
            res = self._http_extract_google_regex(html, steps_log)
            if res:
                steps_log.append(f"[http] Extracted {len(res)} results via HTTP Google (regex).")
                return res
            # try duckduckgo as alternative
            ddg_url = url.replace("www.google.com/search", "duckduckgo.com/html").replace("q=", "q=")
            try:
                resp2 = requests.get(ddg_url, headers=headers, timeout=12)
                html2 = resp2.text or ""
                res2 = self._http_extract_duckduckgo(html2, steps_log)
                if res2:
                    steps_log.append(f"[http] Extracted {len(res2)} results via DuckDuckGo fallback.")
                    return res2
            except Exception as e:
                steps_log.append(f"[http] DuckDuckGo fetch error: {repr(e)}")
            # nothing
            return None
        except Exception as e:
            steps_log.append(f"[fallback] HTTP fetch failed: {repr(e)}")
            return None

    # ----------- Search execution -----------
    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        search_term = step.parameters.get("search_term", "")
        if not search_term:
            return TaskResult(success=False, data=None, error_message="No search term provided", steps_taken=[])

        steps_log: List[str] = [f"🔎 Start search for: {search_term}"]
        # create Google URL with safe params
        import urllib.parse as up
        q = up.quote_plus(search_term)
        url = f"https://www.google.com/search?q={q}&num=10&hl=en&pws=0"

        # Playwright path
        if self.page is not None:
            try:
                results = await self._extract_with_playwright_google(url, search_term, steps_log)
                if results:
                    # save & return
                    payload = {"search_term": search_term, "results": results}
                    self.memory.save_task(search_term, payload, steps_taken=steps_log)
                    return TaskResult(success=True, data=payload, steps_taken=steps_log)
                else:
                    steps_log.append("[chrome] No results extracted from Playwright DOM; falling back to HTTP fetch.")
            except Exception as e:
                steps_log.append(f"[playwright] extraction exception: {repr(e)}")

        # HTTP fallback
        http_res = await self._http_fetch_and_extract(url, steps_log)
        if http_res:
            payload = {"search_term": search_term, "results": http_res}
            self.memory.save_task(search_term, payload, steps_taken=steps_log)
            return TaskResult(success=True, data=payload, steps_taken=steps_log)

        # Last resort: try DuckDuckGo direct (in case Google blocks)
        try:
            ddg_q = up.quote_plus(search_term)
            ddg_url = f"https://duckduckgo.com/html?q={ddg_q}"
            ddg_res = await self._http_fetch_and_extract(ddg_url, steps_log)
            if ddg_res:
                payload = {"search_term": search_term, "results": ddg_res}
                self.memory.save_task(search_term, payload, steps_taken=steps_log)
                return TaskResult(success=True, data=payload, steps_taken=steps_log)
        except Exception as e:
            steps_log.append(f"[fallback] DuckDuckGo attempt failed: {repr(e)}")

        # final failure
        steps_log.append("[fallback] All attempts (Playwright + HTTP) failed or returned no results")
        self.memory.save_task(search_term, None, steps_taken=steps_log)
        return TaskResult(success=False, data=None, error_message="All attempts (Playwright + HTTP) failed or returned no results", steps_taken=steps_log)

    async def execute_instruction(self, instruction: str) -> TaskResult:
        """
        Public entrypoint that orchestrates the instruction into steps.
        For now we support a single-step 'search' workflow.
        """
        step = TaskStep(step_id=1, description=f"Search for {instruction}", action="search",
                        parameters={"search_term": instruction})
        result = await self._execute_search_step(step)
        # ensure result.steps_taken is list
        if result.steps_taken is None:
            result.steps_taken = []
        return result

# If executed directly, provide a simple interactive CLI for debugging
async def main():
    agent = EnhancedWebAgent(headless=False)
    # optionally auto-load local model here; change path as needed
    # agent.load_local_mistral_model(r"D:\YASH\model\mistral-7b-openorca.gguf2.Q4_0.gguf")
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
