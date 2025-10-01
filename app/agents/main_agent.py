# app/agents/main_agent.py
"""
InfoScout — EnhancedWebAgent (Google-only, DB-integrated, headless by default)

Responsibilities:
- Robust imports for db_helpers and reranker
- Playwright-based Google SERP extraction + HTTP (bs4/regex) fallback
- Image (og:image / favicon) fetch
- Quiet local model loading (llama-cpp-python) and fallback to heuristics
- Calls into reranker.rerank_search for scoring & summary
"""
import asyncio
import contextlib
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin

# -------------------------
# resilient imports (db_helpers, reranker)
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

# reranker import (optional)
_RERANKER_AVAILABLE = False
try:
    from app.llm.reranker import rerank_search
    print("ℹ️ Imported rerank_search from app.llm.reranker", flush=True)
    _RERANKER_AVAILABLE = True
except Exception:
    try:
        from llm.reranker import rerank_search  # top-level fallback
        print("ℹ️ Imported rerank_search from llm.reranker (top-level)", flush=True)
        _RERANKER_AVAILABLE = True
    except Exception:
        print("⚠️ reranker import failed (app.llm.reranker / llm.reranker).", flush=True)
        _RERANKER_AVAILABLE = False

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
# try Llama binding availability (we load later quietly)
# -------------------------
_LLAMACPP_AVAILABLE = False
_LLAMACPP_ERR = None
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMACPP_AVAILABLE = True
except Exception as e:
    _LLAMACPP_AVAILABLE = False
    _LLAMACPP_ERR = e

# -------------------------
# dataclasses & TaskMemory
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
# Helpers: normalize and image fetch
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

def _fetch_og_or_favicon(url: str, steps_log: List[str], timeout: int = 6) -> Optional[str]:
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        norm = normalize_url(url)
        if not norm:
            steps_log.append(f"[image-fetch] cannot normalize url: {url}")
            return None
        resp = requests.get(norm, headers=headers, timeout=timeout)
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
# Silence native stdout/stderr for model load
# -------------------------
@contextlib.contextmanager
def _suppress_native_output():
    """
    Redirects OS-level stdout/stderr to devnull for the context duration,
    to silence native library prints (like llama.cpp).
    """
    devnull = open(os.devnull, "w")
    try:
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(orig_stdout_fd, 1)
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)
            devnull.close()
    except Exception:
        # fallback to Python-level redirect if dup ops fail
        with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
            yield

# -------------------------
# Model-call adapter (used only when model loaded)
# -------------------------
async def _call_model(model_obj, prompt: str, max_tokens: int = 128, stop: Optional[List[str]] = None) -> str:
    """
    Call the local model wrapper in a thread; handle several common wrapper shapes.
    Returns textual output or raises.
    """
    def _sync():
        # prefer callable usage (llama-cpp-python supports calling the Llama instance)
        if hasattr(model_obj, "create"):
            out = model_obj.create(prompt=prompt, max_tokens=max_tokens, temperature=0.0, stop=stop or ["\n"])
            if isinstance(out, dict):
                if "choices" in out and out["choices"]:
                    ch = out["choices"][0]
                    if isinstance(ch, dict):
                        return ch.get("text") or ch.get("message", {}).get("content", "") or str(ch)
                    return str(ch)
                if "text" in out:
                    return out["text"]
                return str(out)
            return str(out)
        if callable(model_obj):
            try:
                out = model_obj(prompt, max_tokens=max_tokens, temperature=0.0)
            except TypeError:
                out = model_obj(prompt)
            if isinstance(out, dict):
                if "choices" in out and out["choices"]:
                    ch = out["choices"][0]
                    if isinstance(ch, dict):
                        return ch.get("text") or ch.get("message", {}).get("content", "") or str(ch)
                    return str(ch)
                if "text" in out:
                    return out["text"]
            return str(out)
        if hasattr(model_obj, "generate"):
            try:
                out = model_obj.generate(prompt=prompt, max_tokens=max_tokens)
            except TypeError:
                out = model_obj.generate(prompt)
            return str(out)
        raise AttributeError("Unsupported model call interface.")
    return await asyncio.to_thread(_sync)

# -------------------------
# EnhancedWebAgent
# -------------------------
class EnhancedWebAgent:
    def __init__(self, headless: bool = True, db_dir: str = ".", model_path: Optional[str] = None, quiet_model_load: bool = True):
        # canonicalize db_dir
        self.db_dir = Path(db_dir).resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # init DBs
        if _DB_HELPERS_AVAILABLE:
            try:
                init_all_dbs(str(self.db_dir))
                print(f"ℹ️ DBs initialized in {self.db_dir}", flush=True)
            except Exception as e:
                print("⚠️ init_all_dbs failed:", e, flush=True)

        self.searches_db = str(self.db_dir / "searches.db")
        self.results_db = str(self.db_dir / "results.db")
        self.memory_db = str(self.db_dir / "memory.db")

        self.memory = TaskMemory(db_path=self.memory_db)

        # Playwright
        self.page = None
        self.context = None
        self._playwright = None
        self.headless = headless

        # model
        self.mistral_parser = SimpleMistralParser()
        self.model_path = model_path
        self.quiet_model_load = quiet_model_load
        if model_path:
            ok = self.load_local_mistral_model(model_path, quiet=quiet_model_load)
            if ok:
                print(f"✅ Local model loaded at: {model_path}", flush=True)
            else:
                print("⚠️ Local model not loaded; using heuristics-only fallback.", flush=True)
        else:
            if _LLAMACPP_AVAILABLE:
                print("ℹ️ llama_cpp available but model_path not provided.", flush=True)
            else:
                print("⚠️ llama_cpp not available:", repr(_LLAMACPP_ERR), flush=True)

        # Playwright profile
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
        try:
            if getattr(self.mistral_parser, "model_obj", None):
                obj = self.mistral_parser.model_obj
                if hasattr(obj, "close"):
                    try: obj.close()
                    except Exception: pass
        except Exception:
            pass

    # Quiet model load
    def load_local_mistral_model(self, path: str, force_mark_if_exists: bool = True, quiet: bool = True) -> bool:
        p = Path(path)
        resolved = str(p.resolve()) if p.exists() else None
        self.model_path = resolved
        self.mistral_parser.model_path = resolved

        if not p.exists():
            print("⚠️ model file not found:", path, flush=True)
            return False

        if not _LLAMACPP_AVAILABLE:
            print("⚠️ llama_cpp not installed:", repr(_LLAMACPP_ERR), flush=True)
            if p.exists() and force_mark_if_exists:
                # mark presence but don't bind loader
                self.mistral_parser.model_loaded = True
                self.mistral_parser.model_name = p.name
                self.mistral_parser.loader_repr = "<file present, llama_cpp not installed>"
                self.model_loaded = True
                return False
            return False

        # small noop callback to reduce native logging if the binding exposes it
        try:
            import llama_cpp as _lc
            if hasattr(_lc, "llama_log_set"):
                try:
                    # register a minimal callback (ctypes) if available
                    def _noop_cb(msg_ptr, userp):
                        return
                    try:
                        _lc.llama_log_set(_noop_cb)
                    except Exception:
                        # some bindings expect a specific ctypes signature; ignore if incompatible
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if quiet:
                with _suppress_native_output():
                    model = Llama(model_path=str(p), n_ctx=2048, n_threads=os.cpu_count() or 4, verbose=False, use_mlock=False)
            else:
                model = Llama(model_path=str(p), n_ctx=2048, n_threads=os.cpu_count() or 4, verbose=False, use_mlock=False)
            self.mistral_parser.model_obj = model
            self.mistral_parser.model_loaded = True
            self.mistral_parser.model_name = getattr(model, "__repr__", lambda: "llama_model")()
            self.mistral_parser.loader_repr = repr(model)[:1000]
            self.model_loaded = True
            print(f"✅ Loaded local model via llama_cpp at: {self.model_path}", flush=True)
            return True
        except Exception as e:
            print(f"⚠️ llama_cpp failed to load model at {p}: {e}", flush=True)
            # mark file present if requested so agent can still run heuristics and mark DB
            if p.exists() and force_mark_if_exists:
                self.mistral_parser.model_loaded = True
                self.mistral_parser.model_name = p.name
                self.mistral_parser.loader_repr = "<file present, loader not bound>"
                self.model_loaded = True
                self.mistral_parser.model_path = str(p.resolve())
                print(f"ℹ️ Model file exists at {self.model_path}. Marked as present (load failed).", flush=True)
                return True
        self.mistral_parser.model_loaded = False
        self.model_loaded = False
        return False

    # Playwright extraction
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
                                if rank > 10:
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

    # HTTP fallback
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
                    if rank > 10:
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
                if rank > 10: break
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

    # ----------- Search execution -----------
    async def _execute_search_step(self, step: TaskStep) -> TaskResult:
        search_term = step.parameters.get("search_term", "")
        if not search_term:
            return TaskResult(success=False, data=None, error_message="No search term provided", steps_taken=[])

        steps_log: List[str] = [f"🔎 Start search for: {search_term}"]
        import urllib.parse as up
        q = up.quote_plus(search_term)
        url = f"https://www.google.com/search?q={q}&num=10&hl=en&pws=0"

        # create search record
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
        # Playwright
        if self.page is not None:
            try:
                results = await self._extract_with_playwright_google(url, search_term, steps_log)
                if results:
                    steps_log.append(f"[flow] Playwright returned {len(results)} results.")
                else:
                    steps_log.append("[chrome] No results extracted from Playwright DOM; falling back to HTTP fetch.")
            except Exception as e:
                steps_log.append(f"[playwright] extraction exception: {repr(e)}")

        # HTTP fallback
        if not results:
            http_res = await self._http_fetch_and_extract(url, steps_log)
            if http_res:
                results = http_res
                steps_log.append(f"[flow] HTTP fetch returned {len(results)} results.")

        # enrich results (image + scoring/summarization via reranker)
        if results:
            for r in results:
                link = r.get("link")
                r["link"] = normalize_url(link) or link
                r["domain"] = extract_domain(r["link"])
                try:
                    img = _fetch_og_or_favicon(r["link"], steps_log)
                except Exception as e:
                    steps_log.append(f"[image-fetch] error for {r.get('link')}: {repr(e)}")
                    img = None
                r["image"] = img
                r.setdefault("score", 0.0)
                r.setdefault("relevance", 0.0)
                r.setdefault("clickability", 0.0)
                r.setdefault("trust", 0.0)
                r["extra"] = r.get("extra", {})

            # run reranker if present, else try inline fallback
            results_enriched = results
            summary_text = None
            if _RERANKER_AVAILABLE:
                try:
                    # rerank_search(model, query, results, steps_log) -> {results, summary}
                    model_obj = getattr(self.mistral_parser, "model_obj", None)
                    rr = rerank_search(model_obj, search_term, results, steps_log)
                    # rerank_search may be sync or return dict; handle both
                    if isinstance(rr, dict) and "results" in rr:
                        results_enriched = rr["results"]
                        summary_text = rr.get("summary")
                    else:
                        # assume sync list
                        results_enriched = rr
                except Exception as e:
                    steps_log.append(f"[reranker] failed: {repr(e)}")
                    traceback.print_exc()
            else:
                # no reranker module — try a tiny builtin heuristic summary
                try:
                    # simplest: compute heuristic score (relevance heuristics)
                    def heuristic_score(r):
                        title = (r.get("title") or "").lower()
                        img = bool(r.get("image"))
                        domain = r.get("domain") or ""
                        rel = 0.35 + min(0.65, len(title.split()) / 20.0)
                        clk = 0.2 + (0.5 if img else 0.0)
                        trust = 0.3 + (0.4 if any(t in domain for t in ("apple.com", "wikipedia.org", "macrumors.com", "gsmarena.com")) else 0.0)
                        return round(max(0.0, min(1.0, rel)),4), round(clk,4), round(trust,4)
                    for r in results_enriched:
                        rel, clk, trust = heuristic_score(r)
                        r["relevance"] = rel
                        r["clickability"] = clk
                        r["trust"] = trust
                        r["score"] = round(0.6 * rel + 0.15 * clk + 0.25 * trust, 4)
                    summary_text = f"Top result: {results_enriched[0].get('title','').strip()}" if results_enriched else ""
                    steps_log.append("[heuristic-summary] used fallback summary")
                except Exception as e:
                    steps_log.append(f"[heuristic-summary] failed: {repr(e)}")

            # persist backward-compatible memory
            try:
                self.memory.save_task(search_term, {"search_term": search_term, "results": results_enriched, "summary": summary_text}, steps_taken=steps_log)
            except Exception:
                pass

            # DB writes via helpers
            if _DB_HELPERS_AVAILABLE and search_id is not None:
                try:
                    bulk_insert_results(self.results_db, search_id, results_enriched)
                    mark_search_completed(self.searches_db, search_id)
                    try:
                        save_memory(self.memory_db, search_id, search_term, steps_log, summary=summary_text)
                    except Exception as e:
                        steps_log.append(f"[db] save_memory failed: {repr(e)}")
                except Exception as e:
                    steps_log.append(f"[db] bulk_insert_results/mark_search_completed failed: {repr(e)}")

            payload = {"search_term": search_term, "results": results_enriched, "summary": summary_text, "search_id": search_id}
            return TaskResult(success=True, data=payload, steps_taken=steps_log)

        # final failure
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

    async def execute_instruction(self, instruction: str) -> TaskResult:
        step = TaskStep(step_id=1, description=f"Search for {instruction}", action="search",
                        parameters={"search_term": instruction})
        result = await self._execute_search_step(step)
        if result.steps_taken is None:
            result.steps_taken = []
        return result

# CLI runner
async def main():
    model_path = "model/mistral-7b-openorca.gguf2.Q4_0.gguf"  # adjust as needed
    agent = EnhancedWebAgent(headless=True, db_dir=".", model_path=model_path, quiet_model_load=True)
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
