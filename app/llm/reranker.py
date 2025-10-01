# app/llm/reranker.py
"""
Async-friendly reranker + summary helper.

Usage:
    from app.llm.reranker import rerank_search

API:
    async def rerank_search(model_obj, query: str, results: List[Dict], max_model_calls: int=5)
    -> returns (results_with_scores, summary_str, steps_log)

Notes:
- If model_obj is None, or if the model-call fails, the function uses heuristics to produce
  relevance/clickability/trust scores and a short heuristic summary.
- If model_obj is provided, we call it in a background thread using asyncio.to_thread() to avoid
  "asyncio.run() cannot be called from a running event loop" problems.
- Supports llama_cpp Llama objects that expose either `.create()`, `.generate()`, `.__call__()` or other sync APIs.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Simple tokenizer for heuristic overlap
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
    # relevance: token-overlap & title similarity (0-1)
    rel = _token_overlap_score(query, (title or "")) * 0.95
    # domain boost for authoritative domains
    authoritative = ("edu", "gov", "ac.", "nature.com", "arxiv.org", "ieee.org", "springer", "ncbi.nlm.nih.gov")
    trust = 0.3
    if domain:
        dom = domain.lower()
        if any(d in dom for d in authoritative):
            trust = 0.75
        elif dom.endswith(".org") or dom.endswith(".com"):
            trust = 0.45
    # clickability: short titles + image help
    click = 0.45 + (0.25 if len((title or "")) < 80 else 0.0) + (0.2 if has_image else 0.0)
    click = min(1.0, click)
    # normalize relevance to reasonable range
    rel = max(0.0, min(1.0, rel))
    return {"relevance": round(rel, 4), "clickability": round(click, 4), "trust": round(trust, 4)}


# ---------- model calling helpers ----------
def _find_sync_model_api(model_obj: Any):
    """
    Return a synchronous callable that accepts (prompt: str, max_tokens: int) -> str
    tries known patterns (.create, .generate, __call__, .completion) and wraps them to return text.
    Returns None if no known API was detected.
    """
    if model_obj is None:
        return None

    # llama_cpp (newer) often exposes .create(messages=..., max_tokens=...)
    if hasattr(model_obj, "create"):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            # map to messages chat style if model wants it
            try:
                # try chat-style call
                resp = model_obj.create(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
                # try to extract text
                if isinstance(resp, dict):
                    # many llama_cpp wrappers return dict-like
                    if "choices" in resp and len(resp["choices"]) > 0:
                        c = resp["choices"][0]
                        if isinstance(c, dict) and "message" in c and "content" in c["message"]:
                            return c["message"]["content"]
                        if isinstance(c, dict) and "text" in c:
                            return c["text"]
                # fallback str()
                return str(resp)
            except Exception:
                # fallback attempt single string completion
                try:
                    resp = model_obj.create(prompt=prompt, max_tokens=max_tokens)
                    return str(resp)
                except Exception as e:
                    raise

        return _call_sync

    # huggingface-like .generate or text-generation API
    if hasattr(model_obj, "generate"):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            try:
                gen = model_obj.generate(prompt, max_tokens=max_tokens)
                return str(gen)
            except Exception as e:
                raise
        return _call_sync

    # callable model obj
    if callable(model_obj):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            try:
                out = model_obj(prompt)
                return str(out)
            except Exception:
                raise
        return _call_sync

    return None


async def _call_model_async(model_obj: Any, prompt: str, max_tokens: int = 256) -> str:
    """
    Safely call a blocking model API from async context via asyncio.to_thread.
    """
    sync_fn = _find_sync_model_api(model_obj)
    if sync_fn is None:
        raise RuntimeError("No supported sync model API found on model_obj")
    # run in threadpool
    return await asyncio.to_thread(sync_fn, prompt, max_tokens)


# ---------- main exposed function ----------
async def rerank_search(
    model_obj: Optional[Any],
    query: str,
    results: List[Dict[str, Any]],
    max_model_calls: int = 4,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """
    Rerank results and optionally summarize using the provided local model.

    Returns:
      - results: same list enriched with 'score','relevance','clickability','trust' fields
      - summary: short summary string
      - steps: list of debug/logging strings
    """
    steps: List[str] = []
    # basic heuristic pass
    for r in results:
        title = r.get("title") or ""
        domain = r.get("domain")
        has_image = bool(r.get("image"))
        hs = _heuristic_scores(query, title, domain, has_image)
        r.setdefault("relevance", hs["relevance"])
        r.setdefault("clickability", hs["clickability"])
        r.setdefault("trust", hs["trust"])
        # combined score weighted
        r["score"] = round(0.55 * r["relevance"] + 0.25 * r["clickability"] + 0.2 * r["trust"], 4)

    # attempt to use model to refine top-N
    if model_obj is None:
        steps.append("[reranker-heuristic] no model provided; used heuristic scoring")
        # produce a heuristic summary
        top = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:3]
        summary = "Top results: " + "; ".join([f"{t.get('title','(no title)')} ({t.get('domain')})" for t in top])
        steps.append("[reranker-heuristic] summary produced")
        return results, summary, steps

    # call model for a compact re-ranking prompt; limit number of candidates to avoid long calls
    candidates = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[: min(len(results), max_model_calls * 3)]
    # build concise prompt
    prompt_lines = [
        "You are an assistant that rates search results for a user query.",
        "For each item produce a JSON list element with fields: relevance (0-1), trust (0-1), clickability (0-1).",
        "Also return a short summary (1-2 sentences) describing the best sources."
    ]
    prompt_lines.append(f"Query: {query}")
    prompt_lines.append("Items:")
    for i, c in enumerate(candidates, start=1):
        prompt_lines.append(f"{i}. Title: {c.get('title','')}")
        prompt_lines.append(f"   Link: {c.get('link','')}")
        if c.get("snippet"):
            s = c.get("snippet")[:300].replace("\n", " ")
            prompt_lines.append(f"   Snippet: {s}")
        if c.get("domain"):
            prompt_lines.append(f"   Domain: {c.get('domain')}")
    prompt_lines.append("Return a JSON object like: {\"scores\": [{\"index\":1,\"relevance\":0.9,\"trust\":0.8,\"clickability\":0.7},...], \"summary\":\"...\"}")
    prompt = "\n".join(prompt_lines)

    # call the model safely
    try:
        steps.append("[reranker-model] attempting model call")
        raw = await _call_model_async(model_obj, prompt, max_tokens=256)
        if not raw:
            raise RuntimeError("empty model output")
        # try to extract JSON blob
        first_json = None
        # heuristic: find first '{' that starts a JSON object
        start = raw.find("{")
        if start != -1:
            candidate = raw[start:]
            try:
                parsed = json.loads(candidate)
                first_json = parsed
            except Exception:
                # attempt to extract trailing JSON-like lines
                try:
                    # look for balanced braces
                    stack = 0
                    end_idx = None
                    for idx, ch in enumerate(candidate):
                        if ch == "{":
                            stack += 1
                        elif ch == "}":
                            stack -= 1
                            if stack == 0:
                                end_idx = idx + 1
                                break
                    if end_idx:
                        candidate2 = candidate[:end_idx]
                        parsed = json.loads(candidate2)
                        first_json = parsed
                except Exception:
                    first_json = None
        if not first_json:
            # maybe the model returned json in multiple lines or wrote list; try to find any JSON array
            try:
                arr_start = raw.find("[")
                if arr_start != -1:
                    arr = raw[arr_start:]
                    parsed = json.loads(arr)
                    first_json = {"scores": parsed}
            except Exception:
                first_json = None

        if first_json and isinstance(first_json, dict) and "scores" in first_json:
            steps.append("[reranker-model] parsed JSON from model output")
            # map model scores into results: assume index refers to presented order (1-based)
            scores = first_json.get("scores", [])
            for sc in scores:
                idx = sc.get("index")
                # map by index into candidates list
                if idx and 1 <= int(idx) <= len(candidates):
                    target = candidates[int(idx) - 1]
                    # update fields if present
                    if "relevance" in sc:
                        target["relevance"] = float(sc["relevance"])
                    if "trust" in sc:
                        target["trust"] = float(sc["trust"])
                    if "clickability" in sc:
                        target["clickability"] = float(sc["clickability"])
                    target["score"] = round(0.55 * target["relevance"] + 0.25 * target["clickability"] + 0.2 * target["trust"], 4)
            # update the main results list with candidate updates
            cand_links = {c["link"]: c for c in candidates}
            for r in results:
                if r.get("link") in cand_links:
                    upd = cand_links[r["link"]]
                    r.update({k: upd[k] for k in ("relevance", "clickability", "trust", "score") if k in upd})
            # summary
            summ = first_json.get("summary")
            summary = summ if summ else "Top results available."
            steps.append("[reranker-model] applied model scores & summary")
            return results, summary, steps
        else:
            steps.append("[reranker-model] could not parse JSON; falling back to heuristic summary")
            top = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:3]
            summary = "Top results: " + "; ".join([f"{t.get('title','(no title)')} ({t.get('domain')})" for t in top])
            return results, summary, steps

    except Exception as e:
        # don't fail the agent — record and fallback
        steps.append(f"[reranker-model] call failed: {repr(e)}")
        top = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:3]
        summary = "Top results: " + "; ".join([f"{t.get('title','(no title)')} ({t.get('domain')})" for t in top])
        steps.append("[reranker-heuristic] fallback used")
        return results, summary, steps
