# app/llm/reranker.py
"""
Async-friendly reranker + summary helper.

Return format is ALWAYS a dict:
{
  "results": [...],
  "summary": "string or None",
  "steps": ["log", "entries"]
}
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

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


# ---------- model calling helpers ----------
def _find_sync_model_api(model_obj: Any):
    """
    Detect and wrap the sync API of the model object.
    Supports llama_cpp (create), huggingface-like (generate), or callable objects.
    Tries both max_tokens and max_new_tokens variants in an order that avoids TypeError
    depending on which the underlying library expects.
    """
    if model_obj is None:
        return None

    # llama_cpp-like create() (chat-style)
    if hasattr(model_obj, "create"):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            try:
                resp = model_obj.create(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
                if isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                    c = resp["choices"][0]
                    if isinstance(c, dict):
                        if "message" in c and "content" in c["message"]:
                            return c["message"]["content"]
                        if "text" in c:
                            return c["text"]
                return str(resp)
            except Exception:
                try:
                    resp = model_obj.create(prompt=prompt, max_tokens=max_tokens)
                    return str(resp)
                except Exception:
                    raise
        return _call_sync

    # generate() style: try passing max_tokens first (some wrappers accept it),
    # then on TypeError try max_new_tokens (common for ggml/llama-cpp python wrappers).
    if hasattr(model_obj, "generate"):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            try:
                # try common param name first
                return str(model_obj.generate(prompt, max_tokens=max_tokens))
            except TypeError:
                # fallback to alternative name used by some libraries
                return str(model_obj.generate(prompt, max_new_tokens=max_tokens))
        return _call_sync

    # generic callable object
    if callable(model_obj):
        def _call_sync(prompt: str, max_tokens: int = 256) -> str:
            return str(model_obj(prompt))
        return _call_sync

    return None


async def _call_model_async(model_obj: Any, prompt: str, max_tokens: int = 256) -> str:
    sync_fn = _find_sync_model_api(model_obj)
    if sync_fn is None:
        raise RuntimeError("No supported sync model API found on model_obj")
    # run blocking sync call in thread to avoid blocking event loop
    return await asyncio.to_thread(sync_fn, prompt, max_tokens)


# ---------- internal reranker impl ----------
async def _reranker_impl(
    model_obj: Optional[Any],
    query: str,
    results: List[Dict],
    max_model_calls: int = 4,
    max_tokens: int = 256,
    use_model_for_summary: bool = True
) -> Dict[str, Any]:
    steps: List[str] = []

    # baseline heuristics applied to all results
    for r in results:
        hs = _heuristic_scores(query, r.get("title", ""), r.get("domain"), bool(r.get("image")))
        r.setdefault("relevance", hs["relevance"])
        r.setdefault("clickability", hs["clickability"])
        r.setdefault("trust", hs["trust"])
        r["score"] = round(0.55 * r["relevance"] + 0.25 * r["clickability"] + 0.2 * r["trust"], 4)

    # If no model available or user disabled model summary, return heuristic summary
    if model_obj is None or not use_model_for_summary:
        steps.append("[reranker-heuristic] no model provided or summary disabled")
        top = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
        summary = "Top results include: " + "; ".join([f"{t.get('title','(no title)')} ({t.get('domain')})" for t in top])
        return {"results": results, "summary": summary, "steps": steps}

    # Build prompt for the model using top candidates
    candidates = sorted(results, key=lambda x: x["score"], reverse=True)[: min(len(results), max_model_calls * 3)]
    prompt_lines = [
        "You are an assistant that rates search results for a user query.",
        "Return JSON: {\"scores\":[{\"index\":1,\"relevance\":0.9,\"trust\":0.8,\"clickability\":0.7},...], \"summary\":\"...\"}",
        f"Query: {query}",
        "Items:"
    ]
    for i, c in enumerate(candidates, start=1):
        prompt_lines.append(f"{i}. Title: {c.get('title','')}")
        if c.get("domain"):
            prompt_lines.append(f"   Domain: {c['domain']}")
    prompt = "\n".join(prompt_lines)

    try:
        steps.append("[reranker-model] calling model")
        raw = await _call_model_async(model_obj, prompt, max_tokens=max_tokens)
        # try to extract JSON from model output robustly
        parsed = None
        try:
            start = raw.find("{")
            if start != -1:
                candidate = raw[start:]
                parsed = json.loads(candidate)
        except Exception:
            try:
                # as a last attempt, try to find a JSON array and wrap it
                start = raw.find("[")
                if start != -1:
                    arr = raw[start:]
                    parsed = {"scores": json.loads(arr)}
            except Exception:
                parsed = None

        if parsed and "scores" in parsed:
            steps.append("[reranker-model] parsed JSON from model output")
            for sc in parsed["scores"]:
                idx = sc.get("index")
                if idx and 1 <= idx <= len(candidates):
                    target = candidates[idx - 1]
                    for k in ("relevance", "trust", "clickability"):
                        if k in sc:
                            try:
                                target[k] = float(sc[k])
                            except Exception:
                                target[k] = target.get(k, 0.0)
                    target["score"] = round(0.55 * target["relevance"] + 0.25 * target["clickability"] + 0.2 * target["trust"], 4)
            summary = parsed.get("summary", "")
            steps.append("[reranker-model] applied model scores & summary")
            return {"results": results, "summary": summary, "steps": steps}
        else:
            steps.append("[reranker-model] parse failed; falling back to heuristic summary")
    except Exception as e:
        steps.append(f"[reranker-model] error: {repr(e)}")

    # fallback: heuristic summary
    top = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    summary = "Top results include: " + "; ".join([f"{t.get('title','(no title)')} ({t.get('domain')})" for t in top])
    steps.append("[reranker-heuristic] fallback summary")
    return {"results": results, "summary": summary, "steps": steps}


# ---------- exported API ----------
async def rerank_search(model_obj: Optional[Any], query: str, results: List[Dict], **kwargs) -> Dict[str, Any]:
    """
    Async API: rerank_search(model_obj, query, results, max_model_calls=4, max_tokens=256, use_model_for_summary=True)
    Always returns dict: {"results": [...], "summary": str, "steps": [...]}
    """
    return await _reranker_impl(
        model_obj,
        query or "",
        results or [],
        max_model_calls=kwargs.get("max_model_calls", 4),
        max_tokens=kwargs.get("max_tokens", 256),
        use_model_for_summary=kwargs.get("use_model_for_summary", True),
    )
