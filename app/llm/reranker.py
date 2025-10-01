# app/llm/reranker.py
"""
Reranker: score results (relevance / clickability / trust) using
- Primary: local LLM via llama_cpp (if installed)
- Fallback: lightweight heuristic scoring (domain trust + image presence + title/snippet matching)

Writes scores back into results.db. Uses db_helpers update function if available,
otherwise performs direct sqlite updates.

Usage examples:
    # rerank last search in DB
    python -m app.llm.reranker --db-dir . --search-id LAST

    # rerank every search (careful)
    python -m app.llm.reranker --db-dir . --all --batch 8
"""
import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import List, Dict, Optional

# try to import db_helpers from package or top-level
try:
    from app.db.db_helpers import update_result_score, fetch_results_for_search, list_searches
    _DB_HELPERS = True
except Exception:
    _DB_HELPERS = False

# try to import llama_cpp
try:
    from llama_cpp import Llama
    _LLAMA_CPP = True
except Exception:
    _LLAMA_CPP = False

SANE_BATCH = 8


def heuristic_score(result: Dict) -> Dict[str, float]:
    """Compute a simple heuristic score if no LLM available."""
    score = {"relevance": 0.0, "clickability": 0.0, "trust": 0.0}
    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()
    domain = (result.get("domain") or "").lower()
    image = result.get("image")
    # relevance: presence of query-like words in title/snippet -> +0.6
    words = title.split() + snippet.split()
    word_score = min(1.0, sum(1 for w in words if len(w) > 3) / 20.0)
    score["relevance"] = 0.3 + 0.4 * word_score
    # clickability: if there's image or youtube link -> boost
    score["clickability"] = 0.2 + (0.5 if image else 0.0)
    if "youtube.com" in domain or "youtu.be" in domain:
        score["clickability"] += 0.2
    score["clickability"] = min(1.0, score["clickability"])
    # trust: domain heuristics (official domains get higher score)
    trusted = ("apple.com", "wikipedia.org", "nytimes.com", "bbc.co", "macrumors.com", "gsmarena.com")
    score["trust"] = 0.3 + (0.5 if any(t in domain for t in trusted) else 0.0)
    # small normalization
    for k in score:
        score[k] = round(float(score[k]), 4)
    return score


def llama_score_batch(model: Llama, prompt_list: List[str]) -> List[float]:
    """
    Send multiple prompts to local Llama model and parse numeric scores.
    This is a heuristic prompt: ask the model to output a float 0.0-1.0.
    Implementation depends on llama_cpp API; we use create() where available.
    If parsing fails, return NaNs to fall back later.
    """
    scores = []
    for prompt in prompt_list:
        try:
            # conservative generation: ask for a single-token numeric answer
            out = model.create(prompt=prompt, max_tokens=16, temperature=0.0, stop=["\n"])
            # depending on version, out may be dict with 'choices'
            text = ""
            if isinstance(out, dict):
                # new llama_cpp returns choices -> text
                choices = out.get("choices")
                if choices:
                    text = choices[0].get("text", "") if isinstance(choices[0], dict) else str(choices[0])
                else:
                    text = str(out.get("text", ""))
            else:
                text = str(out)
            # try to extract float
            toks = text.strip().split()
            # find first token that looks like a float
            val = None
            for t in toks:
                try:
                    v = float(t.strip().strip(".,"))
                    val = v
                    break
                except Exception:
                    continue
            if val is None:
                # try to parse words like "0.73" inside text
                import re
                m = re.search(r"([0-9]*\.[0-9]+|[01])", text)
                if m:
                    val = float(m.group(1))
            if val is None:
                scores.append(float("nan"))
            else:
                # clamp to [0,1]
                scores.append(max(0.0, min(1.0, float(val))))
        except Exception:
            scores.append(float("nan"))
    return scores


def fetch_results_sqlite(results_db: str, search_id: int) -> List[Dict]:
    """Fallback: fetch results rows using sqlite directly if db_helpers missing."""
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    cur.execute("SELECT id, rank, title, link, snippet, image, extra FROM results WHERE search_id = ? ORDER BY rank ASC", (search_id,))
    rows = cur.fetchall()
    results = []
    for r in rows:
        rid, rank, title, link, snippet, image, extra = r
        try:
            extra_j = json.loads(extra) if extra else {}
        except Exception:
            extra_j = {}
        results.append({"row_id": rid, "rank": rank, "title": title, "link": link, "snippet": snippet, "image": image, "extra": extra_j})
    conn.close()
    return results


def update_result_scores_sqlite(results_db: str, updates: List[Dict]):
    """Fallback update function to write scores into results.db directly."""
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    for u in updates:
        # u expected: {"row_id": int, "relevance": float, ...}
        cur.execute("""
            UPDATE results SET relevance = ?, clickability = ?, trust = ?, score = ?
            WHERE id = ?
        """, (u["relevance"], u["clickability"], u["trust"], u.get("score", None), u["row_id"]))
    conn.commit()
    conn.close()


def compute_overall_score(relevance: float, clickability: float, trust: float) -> float:
    # weighted aggregate — tweakable
    return round(0.6 * relevance + 0.25 * trust + 0.15 * clickability, 4)


def rerank_search(results_db: str, searches_db: str, search_id: int, model_path: Optional[str] = None, batch: int = SANE_BATCH):
    print(f"-> Reranking search_id={search_id} (results_db={results_db})")
    if _DB_HELPERS:
        rows = fetch_results_for_search(results_db, search_id)
        # expected each row is dict with at least row_id, title, snippet, image, domain
        results = rows
    else:
        results = fetch_results_sqlite(results_db, search_id)

    if not results:
        print("  no results for search_id:", search_id)
        return

    # If llama_cpp available and model_path provided (or default), use it
    use_llama = _LLAMA_CPP and model_path is not None
    model = None
    if use_llama:
        try:
            model = Llama(model_path=model_path)
            print("  loaded Llama model for reranking.")
        except Exception as e:
            print("  llama_cpp load failed, falling back to heuristic. error:", e)
            model = None
            use_llama = False

    updates = []
    # batch prompts
    for i in range(0, len(results), batch):
        batch_results = results[i:i+batch]
        # prepare prompts if using llama
        if use_llama and model:
            prompt_relevance = []
            prompt_trust = []
            prompt_click = []
            for r in batch_results:
                title = r.get("title") or ""
                snippet = r.get("snippet") or ""
                domain = r.get("domain") or ""
                prompt_relevance.append(
                    f"Given the query and the result, rate the RELEVANCE from 0.0 (not relevant) to 1.0 (perfectly relevant).\n\nResult title: {title}\nResult snippet: {snippet}\n\nAnswer with a single number between 0 and 1."
                )
                prompt_trust.append(
                    f"Given the result link domain '{domain}', rate the TRUSTWORTHINESS from 0.0 to 1.0 (1.0 = highly trustworthy).\n\nAnswer with a single number between 0 and 1."
                )
                prompt_click.append(
                    f"Given title and snippet, rate the CLICKABILITY (how likely a user is to click) from 0.0 to 1.0.\n\nTitle: {title}\nSnippet: {snippet}\n\nAnswer with a single number 0-1."
                )
            # generate scores
            rel_scores = llama_score_batch(model, prompt_relevance)
            trust_scores = llama_score_batch(model, prompt_trust)
            click_scores = llama_score_batch(model, prompt_click)
            # assign (fall back to heuristic for NaNs)
            for idx, r in enumerate(batch_results):
                rel = rel_scores[idx] if not math.isnan(rel_scores[idx]) else None
                tr = trust_scores[idx] if not math.isnan(trust_scores[idx]) else None
                cl = click_scores[idx] if not math.isnan(click_scores[idx]) else None
                if rel is None or tr is None or cl is None:
                    h = heuristic_score(r)
                    rel = rel if rel is not None else h["relevance"]
                    tr = tr if tr is not None else h["trust"]
                    cl = cl if cl is not None else h["clickability"]
                overall = compute_overall_score(rel, cl, tr)
                updates.append({
                    "row_id": r.get("row_id"),
                    "relevance": round(rel, 4),
                    "clickability": round(cl, 4),
                    "trust": round(tr, 4),
                    "score": overall
                })
        else:
            # heuristic-only path
            for r in batch_results:
                h = heuristic_score(r)
                overall = compute_overall_score(h["relevance"], h["clickability"], h["trust"])
                updates.append({
                    "row_id": r.get("row_id"),
                    "relevance": h["relevance"],
                    "clickability": h["clickability"],
                    "trust": h["trust"],
                    "score": overall
                })

    # write updates back to DB
    if _DB_HELPERS:
        try:
            # prefer a db_helpers update function if present
            update_result_score(results_db, updates)
            print(f"  wrote {len(updates)} score(s) via db_helpers.update_result_score")
            return
        except Exception as e:
            print("  db_helpers.update_result_score failed, falling back to sqlite. error:", e)

    # fallback: direct sqlite
    update_result_scores_sqlite(results_db, updates)
    print(f"  wrote {len(updates)} score(s) via direct sqlite update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=str, default=".", help="base db dir (where results.db lives)")
    parser.add_argument("--search-id", type=str, default="LAST", help="search id to rerank or LAST")
    parser.add_argument("--all", action="store_true", help="rerank all searches")
    parser.add_argument("--model-path", type=str, default=None, help="path to local gguf/ggml model for llama_cpp")
    parser.add_argument("--batch", type=int, default=SANE_BATCH, help="batch size for model calls")
    args = parser.parse_args()

    db_dir = Path(args.db_dir).resolve()
    results_db = str(db_dir / "results.db")
    searches_db = str(db_dir / "searches.db")

    # if _DB_HELPERS and user asked for all, we can list searches
    if args.all and _DB_HELPERS:
        try:
            searches = list_searches(searches_db)
            for s in searches:
                sid = s["id"]
                rerank_search(results_db, searches_db, sid, model_path=args.model_path, batch=args.batch)
            return
        except Exception as e:
            print("list_searches failed:", e)

    if args.search_id.upper() == "LAST":
        # fetch newest search id from searches.db
        conn = sqlite3.connect(str(searches_db))
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM searches ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                print("no searches found in", searches_db)
                return
            sid = int(row[0])
        finally:
            conn.close()
    else:
        sid = int(args.search_id)

    rerank_search(str(results_db), str(searches_db), sid, model_path=args.model_path, batch=args.batch)


if __name__ == "__main__":
    main()
