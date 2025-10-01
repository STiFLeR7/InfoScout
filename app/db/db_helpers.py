# db_helpers.py
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any
import time

DEFAULT_TIMEOUT = 30

def _conn(path: str):
    # enable WAL and foreign keys
    conn = sqlite3.connect(path, timeout=DEFAULT_TIMEOUT, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")  # keep simple; you can enable if needed
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_all_dbs(base_dir: str = "."):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    init_searches_db(Path(base_dir) / "searches.db")
    init_results_db(Path(base_dir) / "results.db")
    init_memory_db(Path(base_dir) / "memory.db")

def init_searches_db(path: Path):
    conn = _conn(str(path))
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        num_results_requested INTEGER DEFAULT 10,
        backend TEXT DEFAULT 'google',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed_at DATETIME,
        status TEXT DEFAULT 'pending',
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_searches_query ON searches(query);
    """)
    conn.close()

def init_results_db(path: Path):
    conn = _conn(str(path))
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_id INTEGER NOT NULL,
        rank INTEGER,
        title TEXT,
        link TEXT,
        snippet TEXT,
        image TEXT,
        domain TEXT,
        source TEXT,
        score REAL DEFAULT 0.0,
        relevance REAL DEFAULT 0.0,
        clickability REAL DEFAULT 0.0,
        trust REAL DEFAULT 0.0,
        cluster_id INTEGER DEFAULT NULL,
        dedup_key TEXT DEFAULT NULL,
        fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        extra JSON DEFAULT NULL,
        UNIQUE(search_id, link)
    );
    CREATE INDEX IF NOT EXISTS idx_results_search_id ON results(search_id);
    CREATE INDEX IF NOT EXISTS idx_results_score ON results(score DESC);
    CREATE INDEX IF NOT EXISTS idx_results_dedup_key ON results(dedup_key);
    """)
    conn.close()

def init_memory_db(path: Path):
    conn = _conn(str(path))
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS task_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_id INTEGER NULL,
        instruction TEXT,
        steps_taken TEXT,
        result_summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_task_memory_search_id ON task_memory(search_id);
    """)
    conn.close()

# --- Utility helpers ---
def dedup_key_from_link(link: str) -> str:
    # normalize and hash
    s = link.strip().lower()
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h

# --- Write helpers ---
def create_search(db_path: str, query: str, num_results: int = 10, backend: str = "google", metadata: Optional[Dict]=None) -> int:
    conn = _conn(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO searches (query, num_results_requested, backend, metadata) VALUES (?, ?, ?, ?)",
                (query, num_results, backend, json.dumps(metadata or {})))
    search_id = cur.lastrowid
    conn.close()
    return search_id

def mark_search_completed(db_path: str, search_id: int):
    conn = _conn(db_path)
    conn.execute("UPDATE searches SET status='done', completed_at=CURRENT_TIMESTAMP WHERE id=?", (search_id,))
    conn.close()

def insert_result(results_db: str, search_id: int, item: Dict[str,Any], compute_dedup=True) -> Optional[int]:
    conn = _conn(results_db)
    dedup = dedup_key_from_link(item.get("link","")) if compute_dedup else None
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT OR IGNORE INTO results
            (search_id, rank, title, link, snippet, image, domain, source, score, extra, dedup_key)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (search_id,
             item.get("rank"),
             item.get("title"),
             item.get("link"),
             item.get("snippet"),
             item.get("image"),
             item.get("domain"),
             item.get("source"),
             item.get("score", 0.0),
             json.dumps(item.get("extra", {}), ensure_ascii=False),
             dedup)
        )
        rid = cur.lastrowid
        conn.commit()
    except Exception as e:
        print("insert_result error:", e)
        rid = None
    finally:
        conn.close()
    return rid

def bulk_insert_results(results_db: str, search_id: int, items: List[Dict[str,Any]]):
    for it in items:
        insert_result(results_db, search_id, it)

# --- Read helpers ---
def fetch_results_for_search(results_db: str, search_id: int, limit: int = 50, order_by_score: bool = True):
    conn = _conn(results_db)
    order = "score DESC" if order_by_score else "rank ASC"
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM results WHERE search_id=? ORDER BY {order} LIMIT ?", (search_id, limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def update_result_score(results_db: str, result_id: int, *, score: float=None, relevance: float=None, clickability: float=None, trust: float=None):
    conn = _conn(results_db)
    # build update dynamically
    updates = []
    params = []
    if score is not None:
        updates.append("score=?"); params.append(score)
    if relevance is not None:
        updates.append("relevance=?"); params.append(relevance)
    if clickability is not None:
        updates.append("clickability=?"); params.append(clickability)
    if trust is not None:
        updates.append("trust=?"); params.append(trust)
    if not updates:
        conn.close(); return
    params.append(result_id)
    sql = "UPDATE results SET " + ", ".join(updates) + " WHERE id=?"
    conn.execute(sql, params)
    conn.close()

def save_memory(memory_db: str, search_id: Optional[int], instruction: str, steps_taken: List[str], summary: Optional[str]=None):
    conn = _conn(memory_db)
    conn.execute("INSERT INTO task_memory (search_id, instruction, steps_taken, result_summary) VALUES (?, ?, ?, ?)",
                 (search_id, instruction, json.dumps(steps_taken, ensure_ascii=False), summary))
    conn.close()
