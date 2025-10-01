# app/utils/image_cache.py
"""
Image cache utility.

- Downloads images referenced in results.db (results.image)
- Saves them under: static/cache/images/<search_id>/
- Updates results.db image field to local relative path (e.g. static/cache/images/123/img-1.png)
- Uses db_helpers.save_local_image_path() if available; else updates sqlite directly.

Usage:
    python -m app.utils.image_cache --db-dir . --search-id LAST --max-workers 6
"""
import argparse
import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# try to import db_helpers helper
try:
    from app.db.db_helpers import fetch_results_for_search, set_result_local_image
    _DB_HELPERS = True
except Exception:
    _DB_HELPERS = False

import requests

DOWNLOAD_TIMEOUT = 8  # seconds


def safe_filename(url: str) -> str:
    # build a stable filename from url
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".jpg"
    return f"{h}{ext}"


def fetch_results_sqlite(results_db: str, search_id: int):
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    cur.execute("SELECT id, rank, link, image FROM results WHERE search_id = ? ORDER BY rank ASC", (search_id,))
    rows = cur.fetchall()
    conn.close()
    results = [{"row_id": r[0], "rank": r[1], "link": r[2], "image": r[3]} for r in rows]
    return results


def set_local_image_sqlite(results_db: str, row_id: int, local_path: str):
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    cur.execute("UPDATE results SET image = ? WHERE id = ?", (local_path, row_id))
    conn.commit()
    conn.close()


def download_image(url: str, dest: Path) -> Optional[Path]:
    try:
        # normalize simple protocol-relative URLs
        if url.startswith("//"):
            url = "https:" + url
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        if resp.status_code != 200:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)
        return dest
    except Exception:
        return None


def cache_images_for_search(db_dir: str, search_id: int, max_workers: int = 4):
    db_dir = Path(db_dir).resolve()
    results_db = str(db_dir / "results.db")
    static_dir = Path("static")  # relative to repo root
    cache_base = static_dir / "cache" / "images" / str(search_id)
    cache_base.mkdir(parents=True, exist_ok=True)

    # fetch results
    if _DB_HELPERS:
        try:
            rows = fetch_results_for_search(results_db, search_id)
            # unify shape: expect row_id and image fields
            results = [{"row_id": r["row_id"], "image": r.get("image")} for r in rows]
        except Exception:
            results = fetch_results_sqlite(results_db, search_id)
    else:
        results = fetch_results_sqlite(results_db, search_id)

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for r in results:
            row_id = r["row_id"]
            image_url = r.get("image")
            if not image_url:
                continue
            fname = safe_filename(image_url)
            dest = cache_base / fname
            # skip if exists
            if dest.exists():
                # update DB to local path immediately
                local_rel = str(dest.as_posix())
                if _DB_HELPERS:
                    try:
                        set_result_local_image(results_db, row_id, local_rel)
                    except Exception:
                        set_local_image_sqlite(results_db, row_id, local_rel)
                else:
                    set_local_image_sqlite(results_db, row_id, local_rel)
                continue
            f = ex.submit(download_image, image_url, dest)
            futures[f] = (row_id, dest)

        # collect
        for fut in as_completed(futures):
            row_id, dest = futures[fut]
            res = fut.result()
            if res:
                # store relative path to repo (static/cache/...)
                local_rel = str(dest.as_posix())
                if _DB_HELPERS:
                    try:
                        set_result_local_image(results_db, row_id, local_rel)
                    except Exception:
                        set_local_image_sqlite(results_db, row_id, local_rel)
                else:
                    set_local_image_sqlite(results_db, row_id, local_rel)
                print(f"[image-cache] cached for row {row_id} -> {local_rel}")
            else:
                print(f"[image-cache] failed to download for row {row_id}")

    print("[image-cache] done for search_id:", search_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", default=".", help="db dir (where results.db lives)")
    parser.add_argument("--search-id", default="LAST", help="search id or LAST")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    db_dir = Path(args.db_dir).resolve()
    results_db = str(db_dir / "results.db")
    searches_db = str(db_dir / "searches.db")

    # find search id if LAST
    if args.search_id.upper() == "LAST":
        conn = sqlite3.connect(str(searches_db))
        cur = conn.cursor()
        cur.execute("SELECT id FROM searches ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row:
            print("no searches in", searches_db)
            return
        sid = int(row[0])
    else:
        sid = int(args.search_id)

    cache_images_for_search(str(db_dir), sid, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
