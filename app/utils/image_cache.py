# app/utils/image_cache.py
"""
Image cache utility.

- Downloads images referenced in results.db (results.image)
- Saves them under: static/cache/images/<search_id>/
- Updates results.db image field to local relative path (e.g. static/cache/images/123/img-1.png)
- Uses db_helpers.set_result_local_image() if available; else updates sqlite directly.

Usage:
    python -m app.utils.image_cache --db-dir . --search-id LAST --max-workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, unquote

import requests

# try to import db_helpers helper (resilient)
_DB_HELPERS = False
try:
    from app.db.db_helpers import fetch_results_for_search, set_result_local_image  # type: ignore
    _DB_HELPERS = True
except Exception:
    try:
        from db_helpers import fetch_results_for_search, set_result_local_image  # type: ignore
        _DB_HELPERS = True
    except Exception:
        _DB_HELPERS = False

DOWNLOAD_TIMEOUT = 8  # seconds
DOWNLOAD_RETRIES = 2


def safe_filename_from_url(url: str, content_type: Optional[str] = None) -> str:
    """
    Create a stable filename for the URL. If content_type is provided, prefer extension from it.
    """
    # canonicalize and hash
    u = unquote(url)
    h = hashlib.sha1(u.encode("utf-8")).hexdigest()
    # try ext from path
    parsed = urlparse(url)
    path = Path(parsed.path)
    ext = path.suffix
    if not ext and content_type:
        ext = _ext_from_content_type(content_type)
    if not ext:
        ext = ".jpg"
    return f"{h}{ext}"


def _ext_from_content_type(content_type: str) -> Optional[str]:
    if not content_type:
        return None
    content_type = content_type.split(";")[0].strip().lower()
    # common images
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/x-icon": ".ico",
        "image/vnd.microsoft.icon": ".ico",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    return mapping.get(content_type) or mimetypes.guess_extension(content_type) or None


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


def _download_once(url: str, dest: Path, timeout: int = DOWNLOAD_TIMEOUT) -> Tuple[bool, Optional[str]]:
    """
    Download URL to dest. Returns (success, content_type_or_none).
    This function does not retry; caller handles retries.
    """
    try:
        if url.startswith("//"):
            url = "https:" + url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, timeout=timeout, stream=True, headers=headers)
        if resp.status_code != 200:
            return False, None
        content_type = resp.headers.get("Content-Type")
        # write to temp and then rename (atomic-ish)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
        return True, content_type
    except Exception:
        return False, None


def download_image_with_retries(url: str, dest: Path, retries: int = DOWNLOAD_RETRIES) -> Optional[str]:
    """
    Attempt to download and return content_type if succeeded; None on failure.
    If ext is unknown in filename, uses content_type to determine proper filename (rename).
    """
    # attempt first download to temp destination (use initial ext guess)
    success, content_type = _download_once(url, dest)
    if success:
        return content_type
    # retry loop
    for i in range(retries):
        time.sleep(0.5 + i * 0.2)
        success, content_type = _download_once(url, dest)
        if success:
            return content_type
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

    # prepare tasks
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for r in results:
            row_id = r["row_id"]
            image_url = r.get("image")
            if not image_url:
                continue
            # normalize protocol-relative immediately
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            # create an initial filename (may be adjusted based on content-type)
            fname = safe_filename_from_url(image_url)
            dest = cache_base / fname
            # if exists already, update DB immediately
            if dest.exists():
                local_rel = dest.as_posix()  # POSIX-style for cross-platform
                try:
                    if _DB_HELPERS:
                        set_result_local_image(results_db, row_id, local_rel)
                    else:
                        set_local_image_sqlite(results_db, row_id, local_rel)
                except Exception:
                    set_local_image_sqlite(results_db, row_id, local_rel)
                print(f"[image-cache] already cached row {row_id} -> {local_rel}")
                continue
            # submit download
            fut = ex.submit(download_image_with_retries, image_url, dest, DOWNLOAD_RETRIES)
            futures[fut] = (row_id, image_url, dest)

        # collect results
        for fut in as_completed(futures):
            row_id, image_url, dest = futures[fut]
            try:
                content_type = fut.result()
            except Exception as e:
                content_type = None
                print(f"[image-cache] exception during download for row {row_id}: {e}")
            if content_type is not None:
                # if extension mismatch (e.g. no ext or wrong ext), rename final file
                final_path = dest
                # compute desired filename with ext from content_type
                desired_name = safe_filename_from_url(image_url, content_type=content_type)
                desired_path = cache_base / desired_name
                if desired_path != dest:
                    try:
                        # rename dest -> desired_path
                        dest.rename(desired_path)
                        final_path = desired_path
                    except Exception:
                        final_path = dest  # fallback if rename fails
                local_rel = final_path.as_posix()
                try:
                    if _DB_HELPERS:
                        set_result_local_image(results_db, row_id, local_rel)
                    else:
                        set_local_image_sqlite(results_db, row_id, local_rel)
                except Exception:
                    # try sqlite fallback
                    set_local_image_sqlite(results_db, row_id, local_rel)
                print(f"[image-cache] cached for row {row_id} -> {local_rel}")
            else:
                print(f"[image-cache] failed to download for row {row_id} (url: {image_url})")

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

    if not Path(results_db).exists() or not Path(searches_db).exists():
        print("ERROR: results.db or searches.db not found in", db_dir)
        return

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
