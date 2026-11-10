"""Download the DocuAsk document set listed in data/sources.csv into data/raw/.

Run from the project root:
    python scripts/download_docs.py            # skips files already downloaded
    python scripts/download_docs.py --force    # re-download everything

Output:
    data/raw/pdf/<doc_id>.pdf
    data/raw/html/<doc_id>.html     (FCC guide pages and Complaint Center articles)
    data/raw/manifest.json          (url, bytes, sha256, download time per file)

Note: www.fcc.gov answers scripted requests with 403 Forbidden (bot protection),
while docs.fcc.gov and the Complaint Center API allow them. The www.fcc.gov files
were therefore saved through a normal web browser and are committed in data/raw/,
so a fresh clone does not need to download anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
import time
from datetime import datetime, timezone

import requests

from docuask.config import PROJECT_ROOT, get_settings

SOURCES_CSV = PROJECT_ROOT / "data" / "sources.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 DocuAsk-portfolio-project"
    ),
    "Accept": "text/html,application/pdf,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def read_sources() -> list[dict]:
    with open(SOURCES_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def target_path(raw_dir, row):
    sub, ext = ("pdf", "pdf") if row["kind"] == "pdf" else ("html", "html")
    return raw_dir / sub / f"{row['doc_id']}.{ext}"


def fetch(session: requests.Session, row: dict) -> bytes:
    resp = session.get(row["url"], timeout=30)
    resp.raise_for_status()
    if row["kind"] == "pdf":
        if not resp.content.startswith(b"%PDF"):
            raise ValueError("response is not a PDF (the site may have returned a block page)")
        return resp.content
    if row["kind"] == "zendesk":
        # Complaint Center articles come from the public Zendesk Help Center API as JSON.
        art = resp.json()["article"]
        page = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(art['title'])}</title>"
            f"<link rel='canonical' href='{html.escape(art['html_url'])}'></head>"
            f"<body><article class='zendesk'><h1>{html.escape(art['title'])}</h1>"
            f"{art['body']}</article></body></html>"
        )
        return page.encode("utf-8")
    if b"<article" not in resp.content:
        raise ValueError("HTML has no <article> element (the site may have returned a block page)")
    return resp.content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    raw_dir = get_settings().paths.raw_dir
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    session = requests.Session()
    session.headers.update(HEADERS)
    rows = read_sources()
    ok, skipped, failed = 0, 0, []

    for i, row in enumerate(rows, 1):
        path = target_path(raw_dir, row)
        label = f"[{i:2d}/{len(rows)}] {row['doc_id']:28s}"
        if path.exists() and path.stat().st_size > 0 and not args.force:
            if row["doc_id"] not in manifest:   # file added by hand / via browser: record it
                data = path.read_bytes()
                manifest[row["doc_id"]] = {
                    "url": row["url"],
                    "file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "downloaded_at": None,
                    "note": "saved via a web browser (www.fcc.gov blocks scripted downloads)",
                }
            print(f"{label} skip (exists)")
            skipped += 1
            continue
        try:
            data = fetch(session, row)
        except Exception as e:  # noqa: BLE001
            print(f"{label} FAIL {type(e).__name__}: {e}")
            failed.append(row["doc_id"])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest[row["doc_id"]] = {
            "url": row["url"],
            "file": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"{label} OK   {len(data) / 1024:7.1f} KB")
        ok += 1
        time.sleep(args.delay)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDownloaded {ok}, skipped {skipped}, failed {len(failed)} of {len(rows)} sources.")
    if failed:
        print("Failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
