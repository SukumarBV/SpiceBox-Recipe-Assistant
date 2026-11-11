"""Optional: download a pre-built recipes.db when it is missing (for free cloud hosting).

Set DB_URL (env var or Streamlit secret) to a direct-download URL of recipes.db or recipes.db.gz that YOU host
(e.g. a GitHub Release asset on a repo where you are allowed to redistribute the data). No URL is assumed or invented.
"""
from __future__ import annotations
import gzip
import os
import shutil
import urllib.request
from pathlib import Path


def get_url(secrets=None) -> str:
    v = os.environ.get("DB_URL", "")
    if not v and secrets is not None:
        try:
            if "DB_URL" in secrets:
                v = str(secrets["DB_URL"])
        except Exception:
            pass
    return v.strip()


def download_db(url: str, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    req = urllib.request.Request(url, headers={"User-Agent": "spicebox-db-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    if url.lower().endswith(".gz"):
        out = dest.with_suffix(".unpacked")
        with gzip.open(tmp, "rb") as g, open(out, "wb") as f:
            shutil.copyfileobj(g, f, length=1024 * 1024)
        tmp.unlink()
        tmp = out
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)
    return dest
