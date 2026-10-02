"""Download the church app's content catalog and every podcast episode's audio.

Writes:
  data/catalog.csv   one row per episode (id, title, speaker, series, language, ...)
  data/audio/<id>.mp3
"""
import json
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import pandas as pd

MANIFEST_URL = (
    "https://firebasestorage.googleapis.com/v0/b/church-app2-7779d.firebasestorage.app"
    "/o/content.json?alt=media"
)
DATA = Path(__file__).resolve().parent.parent / "data"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, context=SSL_CONTEXT, timeout=120) as response:
        return response.read()


def load_catalog() -> pd.DataFrame:
    manifest = json.loads(fetch(MANIFEST_URL))
    episodes = pd.DataFrame([item for item in manifest["items"] if item["type"] == "podcast"])

    # Sub-folders like "Genesis" live under a parent folder ("Pre-Study").
    parents = {s["name"]: s.get("parent") for s in manifest.get("series", [])}
    episodes["parent_series"] = episodes["series"].map(parents)

    columns = ["id", "title", "speaker", "series", "parent_series", "language", "order", "dateAdded", "audioURL"]
    return episodes[columns].sort_values(["series", "language", "order"]).reset_index(drop=True)


def download(row) -> str:
    path = DATA / "audio" / f"{row.id}.mp3"
    if not path.exists():
        path.write_bytes(fetch(row.audioURL))
    return row.id


def main():
    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    catalog.to_csv(DATA / "catalog.csv", index=False)
    print(f"{len(catalog)} episodes in catalog")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, episode_id in enumerate(pool.map(download, catalog.itertuples()), start=1):
            print(f"[{done}/{len(catalog)}] {episode_id}", flush=True)


if __name__ == "__main__":
    main()
