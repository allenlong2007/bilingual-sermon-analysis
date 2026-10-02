"""Load the transcripts into a pandas DataFrame and clean them up."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Whisper has a known habit of "hearing" these phrases during music or silence,
# because they show up so often at the ends of online videos it was trained on.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thank you for watching.", "you",
    "谢谢观看", "谢谢观看。", "谢谢大家", "字幕由amara.org社区提供", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
}

# During background music Whisper also invents song credits and subtitle notices,
# e.g. "作词曲 李宗盛" (a famous songwriter). Any segment containing these is dropped.
HALLUCINATION_MARKERS = ("作词", "作曲", "字幕", "subtitles by")


def clean_segments(segments: list[dict]) -> list[dict]:
    """Drop hallucinated filler and segments that just repeat the one before."""
    cleaned = []
    for seg in segments:
        text = seg["text"].strip()
        if not text or text.lower() in HALLUCINATIONS:
            continue
        if any(marker in text.lower() for marker in HALLUCINATION_MARKERS):
            continue
        if cleaned and text == cleaned[-1]["text"]:
            continue
        cleaned.append({**seg, "text": text})
    return cleaned


def text_length(text: str, language: str) -> int:
    """Words for English. Characters for Chinese, which has no spaces between words."""
    if language == "en":
        return len(text.split())
    return sum(1 for ch in text if "一" <= ch <= "鿿")


def load_corpus() -> pd.DataFrame:
    """One row per episode: catalog metadata plus the cleaned transcript."""
    catalog = pd.read_csv(DATA / "catalog.csv")
    rows = []
    for episode in catalog.itertuples():
        path = DATA / "transcripts" / f"{episode.id}.json"
        if not path.exists():
            continue
        transcript = json.loads(path.read_text())
        segments = clean_segments(transcript["segments"])
        text = " ".join(s["text"] for s in segments)
        rows.append({
            "id": episode.id,
            "title": episode.title,
            "series": episode.series,
            "language": episode.language,
            "duration_min": transcript["duration_seconds"] / 60,
            "segments": segments,
            "text": text,
            "length": text_length(text, episode.language),
            "segments_removed": len(transcript["segments"]) - len(segments),
        })
    corpus = pd.DataFrame(rows)
    # Words per minute for English, characters per minute for Chinese.
    corpus["rate_per_min"] = corpus["length"] / corpus["duration_min"]
    return corpus


def chunk_segments(segments: list[dict], language: str, target: int | None = None) -> list[dict]:
    """Group consecutive segments into passages of roughly `target` words (or characters).

    Embedding models only read a limited amount of text at once, so a 20 minute
    episode gets split into passages that each keep their start time.
    """
    target = target or (120 if language == "en" else 250)
    chunks, current, size = [], [], 0
    for seg in segments:
        current.append(seg)
        size += text_length(seg["text"], language)
        if size >= target:
            chunks.append(_join(current))
            current, size = [], 0
    if current:
        chunks.append(_join(current))
    return chunks


def _join(segments: list[dict]) -> dict:
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(s["text"] for s in segments),
    }


def load_pairs() -> pd.DataFrame:
    """Hand-labeled answer key: each English episode and its Chinese counterpart(s)."""
    pairs = pd.read_csv(ROOT / "pairs.csv")
    pairs["zh_ids"] = pairs["zh_ids"].str.split()
    return pairs
