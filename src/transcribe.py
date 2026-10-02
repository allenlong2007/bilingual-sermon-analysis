"""Transcribe every downloaded episode with Whisper (large-v3-turbo, run locally through MLX).

Writes data/transcripts/<id>.json with the full text, the timed segments, and the
audio length. Episodes that already have a transcript are skipped, so this can be
stopped and restarted.
"""
import json
import time
from pathlib import Path

import mlx_whisper
import pandas as pd

MODEL = "mlx-community/whisper-large-v3-turbo"
DATA = Path(__file__).resolve().parent.parent / "data"

# Whisper sometimes writes Mandarin in traditional characters. A short prompt in
# simplified Chinese nudges it toward simplified, which matches the app's text.
CHINESE_PROMPT = "以下是普通话的句子，使用简体中文。"


def main():
    catalog = pd.read_csv(DATA / "catalog.csv")
    out_dir = DATA / "transcripts"
    out_dir.mkdir(exist_ok=True)

    for n, row in enumerate(catalog.itertuples(), start=1):
        out_path = out_dir / f"{row.id}.json"
        if out_path.exists():
            continue

        started = time.time()
        result = mlx_whisper.transcribe(
            str(DATA / "audio" / f"{row.id}.mp3"),
            path_or_hf_repo=MODEL,
            language=row.language,
            initial_prompt=CHINESE_PROMPT if row.language == "zh" else None,
        )
        segments = [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result["segments"]
        ]
        out_path.write_text(json.dumps({
            "id": row.id,
            "language": row.language,
            "duration_seconds": segments[-1]["end"] if segments else 0,
            "text": result["text"].strip(),
            "segments": segments,
        }, ensure_ascii=False, indent=1))
        print(f"[{n}/{len(catalog)}] {row.id} {row.language} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
