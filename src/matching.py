"""Cross-lingual matching: can a model find an English episode's Chinese version?"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from corpus import chunk_segments

MODELS = {
    "MiniLM (multilingual)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "BGE-M3": "BAAI/bge-m3",
}


def embed_corpus(corpus: pd.DataFrame, model: SentenceTransformer) -> dict:
    """Embed every passage of every episode.

    Returns {episode id: (passage vectors, passages)}. Vectors are normalized, so a
    dot product between two of them is their cosine similarity.
    """
    out = {}
    for row in corpus.itertuples():
        passages = chunk_segments(row.segments, row.language)
        vectors = model.encode([p["text"] for p in passages], normalize_embeddings=True, batch_size=16)
        out[row.id] = (vectors, passages)
    return out


def episode_vectors(embedded: dict) -> dict:
    """One vector per episode: the average of its passage vectors, re-normalized."""
    result = {}
    for episode_id, (vectors, _) in embedded.items():
        mean = vectors.mean(axis=0)
        result[episode_id] = mean / np.linalg.norm(mean)
    return result


def center_by_language(vectors: dict, languages: dict) -> dict:
    """Subtract each language's average vector, then re-normalize.

    Multilingual models still leave a "this is Chinese" direction in every vector.
    Removing each language's average takes that shared offset out, so what's left
    is closer to pure meaning.
    """
    means = {}
    for lang in set(languages.values()):
        means[lang] = np.mean([v for i, v in vectors.items() if languages[i] == lang], axis=0)
    centered = {}
    for i, v in vectors.items():
        c = v - means[languages[i]]
        centered[i] = c / np.linalg.norm(c)
    return centered


def evaluate(pairs: pd.DataFrame, vectors: dict, zh_ids: list[str]) -> pd.DataFrame:
    """For each English episode, rank every Chinese episode by similarity.

    `rank` is the position of the first correct Chinese episode (1 = best).
    """
    zh_matrix = np.stack([vectors[i] for i in zh_ids])
    rows = []
    for pair in pairs.itertuples():
        scores = zh_matrix @ vectors[pair.en_id]
        order = [zh_ids[i] for i in np.argsort(-scores)]
        rank = min(order.index(z) for z in pair.zh_ids) + 1
        rows.append({"en_id": pair.en_id, "series": pair.series, "rank": rank, "top_match": order[0]})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    return {
        "top-1 accuracy": (results["rank"] == 1).mean(),
        "top-3 accuracy": (results["rank"] <= 3).mean(),
        "mean reciprocal rank": (1 / results["rank"]).mean(),
    }


def random_baseline(pairs: pd.DataFrame, zh_ids: list[str], trials: int = 2000, seed: int = 0) -> dict:
    """What guessing at random would score, estimated by shuffling the ranking many times."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(trials):
        ranks = []
        for correct in pairs["zh_ids"]:
            order = list(rng.permutation(zh_ids))
            ranks.append(min(order.index(z) for z in correct) + 1)
        runs.append(summarize(pd.DataFrame({"rank": ranks})))
    return pd.DataFrame(runs).mean().to_dict()


def search(query: str, model: SentenceTransformer, embedded: dict, corpus: pd.DataFrame, language: str, top_k: int = 5):
    """Find the passages closest to `query`, only looking in episodes of `language`."""
    q = model.encode([query], normalize_embeddings=True)[0]
    titles = corpus.set_index("id")["title"]
    langs = corpus.set_index("id")["language"]
    hits = []
    for episode_id, (vectors, passages) in embedded.items():
        if langs[episode_id] != language:
            continue
        for score, passage in zip(vectors @ q, passages):
            hits.append((float(score), episode_id, titles[episode_id], passage["start"]))
    hits.sort(reverse=True)
    return pd.DataFrame(hits[:top_k], columns=["similarity", "id", "episode", "starts_at_sec"])
