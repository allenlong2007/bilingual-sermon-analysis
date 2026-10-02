"""Distinctive words for each series, separately for English and Chinese, using TF-IDF.

TF-IDF scores a word highly when it shows up a lot in one series but not in the
others, so it surfaces what makes each series different rather than words every
sermon uses.
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

jieba.setLogLevel(60)

# Common Chinese function words that would otherwise crowd the results.
CHINESE_STOPWORDS = set(
    "的 了 是 我 你 他 她 它 们 我们 你们 他们 这 那 这个 那个 在 有 和 与 就 都 也 要 会 能 不 没有 "
    "一个 一 个 上 下 里 中 说 对 到 来 去 着 过 被 把 让 给 从 很 还 又 而 但 但是 所以 因为 如果 "
    "什么 怎么 这样 那么 可以 自己 就是 还是 已经 所 为 以 之 其 及 或 等 吗 呢 吧 啊 呀 嗯 哦 "
    "这些 那些 一些 时候 现在 今天 我们的 他的 你的 弟兄 姊妹 各位 "
    "看到 接着 以后 能够 真正 不同 神要 心而 小点 怎样 这里 那里 知道 觉得 应该 需要".split()
)

# Old King James style words fill the English top terms without saying much.
ENGLISH_STOPWORDS = list(ENGLISH_STOP_WORDS | {"thee", "thy", "thou", "thine", "ye", "unto", "hath", "shalt", "doth", "going", "did"})


def chinese_tokenizer(text: str) -> list[str]:
    return [
        w for w in jieba.lcut(text)
        if len(w) > 1 and w not in CHINESE_STOPWORDS and not w.isascii()
        and len(set(w)) > 1  # skips stutters Whisper wrote out, like 哲哲哲
    ]


def top_terms_by_series(corpus: pd.DataFrame, language: str, n: int = 8, min_episodes: int = 2) -> pd.DataFrame:
    subset = corpus[corpus["language"] == language]
    counts = subset["series"].value_counts()
    subset = subset[subset["series"].isin(counts[counts >= min_episodes].index)]

    # Treat each series as one long document.
    docs = subset.groupby("series")["text"].apply(" ".join)

    # Ignore words that appear in nearly every series (only possible with 3+ series).
    max_df = 0.8 if len(docs) >= 3 else 1.0
    if language == "en":
        vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOPWORDS, max_df=max_df, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    else:
        vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None, lowercase=False, max_df=max_df)

    matrix = vectorizer.fit_transform(docs.values)
    vocab = vectorizer.get_feature_names_out()
    rows = []
    for i, series in enumerate(docs.index):
        scores = matrix[i].toarray().ravel()
        top = scores.argsort()[::-1][:n]
        rows.append({"series": series, "language": language, "top_terms": ", ".join(vocab[j] for j in top)})
    return pd.DataFrame(rows)
