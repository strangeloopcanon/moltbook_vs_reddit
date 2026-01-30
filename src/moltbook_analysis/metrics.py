from __future__ import annotations

import gzip
import math
import random
import re
from dataclasses import dataclass

_FENCED_CODE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://\\S+")
_QUOTE_LINE_RE = re.compile(r"(^|\\n)>[^\\n]*")


def clean_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _FENCED_CODE_RE.sub(" ", t)
    t = _MARKDOWN_LINK_RE.sub(r"\\1", t)
    t = _INLINE_CODE_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _QUOTE_LINE_RE.sub(" ", t)
    return " ".join(t.split())


def gzip_bits_per_char(text: str) -> float:
    raw = clean_text(text).encode("utf-8")
    if not raw:
        return 0.0
    compressed = gzip.compress(raw, compresslevel=9)
    return (len(compressed) * 8) / len(raw)


def shannon_entropy_bits_per_char(text: str) -> float:
    raw = clean_text(text)
    if not raw:
        return 0.0
    counts: dict[str, int] = {}
    for ch in raw:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(raw)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def type_token_ratio(text: str, *, max_tokens: int = 50) -> float:
    cleaned = clean_text(text).lower()
    if not cleaned:
        return 0.0
    tokens = [tok.strip(".,!?;:\"'()[]{}") for tok in cleaned.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    tokens = tokens[:max_tokens]
    return len(set(tokens)) / len(tokens)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
_HEXISH_RE = re.compile(r"^[0-9a-f]{6,}$")
_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "more",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "people",
    "really",
    "so",
    "some",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "up",
    "use",
    "was",
    "we",
    "were",
    "what",
    "when",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _is_noise_token(token: str) -> bool:
    if len(token) > 30:
        return True
    if _HEXISH_RE.fullmatch(token):
        return True
    digit_ratio = sum(ch.isdigit() for ch in token) / len(token)
    return digit_ratio >= 0.5


def tokenize(text: str, *, max_tokens: int = 200) -> list[str]:
    cleaned = clean_text(text).lower()
    toks = [
        t
        for t in _TOKEN_RE.findall(cleaned)
        if len(t) >= 3 and t not in _STOPWORDS and not _is_noise_token(t)
    ]
    return toks[:max_tokens]


def soft_duplicate_key(text: str, *, max_tokens: int = 80) -> str | None:
    toks = tokenize(text, max_tokens=max_tokens)
    if not toks:
        return None
    toks.sort()
    return " ".join(toks)


def exact_duplicate_rate(texts: list[str]) -> float | None:
    if not texts:
        return None
    cleaned = [clean_text(t) for t in texts]
    return 1.0 - (len(set(cleaned)) / len(cleaned))


def distinct_n(texts: list[str], *, n: int, max_tokens: int = 200) -> float | None:
    if n <= 0:
        raise ValueError("n must be positive")
    total = 0
    uniq: set[tuple[str, ...]] = set()
    for t in texts:
        toks = tokenize(t, max_tokens=max_tokens)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            uniq.add(ng)
            total += 1
    if total == 0:
        return None
    return len(uniq) / total


def ngram_counts(texts: list[str], *, n: int, max_tokens: int = 200) -> dict[tuple[str, ...], int]:
    if n <= 0:
        raise ValueError("n must be positive")
    counts: dict[tuple[str, ...], int] = {}
    for t in texts:
        toks = tokenize(t, max_tokens=max_tokens)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            counts[ng] = counts.get(ng, 0) + 1
    return counts


def shannon_entropy_bits_from_counts(counts: dict[object, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def top_k_coverage(counts: dict[object, int], *, k: int) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    if k <= 0:
        return 0.0
    top = sum(sorted(counts.values(), reverse=True)[:k])
    return top / total


def topic_signatures(
    texts: list[str],
    *,
    sig_k: int = 3,
    max_tokens: int = 200,
    min_tokens: int = 5,
    min_df: int = 2,
    max_df_ratio: float = 0.3,
) -> list[tuple[str, ...]]:
    docs: list[list[str]] = []
    for t in texts:
        toks = tokenize(t, max_tokens=max_tokens)
        if len(toks) < min_tokens:
            continue
        docs.append(toks)

    n = len(docs)
    if n == 0:
        return []

    df: dict[str, int] = {}
    for toks in docs:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1

    max_df = int(max_df_ratio * n)
    if max_df < min_df:
        max_df = n
    allowed = {tok for tok, dfc in df.items() if dfc >= min_df and dfc <= max_df}
    idf = {tok: math.log((n + 1) / (dfc + 1)) + 1.0 for tok, dfc in df.items() if tok in allowed}

    sigs: list[tuple[str, ...]] = []
    for toks in docs:
        tf: dict[str, int] = {}
        for tok in toks:
            if tok not in allowed:
                continue
            tf[tok] = tf.get(tok, 0) + 1
        scored = sorted(
            ((tok, tfc * idf.get(tok, 0.0)) for tok, tfc in tf.items()),
            key=lambda x: (-x[1], x[0]),
        )
        top = tuple(tok for tok, _ in scored[:sig_k])
        if len(top) == sig_k:
            sigs.append(top)
    return sigs


def distribution_entropy_and_coverage(
    items: list[tuple[str, ...]] | list[str], *, top_ks: list[int]
) -> dict:
    if not items:
        return {"n": 0}

    counts: dict[object, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    total = sum(counts.values())
    ent = shannon_entropy_bits_from_counts(counts)
    top_counts = sorted(counts.values(), reverse=True)
    coverage = {str(k): sum(top_counts[:k]) / total for k in top_ks}
    return {
        "n": total,
        "unique": len(counts),
        "entropy_bits": ent,
        "effective_topics": (2**ent) if ent is not None else None,
        "topk_coverage": coverage,
    }


def sampled_pairwise_jaccard_similarity(
    texts: list[str],
    *,
    pairs: int = 2000,
    max_tokens: int = 80,
    seed: int = 1337,
) -> dict:
    if pairs <= 0:
        return {"pairs": 0, "max_tokens": max_tokens}

    token_sets = [set(tokenize(t, max_tokens=max_tokens)) for t in texts]
    token_sets = [s for s in token_sets if s]

    n = len(token_sets)
    if n < 2:
        return {"pairs": 0, "max_tokens": max_tokens}

    rng = random.Random(seed)
    sims: list[float] = []
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        a = token_sets[i]
        b = token_sets[j]
        union = len(a | b)
        if union == 0:
            continue
        sims.append(len(a & b) / union)

    if not sims:
        return {"pairs": 0, "max_tokens": max_tokens}

    sims.sort()
    mean = sum(sims) / len(sims)
    mid = len(sims) // 2
    median = sims[mid] if len(sims) % 2 else (sims[mid - 1] + sims[mid]) / 2

    def percentile(p: float) -> float:
        return sims[int(p * (len(sims) - 1))]

    return {
        "pairs": len(sims),
        "max_tokens": max_tokens,
        "mean": mean,
        "median": median,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def sampled_pairwise_jaccard_similarity_sets(
    sets: list[set[str]],
    *,
    pairs: int = 2000,
    seed: int = 1337,
) -> dict:
    if pairs <= 0:
        return {"pairs": 0}

    items = [s for s in sets if s]
    n = len(items)
    if n < 2:
        return {"pairs": 0}

    rng = random.Random(seed)
    sims: list[float] = []
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        a = items[i]
        b = items[j]
        union = len(a | b)
        if union == 0:
            continue
        sims.append(len(a & b) / union)

    if not sims:
        return {"pairs": 0}

    sims.sort()
    mean = sum(sims) / len(sims)
    mid = len(sims) // 2
    median = sims[mid] if len(sims) % 2 else (sims[mid - 1] + sims[mid]) / 2

    def percentile(p: float) -> float:
        return sims[int(p * (len(sims) - 1))]

    return {
        "pairs": len(sims),
        "mean": mean,
        "median": median,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


@dataclass(frozen=True)
class ThreadMetrics:
    thread_id: str
    source: str
    message_count: int
    char_len: int
    gzip_bpc: float
    shannon_bpc: float
