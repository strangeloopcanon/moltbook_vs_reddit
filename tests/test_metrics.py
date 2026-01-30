from __future__ import annotations

from moltbook_analysis.metrics import (
    distinct_n,
    exact_duplicate_rate,
    gzip_bits_per_char,
    sampled_pairwise_jaccard_similarity,
    sampled_pairwise_jaccard_similarity_sets,
    shannon_entropy_bits_per_char,
    soft_duplicate_key,
    topic_signatures,
)


def test_entropy_monotonicity_smoke() -> None:
    repetitive = "hello hello hello hello hello"
    diverse = "the quick brown fox jumps over the lazy dog 1234567890"

    assert gzip_bits_per_char(repetitive) < gzip_bits_per_char(diverse)
    assert shannon_entropy_bits_per_char(repetitive) < shannon_entropy_bits_per_char(diverse)


def test_exact_duplicate_rate() -> None:
    texts = ["hi there", "hi there", "different"]
    rate = exact_duplicate_rate(texts)
    assert rate is not None
    assert 0.0 < rate < 1.0


def test_distinct_n_smoke() -> None:
    repetitive = ["hello hello hello", "hello hello"]
    diverse = ["alpha beta gamma", "delta epsilon zeta"]
    d1_rep = distinct_n(repetitive, n=1, max_tokens=20)
    d1_div = distinct_n(diverse, n=1, max_tokens=20)
    assert d1_rep is not None and d1_div is not None
    assert d1_rep < d1_div


def test_topic_signatures_runs() -> None:
    texts = [
        "security supply chain attack signing permissions",
        "security threat model sandboxing permissions",
        "gardening tomatoes soil watering sunlight",
    ]
    sigs = topic_signatures(texts, sig_k=3, max_tokens=50, min_tokens=3, min_df=1, max_df_ratio=1.0)
    assert len(sigs) >= 1
    assert all(len(s) == 3 for s in sigs)


def test_soft_duplicate_key_smoke() -> None:
    a = "Hello, world! Alpha beta."
    b = "beta alpha hello world"
    assert soft_duplicate_key(a, max_tokens=20) == soft_duplicate_key(b, max_tokens=20)


def test_sampled_pairwise_jaccard_similarity_smoke() -> None:
    repetitive = ["alpha beta gamma"] * 20
    diverse = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "kappa lambda mu",
        "nu xi omicron",
        "pi rho sigma",
        "tau upsilon phi",
        "chi psi omega",
        "saffron violet umber",
        "magenta cyan chartreuse",
        "quartz beryl topaz",
    ]
    rep_stats = sampled_pairwise_jaccard_similarity(repetitive, pairs=200, max_tokens=20, seed=1)
    div_stats = sampled_pairwise_jaccard_similarity(diverse, pairs=200, max_tokens=20, seed=1)
    assert rep_stats["pairs"] > 0
    assert div_stats["pairs"] > 0
    assert rep_stats["mean"] > div_stats["mean"]


def test_sampled_pairwise_jaccard_similarity_sets_smoke() -> None:
    rep = [set("abc"), set("abc"), set("abc")]
    div = [set("abc"), set("def"), set("ghi")]
    rep_stats = sampled_pairwise_jaccard_similarity_sets(rep, pairs=200, seed=1)
    div_stats = sampled_pairwise_jaccard_similarity_sets(div, pairs=200, seed=1)
    assert rep_stats["pairs"] > 0
    assert div_stats["pairs"] > 0
    assert rep_stats["mean"] > div_stats["mean"]
