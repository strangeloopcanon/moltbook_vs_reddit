from __future__ import annotations

import json
import random
import sqlite3
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .db import init_db
from .havelock_client import HavelockClient
from .metrics import (
    ThreadMetrics,
    clean_text,
    distinct_n,
    distribution_entropy_and_coverage,
    exact_duplicate_rate,
    gzip_bits_per_char,
    ngram_counts,
    sampled_pairwise_jaccard_similarity,
    sampled_pairwise_jaccard_similarity_sets,
    shannon_entropy_bits_from_counts,
    shannon_entropy_bits_per_char,
    soft_duplicate_key,
    top_k_coverage,
    topic_signatures,
    type_token_ratio,
)
from .orality import havelock_orality_by_section


@dataclass(frozen=True)
class AnalysisReport:
    moltbook_threads: int
    reddit_threads: int
    moltbook_messages: int
    reddit_messages: int
    metrics: dict
    sources: dict[str, str] | None = None
    reddit_domain_threads: int | None = None
    reddit_domain_messages: int | None = None
    metrics_domain: dict | None = None

    def to_json(self, *, indent: int | None = None) -> str:
        obj: dict[str, object] = {
            "moltbook_threads": self.moltbook_threads,
            "reddit_threads": self.reddit_threads,
            "moltbook_messages": self.moltbook_messages,
            "reddit_messages": self.reddit_messages,
            "metrics": self.metrics,
        }
        if self.sources is not None:
            obj["sources"] = self.sources
        if self.reddit_domain_threads is not None:
            obj["reddit_domain_threads"] = self.reddit_domain_threads
        if self.reddit_domain_messages is not None:
            obj["reddit_domain_messages"] = self.reddit_domain_messages
        if self.metrics_domain is not None:
            obj["metrics_domain"] = self.metrics_domain
        return json.dumps(obj, ensure_ascii=False, indent=indent)


@dataclass(frozen=True)
class HavelockParams:
    enabled: bool = False
    base_url: str = "https://thestalwart-havelock-demo.hf.space"
    top_n_sections: int = 8
    sample_messages: int = 200
    max_chars: int = 8000
    include_sentences: bool = False
    seed: int = 1337
    include_domain: bool = False


def _fetch_thread_texts(
    conn: sqlite3.Connection, *, source: str, min_thread_messages: int
) -> list[ThreadMetrics]:
    thread_rows = conn.execute(
        """
        SELECT t.id AS thread_id, t.source AS source, COUNT(m.id) AS message_count
        FROM threads t
        JOIN messages m ON m.thread_id = t.id
        WHERE t.source = ?
        GROUP BY t.id, t.source
        HAVING COUNT(m.id) >= ?
        """,
        (source, min_thread_messages),
    ).fetchall()

    out: list[ThreadMetrics] = []
    for row in thread_rows:
        thread_id = str(row["thread_id"])
        messages = conn.execute(
            "SELECT body FROM messages WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
        text = "\n".join((m["body"] or "") for m in messages)
        cleaned = clean_text(text)
        out.append(
            ThreadMetrics(
                thread_id=thread_id,
                source=source,
                message_count=int(row["message_count"]),
                char_len=len(cleaned),
                gzip_bpc=gzip_bits_per_char(cleaned),
                shannon_bpc=shannon_entropy_bits_per_char(cleaned),
            )
        )
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _bootstrap_ci_diff(
    a: list[float], b: list[float], *, iters: int = 1000, seed: int = 1337
) -> dict:
    rng = random.Random(seed)
    if not a or not b:
        return {"error": "empty-sample"}

    obs = _mean(a) - _mean(b)
    boot: list[float] = []
    for _ in range(iters):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        boot.append(_mean(sa) - _mean(sb))
    boot.sort()
    lo = boot[int(0.025 * iters)]
    hi = boot[int(0.975 * iters)]
    return {"observed_diff": obs, "ci95": [lo, hi]}


def _permutation_p_value_diff(
    a: list[float], b: list[float], *, iters: int = 1000, seed: int = 1337
) -> float | None:
    rng = random.Random(seed)
    if not a or not b:
        return None

    obs = _mean(a) - _mean(b)
    combined = list(a) + list(b)
    n = len(a)
    extreme = 0
    for _ in range(iters):
        rng.shuffle(combined)
        diff = _mean(combined[:n]) - _mean(combined[n:])
        if abs(diff) >= abs(obs):
            extreme += 1
    return (extreme + 1) / (iters + 1)


def _cliffs_delta(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    b_sorted = sorted(b)
    m = len(b_sorted)
    wins = 0
    losses = 0
    for x in a:
        wins += bisect_left(b_sorted, x)
        losses += m - bisect_right(b_sorted, x)
    return (wins - losses) / (len(a) * m)


def _sample_length_matched(
    molt_texts: list[str],
    reddit_texts: list[str],
    *,
    bin_size: int = 50,
    seed: int = 1337,
    min_len: int = 40,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)

    def bins(texts: list[str]) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for t in texts:
            ct = clean_text(t)
            if len(ct) < min_len:
                continue
            b = len(ct) // bin_size
            out.setdefault(b, []).append(ct)
        return out

    molt_bins = bins(molt_texts)
    red_bins = bins(reddit_texts)
    common = sorted(set(molt_bins) & set(red_bins))

    molt_out: list[str] = []
    red_out: list[str] = []
    for b in common:
        n = min(len(molt_bins[b]), len(red_bins[b]))
        if n <= 0:
            continue
        molt_out.extend(rng.sample(molt_bins[b], k=n))
        red_out.extend(rng.sample(red_bins[b], k=n))
    return molt_out, red_out


def _sample_length_matched_with_thread_ids(
    molt_rows: list[tuple[str, str]],
    reddit_rows: list[tuple[str, str]],
    *,
    bin_size: int = 50,
    seed: int = 1337,
    min_len: int = 40,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)

    def bins(rows: list[tuple[str, str]]) -> dict[int, list[tuple[str, str]]]:
        out: dict[int, list[tuple[str, str]]] = {}
        for tid, body in rows:
            ct = clean_text(body)
            if len(ct) < min_len:
                continue
            b = len(ct) // bin_size
            out.setdefault(b, []).append((tid, ct))
        return out

    molt_bins = bins(molt_rows)
    red_bins = bins(reddit_rows)
    common = sorted(set(molt_bins) & set(red_bins))

    molt_out: list[tuple[str, str]] = []
    red_out: list[tuple[str, str]] = []
    for b in common:
        n = min(len(molt_bins[b]), len(red_bins[b]))
        if n <= 0:
            continue
        molt_out.extend(rng.sample(molt_bins[b], k=n))
        red_out.extend(rng.sample(red_bins[b], k=n))
    return molt_out, red_out


def _sample_thread_metrics_length_matched(
    molt: list[ThreadMetrics],
    red: list[ThreadMetrics],
    *,
    bin_size: int = 500,
    seed: int = 1337,
    min_len: int = 500,
) -> tuple[list[ThreadMetrics], list[ThreadMetrics]]:
    rng = random.Random(seed)

    def bins(items: list[ThreadMetrics]) -> dict[int, list[ThreadMetrics]]:
        out: dict[int, list[ThreadMetrics]] = {}
        for tm in items:
            if tm.char_len < min_len:
                continue
            b = tm.char_len // bin_size
            out.setdefault(b, []).append(tm)
        return out

    m_bins = bins(molt)
    r_bins = bins(red)
    common = sorted(set(m_bins) & set(r_bins))

    m_out: list[ThreadMetrics] = []
    r_out: list[ThreadMetrics] = []
    for b in common:
        n = min(len(m_bins[b]), len(r_bins[b]))
        if n <= 0:
            continue
        m_out.extend(rng.sample(m_bins[b], k=n))
        r_out.extend(rng.sample(r_bins[b], k=n))
    return m_out, r_out


def _build_thread_texts(
    conn: sqlite3.Connection, *, source: str, skip_bot_authors: bool = False
) -> dict[str, str]:
    if skip_bot_authors:
        rows = conn.execute(
            "SELECT thread_id, body, author FROM messages WHERE source = ? ORDER BY thread_id",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT thread_id, body FROM messages WHERE source = ? ORDER BY thread_id",
            (source,),
        ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        if skip_bot_authors and _is_probably_bot_author(r["author"]):
            continue
        tid = str(r["thread_id"])
        out.setdefault(tid, []).append(str(r["body"] or ""))
    return {tid: "\n".join(parts) for tid, parts in out.items()}


def _top_counts(counts: dict[object, int], *, top_n: int) -> list[dict]:
    total = sum(counts.values())
    items = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))[:top_n]
    out = []
    for k, v in items:
        key = "|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)
        out.append({"item": key, "count": v, "share": (v / total) if total else None})
    return out


def _top_duplicate_keys(
    counts: dict[str, int],
    threads: dict[str, set[str]],
    previews: dict[str, str],
    *,
    top_n: int,
    max_preview_chars: int = 140,
) -> list[dict]:
    total = sum(counts.values())
    items = [(k, c) for k, c in counts.items() if c >= 2]
    items.sort(key=lambda x: (-x[1], x[0]))

    out = []
    for key, count in items[:top_n]:
        text = previews.get(key, key)
        preview = text if len(text) <= max_preview_chars else f"{text[: max_preview_chars - 1]}…"
        out.append(
            {
                "key": key,
                "preview": preview,
                "count": count,
                "share": (count / total) if total else None,
                "distinct_threads": len(threads.get(key, set())),
            }
        )
    return out


def _duplicate_breakdown(
    rows: list[tuple[str, str]],
    *,
    key_fn: Callable[[str], str | None],
    top_n: int = 20,
) -> dict:
    counts: dict[str, int] = {}
    threads: dict[str, set[str]] = {}
    previews: dict[str, str] = {}
    for tid, text in rows:
        key = key_fn(text)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        threads.setdefault(key, set()).add(tid)
        previews.setdefault(key, text)

    total = sum(counts.values())
    if total == 0:
        return {"total": 0}

    dup_message_share = sum(c for c in counts.values() if c >= 2) / total
    cross_thread_message_share = (
        sum(c for k, c in counts.items() if len(threads.get(k, set())) >= 2) / total
    )
    within_thread_message_share = (
        sum(c for k, c in counts.items() if c >= 2 and len(threads.get(k, set())) == 1) / total
    )

    return {
        "total": total,
        "unique": len(counts),
        "duplicate_message_share": dup_message_share,
        "cross_thread_duplicate_message_share": cross_thread_message_share,
        "within_thread_duplicate_message_share": within_thread_message_share,
        "top_duplicates": _top_duplicate_keys(counts, threads, previews, top_n=top_n),
    }


def _compute_message_metrics(
    *,
    molt_rows: list[tuple[str, str]],
    red_rows: list[tuple[str, str]],
) -> dict:
    molt_msg_rows, red_msg_rows = _sample_length_matched_with_thread_ids(molt_rows, red_rows)
    molt_msg_sample = [t for _, t in molt_msg_rows]
    red_msg_sample = [t for _, t in red_msg_rows]

    molt_msg_gzip = [gzip_bits_per_char(t) for t in molt_msg_sample]
    red_msg_gzip = [gzip_bits_per_char(t) for t in red_msg_sample]
    molt_msg_shan = [shannon_entropy_bits_per_char(t) for t in molt_msg_sample]
    red_msg_shan = [shannon_entropy_bits_per_char(t) for t in red_msg_sample]
    molt_msg_ttr = [type_token_ratio(t) for t in molt_msg_sample]
    red_msg_ttr = [type_token_ratio(t) for t in red_msg_sample]

    molt_msg_dupe = exact_duplicate_rate(molt_msg_sample)
    red_msg_dupe = exact_duplicate_rate(red_msg_sample)
    molt_msg_jacc = sampled_pairwise_jaccard_similarity(molt_msg_sample, pairs=2000, max_tokens=80)
    red_msg_jacc = sampled_pairwise_jaccard_similarity(red_msg_sample, pairs=2000, max_tokens=80)

    def soft_key(text: str) -> str | None:
        return soft_duplicate_key(text, max_tokens=80)

    molt_msg_soft_keys = [k for k in (soft_key(t) for t in molt_msg_sample) if k]
    red_msg_soft_keys = [k for k in (soft_key(t) for t in red_msg_sample) if k]

    molt_msg_soft_dupe = exact_duplicate_rate(molt_msg_soft_keys)
    red_msg_soft_dupe = exact_duplicate_rate(red_msg_soft_keys)

    molt_dup_breakdown = _duplicate_breakdown(molt_msg_rows, key_fn=lambda t: t)
    red_dup_breakdown = _duplicate_breakdown(red_msg_rows, key_fn=lambda t: t)
    molt_soft_breakdown = _duplicate_breakdown(molt_msg_rows, key_fn=soft_key)
    red_soft_breakdown = _duplicate_breakdown(red_msg_rows, key_fn=soft_key)

    molt_dist1 = distinct_n(molt_msg_sample, n=1, max_tokens=80)
    red_dist1 = distinct_n(red_msg_sample, n=1, max_tokens=80)
    molt_dist2 = distinct_n(molt_msg_sample, n=2, max_tokens=80)
    red_dist2 = distinct_n(red_msg_sample, n=2, max_tokens=80)

    molt_uni = ngram_counts(molt_msg_sample, n=1, max_tokens=80)
    red_uni = ngram_counts(red_msg_sample, n=1, max_tokens=80)
    molt_uni_ent = shannon_entropy_bits_from_counts(molt_uni)
    red_uni_ent = shannon_entropy_bits_from_counts(red_uni)
    molt_uni_top = {str(k): top_k_coverage(molt_uni, k=k) for k in (50, 100, 500)}
    red_uni_top = {str(k): top_k_coverage(red_uni, k=k) for k in (50, 100, 500)}
    molt_uni_top_items = _top_counts(molt_uni, top_n=30)
    red_uni_top_items = _top_counts(red_uni, top_n=30)

    molt_sig = topic_signatures(molt_msg_sample, sig_k=3, max_tokens=120, min_tokens=6)
    red_sig = topic_signatures(red_msg_sample, sig_k=3, max_tokens=120, min_tokens=6)
    molt_sig_counts: dict[object, int] = {}
    red_sig_counts: dict[object, int] = {}
    for s in molt_sig:
        molt_sig_counts[s] = molt_sig_counts.get(s, 0) + 1
    for s in red_sig:
        red_sig_counts[s] = red_sig_counts.get(s, 0) + 1

    coverage_ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

    def coverage_curve(counts: dict[object, int]) -> dict:
        top_counts = sorted(counts.values(), reverse=True)
        total = sum(top_counts)
        if total <= 0:
            return {"k_values": coverage_ks, "points": [], "k_at_coverage": {}}

        points = []
        for k in coverage_ks:
            points.append({"k": k, "coverage": sum(top_counts[:k]) / total})

        k_at: dict[str, int] = {}
        cum = 0
        for i, c in enumerate(top_counts, start=1):
            cum += c
            cov = cum / total
            if "50" not in k_at and cov >= 0.50:
                k_at["50"] = i
            if "80" not in k_at and cov >= 0.80:
                k_at["80"] = i
            if "90" not in k_at and cov >= 0.90:
                k_at["90"] = i
            if len(k_at) == 3:
                break

        return {"k_values": coverage_ks, "points": points, "k_at_coverage": k_at}

    molt_sig_curve = coverage_curve(molt_sig_counts)
    red_sig_curve = coverage_curve(red_sig_counts)

    molt_sig_top_items = _top_counts(molt_sig_counts, top_n=20)
    red_sig_top_items = _top_counts(red_sig_counts, top_n=20)
    molt_sig_stats = distribution_entropy_and_coverage(molt_sig, top_ks=[10, 20, 50, 100])
    red_sig_stats = distribution_entropy_and_coverage(red_sig, top_ks=[10, 20, 50, 100])

    return {
        "message_sample": {"size": len(molt_msg_sample), "bin_size": 50, "min_len": 40},
        "message_gzip_bits_per_char": {
            "moltbook_mean": _mean(molt_msg_gzip) if molt_msg_gzip else None,
            "reddit_mean": _mean(red_msg_gzip) if red_msg_gzip else None,
            "bootstrap_ci": _bootstrap_ci_diff(molt_msg_gzip, red_msg_gzip),
            "p_perm": _permutation_p_value_diff(molt_msg_gzip, red_msg_gzip),
            "cliffs_delta": _cliffs_delta(molt_msg_gzip, red_msg_gzip),
        },
        "message_shannon_bits_per_char": {
            "moltbook_mean": _mean(molt_msg_shan) if molt_msg_shan else None,
            "reddit_mean": _mean(red_msg_shan) if red_msg_shan else None,
            "bootstrap_ci": _bootstrap_ci_diff(molt_msg_shan, red_msg_shan),
            "p_perm": _permutation_p_value_diff(molt_msg_shan, red_msg_shan),
            "cliffs_delta": _cliffs_delta(molt_msg_shan, red_msg_shan),
        },
        "message_type_token_ratio_50w": {
            "moltbook_mean": _mean(molt_msg_ttr) if molt_msg_ttr else None,
            "reddit_mean": _mean(red_msg_ttr) if red_msg_ttr else None,
            "bootstrap_ci": _bootstrap_ci_diff(molt_msg_ttr, red_msg_ttr),
            "p_perm": _permutation_p_value_diff(molt_msg_ttr, red_msg_ttr),
            "cliffs_delta": _cliffs_delta(molt_msg_ttr, red_msg_ttr),
        },
        "message_exact_duplicate_rate": {"moltbook": molt_msg_dupe, "reddit": red_msg_dupe},
        "message_soft_duplicate_rate": {
            "moltbook": molt_msg_soft_dupe,
            "reddit": red_msg_soft_dupe,
        },
        "message_duplicate_breakdown": {
            "moltbook": molt_dup_breakdown,
            "reddit": red_dup_breakdown,
        },
        "message_soft_duplicate_breakdown": {
            "moltbook": molt_soft_breakdown,
            "reddit": red_soft_breakdown,
        },
        "message_pairwise_jaccard_similarity": {
            "pairs": 2000,
            "max_tokens": 80,
            "moltbook": molt_msg_jacc,
            "reddit": red_msg_jacc,
        },
        "message_distinct_n": {
            "max_tokens": 80,
            "distinct_1": {"moltbook": molt_dist1, "reddit": red_dist1},
            "distinct_2": {"moltbook": molt_dist2, "reddit": red_dist2},
        },
        "message_unigram_entropy_bits": {"moltbook": molt_uni_ent, "reddit": red_uni_ent},
        "message_unigram_topk_coverage": {"moltbook": molt_uni_top, "reddit": red_uni_top},
        "message_top_unigrams": {"moltbook": molt_uni_top_items, "reddit": red_uni_top_items},
        "message_topic_signatures": {
            "sig_k": 3,
            "max_tokens": 120,
            "min_tokens": 6,
            "moltbook": molt_sig_stats,
            "reddit": red_sig_stats,
        },
        "message_topic_signature_coverage_curve": {
            "k_values": molt_sig_curve["k_values"],
            "moltbook": molt_sig_curve,
            "reddit": red_sig_curve,
        },
        "message_top_topic_signatures": {
            "moltbook": molt_sig_top_items,
            "reddit": red_sig_top_items,
        },
    }


def _is_probably_bot_author(author: str | None) -> bool:
    if not author:
        return False
    a = author.lower()
    if a == "automoderator" or "automod" in a:
        return True
    return a.endswith("bot") or a.endswith("_bot")


def run_analysis(
    *,
    db_path: Path,
    min_thread_messages: int = 5,
    reddit_source: str = "reddit",
    reddit_domain_source: str = "reddit_domain",
    havelock: HavelockParams | None = None,
) -> AnalysisReport:
    db = init_db(db_path)
    with db.connect() as conn:
        molt = _fetch_thread_texts(conn, source="moltbook", min_thread_messages=min_thread_messages)
        red = _fetch_thread_texts(
            conn, source=reddit_source, min_thread_messages=min_thread_messages
        )

        molt_gzip = [m.gzip_bpc for m in molt]
        red_gzip = [m.gzip_bpc for m in red]
        molt_shan = [m.shannon_bpc for m in molt]
        red_shan = [m.shannon_bpc for m in red]

        molt_m, red_m = _sample_thread_metrics_length_matched(molt, red)
        molt_gzip_m = [m.gzip_bpc for m in molt_m]
        red_gzip_m = [m.gzip_bpc for m in red_m]
        molt_shan_m = [m.shannon_bpc for m in molt_m]
        red_shan_m = [m.shannon_bpc for m in red_m]

        molt_msgs = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE source='moltbook'"
        ).fetchone()["c"]
        red_msgs = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE source=?",
            (reddit_source,),
        ).fetchone()["c"]

        molt_rows = [
            (str(r["thread_id"]), str(r["body"]))
            for r in conn.execute(
                "SELECT thread_id, body FROM messages WHERE source='moltbook'"
            ).fetchall()
            if r["body"] is not None
        ]
        red_rows = [
            (str(r["thread_id"]), str(r["body"]))
            for r in conn.execute(
                "SELECT thread_id, body, author FROM messages WHERE source=?",
                (reddit_source,),
            ).fetchall()
            if r["body"] is not None and not _is_probably_bot_author(r["author"])
        ]

        molt_msg_rows, red_msg_rows = _sample_length_matched_with_thread_ids(molt_rows, red_rows)
        molt_msg_sample = [t for _, t in molt_msg_rows]
        red_msg_sample = [t for _, t in red_msg_rows]
        molt_msg_gzip = [gzip_bits_per_char(t) for t in molt_msg_sample]
        red_msg_gzip = [gzip_bits_per_char(t) for t in red_msg_sample]
        molt_msg_shan = [shannon_entropy_bits_per_char(t) for t in molt_msg_sample]
        red_msg_shan = [shannon_entropy_bits_per_char(t) for t in red_msg_sample]
        molt_msg_ttr = [type_token_ratio(t) for t in molt_msg_sample]
        red_msg_ttr = [type_token_ratio(t) for t in red_msg_sample]

        molt_msg_dupe = exact_duplicate_rate(molt_msg_sample)
        red_msg_dupe = exact_duplicate_rate(red_msg_sample)
        molt_msg_jacc = sampled_pairwise_jaccard_similarity(
            molt_msg_sample, pairs=2000, max_tokens=80
        )
        red_msg_jacc = sampled_pairwise_jaccard_similarity(
            red_msg_sample, pairs=2000, max_tokens=80
        )

        def soft_key(text: str) -> str | None:
            return soft_duplicate_key(text, max_tokens=80)

        molt_msg_soft_keys = [k for k in (soft_key(t) for t in molt_msg_sample) if k]
        red_msg_soft_keys = [k for k in (soft_key(t) for t in red_msg_sample) if k]

        molt_msg_soft_dupe = exact_duplicate_rate(molt_msg_soft_keys)
        red_msg_soft_dupe = exact_duplicate_rate(red_msg_soft_keys)

        molt_dup_breakdown = _duplicate_breakdown(molt_msg_rows, key_fn=lambda t: t)
        red_dup_breakdown = _duplicate_breakdown(red_msg_rows, key_fn=lambda t: t)
        molt_soft_breakdown = _duplicate_breakdown(molt_msg_rows, key_fn=soft_key)
        red_soft_breakdown = _duplicate_breakdown(red_msg_rows, key_fn=soft_key)
        molt_dist1 = distinct_n(molt_msg_sample, n=1, max_tokens=80)
        red_dist1 = distinct_n(red_msg_sample, n=1, max_tokens=80)
        molt_dist2 = distinct_n(molt_msg_sample, n=2, max_tokens=80)
        red_dist2 = distinct_n(red_msg_sample, n=2, max_tokens=80)

        molt_uni = ngram_counts(molt_msg_sample, n=1, max_tokens=80)
        red_uni = ngram_counts(red_msg_sample, n=1, max_tokens=80)
        molt_uni_ent = shannon_entropy_bits_from_counts(molt_uni)
        red_uni_ent = shannon_entropy_bits_from_counts(red_uni)
        molt_uni_top = {str(k): top_k_coverage(molt_uni, k=k) for k in (50, 100, 500)}
        red_uni_top = {str(k): top_k_coverage(red_uni, k=k) for k in (50, 100, 500)}
        molt_uni_top_items = _top_counts(molt_uni, top_n=30)
        red_uni_top_items = _top_counts(red_uni, top_n=30)

        molt_sig = topic_signatures(molt_msg_sample, sig_k=3, max_tokens=120, min_tokens=6)
        red_sig = topic_signatures(red_msg_sample, sig_k=3, max_tokens=120, min_tokens=6)
        molt_sig_counts: dict[object, int] = {}
        red_sig_counts: dict[object, int] = {}
        for s in molt_sig:
            molt_sig_counts[s] = molt_sig_counts.get(s, 0) + 1
        for s in red_sig:
            red_sig_counts[s] = red_sig_counts.get(s, 0) + 1
        molt_sig_top_items = _top_counts(molt_sig_counts, top_n=20)
        red_sig_top_items = _top_counts(red_sig_counts, top_n=20)
        molt_sig_stats = distribution_entropy_and_coverage(molt_sig, top_ks=[10, 20, 50, 100])
        red_sig_stats = distribution_entropy_and_coverage(red_sig, top_ks=[10, 20, 50, 100])

        coverage_ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

        def coverage_curve(counts: dict[object, int]) -> dict:
            top_counts = sorted(counts.values(), reverse=True)
            total = sum(top_counts)
            if total <= 0:
                return {"k_values": coverage_ks, "points": [], "k_at_coverage": {}}

            points = []
            for k in coverage_ks:
                covered = sum(top_counts[:k])
                points.append({"k": k, "coverage": covered / total})

            k_at: dict[str, int] = {}
            cum = 0
            for i, c in enumerate(top_counts, start=1):
                cum += c
                cov = cum / total
                if "50" not in k_at and cov >= 0.50:
                    k_at["50"] = i
                if "80" not in k_at and cov >= 0.80:
                    k_at["80"] = i
                if "90" not in k_at and cov >= 0.90:
                    k_at["90"] = i
                if len(k_at) == 3:
                    break

            return {"k_values": coverage_ks, "points": points, "k_at_coverage": k_at}

        molt_sig_curve = coverage_curve(molt_sig_counts)
        red_sig_curve = coverage_curve(red_sig_counts)

        molt_thread_texts = _build_thread_texts(conn, source="moltbook")
        red_thread_texts = _build_thread_texts(conn, source=reddit_source, skip_bot_authors=True)
        molt_thread_docs = [
            molt_thread_texts[m.thread_id] for m in molt_m if m.thread_id in molt_thread_texts
        ]
        red_thread_docs = [
            red_thread_texts[m.thread_id] for m in red_m if m.thread_id in red_thread_texts
        ]
        molt_thread_terms = topic_signatures(
            molt_thread_docs, sig_k=10, max_tokens=300, min_tokens=20
        )
        red_thread_terms = topic_signatures(
            red_thread_docs, sig_k=10, max_tokens=300, min_tokens=20
        )
        molt_thread_sig = topic_signatures(molt_thread_docs, sig_k=3, max_tokens=200, min_tokens=10)
        red_thread_sig = topic_signatures(red_thread_docs, sig_k=3, max_tokens=200, min_tokens=10)
        molt_thread_sig_stats = distribution_entropy_and_coverage(
            molt_thread_sig, top_ks=[5, 10, 20, 50]
        )
        red_thread_sig_stats = distribution_entropy_and_coverage(
            red_thread_sig, top_ks=[5, 10, 20, 50]
        )
        molt_thread_sig_jacc = sampled_pairwise_jaccard_similarity_sets(
            [set(s) for s in molt_thread_sig], pairs=2000
        )
        red_thread_sig_jacc = sampled_pairwise_jaccard_similarity_sets(
            [set(s) for s in red_thread_sig], pairs=2000
        )
        molt_thread_terms_jacc = sampled_pairwise_jaccard_similarity_sets(
            [set(s) for s in molt_thread_terms], pairs=2000
        )
        red_thread_terms_jacc = sampled_pairwise_jaccard_similarity_sets(
            [set(s) for s in red_thread_terms], pairs=2000
        )

        metrics = {
            "thread_gzip_bits_per_char": {
                "moltbook_mean": _mean(molt_gzip) if molt_gzip else None,
                "reddit_mean": _mean(red_gzip) if red_gzip else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_gzip, red_gzip),
                "p_perm": _permutation_p_value_diff(molt_gzip, red_gzip),
                "cliffs_delta": _cliffs_delta(molt_gzip, red_gzip),
            },
            "thread_shannon_bits_per_char": {
                "moltbook_mean": _mean(molt_shan) if molt_shan else None,
                "reddit_mean": _mean(red_shan) if red_shan else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_shan, red_shan),
                "p_perm": _permutation_p_value_diff(molt_shan, red_shan),
                "cliffs_delta": _cliffs_delta(molt_shan, red_shan),
            },
            "thread_matched_sample": {"size": len(molt_m), "bin_size": 500, "min_len": 500},
            "thread_matched_gzip_bits_per_char": {
                "moltbook_mean": _mean(molt_gzip_m) if molt_gzip_m else None,
                "reddit_mean": _mean(red_gzip_m) if red_gzip_m else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_gzip_m, red_gzip_m),
                "p_perm": _permutation_p_value_diff(molt_gzip_m, red_gzip_m),
                "cliffs_delta": _cliffs_delta(molt_gzip_m, red_gzip_m),
            },
            "thread_matched_shannon_bits_per_char": {
                "moltbook_mean": _mean(molt_shan_m) if molt_shan_m else None,
                "reddit_mean": _mean(red_shan_m) if red_shan_m else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_shan_m, red_shan_m),
                "p_perm": _permutation_p_value_diff(molt_shan_m, red_shan_m),
                "cliffs_delta": _cliffs_delta(molt_shan_m, red_shan_m),
            },
            "message_sample": {"size": len(molt_msg_sample), "bin_size": 50, "min_len": 40},
            "message_gzip_bits_per_char": {
                "moltbook_mean": _mean(molt_msg_gzip) if molt_msg_gzip else None,
                "reddit_mean": _mean(red_msg_gzip) if red_msg_gzip else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_msg_gzip, red_msg_gzip),
                "p_perm": _permutation_p_value_diff(molt_msg_gzip, red_msg_gzip),
                "cliffs_delta": _cliffs_delta(molt_msg_gzip, red_msg_gzip),
            },
            "message_shannon_bits_per_char": {
                "moltbook_mean": _mean(molt_msg_shan) if molt_msg_shan else None,
                "reddit_mean": _mean(red_msg_shan) if red_msg_shan else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_msg_shan, red_msg_shan),
                "p_perm": _permutation_p_value_diff(molt_msg_shan, red_msg_shan),
                "cliffs_delta": _cliffs_delta(molt_msg_shan, red_msg_shan),
            },
            "message_type_token_ratio_50w": {
                "moltbook_mean": _mean(molt_msg_ttr) if molt_msg_ttr else None,
                "reddit_mean": _mean(red_msg_ttr) if red_msg_ttr else None,
                "bootstrap_ci": _bootstrap_ci_diff(molt_msg_ttr, red_msg_ttr),
                "p_perm": _permutation_p_value_diff(molt_msg_ttr, red_msg_ttr),
                "cliffs_delta": _cliffs_delta(molt_msg_ttr, red_msg_ttr),
            },
            "message_exact_duplicate_rate": {"moltbook": molt_msg_dupe, "reddit": red_msg_dupe},
            "message_soft_duplicate_rate": {
                "moltbook": molt_msg_soft_dupe,
                "reddit": red_msg_soft_dupe,
            },
            "message_duplicate_breakdown": {
                "moltbook": molt_dup_breakdown,
                "reddit": red_dup_breakdown,
            },
            "message_soft_duplicate_breakdown": {
                "moltbook": molt_soft_breakdown,
                "reddit": red_soft_breakdown,
            },
            "message_pairwise_jaccard_similarity": {
                "pairs": 2000,
                "max_tokens": 80,
                "moltbook": molt_msg_jacc,
                "reddit": red_msg_jacc,
            },
            "message_distinct_n": {
                "max_tokens": 80,
                "distinct_1": {"moltbook": molt_dist1, "reddit": red_dist1},
                "distinct_2": {"moltbook": molt_dist2, "reddit": red_dist2},
            },
            "message_unigram_entropy_bits": {"moltbook": molt_uni_ent, "reddit": red_uni_ent},
            "message_unigram_topk_coverage": {"moltbook": molt_uni_top, "reddit": red_uni_top},
            "message_top_unigrams": {"moltbook": molt_uni_top_items, "reddit": red_uni_top_items},
            "message_topic_signatures": {
                "sig_k": 3,
                "max_tokens": 120,
                "min_tokens": 6,
                "moltbook": molt_sig_stats,
                "reddit": red_sig_stats,
            },
            "message_topic_signature_coverage_curve": {
                "k_values": molt_sig_curve["k_values"],
                "moltbook": molt_sig_curve,
                "reddit": red_sig_curve,
            },
            "message_top_topic_signatures": {
                "moltbook": molt_sig_top_items,
                "reddit": red_sig_top_items,
            },
            "thread_matched_topic_signatures": {
                "sig_k": 3,
                "max_tokens": 200,
                "min_tokens": 10,
                "moltbook": molt_thread_sig_stats,
                "reddit": red_thread_sig_stats,
            },
            "thread_signature_pairwise_jaccard": {
                "pairs": 2000,
                "moltbook": molt_thread_sig_jacc,
                "reddit": red_thread_sig_jacc,
            },
            "thread_top_terms_pairwise_jaccard": {
                "sig_k": 10,
                "max_tokens": 300,
                "min_tokens": 20,
                "pairs": 2000,
                "moltbook": molt_thread_terms_jacc,
                "reddit": red_thread_terms_jacc,
            },
        }

        red_dom_msgs = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE source=?", (reddit_domain_source,)
        ).fetchone()["c"]

        metrics_domain: dict | None = None
        red_dom_threads: int | None = None
        if int(red_dom_msgs) > 0:
            red_dom = _fetch_thread_texts(
                conn, source=reddit_domain_source, min_thread_messages=min_thread_messages
            )
            red_dom_threads = len(red_dom)
            red_dom_rows = [
                (str(r["thread_id"]), str(r["body"]))
                for r in conn.execute(
                    "SELECT thread_id, body, author FROM messages WHERE source=?",
                    (reddit_domain_source,),
                ).fetchall()
                if r["body"] is not None and not _is_probably_bot_author(r["author"])
            ]

            metrics_domain = _compute_message_metrics(molt_rows=molt_rows, red_rows=red_dom_rows)

            molt_dm, red_dm = _sample_thread_metrics_length_matched(molt, red_dom)
            molt_dom_thread_texts = _build_thread_texts(conn, source="moltbook")
            red_dom_thread_texts = _build_thread_texts(
                conn, source=reddit_domain_source, skip_bot_authors=True
            )
            molt_dom_docs = [
                molt_dom_thread_texts[m.thread_id]
                for m in molt_dm
                if m.thread_id in molt_dom_thread_texts
            ]
            red_dom_docs = [
                red_dom_thread_texts[m.thread_id]
                for m in red_dm
                if m.thread_id in red_dom_thread_texts
            ]
            molt_dom_sig = topic_signatures(molt_dom_docs, sig_k=3, max_tokens=200, min_tokens=10)
            red_dom_sig = topic_signatures(red_dom_docs, sig_k=3, max_tokens=200, min_tokens=10)
            molt_dom_sig_stats = distribution_entropy_and_coverage(
                molt_dom_sig, top_ks=[5, 10, 20, 50]
            )
            red_dom_sig_stats = distribution_entropy_and_coverage(
                red_dom_sig, top_ks=[5, 10, 20, 50]
            )

            molt_dom_terms = topic_signatures(
                molt_dom_docs, sig_k=10, max_tokens=300, min_tokens=20
            )
            red_dom_terms = topic_signatures(red_dom_docs, sig_k=10, max_tokens=300, min_tokens=20)
            molt_dom_terms_jacc = sampled_pairwise_jaccard_similarity_sets(
                [set(s) for s in molt_dom_terms], pairs=2000
            )
            red_dom_terms_jacc = sampled_pairwise_jaccard_similarity_sets(
                [set(s) for s in red_dom_terms], pairs=2000
            )

            metrics_domain.update(
                {
                    "thread_matched_sample": {
                        "size": len(molt_dm),
                        "bin_size": 500,
                        "min_len": 500,
                    },
                    "thread_matched_topic_signatures": {
                        "sig_k": 3,
                        "max_tokens": 200,
                        "min_tokens": 10,
                        "moltbook": molt_dom_sig_stats,
                        "reddit": red_dom_sig_stats,
                    },
                    "thread_top_terms_pairwise_jaccard": {
                        "sig_k": 10,
                        "max_tokens": 300,
                        "min_tokens": 20,
                        "pairs": 2000,
                        "moltbook": molt_dom_terms_jacc,
                        "reddit": red_dom_terms_jacc,
                    },
                }
            )

        if havelock is not None and havelock.enabled:
            client = HavelockClient(base_url=havelock.base_url)
            molt_orality = havelock_orality_by_section(
                conn,
                client=client,
                source="moltbook",
                top_n_sections=havelock.top_n_sections,
                sample_messages=havelock.sample_messages,
                max_chars=havelock.max_chars,
                include_sentences=havelock.include_sentences,
                seed=havelock.seed,
            )
            red_orality = havelock_orality_by_section(
                conn,
                client=client,
                source=reddit_source,
                top_n_sections=havelock.top_n_sections,
                sample_messages=havelock.sample_messages,
                max_chars=havelock.max_chars,
                include_sentences=havelock.include_sentences,
                seed=havelock.seed,
            )
            metrics["havelock_orality_literacy"] = {
                "provider": "havelock.ai",
                "base_url": havelock.base_url,
                "moltbook": molt_orality,
                "reddit": red_orality,
            }

            if metrics_domain is not None and havelock.include_domain:
                red_dom_orality = havelock_orality_by_section(
                    conn,
                    client=client,
                    source=reddit_domain_source,
                    top_n_sections=havelock.top_n_sections,
                    sample_messages=havelock.sample_messages,
                    max_chars=havelock.max_chars,
                    include_sentences=havelock.include_sentences,
                    seed=havelock.seed,
                )
                metrics_domain["havelock_orality_literacy"] = {
                    "provider": "havelock.ai",
                    "base_url": havelock.base_url,
                    "moltbook": molt_orality,
                    "reddit": red_dom_orality,
                }

        return AnalysisReport(
            moltbook_threads=len(molt),
            reddit_threads=len(red),
            moltbook_messages=int(molt_msgs),
            reddit_messages=int(red_msgs),
            metrics=metrics,
            sources={
                "moltbook": "moltbook",
                "reddit": reddit_source,
                "reddit_domain": reddit_domain_source,
            },
            reddit_domain_threads=red_dom_threads,
            reddit_domain_messages=int(red_dom_msgs) if int(red_dom_msgs) > 0 else None,
            metrics_domain=metrics_domain,
        )
