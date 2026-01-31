from __future__ import annotations

import math
import random
import sqlite3
import zlib
from typing import Any

from .havelock_client import HavelockClient
from .metrics import clean_text


def _stable_seed(seed: int, salt: str) -> int:
    return (seed ^ zlib.adler32(salt.encode("utf-8", errors="ignore"))) & 0xFFFFFFFF


def _trim_havelock_result(result: dict[str, Any], *, max_sentences: int = 12) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "score",
        "doc_score",
        "sentence_ratio",
        "sentence_ratio_binary",
        "oral_count",
        "literate_count",
        "oral_weighted",
        "literate_weighted",
        "word_count",
        "interpretation",
        "label",
        "explanation",
        "example",
    ):
        if key in result:
            out[key] = result[key]

    sentences = result.get("sentences")
    if isinstance(sentences, list) and sentences:
        trimmed: list[dict[str, Any]] = []
        for s in sentences[:max_sentences]:
            if not isinstance(s, dict):
                continue
            t: dict[str, Any] = {}
            for k in (
                "text",
                "category",
                "category_confidence",
                "marker",
                "confidence",
            ):
                if k in s:
                    t[k] = s[k]
            markers = s.get("markers")
            if isinstance(markers, list) and markers:
                m2: list[dict[str, Any]] = []
                for m in markers[:3]:
                    if isinstance(m, dict):
                        m2.append({"marker": m.get("marker"), "confidence": m.get("confidence")})
                if m2:
                    t["markers"] = m2
            if t:
                trimmed.append(t)
        if trimmed:
            out["sentences"] = trimmed

    return out


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_safe_float((r.get("havelock") or {}).get("score")) for r in rows]
    pairs = [
        (s, _safe_float(r.get("message_count")))
        for s, r in zip(scores, rows, strict=False)
        if s is not None
    ]
    score_vals = [s for s in scores if s is not None]
    if not score_vals:
        return {"n": 0}

    score_vals_sorted = sorted(score_vals)
    mid = len(score_vals_sorted) // 2
    median = (
        score_vals_sorted[mid]
        if len(score_vals_sorted) % 2
        else (score_vals_sorted[mid - 1] + score_vals_sorted[mid]) / 2
    )

    weighted_num = 0.0
    weighted_den = 0.0
    for s, w in pairs:
        if w is None or w <= 0:
            continue
        weighted_num += s * w
        weighted_den += w

    return {
        "n": len(score_vals),
        "mean_score": sum(score_vals) / len(score_vals),
        "median_score": median,
        "min_score": min(score_vals),
        "max_score": max(score_vals),
        "weighted_mean_score": (weighted_num / weighted_den) if weighted_den > 0 else None,
    }


def _collect_sample_text(
    bodies: list[str],
    *,
    seed: int,
    sample_messages: int,
    max_chars: int,
) -> tuple[str, int]:
    cleaned = [clean_text(b) for b in bodies]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return "", 0

    rng = random.Random(seed)
    k = min(len(cleaned), max(1, sample_messages))
    sampled = rng.sample(cleaned, k=k)
    parts: list[str] = []
    total = 0
    for s in sampled:
        if not s:
            continue
        add = len(s) + (1 if parts else 0)
        if total + add > max_chars:
            break
        parts.append(s)
        total += add
    return "\n".join(parts), len(parts)


def _section_counts(
    conn: sqlite3.Connection, *, source: str, top_n_sections: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          t.subcommunity AS section,
          COUNT(m.id) AS message_count,
          COUNT(DISTINCT t.id) AS thread_count
        FROM messages m
        JOIN threads t ON t.id = m.thread_id
        WHERE m.source = ?
        GROUP BY t.subcommunity
        ORDER BY message_count DESC
        LIMIT ?
        """,
        (source, top_n_sections),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        section = r["section"]
        if section is None:
            continue
        out.append(
            {
                "section": str(section),
                "message_count": int(r["message_count"]),
                "thread_count": int(r["thread_count"]),
            }
        )
    return out


def _fetch_section_bodies(conn: sqlite3.Connection, *, source: str, section: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT m.body AS body
        FROM messages m
        JOIN threads t ON t.id = m.thread_id
        WHERE m.source = ? AND t.subcommunity = ?
        """,
        (source, section),
    ).fetchall()
    return [str(r["body"]) for r in rows if r["body"] is not None]


def _fetch_source_bodies(conn: sqlite3.Connection, *, source: str) -> list[str]:
    rows = conn.execute("SELECT body FROM messages WHERE source = ?", (source,)).fetchall()
    return [str(r["body"]) for r in rows if r["body"] is not None]


def havelock_orality_by_section(
    conn: sqlite3.Connection,
    *,
    client: HavelockClient,
    source: str,
    top_n_sections: int,
    sample_messages: int,
    max_chars: int,
    include_sentences: bool,
    seed: int,
) -> dict[str, Any]:
    sections = _section_counts(conn, source=source, top_n_sections=top_n_sections)
    out_rows: list[dict[str, Any]] = []
    for s in sections:
        section = str(s["section"])
        bodies = _fetch_section_bodies(conn, source=source, section=section)
        sample_text, sampled_messages = _collect_sample_text(
            bodies,
            seed=_stable_seed(seed, f"{source}:{section}"),
            sample_messages=sample_messages,
            max_chars=max_chars,
        )
        if not sample_text:
            continue
        result = client.analyze(text=sample_text, include_sentences=include_sentences)
        trimmed = _trim_havelock_result(result)
        out_rows.append(
            {
                "section": section,
                "thread_count": int(s["thread_count"]),
                "message_count": int(s["message_count"]),
                "sampled_messages": sampled_messages,
                "sample_char_len": len(sample_text),
                "havelock": trimmed,
            }
        )

    overall_text, overall_sampled_messages = _collect_sample_text(
        _fetch_source_bodies(conn, source=source),
        seed=_stable_seed(seed, f"{source}:__overall__"),
        sample_messages=max(1, sample_messages),
        max_chars=max_chars,
    )
    overall: dict[str, Any] | None = None
    if overall_text:
        overall = {
            "sampled_messages": overall_sampled_messages,
            "sample_char_len": len(overall_text),
            "havelock": _trim_havelock_result(
                client.analyze(text=overall_text, include_sentences=include_sentences),
                max_sentences=24,
            ),
        }

    return {
        "source": source,
        "params": {
            "top_n_sections": top_n_sections,
            "sample_messages": sample_messages,
            "max_chars": max_chars,
            "include_sentences": include_sentences,
            "seed": seed,
        },
        "overall": overall,
        "sections": out_rows,
        "summary": _summary_stats(out_rows),
    }
