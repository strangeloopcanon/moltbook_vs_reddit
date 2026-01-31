from __future__ import annotations

import datetime
import html
import json
import math
from pathlib import Path
from typing import Any


def render_html_report(report: dict[str, Any]) -> str:
    title = "Moltbook vs Reddit – Conversation Diversity Report"
    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    def get(path: list[str], default: Any = None) -> Any:
        cur: Any = report
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def esc(s: Any) -> str:
        return html.escape("" if s is None else str(s), quote=True)

    def pct(x: Any) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x) * 100:.2f}%"
        except (TypeError, ValueError):
            return "—"

    def num(x: Any, *, digits: int = 3) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    def log_x(k: int, *, min_k: int, max_k: int, width: float) -> float:
        if k <= 0 or max_k <= 0 or min_k <= 0:
            return 0.0
        if max_k == min_k:
            return 0.0
        lk = math.log10(k)
        lmin = math.log10(min_k)
        lmax = math.log10(max_k)
        return (lk - lmin) / (lmax - lmin) * width

    def coverage_curve_svg(
        molt_points: list[dict[str, Any]], reddit_points: list[dict[str, Any]]
    ) -> str:
        if not molt_points and not reddit_points:
            return '<p class="muted">No curve data.</p>'

        all_ks = [int(p["k"]) for p in molt_points + reddit_points if isinstance(p.get("k"), int)]
        if not all_ks:
            return '<p class="muted">No curve data.</p>'

        w = 620.0
        h = 180.0
        pad = 28.0
        min_k = max(1, min(all_ks))
        max_k = max(all_ks)

        def to_poly(points: list[dict[str, Any]]) -> str:
            out = []
            for p in points:
                k = p.get("k")
                cov = p.get("coverage")
                if not isinstance(k, int) or cov is None:
                    continue
                try:
                    yv = float(cov)
                except (TypeError, ValueError):
                    continue
                x = pad + log_x(k, min_k=min_k, max_k=max_k, width=w - 2 * pad)
                y = pad + (1.0 - max(0.0, min(1.0, yv))) * (h - 2 * pad)
                out.append(f"{x:.1f},{y:.1f}")
            return " ".join(out)

        molt_poly = to_poly(molt_points)
        reddit_poly = to_poly(reddit_points)
        if not molt_poly and not reddit_poly:
            return '<p class="muted">No curve data.</p>'

        x1 = pad
        y1 = h - pad
        x2 = w - pad
        y2 = pad
        return (
            f'<svg viewBox="0 0 {w:.0f} {h:.0f}" class="chart" role="img" '
            'aria-label="Topic coverage curve">'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y1:.1f}" class="axis" />'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y2:.1f}" class="axis" />'
            f'<polyline points="{molt_poly}" class="line moltbook" />'
            f'<polyline points="{reddit_poly}" class="line reddit" />'
            "</svg>"
        )

    def bar(value: Any, *, cls: str, max_value: float = 1.0) -> str:
        if value is None:
            width = 0.0
        else:
            try:
                v = float(value)
                width = 0.0 if max_value <= 0 else max(0.0, min(1.0, v / max_value))
            except (TypeError, ValueError):
                width = 0.0
        return (
            '<div class="bar"><div class="fill '
            f'{cls}" style="width: {width * 100:.1f}%"></div></div>'
        )

    def metric_row(
        name: str,
        molt_value: Any,
        reddit_value: Any,
        *,
        value_fmt: str,
        bar_max: float = 1.0,
    ) -> str:
        if value_fmt == "pct":
            molt_txt = pct(molt_value)
            red_txt = pct(reddit_value)
        elif value_fmt == "num3":
            molt_txt = num(molt_value, digits=3)
            red_txt = num(reddit_value, digits=3)
        elif value_fmt == "num4":
            molt_txt = num(molt_value, digits=4)
            red_txt = num(reddit_value, digits=4)
        else:
            molt_txt = esc(molt_value)
            red_txt = esc(reddit_value)

        return "\n".join(
            [
                "<tr>",
                f"<th>{esc(name)}</th>",
                "<td>",
                bar(molt_value, cls="moltbook", max_value=bar_max),
                f'<div class="val">{esc(molt_txt)}</div>',
                "</td>",
                "<td>",
                bar(reddit_value, cls="reddit", max_value=bar_max),
                f'<div class="val">{esc(red_txt)}</div>',
                "</td>",
                "</tr>",
            ]
        )

    def top_items_table(rows: list[dict[str, Any]], *, kind: str) -> str:
        if not rows:
            return '<p class="muted">No data.</p>'

        if kind == "duplicates":
            head = "<tr><th>Preview</th><th>Count</th><th>Threads</th><th>Share</th></tr>"
            body = "\n".join(
                "<tr>"
                f'<td class="mono">{esc(r.get("preview"))}</td>'
                f"<td>{esc(r.get('count'))}</td>"
                f"<td>{esc(r.get('distinct_threads'))}</td>"
                f"<td>{esc(pct(r.get('share')))}</td>"
                "</tr>"
                for r in rows
            )
        elif kind == "duplicates_with_key":
            head = (
                "<tr><th>Key</th><th>Preview</th><th>Count</th><th>Threads</th><th>Share</th></tr>"
            )
            body = "\n".join(
                "<tr>"
                f'<td class="mono">{esc(r.get("key"))}</td>'
                f'<td class="mono">{esc(r.get("preview"))}</td>'
                f"<td>{esc(r.get('count'))}</td>"
                f"<td>{esc(r.get('distinct_threads'))}</td>"
                f"<td>{esc(pct(r.get('share')))}</td>"
                "</tr>"
                for r in rows
            )
        else:
            head = "<tr><th>Item</th><th>Count</th><th>Share</th></tr>"
            body = "\n".join(
                "<tr>"
                f'<td class="mono">{esc(r.get("item"))}</td>'
                f"<td>{esc(r.get('count'))}</td>"
                f"<td>{esc(pct(r.get('share')))}</td>"
                "</tr>"
                for r in rows
            )

        return f'<table class="items"><thead>{head}</thead><tbody>{body}</tbody></table>'

    def _havelock_score_cell(row: dict[str, Any], *, cls: str) -> str:
        h = row.get("havelock") if isinstance(row.get("havelock"), dict) else {}
        score = h.get("score")
        return "\n".join(
            [
                bar(score, cls=cls, max_value=100.0),
                f'<div class="val">{esc(score)}</div>',
            ]
        )

    def _havelock_ratio_cell(row: dict[str, Any], *, cls: str) -> str:
        h = row.get("havelock") if isinstance(row.get("havelock"), dict) else {}
        ratio = h.get("sentence_ratio")
        return "\n".join(
            [
                bar(ratio, cls=cls, max_value=1.0),
                f'<div class="val">{esc(pct(ratio))}</div>',
            ]
        )

    def havelock_sections_table(rows: Any, *, cls: str) -> str:
        if not isinstance(rows, list) or not rows:
            return '<p class="muted">No section data.</p>'

        head = (
            "<tr>"
            "<th>Section</th>"
            "<th>Havelock score</th>"
            "<th>Oral sentence ratio</th>"
            "<th>Sample</th>"
            "<th>Corpus msgs</th>"
            "</tr>"
        )
        body_rows = []
        for r in rows:
            sample_txt = (
                f"{esc(r.get('sampled_messages'))} msgs / {esc(r.get('sample_char_len'))} chars"
            )
            body_rows.append(
                "<tr>"
                f"<td>{esc(r.get('section'))}</td>"
                f"<td>{_havelock_score_cell(r, cls=cls)}</td>"
                f"<td>{_havelock_ratio_cell(r, cls=cls)}</td>"
                f"<td>{sample_txt}</td>"
                f"<td>{esc(r.get('message_count'))}</td>"
                "</tr>"
            )
        body = "\n".join(body_rows)
        return f'<table class="items"><thead>{head}</thead><tbody>{body}</tbody></table>'

    counts = {
        "moltbook_threads": get(["moltbook_threads"]),
        "reddit_threads": get(["reddit_threads"]),
        "moltbook_messages": get(["moltbook_messages"]),
        "reddit_messages": get(["reddit_messages"]),
    }

    sources = get(["sources"], default={}) or {}
    molt_source = esc(sources.get("moltbook", "moltbook"))
    reddit_source = esc(sources.get("reddit", "reddit"))
    reddit_domain_source = esc(sources.get("reddit_domain", "reddit_domain"))

    molt_threads = esc(counts["moltbook_threads"])
    reddit_threads = esc(counts["reddit_threads"])
    molt_messages = esc(counts["moltbook_messages"])
    reddit_messages = esc(counts["reddit_messages"])

    m_dupe = get(["metrics", "message_exact_duplicate_rate", "moltbook"])
    r_dupe = get(["metrics", "message_exact_duplicate_rate", "reddit"])
    m_soft_dupe = get(["metrics", "message_soft_duplicate_rate", "moltbook"])
    r_soft_dupe = get(["metrics", "message_soft_duplicate_rate", "reddit"])
    m_dupe_breakdown = get(["metrics", "message_duplicate_breakdown", "moltbook"], default={}) or {}
    r_dupe_breakdown = get(["metrics", "message_duplicate_breakdown", "reddit"], default={}) or {}
    m_soft_breakdown = (
        get(["metrics", "message_soft_duplicate_breakdown", "moltbook"], default={}) or {}
    )
    r_soft_breakdown = (
        get(["metrics", "message_soft_duplicate_breakdown", "reddit"], default={}) or {}
    )

    m_dist1 = get(["metrics", "message_distinct_n", "distinct_1", "moltbook"])
    r_dist1 = get(["metrics", "message_distinct_n", "distinct_1", "reddit"])
    m_dist2 = get(["metrics", "message_distinct_n", "distinct_2", "moltbook"])
    r_dist2 = get(["metrics", "message_distinct_n", "distinct_2", "reddit"])

    m_uni_ent = get(["metrics", "message_unigram_entropy_bits", "moltbook"])
    r_uni_ent = get(["metrics", "message_unigram_entropy_bits", "reddit"])
    m_uni_cov = get(["metrics", "message_unigram_topk_coverage", "moltbook"], default={}) or {}
    r_uni_cov = get(["metrics", "message_unigram_topk_coverage", "reddit"], default={}) or {}

    m_sig = get(["metrics", "message_topic_signatures", "moltbook"], default={}) or {}
    r_sig = get(["metrics", "message_topic_signatures", "reddit"], default={}) or {}
    sig_curve = get(["metrics", "message_topic_signature_coverage_curve"], default={}) or {}
    m_sig_curve = (sig_curve.get("moltbook", {}) or {}).get("points", []) or []
    r_sig_curve = (sig_curve.get("reddit", {}) or {}).get("points", []) or []
    m_sig_k_at = (sig_curve.get("moltbook", {}) or {}).get("k_at_coverage", {}) or {}
    r_sig_k_at = (sig_curve.get("reddit", {}) or {}).get("k_at_coverage", {}) or {}

    m_thread_sig = get(["metrics", "thread_matched_topic_signatures", "moltbook"], default={}) or {}
    r_thread_sig = get(["metrics", "thread_matched_topic_signatures", "reddit"], default={}) or {}
    thread_terms_jacc = get(["metrics", "thread_top_terms_pairwise_jaccard"], default={}) or {}
    m_thread_jacc = thread_terms_jacc.get("moltbook", {}) or {}
    r_thread_jacc = thread_terms_jacc.get("reddit", {}) or {}

    m_jacc = get(["metrics", "message_pairwise_jaccard_similarity", "moltbook"], default={}) or {}
    r_jacc = get(["metrics", "message_pairwise_jaccard_similarity", "reddit"], default={}) or {}

    m_top_unigrams = get(["metrics", "message_top_unigrams", "moltbook"], default=[]) or []
    r_top_unigrams = get(["metrics", "message_top_unigrams", "reddit"], default=[]) or []
    m_top_sigs = get(["metrics", "message_top_topic_signatures", "moltbook"], default=[]) or []
    r_top_sigs = get(["metrics", "message_top_topic_signatures", "reddit"], default=[]) or []

    havelock = get(["metrics", "havelock_orality_literacy"], default=None)
    havelock_html = ""
    if isinstance(havelock, dict):
        hm = havelock.get("moltbook") if isinstance(havelock.get("moltbook"), dict) else {}
        hr = havelock.get("reddit") if isinstance(havelock.get("reddit"), dict) else {}

        hm_over = hm.get("overall") if isinstance(hm.get("overall"), dict) else {}
        hr_over = hr.get("overall") if isinstance(hr.get("overall"), dict) else {}
        hm_over_h = hm_over.get("havelock") if isinstance(hm_over.get("havelock"), dict) else {}
        hr_over_h = hr_over.get("havelock") if isinstance(hr_over.get("havelock"), dict) else {}

        hm_sum = hm.get("summary") if isinstance(hm.get("summary"), dict) else {}
        hr_sum = hr.get("summary") if isinstance(hr.get("summary"), dict) else {}

        havelock_rows = "\n".join(
            [
                metric_row(
                    "Overall score (sample)",
                    hm_over_h.get("score"),
                    hr_over_h.get("score"),
                    value_fmt="raw",
                    bar_max=100.0,
                ),
                metric_row(
                    "Oral sentence ratio (sample)",
                    hm_over_h.get("sentence_ratio"),
                    hr_over_h.get("sentence_ratio"),
                    value_fmt="pct",
                    bar_max=1.0,
                ),
                metric_row(
                    "Mean section score",
                    hm_sum.get("mean_score"),
                    hr_sum.get("mean_score"),
                    value_fmt="raw",
                    bar_max=100.0,
                ),
                metric_row(
                    "Weighted mean section score",
                    hm_sum.get("weighted_mean_score"),
                    hr_sum.get("weighted_mean_score"),
                    value_fmt="raw",
                    bar_max=100.0,
                ),
            ]
        )

        hm_sections = hm.get("sections") if isinstance(hm.get("sections"), list) else []
        hr_sections = hr.get("sections") if isinstance(hr.get("sections"), list) else []

        havelock_html = "\n".join(
            [
                '      <div class="card">',
                "        <h2>Orality vs literacy (Havelock)</h2>",
                '        <p class="muted">',
                "          Havelock scores text on a 0–100 scale (0=highly literate,",
                "          100=highly oral).",
                "          Sections are top subcommunities by message count.",
                "          Each section is scored on a deterministic sample of messages.",
                "        </p>",
                "        <table>",
                "          <thead>",
                "            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>",
                "          </thead>",
                "          <tbody>",
                havelock_rows,
                "          </tbody>",
                "        </table>",
                "        <details>",
                "          <summary>Section breakdown</summary>",
                "          <h2>Moltbook sections</h2>",
                havelock_sections_table(hm_sections, cls="moltbook"),
                "          <h2>Reddit sections</h2>",
                havelock_sections_table(hr_sections, cls="reddit"),
                "        </details>",
                "      </div>",
            ]
        )
    else:
        havelock_html = "\n".join(
            [
                '      <div class="card">',
                "        <h2>Orality vs literacy (Havelock)</h2>",
                '        <p class="muted">',
                "          No Havelock results in this report.",
                "          Re-run analyze with --havelock to fetch them.",
                "        </p>",
                "      </div>",
            ]
        )

    redundancy_rows = "\n".join(
        [
            metric_row("Exact duplicate rate (message)", m_dupe, r_dupe, value_fmt="pct"),
            metric_row(
                "Soft duplicate rate (bag-of-words)", m_soft_dupe, r_soft_dupe, value_fmt="pct"
            ),
            metric_row(
                "Duplicate message share (any duplicate)",
                m_dupe_breakdown.get("duplicate_message_share"),
                r_dupe_breakdown.get("duplicate_message_share"),
                value_fmt="pct",
            ),
            metric_row(
                "Soft duplicate message share (any duplicate)",
                m_soft_breakdown.get("duplicate_message_share"),
                r_soft_breakdown.get("duplicate_message_share"),
                value_fmt="pct",
            ),
            metric_row(
                "Cross-thread duplicate message share",
                m_dupe_breakdown.get("cross_thread_duplicate_message_share"),
                r_dupe_breakdown.get("cross_thread_duplicate_message_share"),
                value_fmt="pct",
            ),
            metric_row(
                "Cross-thread soft duplicate message share",
                m_soft_breakdown.get("cross_thread_duplicate_message_share"),
                r_soft_breakdown.get("cross_thread_duplicate_message_share"),
                value_fmt="pct",
            ),
            metric_row(
                "Within-thread duplicate message share",
                m_dupe_breakdown.get("within_thread_duplicate_message_share"),
                r_dupe_breakdown.get("within_thread_duplicate_message_share"),
                value_fmt="pct",
            ),
            metric_row(
                "Pairwise Jaccard similarity (mean, sampled)",
                m_jacc.get("mean"),
                r_jacc.get("mean"),
                value_fmt="num4",
                bar_max=0.2,
            ),
            metric_row(
                "Pairwise Jaccard similarity (p99, sampled)",
                m_jacc.get("p99"),
                r_jacc.get("p99"),
                value_fmt="num4",
                bar_max=0.2,
            ),
        ]
    )

    lexical_rows = "\n".join(
        [
            metric_row("Distinct-1 (token)", m_dist1, r_dist1, value_fmt="num3"),
            metric_row("Distinct-2 (token bigram)", m_dist2, r_dist2, value_fmt="num3"),
            metric_row(
                "Unigram entropy (bits)",
                m_uni_ent,
                r_uni_ent,
                value_fmt="num3",
                bar_max=16.0,
            ),
            metric_row(
                "Top-50 unigram coverage",
                m_uni_cov.get("50"),
                r_uni_cov.get("50"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-100 unigram coverage",
                m_uni_cov.get("100"),
                r_uni_cov.get("100"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-500 unigram coverage",
                m_uni_cov.get("500"),
                r_uni_cov.get("500"),
                value_fmt="pct",
            ),
        ]
    )

    message_topic_rows = "\n".join(
        [
            metric_row(
                "Signature entropy (bits)",
                m_sig.get("entropy_bits"),
                r_sig.get("entropy_bits"),
                value_fmt="num3",
                bar_max=20.0,
            ),
            metric_row(
                "Effective topics (2^H)",
                m_sig.get("effective_topics"),
                r_sig.get("effective_topics"),
                value_fmt="num3",
                bar_max=20000.0,
            ),
            metric_row(
                "Unique signatures",
                m_sig.get("unique"),
                r_sig.get("unique"),
                value_fmt="raw",
                bar_max=20000.0,
            ),
            metric_row(
                "Top-10 signature coverage",
                (m_sig.get("topk_coverage", {}) or {}).get("10"),
                (r_sig.get("topk_coverage", {}) or {}).get("10"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-20 signature coverage",
                (m_sig.get("topk_coverage", {}) or {}).get("20"),
                (r_sig.get("topk_coverage", {}) or {}).get("20"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-50 signature coverage",
                (m_sig.get("topk_coverage", {}) or {}).get("50"),
                (r_sig.get("topk_coverage", {}) or {}).get("50"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-100 signature coverage",
                (m_sig.get("topk_coverage", {}) or {}).get("100"),
                (r_sig.get("topk_coverage", {}) or {}).get("100"),
                value_fmt="pct",
            ),
        ]
    )

    thread_topic_rows = "\n".join(
        [
            metric_row(
                "Signature entropy (bits)",
                m_thread_sig.get("entropy_bits"),
                r_thread_sig.get("entropy_bits"),
                value_fmt="num3",
                bar_max=12.0,
            ),
            metric_row(
                "Effective topics (2^H)",
                m_thread_sig.get("effective_topics"),
                r_thread_sig.get("effective_topics"),
                value_fmt="num3",
                bar_max=2000.0,
            ),
            metric_row(
                "Top-10 signature coverage",
                (m_thread_sig.get("topk_coverage", {}) or {}).get("10"),
                (r_thread_sig.get("topk_coverage", {}) or {}).get("10"),
                value_fmt="pct",
            ),
            metric_row(
                "Top-50 signature coverage",
                (m_thread_sig.get("topk_coverage", {}) or {}).get("50"),
                (r_thread_sig.get("topk_coverage", {}) or {}).get("50"),
                value_fmt="pct",
            ),
        ]
    )

    raw_json = esc(json.dumps(report, ensure_ascii=False, indent=2))
    coverage_svg = coverage_curve_svg(m_sig_curve, r_sig_curve)
    m_k50 = esc(m_sig_k_at.get("50"))
    r_k50 = esc(r_sig_k_at.get("50"))
    thread_overlap_rows = "\n".join(
        [
            metric_row(
                "Top-terms Jaccard (mean, sampled)",
                m_thread_jacc.get("mean"),
                r_thread_jacc.get("mean"),
                value_fmt="num4",
                bar_max=1.0,
            ),
            metric_row(
                "Top-terms Jaccard (p99, sampled)",
                m_thread_jacc.get("p99"),
                r_thread_jacc.get("p99"),
                value_fmt="num4",
                bar_max=1.0,
            ),
        ]
    )

    domain_html = ""
    if isinstance(get(["metrics_domain"]), dict):
        dom_threads = esc(get(["reddit_domain_threads"]))
        dom_messages = esc(get(["reddit_domain_messages"]))

        dm_dupe = get(["metrics_domain", "message_exact_duplicate_rate", "moltbook"])
        dr_dupe = get(["metrics_domain", "message_exact_duplicate_rate", "reddit"])
        dm_soft_dupe = get(["metrics_domain", "message_soft_duplicate_rate", "moltbook"])
        dr_soft_dupe = get(["metrics_domain", "message_soft_duplicate_rate", "reddit"])

        dm_dupe_breakdown = (
            get(["metrics_domain", "message_duplicate_breakdown", "moltbook"], default={}) or {}
        )
        dr_dupe_breakdown = (
            get(["metrics_domain", "message_duplicate_breakdown", "reddit"], default={}) or {}
        )
        dm_soft_breakdown = (
            get(["metrics_domain", "message_soft_duplicate_breakdown", "moltbook"], default={})
            or {}
        )
        dr_soft_breakdown = (
            get(["metrics_domain", "message_soft_duplicate_breakdown", "reddit"], default={}) or {}
        )

        dm_dist1 = get(["metrics_domain", "message_distinct_n", "distinct_1", "moltbook"])
        dr_dist1 = get(["metrics_domain", "message_distinct_n", "distinct_1", "reddit"])
        dm_dist2 = get(["metrics_domain", "message_distinct_n", "distinct_2", "moltbook"])
        dr_dist2 = get(["metrics_domain", "message_distinct_n", "distinct_2", "reddit"])

        dm_uni_ent = get(["metrics_domain", "message_unigram_entropy_bits", "moltbook"])
        dr_uni_ent = get(["metrics_domain", "message_unigram_entropy_bits", "reddit"])
        dm_uni_cov = (
            get(["metrics_domain", "message_unigram_topk_coverage", "moltbook"], default={}) or {}
        )
        dr_uni_cov = (
            get(["metrics_domain", "message_unigram_topk_coverage", "reddit"], default={}) or {}
        )

        dm_sig = get(["metrics_domain", "message_topic_signatures", "moltbook"], default={}) or {}
        dr_sig = get(["metrics_domain", "message_topic_signatures", "reddit"], default={}) or {}
        d_sig_curve = (
            get(["metrics_domain", "message_topic_signature_coverage_curve"], default={}) or {}
        )
        dm_sig_curve = (d_sig_curve.get("moltbook", {}) or {}).get("points", []) or []
        dr_sig_curve = (d_sig_curve.get("reddit", {}) or {}).get("points", []) or []
        dm_sig_k_at = (d_sig_curve.get("moltbook", {}) or {}).get("k_at_coverage", {}) or {}
        dr_sig_k_at = (d_sig_curve.get("reddit", {}) or {}).get("k_at_coverage", {}) or {}

        dm_thread_sig = (
            get(["metrics_domain", "thread_matched_topic_signatures", "moltbook"], default={}) or {}
        )
        dr_thread_sig = (
            get(["metrics_domain", "thread_matched_topic_signatures", "reddit"], default={}) or {}
        )
        d_thread_terms_jacc = (
            get(["metrics_domain", "thread_top_terms_pairwise_jaccard"], default={}) or {}
        )
        dm_thread_jacc = d_thread_terms_jacc.get("moltbook", {}) or {}
        dr_thread_jacc = d_thread_terms_jacc.get("reddit", {}) or {}

        dm_jacc = (
            get(["metrics_domain", "message_pairwise_jaccard_similarity", "moltbook"], default={})
            or {}
        )
        dr_jacc = (
            get(["metrics_domain", "message_pairwise_jaccard_similarity", "reddit"], default={})
            or {}
        )

        dm_top_unigrams = (
            get(["metrics_domain", "message_top_unigrams", "moltbook"], default=[]) or []
        )
        dr_top_unigrams = (
            get(["metrics_domain", "message_top_unigrams", "reddit"], default=[]) or []
        )
        dm_top_sigs = (
            get(["metrics_domain", "message_top_topic_signatures", "moltbook"], default=[]) or []
        )
        dr_top_sigs = (
            get(["metrics_domain", "message_top_topic_signatures", "reddit"], default=[]) or []
        )

        dom_redundancy_rows = "\n".join(
            [
                metric_row("Exact duplicate rate (message)", dm_dupe, dr_dupe, value_fmt="pct"),
                metric_row(
                    "Soft duplicate rate (bag-of-words)",
                    dm_soft_dupe,
                    dr_soft_dupe,
                    value_fmt="pct",
                ),
                metric_row(
                    "Duplicate message share (any duplicate)",
                    dm_dupe_breakdown.get("duplicate_message_share"),
                    dr_dupe_breakdown.get("duplicate_message_share"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Soft duplicate message share (any duplicate)",
                    dm_soft_breakdown.get("duplicate_message_share"),
                    dr_soft_breakdown.get("duplicate_message_share"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Cross-thread duplicate message share",
                    dm_dupe_breakdown.get("cross_thread_duplicate_message_share"),
                    dr_dupe_breakdown.get("cross_thread_duplicate_message_share"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Cross-thread soft duplicate message share",
                    dm_soft_breakdown.get("cross_thread_duplicate_message_share"),
                    dr_soft_breakdown.get("cross_thread_duplicate_message_share"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Pairwise Jaccard similarity (mean, sampled)",
                    dm_jacc.get("mean"),
                    dr_jacc.get("mean"),
                    value_fmt="num4",
                    bar_max=0.2,
                ),
                metric_row(
                    "Pairwise Jaccard similarity (p99, sampled)",
                    dm_jacc.get("p99"),
                    dr_jacc.get("p99"),
                    value_fmt="num4",
                    bar_max=0.2,
                ),
            ]
        )

        dom_lexical_rows = "\n".join(
            [
                metric_row("Distinct-1 (token)", dm_dist1, dr_dist1, value_fmt="num3"),
                metric_row("Distinct-2 (token bigram)", dm_dist2, dr_dist2, value_fmt="num3"),
                metric_row(
                    "Unigram entropy (bits)",
                    dm_uni_ent,
                    dr_uni_ent,
                    value_fmt="num3",
                    bar_max=16.0,
                ),
                metric_row(
                    "Top-50 unigram coverage",
                    dm_uni_cov.get("50"),
                    dr_uni_cov.get("50"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Top-100 unigram coverage",
                    dm_uni_cov.get("100"),
                    dr_uni_cov.get("100"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Top-500 unigram coverage",
                    dm_uni_cov.get("500"),
                    dr_uni_cov.get("500"),
                    value_fmt="pct",
                ),
            ]
        )

        dom_message_topic_rows = "\n".join(
            [
                metric_row(
                    "Signature entropy (bits)",
                    dm_sig.get("entropy_bits"),
                    dr_sig.get("entropy_bits"),
                    value_fmt="num3",
                    bar_max=20.0,
                ),
                metric_row(
                    "Effective topics (2^H)",
                    dm_sig.get("effective_topics"),
                    dr_sig.get("effective_topics"),
                    value_fmt="num3",
                    bar_max=20000.0,
                ),
                metric_row(
                    "Top-10 signature coverage",
                    (dm_sig.get("topk_coverage", {}) or {}).get("10"),
                    (dr_sig.get("topk_coverage", {}) or {}).get("10"),
                    value_fmt="pct",
                ),
                metric_row(
                    "Top-100 signature coverage",
                    (dm_sig.get("topk_coverage", {}) or {}).get("100"),
                    (dr_sig.get("topk_coverage", {}) or {}).get("100"),
                    value_fmt="pct",
                ),
            ]
        )

        dom_thread_topic_rows = "\n".join(
            [
                metric_row(
                    "Signature entropy (bits)",
                    dm_thread_sig.get("entropy_bits"),
                    dr_thread_sig.get("entropy_bits"),
                    value_fmt="num3",
                    bar_max=12.0,
                ),
                metric_row(
                    "Top-10 signature coverage",
                    (dm_thread_sig.get("topk_coverage", {}) or {}).get("10"),
                    (dr_thread_sig.get("topk_coverage", {}) or {}).get("10"),
                    value_fmt="pct",
                ),
            ]
        )

        d_coverage_svg = coverage_curve_svg(dm_sig_curve, dr_sig_curve)
        dm_k50 = esc(dm_sig_k_at.get("50"))
        dr_k50 = esc(dr_sig_k_at.get("50"))
        dom_thread_overlap_rows = "\n".join(
            [
                metric_row(
                    "Top-terms Jaccard (mean, sampled)",
                    dm_thread_jacc.get("mean"),
                    dr_thread_jacc.get("mean"),
                    value_fmt="num4",
                    bar_max=1.0,
                ),
                metric_row(
                    "Top-terms Jaccard (p99, sampled)",
                    dm_thread_jacc.get("p99"),
                    dr_thread_jacc.get("p99"),
                    value_fmt="num4",
                    bar_max=1.0,
                ),
            ]
        )

        dom_dupe_m = top_items_table(dm_dupe_breakdown.get("top_duplicates", []), kind="duplicates")
        dom_dupe_r = top_items_table(dr_dupe_breakdown.get("top_duplicates", []), kind="duplicates")
        dom_soft_m = top_items_table(
            dm_soft_breakdown.get("top_duplicates", []), kind="duplicates_with_key"
        )
        dom_soft_r = top_items_table(
            dr_soft_breakdown.get("top_duplicates", []), kind="duplicates_with_key"
        )
        dom_uni_m = top_items_table(dm_top_unigrams, kind="items")
        dom_uni_r = top_items_table(dr_top_unigrams, kind="items")
        dom_sig_m = top_items_table(dm_top_sigs, kind="items")
        dom_sig_r = top_items_table(dr_top_sigs, kind="items")

        domain_html = "\n".join(
            [
                "      <details>",
                "        <summary>Domain baseline (Reddit subset)</summary>",
                '        <div class="grid">',
                '          <div class="card">',
                "            <h2>Domain baseline</h2>",
                '            <div class="legend">',
                '              <span class="pill"><span class="dot moltbook"></span>',
                "                Moltbook</span>",
                '              <span class="pill"><span class="dot reddit"></span>',
                "                Reddit (domain)</span>",
                f'              <span class="pill">Reddit source: {reddit_domain_source}</span>',
                f'              <span class="pill">Threads: {molt_threads} vs {dom_threads}</span>',
                f'              <span class="pill">Msgs: {molt_messages} vs {dom_messages}</span>',
                "            </div>",
                "          </div>",
                "",
                '          <div class="card">',
                "            <h2>Redundancy</h2>",
                "            <table>",
                "              <thead>",
                "                <tr><th></th><th>Moltbook</th><th>Reddit (domain)</th></tr>",
                "              </thead>",
                "              <tbody>",
                dom_redundancy_rows,
                "              </tbody>",
                "            </table>",
                "",
                "            <details>",
                "              <summary>Top duplicate messages</summary>",
                "              <h2>Top duplicates – Moltbook</h2>",
                dom_dupe_m,
                "              <h2>Top duplicates – Reddit (domain)</h2>",
                dom_dupe_r,
                "            </details>",
                "",
                "            <details>",
                "              <summary>Top soft duplicates (bag-of-words)</summary>",
                "              <h2>Top soft duplicates – Moltbook</h2>",
                dom_soft_m,
                "              <h2>Top soft duplicates – Reddit (domain)</h2>",
                dom_soft_r,
                "            </details>",
                "          </div>",
                "",
                '          <div class="card">',
                "            <h2>Lexical diversity</h2>",
                "            <table>",
                "              <thead>",
                "                <tr><th></th><th>Moltbook</th><th>Reddit (domain)</th></tr>",
                "              </thead>",
                "              <tbody>",
                dom_lexical_rows,
                "              </tbody>",
                "            </table>",
                "",
                "            <details>",
                "              <summary>Top unigrams</summary>",
                "              <h2>Top unigrams – Moltbook</h2>",
                dom_uni_m,
                "              <h2>Top unigrams – Reddit (domain)</h2>",
                dom_uni_r,
                "            </details>",
                "          </div>",
                "",
                '          <div class="card">',
                "            <h2>Topic bucket concentration (message-level signatures)</h2>",
                "            <table>",
                "              <thead>",
                "                <tr><th></th><th>Moltbook</th><th>Reddit (domain)</th></tr>",
                "              </thead>",
                "              <tbody>",
                dom_message_topic_rows,
                "              </tbody>",
                "            </table>",
                "",
                "            <h2>Topic coverage curve (message signatures)</h2>",
                d_coverage_svg,
                '            <p class="muted">',
                f"              Buckets needed to cover 50% of messages: Moltbook {dm_k50},",
                f"              Reddit (domain) {dr_k50}.",
                "            </p>",
                "",
                "            <details>",
                "              <summary>Top message-level signatures</summary>",
                "              <h2>Top signatures – Moltbook</h2>",
                dom_sig_m,
                "              <h2>Top signatures – Reddit (domain)</h2>",
                dom_sig_r,
                "            </details>",
                "          </div>",
                "",
                '          <div class="card">',
                "            <h2>Thread-doc signatures (length-matched)</h2>",
                "            <table>",
                "              <thead>",
                "                <tr><th></th><th>Moltbook</th><th>Reddit (domain)</th></tr>",
                "              </thead>",
                "              <tbody>",
                dom_thread_topic_rows,
                "              </tbody>",
                "            </table>",
                "            <h2>Cross-thread topic overlap (thread top terms)</h2>",
                "            <table>",
                "              <thead>",
                "                <tr><th></th><th>Moltbook</th><th>Reddit (domain)</th></tr>",
                "              </thead>",
                "              <tbody>",
                dom_thread_overlap_rows,
                "              </tbody>",
                "            </table>",
                '            <p class="muted">',
                "              Note: Reddit thread-docs are aggregates of sampled comments",
                "              (not full threads).",
                "            </p>",
                "          </div>",
                "        </div>",
                "      </details>",
            ]
        )
    html_out = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #0b1220;
        --panel: #121b2e;
        --text: #e6edf7;
        --muted: #9bb0d0;
        --border: rgba(255,255,255,0.09);
        --molt: #a855f7;
        --reddit: #22c55e;
      }}
      body {{
        margin: 0;
        padding: 24px;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial,
          "Apple Color Emoji", "Segoe UI Emoji";
        background: var(--bg);
        color: var(--text);
      }}
      h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
      h2 {{ margin: 18px 0 8px 0; font-size: 16px; }}
      .muted {{ color: var(--muted); }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        max-width: 1100px;
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px;
      }}
      .legend {{
        display: flex;
        gap: 14px;
        align-items: center;
        margin-top: 6px;
      }}
      .pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 10px;
        border: 1px solid var(--border);
        border-radius: 999px;
        font-size: 12px;
        color: var(--muted);
      }}
      .dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
      .dot.moltbook {{ background: var(--molt); }}
      .dot.reddit {{ background: var(--reddit); }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      th, td {{
        border-top: 1px solid var(--border);
        padding: 8px 8px;
        vertical-align: top;
      }}
      th {{ text-align: left; font-weight: 600; color: var(--text); width: 38%; }}
      td {{ width: 31%; }}
      .bar {{
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
      }}
      .fill {{ height: 100%; }}
      .fill.moltbook {{ background: var(--molt); }}
      .fill.reddit {{ background: var(--reddit); }}
      .val {{ margin-top: 4px; color: var(--muted); font-variant-numeric: tabular-nums; }}
      .items th {{ width: auto; }}
      .items td {{ width: auto; }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
          "Courier New", monospace;
        font-size: 12px;
        white-space: pre-wrap;
        word-break: break-word;
      }}
      details > summary {{
        cursor: pointer;
        color: var(--muted);
        margin-top: 8px;
      }}
      pre {{
        margin: 10px 0 0 0;
        padding: 12px;
        border-radius: 10px;
        background: rgba(0,0,0,0.25);
        border: 1px solid var(--border);
        overflow: auto;
        max-height: 380px;
      }}
      .chart {{
        width: 100%;
        height: auto;
        margin-top: 10px;
      }}
      .axis {{
        stroke: rgba(255,255,255,0.25);
        stroke-width: 1;
      }}
      .line {{
        fill: none;
        stroke-width: 2.25;
      }}
      .line.moltbook {{
        stroke: var(--molt);
      }}
      .line.reddit {{
        stroke: var(--reddit);
      }}
    </style>
  </head>
  <body>
    <div class="grid">
      <div class="card">
        <h1>{esc(title)}</h1>
        <div class="muted">Generated {esc(generated_at)} (UTC)</div>
        <div class="legend">
          <span class="pill"><span class="dot moltbook"></span>Moltbook</span>
          <span class="pill"><span class="dot reddit"></span>Reddit</span>
          <span class="pill">Sources: {molt_source} vs {reddit_source}</span>
          <span class="pill">Threads: {molt_threads} vs {reddit_threads}</span>
          <span class="pill">Messages: {molt_messages} vs {reddit_messages}</span>
        </div>
      </div>

      <div class="card">
        <h2>Redundancy</h2>
        <table>
          <thead>
            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>
          </thead>
          <tbody>
            {redundancy_rows}
          </tbody>
        </table>

        <details>
          <summary>Top duplicate messages</summary>
          <h2>Top duplicates – Moltbook</h2>
          {top_items_table(m_dupe_breakdown.get("top_duplicates", []), kind="duplicates")}
          <h2>Top duplicates – Reddit</h2>
          {top_items_table(r_dupe_breakdown.get("top_duplicates", []), kind="duplicates")}
        </details>

        <details>
          <summary>Top soft duplicates (bag-of-words)</summary>
          <h2>Top soft duplicates – Moltbook</h2>
          {top_items_table(m_soft_breakdown.get("top_duplicates", []), kind="duplicates_with_key")}
          <h2>Top soft duplicates – Reddit</h2>
          {top_items_table(r_soft_breakdown.get("top_duplicates", []), kind="duplicates_with_key")}
        </details>
      </div>

      <div class="card">
        <h2>Lexical diversity</h2>
        <table>
          <thead>
            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>
          </thead>
          <tbody>
            {lexical_rows}
          </tbody>
        </table>

        <details>
          <summary>Top unigrams</summary>
          <h2>Top unigrams – Moltbook</h2>
          {top_items_table(m_top_unigrams, kind="items")}
          <h2>Top unigrams – Reddit</h2>
          {top_items_table(r_top_unigrams, kind="items")}
        </details>
      </div>

{havelock_html}

      <div class="card">
        <h2>Topic bucket concentration (message-level signatures)</h2>
        <table>
          <thead>
            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>
          </thead>
          <tbody>
            {message_topic_rows}
          </tbody>
        </table>

        <h2>Topic coverage curve (message signatures)</h2>
        {coverage_svg}
        <p class="muted">
          Buckets needed to cover 50% of messages: Moltbook {m_k50}, Reddit {r_k50}.
        </p>

        <details>
          <summary>Top message-level signatures</summary>
          <h2>Top signatures – Moltbook</h2>
          {top_items_table(m_top_sigs, kind="items")}
          <h2>Top signatures – Reddit</h2>
          {top_items_table(r_top_sigs, kind="items")}
        </details>
      </div>

      <div class="card">
        <h2>Topic bucket concentration (thread-doc signatures, length-matched)</h2>
        <table>
          <thead>
            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>
          </thead>
          <tbody>
            {thread_topic_rows}
          </tbody>
        </table>
        <h2>Cross-thread topic overlap (thread top terms)</h2>
        <table>
          <thead>
            <tr><th></th><th>Moltbook</th><th>Reddit</th></tr>
          </thead>
          <tbody>
            {thread_overlap_rows}
          </tbody>
        </table>
        <p class="muted">
          Note: Reddit thread-docs are aggregates of sampled comments (not full threads).
        </p>
      </div>

      <div class="card">
        <h2>Raw JSON</h2>
        <details>
          <summary>Show full report JSON</summary>
          <pre class="mono">{raw_json}</pre>
        </details>
      </div>
{domain_html}
    </div>
  </body>
</html>
"""
    return html_out


def render_html_file(*, report_json: Path, out_html: Path) -> None:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html_report(report) + "\n", encoding="utf-8")
