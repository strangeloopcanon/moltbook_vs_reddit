from __future__ import annotations

from moltbook_analysis.render_html import render_html_report


def test_render_html_escapes_content() -> None:
    report = {
        "moltbook_threads": 1,
        "reddit_threads": 1,
        "moltbook_messages": 1,
        "reddit_messages": 1,
        "metrics": {
            "message_exact_duplicate_rate": {"moltbook": 0.1, "reddit": 0.2},
            "message_duplicate_breakdown": {
                "moltbook": {
                    "top_duplicates": [{"preview": "<script>alert(1)</script>", "count": 2}]
                },
                "reddit": {"top_duplicates": []},
            },
        },
    }
    html_out = render_html_report(report)
    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
