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
            "havelock_orality_literacy": {
                "provider": "havelock.ai",
                "base_url": "https://example.invalid",
                "moltbook": {
                    "overall": {"havelock": {"score": 10, "sentence_ratio": 0.2}},
                    "summary": {"mean_score": 11, "weighted_mean_score": 12},
                    "sections": [
                        {
                            "section": "<script>section</script>",
                            "thread_count": 1,
                            "message_count": 2,
                            "sampled_messages": 2,
                            "sample_char_len": 10,
                            "havelock": {"score": 13, "sentence_ratio": 0.1},
                        }
                    ],
                },
                "reddit": {
                    "overall": {"havelock": {"score": 20, "sentence_ratio": 0.3}},
                    "summary": {"mean_score": 21, "weighted_mean_score": 22},
                    "sections": [],
                },
            },
        },
    }
    html_out = render_html_report(report)
    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&lt;script&gt;section&lt;/script&gt;" in html_out
