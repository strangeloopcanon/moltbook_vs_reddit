from __future__ import annotations

import pytest

from moltbook_analysis.havelock_client import parse_havelock_sse


def test_parse_havelock_sse_object() -> None:
    sse = 'event: complete\ndata: {"score": 76}\n'
    assert parse_havelock_sse(sse)["score"] == 76


def test_parse_havelock_sse_list_of_object() -> None:
    sse = 'event: complete\ndata: [{"score": 76}]\n'
    assert parse_havelock_sse(sse)["score"] == 76


def test_parse_havelock_sse_list_with_header() -> None:
    sse = 'event: complete\ndata: ["## Results:", {"score": 76}]\n'
    assert parse_havelock_sse(sse)["score"] == 76


def test_parse_havelock_sse_missing_data_line_raises() -> None:
    with pytest.raises(ValueError, match="data"):
        parse_havelock_sse("event: ping\n")
