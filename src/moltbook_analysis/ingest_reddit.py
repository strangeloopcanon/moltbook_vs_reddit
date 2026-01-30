from __future__ import annotations

import datetime as dt
import json
import random
import re
import subprocess
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

from .db import init_db, upsert_message, upsert_thread


def _ns_id(source: str, kind: str, source_id: str) -> str:
    return f"{source}_{kind}:{source_id}"


_FNV1A_32_OFFSET = 2166136261
_FNV1A_32_PRIME = 16777619


def _fnv1a_32(text: str) -> int:
    h = _FNV1A_32_OFFSET
    for b in text.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * _FNV1A_32_PRIME) & 0xFFFFFFFF
    return h


def _subcommunity_ok(
    subreddit: object,
    *,
    allowlist: set[str] | None,
    pattern: re.Pattern[str] | None,
) -> bool:
    if allowlist is None and pattern is None:
        return True
    if not isinstance(subreddit, str) or not subreddit:
        return False
    sub = subreddit.lower()
    if allowlist is not None and sub in allowlist:
        return True
    return bool(pattern.search(subreddit) if pattern is not None else False)


def _iter_ndjson_from_zst_path(path: Path) -> Iterable[dict]:
    # Stream decompression via zstd CLI for portability (no extra Python deps).
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        with suppress(Exception):
            proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=10)


def _iter_ndjson_from_zst_url(url: str) -> Iterable[dict]:
    curl = subprocess.Popen(
        ["curl", "-L", "--fail", "--silent", "--show-error", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert curl.stdout is not None
    zstd = subprocess.Popen(
        ["zstd", "-dc"],
        stdin=curl.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    curl.stdout.close()
    assert zstd.stdout is not None
    try:
        for line in zstd.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        with suppress(Exception):
            zstd.stdout.close()
        zstd.terminate()
        curl.terminate()
        with suppress(Exception):
            zstd.wait(timeout=10)
        with suppress(Exception):
            curl.wait(timeout=10)


def ingest_reddit(
    *,
    db_path: Path,
    source: str = "reddit",
    source_url: str,
    target_comments: int = 20000,
    seed: int = 1337,
    scan_comments: int | None = None,
    mode: str = "comments",
    keep_mod: int = 1000,
    keep_threshold: int = 1,
    subcommunities: list[str] | None = None,
    subcommunity_pattern: str | None = None,
) -> None:
    db = init_db(db_path)
    rng = random.Random(seed)
    allowlist = {s.lower() for s in subcommunities} if subcommunities else None
    pattern = re.compile(subcommunity_pattern) if subcommunity_pattern else None

    started_at = dt.datetime.now(dt.UTC).isoformat()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO ingestions(source, started_at, params_json) VALUES(?, ?, ?)",
            (
                source,
                started_at,
                json.dumps(
                    {
                        "source_url": source_url,
                        "seed": seed,
                        "target_comments": target_comments,
                        "mode": mode,
                        "keep_mod": keep_mod,
                        "keep_threshold": keep_threshold,
                        "subcommunities": subcommunities,
                        "subcommunity_pattern": subcommunity_pattern,
                    },
                    separators=(",", ":"),
                ),
            ),
        )

    if scan_comments is None:
        scan_comments = max(200_000, target_comments * 30)

    if source_url.startswith(("http://", "https://")):
        it = _iter_ndjson_from_zst_url(source_url)
    else:
        it = _iter_ndjson_from_zst_path(Path(source_url))

    if mode not in ("comments", "threads"):
        raise ValueError("mode must be 'comments' or 'threads'")

    if mode == "comments":
        eligible_seen = 0
        sample: list[dict] = []
        for row in it:
            body = row.get("body")
            if not isinstance(body, str) or not body or body in ("[deleted]", "[removed]"):
                continue

            link_id = row.get("link_id")
            comment_id = row.get("id")
            if not isinstance(link_id, str) or not isinstance(comment_id, str):
                continue

            if not _subcommunity_ok(row.get("subreddit"), allowlist=allowlist, pattern=pattern):
                continue

            eligible_seen += 1

            # Reservoir sample over the scanned prefix. Keeps this cheap and avoids needing to
            # download the full dump to get a random-ish slice.
            if len(sample) < target_comments:
                sample.append(row)
            else:
                j = rng.randrange(eligible_seen)
                if j < target_comments:
                    sample[j] = row

            if eligible_seen >= scan_comments:
                break

        seen_threads: set[str] = set()
        comment_ids = {str(r.get("id")) for r in sample if isinstance(r.get("id"), str)}
        parent_updates: list[tuple[str, str]] = []

        with db.connect() as conn:
            conn.execute("BEGIN")
            for row in sample:
                body = row.get("body")
                link_id = row.get("link_id")
                comment_id = row.get("id")
                if (
                    not isinstance(body, str)
                    or not isinstance(link_id, str)
                    or not isinstance(comment_id, str)
                ):
                    continue

                thread_pk = _ns_id(source, "link", link_id)
                if thread_pk not in seen_threads:
                    seen_threads.add(thread_pk)
                    upsert_thread(
                        conn,
                        thread_id=thread_pk,
                        source=source,
                        source_id=link_id,
                        subcommunity=row.get("subreddit"),
                        title=None,
                        body=None,
                        author=None,
                        created_at=None,
                        metadata={"source_url": source_url},
                    )

                msg_id = _ns_id(source, "comment", comment_id)
                upsert_message(
                    conn,
                    message_id=msg_id,
                    source=source,
                    source_id=comment_id,
                    thread_id=thread_pk,
                    parent_id=None,
                    author=row.get("author"),
                    body=body,
                    created_at=str(row.get("created_utc"))
                    if row.get("created_utc") is not None
                    else None,
                    upvotes=None,
                    downvotes=None,
                    metadata={"raw": row},
                )

                parent_full = row.get("parent_id")
                if isinstance(parent_full, str) and parent_full.startswith("t1_"):
                    parent_cid = parent_full[3:]
                    if parent_cid in comment_ids:
                        parent_updates.append((msg_id, _ns_id(source, "comment", parent_cid)))

            for child_id, parent_id in parent_updates:
                conn.execute("UPDATE messages SET parent_id=? WHERE id=?", (parent_id, child_id))

            conn.execute("COMMIT")
    else:
        eligible_seen = 0
        inserted = 0
        inserted_comment_ids: set[str] = set()
        seen_threads: set[str] = set()

        with db.connect() as conn:
            conn.execute("BEGIN")
            for row in it:
                if inserted >= target_comments:
                    break

                body = row.get("body")
                if not isinstance(body, str) or not body or body in ("[deleted]", "[removed]"):
                    continue

                link_id = row.get("link_id")
                comment_id = row.get("id")
                if not isinstance(link_id, str) or not isinstance(comment_id, str):
                    continue

                if not _subcommunity_ok(row.get("subreddit"), allowlist=allowlist, pattern=pattern):
                    continue

                eligible_seen += 1
                keep = (_fnv1a_32(link_id) % keep_mod) < keep_threshold
                if not keep:
                    if eligible_seen >= scan_comments:
                        break
                    continue

                thread_pk = _ns_id(source, "link", link_id)
                if thread_pk not in seen_threads:
                    seen_threads.add(thread_pk)
                    upsert_thread(
                        conn,
                        thread_id=thread_pk,
                        source=source,
                        source_id=link_id,
                        subcommunity=row.get("subreddit"),
                        title=None,
                        body=None,
                        author=None,
                        created_at=None,
                        metadata={"source_url": source_url},
                    )

                parent_id = None
                parent_full = row.get("parent_id")
                if isinstance(parent_full, str) and parent_full.startswith("t1_"):
                    parent_cid = parent_full[3:]
                    if parent_cid in inserted_comment_ids:
                        parent_id = _ns_id(source, "comment", parent_cid)

                msg_id = _ns_id(source, "comment", comment_id)
                upsert_message(
                    conn,
                    message_id=msg_id,
                    source=source,
                    source_id=comment_id,
                    thread_id=thread_pk,
                    parent_id=parent_id,
                    author=row.get("author"),
                    body=body,
                    created_at=str(row.get("created_utc"))
                    if row.get("created_utc") is not None
                    else None,
                    upvotes=None,
                    downvotes=None,
                    metadata={"raw": row},
                )
                inserted += 1
                inserted_comment_ids.add(comment_id)

                if inserted >= target_comments:
                    break
                if eligible_seen >= scan_comments:
                    break
            conn.execute("COMMIT")

    finished_at = dt.datetime.now(dt.UTC).isoformat()
    with db.connect() as conn:
        conn.execute(
            "UPDATE ingestions SET finished_at=? WHERE source=? AND started_at=?",
            (finished_at, source, started_at),
        )
