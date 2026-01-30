from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
from pathlib import Path

from .db import init_db, upsert_message, upsert_thread
from .http_client import HttpClient


def _ns_id(kind: str, source_id: str) -> str:
    return f"moltbook_{kind}:{source_id}"


def _fetch_post_with_comments(client: HttpClient, base_url: str, post_id: str) -> dict:
    return client.get_json(f"{base_url}/api/v1/posts/{post_id}")


def ingest_moltbook(
    *,
    db_path: Path,
    base_url: str = "https://www.moltbook.com",
    max_posts: int | None = None,
    concurrency: int = 8,
) -> None:
    db = init_db(db_path)
    client = HttpClient()

    with db.connect() as conn:
        existing_post_ids = {
            str(r["source_id"])
            for r in conn.execute("SELECT source_id FROM threads WHERE source='moltbook'")
        }

    posts: list[dict] = []
    offset = 0
    limit = 100
    while True:
        url = f"{base_url}/api/v1/posts?limit={limit}&offset={offset}"
        payload = client.get_json(url)
        batch = payload.get("posts") or []
        posts.extend(batch)

        if max_posts is not None and len(posts) >= max_posts:
            posts = posts[:max_posts]
            break

        if not payload.get("has_more"):
            break
        offset = int(payload.get("next_offset") or (offset + len(batch)))

    if existing_post_ids:
        posts = [p for p in posts if str(p.get("id")) not in existing_post_ids]

    started_at = dt.datetime.now(dt.UTC).isoformat()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO ingestions(source, started_at, params_json) VALUES(?, ?, ?)",
            (
                "moltbook",
                started_at,
                json.dumps(
                    {"base_url": base_url, "max_posts": max_posts, "concurrency": concurrency},
                    separators=(",", ":"),
                ),
            ),
        )

    def ingest_one(post_summary: dict) -> tuple[str, dict]:
        post_id = str(post_summary["id"])
        try:
            return post_id, _fetch_post_with_comments(client, base_url, post_id)
        except Exception as e:  # noqa: BLE001
            return post_id, {"success": False, "error": str(e)}

    try:
        total_comments = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(ingest_one, p) for p in posts]
            for fut in concurrent.futures.as_completed(futures):
                post_id, data = fut.result()
                if not data.get("success"):
                    continue

                post = data.get("post") or {}
                comments = data.get("comments") or []

                thread_id = _ns_id("post", post_id)
                submolt = post.get("submolt") or {}
                with db.connect() as conn:
                    conn.execute("BEGIN")
                    upsert_thread(
                        conn,
                        thread_id=thread_id,
                        source="moltbook",
                        source_id=post_id,
                        subcommunity=submolt.get("name"),
                        title=post.get("title"),
                        body=post.get("content") or post.get("body"),
                        author=(post.get("author") or {}).get("name"),
                        created_at=post.get("created_at"),
                        metadata={"raw": post},
                    )

                    def walk(
                        nodes: list[dict], parent_comment_id: str | None, *, thread_id: str
                    ) -> int:
                        total = 0
                        for c in nodes:
                            cid = str(c["id"])
                            upsert_message(
                                conn,
                                message_id=_ns_id("comment", cid),
                                source="moltbook",
                                source_id=cid,
                                thread_id=thread_id,
                                parent_id=_ns_id("comment", parent_comment_id)
                                if parent_comment_id
                                else None,
                                author=(c.get("author") or {}).get("name"),
                                body=c.get("content"),
                                created_at=c.get("created_at"),
                                upvotes=c.get("upvotes"),
                                downvotes=c.get("downvotes"),
                                metadata={"raw": c},
                            )
                            total += 1
                            total += walk(c.get("replies") or [], cid, thread_id=thread_id)
                        return total

                    total_comments += walk(comments, None, thread_id=thread_id)
                    conn.execute("COMMIT")
    finally:
        finished_at = dt.datetime.now(dt.UTC).isoformat()
        with db.connect() as conn:
            conn.execute(
                "UPDATE ingestions SET finished_at=? WHERE source=? AND started_at=?",
                (finished_at, "moltbook", started_at),
            )
