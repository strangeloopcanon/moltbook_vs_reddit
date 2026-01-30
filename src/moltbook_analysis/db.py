from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Db:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  params_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  subcommunity TEXT,
  title TEXT,
  body TEXT,
  author TEXT,
  created_at TEXT,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_threads_source ON threads(source);
CREATE INDEX IF NOT EXISTS idx_threads_source_id ON threads(source, source_id);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
  parent_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
  author TEXT,
  body TEXT,
  created_at TEXT,
  upvotes INTEGER,
  downvotes INTEGER,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_source ON messages(source);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_source_id ON messages(source, source_id);
"""


def init_db(db_path: Path) -> Db:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Db(path=db_path)
    with db.connect() as conn:
        conn.executescript(SCHEMA_SQL)
    return db


def json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def upsert_thread(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    source: str,
    source_id: str,
    subcommunity: str | None,
    title: str | None,
    body: str | None,
    author: str | None,
    created_at: str | None,
    metadata: dict | None,
) -> None:
    conn.execute(
        """
        INSERT INTO threads(
          id, source, source_id, subcommunity, title, body, author, created_at, metadata_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          source=excluded.source,
          source_id=excluded.source_id,
          subcommunity=excluded.subcommunity,
          title=excluded.title,
          body=excluded.body,
          author=excluded.author,
          created_at=excluded.created_at,
          metadata_json=excluded.metadata_json
        """,
        (
            thread_id,
            source,
            source_id,
            subcommunity,
            title,
            body,
            author,
            created_at,
            json_dumps(metadata) if metadata is not None else None,
        ),
    )


def upsert_message(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    source: str,
    source_id: str,
    thread_id: str,
    parent_id: str | None,
    author: str | None,
    body: str | None,
    created_at: str | None,
    upvotes: int | None,
    downvotes: int | None,
    metadata: dict | None,
) -> None:
    conn.execute(
        """
        INSERT INTO messages(
          id,
          source,
          source_id,
          thread_id,
          parent_id,
          author,
          body,
          created_at,
          upvotes,
          downvotes,
          metadata_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          source=excluded.source,
          source_id=excluded.source_id,
          thread_id=excluded.thread_id,
          parent_id=excluded.parent_id,
          author=excluded.author,
          body=excluded.body,
          created_at=excluded.created_at,
          upvotes=excluded.upvotes,
          downvotes=excluded.downvotes,
          metadata_json=excluded.metadata_json
        """,
        (
            message_id,
            source,
            source_id,
            thread_id,
            parent_id,
            author,
            body,
            created_at,
            upvotes,
            downvotes,
            json_dumps(metadata) if metadata is not None else None,
        ),
    )
