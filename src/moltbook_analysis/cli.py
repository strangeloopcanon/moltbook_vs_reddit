from __future__ import annotations

import argparse
from pathlib import Path

from .ingest_moltbook import ingest_moltbook
from .ingest_reddit import ingest_reddit
from .render_html import render_html_file
from .run_analysis import HavelockParams, run_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moltbook_analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest_molt = sub.add_parser(
        "ingest-moltbook", help="Ingest Moltbook posts + comments into SQLite"
    )
    p_ingest_molt.add_argument("--db", type=Path, required=True)
    p_ingest_molt.add_argument("--base-url", default="https://www.moltbook.com")
    p_ingest_molt.add_argument("--max-posts", type=int, default=None)
    p_ingest_molt.add_argument("--concurrency", type=int, default=8)

    p_ingest_red = sub.add_parser(
        "ingest-reddit", help="Ingest a Reddit sample from a .zst NDJSON dump"
    )
    p_ingest_red.add_argument("--db", type=Path, required=True)
    p_ingest_red.add_argument("--source", default="reddit")
    p_ingest_red.add_argument("--source-url", required=True)
    p_ingest_red.add_argument("--target-comments", type=int, default=20000)
    p_ingest_red.add_argument("--seed", type=int, default=1337)
    p_ingest_red.add_argument("--scan-comments", type=int, default=None)
    p_ingest_red.add_argument(
        "--mode", choices=["comments", "threads"], default="comments", help="Sampling mode"
    )
    p_ingest_red.add_argument("--keep-mod", type=int, default=1000)
    p_ingest_red.add_argument("--keep-threshold", type=int, default=1)
    p_ingest_red.add_argument(
        "--subcommunities",
        default=None,
        help="Comma-separated subreddit allowlist (e.g., programming,learnpython)",
    )
    p_ingest_red.add_argument(
        "--subcommunity-pattern",
        default=None,
        help="Regex to match subreddit names (e.g., '(?i)python|program|tech')",
    )

    p_analyze = sub.add_parser("analyze", help="Compare Moltbook vs Reddit entropy proxies")
    p_analyze.add_argument("--db", type=Path, required=True)
    p_analyze.add_argument("--out", type=Path, default=None)
    p_analyze.add_argument("--min-thread-messages", type=int, default=5)
    p_analyze.add_argument(
        "--reddit-source",
        default="reddit",
        help="DB source name for the Reddit baseline (default: reddit)",
    )
    p_analyze.add_argument(
        "--reddit-domain-source",
        default="reddit_domain",
        help="DB source name for the domain-matched Reddit baseline (default: reddit_domain)",
    )
    p_analyze.add_argument(
        "--havelock",
        action="store_true",
        help="Add Havelock orality/literacy scores per section (calls https://havelock.ai/api)",
    )
    p_analyze.add_argument(
        "--havelock-base-url",
        default=HavelockParams.base_url,
        help="Override Havelock base URL (default: demo Hugging Face Space)",
    )
    p_analyze.add_argument(
        "--havelock-top-sections",
        type=int,
        default=HavelockParams.top_n_sections,
        help="Number of top subcommunities/subreddits to score per source",
    )
    p_analyze.add_argument(
        "--havelock-sample-messages",
        type=int,
        default=HavelockParams.sample_messages,
        help="How many messages to sample per section before truncation",
    )
    p_analyze.add_argument(
        "--havelock-max-chars",
        type=int,
        default=HavelockParams.max_chars,
        help="Max characters sent to Havelock per section",
    )
    p_analyze.add_argument(
        "--havelock-include-sentences",
        action="store_true",
        default=HavelockParams.include_sentences,
        help="Request per-sentence output (may increase response size)",
    )
    p_analyze.add_argument(
        "--havelock-seed",
        type=int,
        default=HavelockParams.seed,
        help="Seed for deterministic section sampling",
    )
    p_analyze.add_argument(
        "--havelock-domain",
        action="store_true",
        default=HavelockParams.include_domain,
        help="Also compute Havelock section scores for --reddit-domain-source",
    )

    p_html = sub.add_parser("render-html", help="Render an HTML report from analyze JSON output")
    p_html.add_argument("--in", dest="report_json", type=Path, required=True)
    p_html.add_argument("--out", dest="out_html", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.cmd == "ingest-moltbook":
        ingest_moltbook(
            db_path=args.db,
            base_url=args.base_url,
            max_posts=args.max_posts,
            concurrency=args.concurrency,
        )
        return 0
    if args.cmd == "ingest-reddit":
        subcommunities = (
            [s.strip() for s in args.subcommunities.split(",") if s.strip()]
            if args.subcommunities
            else None
        )
        ingest_reddit(
            db_path=args.db,
            source=args.source,
            source_url=args.source_url,
            target_comments=args.target_comments,
            seed=args.seed,
            scan_comments=args.scan_comments,
            mode=args.mode,
            keep_mod=args.keep_mod,
            keep_threshold=args.keep_threshold,
            subcommunities=subcommunities,
            subcommunity_pattern=args.subcommunity_pattern,
        )
        return 0
    if args.cmd == "analyze":
        report = run_analysis(
            db_path=args.db,
            min_thread_messages=args.min_thread_messages,
            reddit_source=args.reddit_source,
            reddit_domain_source=args.reddit_domain_source,
            havelock=HavelockParams(
                enabled=bool(args.havelock),
                base_url=str(args.havelock_base_url),
                top_n_sections=int(args.havelock_top_sections),
                sample_messages=int(args.havelock_sample_messages),
                max_chars=int(args.havelock_max_chars),
                include_sentences=bool(args.havelock_include_sentences),
                seed=int(args.havelock_seed),
                include_domain=bool(args.havelock_domain),
            ),
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report.to_json(indent=2) + "\n", encoding="utf-8")
        else:
            print(report.to_json(indent=2))
        return 0
    if args.cmd == "render-html":
        render_html_file(report_json=args.report_json, out_html=args.out_html)
        return 0

    raise RuntimeError(f"Unhandled cmd: {args.cmd}")
