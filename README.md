# moltbook_analysis

So what: this repo builds a local SQLite corpus of Moltbook agent conversations and a similarly-sized
baseline from Reddit comment dumps, then compares **redundancy**, **lexical diversity**, and **topic
concentration**. The output is a self-contained HTML report you can open in a browser.

## Quickstart

```bash
make setup
make all
```

## Build the corpora

### Moltbook

```bash
uv run python -m moltbook_analysis ingest-moltbook --db data/conversations.sqlite
```

### Reddit baseline (downloadable dump; no scraping)

This reads a random-ish sample of comments from a `.zst` NDJSON dump by streaming `curl | zstd -dc`.

```bash
uv run python -m moltbook_analysis ingest-reddit \
  --db data/conversations.sqlite \
  --source reddit \
  --source-url 'https://zenodo.org/records/3608135/files/RC_2019-04.zst?download=1' \
  --target-comments 35589 \
  --scan-comments 200000
```

<details>
  <summary>Optional: a more thread-coherent Reddit sample</summary>

The default Reddit mode is comment sampling, so many threads are partial. If you want a “more like
threads” baseline (still not perfect full threads), use `--mode threads`, which hash-samples whole
`link_id` threads until it hits the requested comment count.

```bash
uv run python -m moltbook_analysis ingest-reddit \
  --db data/conversations.sqlite \
  --source reddit_threads \
  --source-url 'https://zenodo.org/records/3608135/files/RC_2019-04.zst?download=1' \
  --mode threads \
  --target-comments 35589 \
  --scan-comments 2000000 \
  --keep-mod 10 \
  --keep-threshold 1
```
  </details>

<details>
  <summary>Optional: a “domain-ish” Reddit subset baseline</summary>

This keeps only a small allowlist of subreddits (news/explainers/science-ish) to be a bit closer to
the Moltbook “agents talking about agents” vibe.

```bash
uv run python -m moltbook_analysis ingest-reddit \
  --db data/conversations.sqlite \
  --source reddit_domain \
  --source-url 'https://zenodo.org/records/3608135/files/RC_2019-04.zst?download=1' \
  --target-comments 35589 \
  --scan-comments 100000 \
  --subcommunities 'todayilearned,news,worldnews,explainlikeimfive,dataisbeautiful,technology,science,askscience,askhistorians'
```
  </details>

## Run analysis + render HTML

Default comparison:

```bash
uv run python -m moltbook_analysis analyze --db data/conversations.sqlite --out data/report.json
uv run python -m moltbook_analysis render-html --in data/report.json --out data/report.html
open data/report.html
```

Thread-coherent Reddit baseline:

```bash
uv run python -m moltbook_analysis analyze \
  --db data/conversations.sqlite \
  --reddit-source reddit_threads \
  --out data/report_threads.json
uv run python -m moltbook_analysis render-html --in data/report_threads.json --out data/report_threads.html
open data/report_threads.html
```

Notes:
- This repo intentionally does **not** commit the raw corpora or generated reports (see `.gitignore`).
