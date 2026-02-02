## arXiv paper draft (LaTeX)

This folder contains an arXiv-ready LaTeX draft derived from `essay.pdf` plus the numbers in
`data/report_v7.json`.

### Build

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output: `paper/main.pdf`

### Reproduce the numbers

From the repo root:

```bash
make all
```

The draft currently cites the snapshot in `data/report_v7.json` (generated on 2026-01-30).

