"""Document loaders: turn a file on disk into text for ingestion. PDF
(the most commonly requested), HTML, and CSV are covered; more loaders
follow the same shape (`load_x(path) -> str`, `load_x_pages(path) ->
list[str]`) and can be added the same way without touching anything else.
Still thin next to the hundreds of connectors a "batteries included"
framework ships — HTML/CSV close the two zero-new-dependency gaps that
matter most; DOCX, web crawling, and cloud-storage loaders each need a
real new dependency and are a reasonable future addition behind this same
shape, not something v1 needs to get right.

PDF loading requires `pypdf` (`pip install kel[pdf]`). `reader=` is
injectable for testing without a real PDF file/dependency, same DI
pattern as the provider adapters' `client=`. HTML and CSV loaders are
stdlib-only (`html.parser`, `csv`) — no extra dependency, no extra.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from kel.tools.web_fetch import extract_text


def _get_reader(path: str | Path | None, reader: Any) -> Any:
    if reader is not None:
        return reader
    if path is None:
        raise ValueError("must provide either `path` or `reader`")
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "The pypdf package is required to load PDF files. Install it with `pip install kel[pdf]`."
        ) from exc
    return pypdf.PdfReader(str(path))


def load_pdf_pages(path: str | Path | None = None, *, reader: Any = None) -> list[str]:
    """One string per page, in order — useful when you want to cite page numbers."""
    pdf_reader = _get_reader(path, reader)
    return [page.extract_text() or "" for page in pdf_reader.pages]


def load_pdf(path: str | Path | None = None, *, reader: Any = None) -> str:
    """All pages concatenated, separated by a blank line — hand this straight to `Retriever.ingest`."""
    return "\n\n".join(load_pdf_pages(path, reader=reader))


def load_html(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Readable text from a local HTML file, using the same generic
    tag-based extraction as `kel.tools.web_fetch` (skip script/style/nav/
    header/footer, keep the rest) — one implementation for "turn HTML
    into text," not a second one duplicated here."""
    return extract_text(Path(path).read_text(encoding=encoding))


def load_csv_rows(path: str | Path, *, encoding: str = "utf-8") -> list[str]:
    """One string per row, in order — useful when you want each row to
    become its own chunk (e.g. one product listing per row) rather than
    the whole file as a single blob. Uses the header row (if present) to
    label each value: `"name: Widget | price: 9.99"`, readable on its own
    without needing the original column order."""
    with Path(path).open(encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return [" | ".join(f"{key}: {value}" for key, value in row.items() if key) for row in reader]


def load_csv(path: str | Path, *, encoding: str = "utf-8") -> str:
    """All rows concatenated, one per line — hand this straight to `Retriever.ingest`."""
    return "\n".join(load_csv_rows(path, encoding=encoding))
