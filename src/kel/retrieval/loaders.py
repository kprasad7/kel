"""Document loaders: turn a file on disk into text for ingestion. v1
ships one format (PDF, the most commonly requested) rather than trying to
cover every format up front; more loaders follow the same shape
(`load_x(path) -> str`, `load_x_pages(path) -> list[str]`) and can be
added the same way without touching anything else.

Requires `pypdf` (`pip install kel[pdf]`). `reader=` is injectable for
testing without a real PDF file/dependency, same DI pattern as the
provider adapters' `client=`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
