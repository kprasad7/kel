from __future__ import annotations

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Sliding-window character chunking — ignores structure entirely
    (can split mid-word, mid-sentence). Kept as the simple fallback and as
    `recursive_split_text`'s base case once no separator is left to try;
    prefer `recursive_split_text` for real documents."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def recursive_split_text(
    text: str, *, chunk_size: int = 500, overlap: int = 50, separators: list[str] | None = None
) -> list[str]:
    """LangChain's `RecursiveCharacterTextSplitter` equivalent: tries to
    split on paragraph breaks first, then lines, then sentences, then
    words, falling back to a hard character split only when a single
    "atomic" piece (e.g. one giant paragraph with no punctuation) still
    doesn't fit. Keeps semantic units intact wherever the text structure
    allows it, unlike `split_text`'s fixed-window approach."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []
    seps = separators if separators is not None else _DEFAULT_SEPARATORS
    chunks = _split_recursive(text, seps, chunk_size, overlap)
    return _apply_overlap(chunks, overlap) if overlap > 0 else chunks


def _split_recursive(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    separator, remaining = separators[0], separators[1:]
    if separator == "":
        return split_text(text, chunk_size=chunk_size, overlap=0)

    pieces = text.split(separator)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > chunk_size:
            chunks.extend(_split_recursive(piece, remaining, chunk_size, overlap))
            current = ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        overlapped.append(tail + chunks[i])
    return overlapped
