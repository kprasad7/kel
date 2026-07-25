import pytest

from kel.retrieval import load_pdf, load_pdf_pages


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, pages: list[str]):
        self.pages = [_FakePage(text) for text in pages]


def test_load_pdf_pages_returns_one_string_per_page():
    reader = _FakeReader(["page one text", "page two text"])
    pages = load_pdf_pages(reader=reader)
    assert pages == ["page one text", "page two text"]


def test_load_pdf_concatenates_pages_with_blank_line():
    reader = _FakeReader(["first page", "second page"])
    text = load_pdf(reader=reader)
    assert text == "first page\n\nsecond page"


def test_load_pdf_pages_handles_page_with_no_extractable_text():
    reader = _FakeReader(["has text", None])  # pypdf returns None for pages with no text
    pages = load_pdf_pages(reader=reader)
    assert pages == ["has text", ""]


def test_load_pdf_requires_path_or_reader():
    with pytest.raises(ValueError):
        load_pdf()
