import tempfile
from pathlib import Path

import pytest

from kel.retrieval import load_csv, load_csv_rows, load_html, load_pdf, load_pdf_pages


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


def test_load_html_extracts_readable_text_and_skips_script_and_style():
    html = """
    <html><head><style>body { color: red; }</style></head>
    <body>
      <nav>Home | About</nav>
      <script>alert('hi')</script>
      <h1>Article Title</h1>
      <p>The actual readable content lives here.</p>
    </body></html>
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.html"
        path.write_text(html, encoding="utf-8")

        text = load_html(path)

    assert "Article Title" in text
    assert "actual readable content" in text
    assert "alert" not in text
    assert "color: red" not in text
    assert "Home | About" not in text


def test_load_csv_rows_labels_each_value_with_its_column_header():
    csv_text = "name,price\nWidget,9.99\nGadget,19.99\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "products.csv"
        path.write_text(csv_text, encoding="utf-8")

        rows = load_csv_rows(path)

    assert rows == ["name: Widget | price: 9.99", "name: Gadget | price: 19.99"]


def test_load_csv_concatenates_rows_one_per_line():
    csv_text = "name,price\nWidget,9.99\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "products.csv"
        path.write_text(csv_text, encoding="utf-8")

        text = load_csv(path)

    assert text == "name: Widget | price: 9.99"


def test_load_csv_rows_on_empty_file_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.csv"
        path.write_text("", encoding="utf-8")

        assert load_csv_rows(path) == []
