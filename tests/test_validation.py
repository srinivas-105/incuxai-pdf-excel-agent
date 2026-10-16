from src.validator import validate_tables
from src.table_stitcher import normalize_table, stitch_tables


def test_repeated_header_removed():
    table = normalize_table([["Name", "Amount"], ["A", "10"], ["Name", "Amount"], ["B", "20"]])
    assert table["data"] == [["A", "10"], ["B", "20"]]


def test_stitch_matching_headers():
    pages = [
        {"page": 1, "table_no": 1, "columns": ["Name", "Amount"], "data": [["A", "10"]], "method": "pdfplumber"},
        {"page": 2, "table_no": 1, "columns": ["Name", "Amount"], "data": [["B", "20"]], "method": "pdfplumber"},
    ]
    result = stitch_tables(pages)
    assert len(result) == 1
    assert result[0]["data"] == [["A", "10"], ["B", "20"]]


def test_ocr_warns_for_review():
    tables = [{"columns": ["A", "B"], "data": [["x", "y"]], "pages": [1], "method": "tesseract OCR"}]
    assert any("OCR" in item["issue"] for item in validate_tables(tables))
