import io
import re
import fitz
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
from src.table_stitcher import normalize_table, stitch_tables


def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _extract_digital_page(page):
    found = []
    try:
        raw_tables = page.extract_tables({
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
        })
        if not raw_tables:
            raw_tables = page.extract_tables({
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
            })
        for raw in raw_tables or []:
            rows = [[_clean_cell(c) for c in row] for row in raw if row and any(_clean_cell(c) for c in row)]
            if len(rows) >= 2 and max((len(r) for r in rows), default=0) >= 2:
                found.append(rows)
    except Exception:
        pass
    return found


def _ocr_page(pdf_bytes, page_index, dpi=220):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(image, config="--psm 6")
    doc.close()
    # Basic OCR fallback: split text rows on repeated spaces or tabs.
    # OCR is not a full table-layout engine; flag this method for human review.
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s{2,}|\t+", line)
        parts = [_clean_cell(p) for p in parts if _clean_cell(p)]
        if len(parts) >= 2:
            rows.append(parts)
    if len(rows) >= 2:
        return rows
    return []


def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50):
    page_log = []
    page_tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total = len(pdf.pages)
        if total > max_pages:
            raise ValueError(f"PDF has {total} pages; configured limit is {max_pages}.")
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = _extract_digital_page(page)
            method = "pdfplumber"
            if not tables and use_ocr:
                try:
                    ocr_rows = _ocr_page(pdf_bytes, page_no - 1)
                    if ocr_rows:
                        tables = [ocr_rows]
                        method = "tesseract OCR (layout heuristic)"
                except Exception as exc:
                    page_log.append({"page": page_no, "method": "OCR failed", "tables": 0, "note": str(exc)[:180]})
                    continue
            page_log.append({
                "page": page_no,
                "method": method if tables else "no table detected",
                "tables": len(tables),
                "note": "" if tables else "No table detected; page may be text-only or difficult to read."
            })
            for table_no, rows in enumerate(tables, start=1):
                normalized = normalize_table(rows)
                if normalized:
                    page_tables.append({
                        "page": page_no,
                        "table_no": table_no,
                        "columns": normalized["columns"],
                        "data": normalized["data"],
                        "method": method,
                    })
    stitched = stitch_tables(page_tables)
    return stitched, page_log
