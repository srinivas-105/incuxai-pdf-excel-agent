
import io
import re

import pdfplumber
from img2table.document import PDF
from img2table.ocr import TesseractOCR

from src.table_stitcher import normalize_table, stitch_tables


def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _digital_tables(page):
    """Extract tables from PDFs that contain selectable text."""
    strategies = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
        },
    ]

    for settings in strategies:
        try:
            raw_tables = page.extract_tables(settings) or []
        except Exception:
            continue

        result = []
        for table in raw_tables:
            rows = [
                [_clean_cell(cell) for cell in row]
                for row in table
                if row and any(_clean_cell(cell) for cell in row)
            ]
            if len(rows) >= 2 and max(map(len, rows), default=0) >= 2:
                result.append(rows)

        if result:
            return result

    return []


def _img2table_rows(table):
    """Convert img2table cell output to a list of rows."""
    rows = []

    for cells in table.content.values():
        row = []
        for cell in cells:
            row.append(_clean_cell(getattr(cell, "value", "")))
        if any(row):
            rows.append(row)

    return rows


def _extract_scanned_page(pdf_bytes, page_index):
    """
    Detect table boundaries and OCR cells using img2table.

    This intentionally does not fall back to treating whole text lines
    as one-column tables.
    """
    document = PDF(
        src=pdf_bytes,
        pages=[page_index],
        detect_rotation=True,
        pdf_text_extraction=False,
    )

    ocr = TesseractOCR(
        n_threads=1,
        lang="eng",
        psm=11,
    )

    extracted = document.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=35,
        max_workers=1,
    )

    page_tables = extracted.get(page_index, []) or []
    tables = []

    for table in page_tables:
        rows = _img2table_rows(table)

        if len(rows) < 2:
            continue

        # Reject output that has collapsed into a single column.
        widths = [len(row) for row in rows]
        if max(widths, default=0) < 2:
            continue

        tables.append(rows)

    return tables


def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50):
    """Extract PDF tables, log failures, and stitch compatible page tables."""
    page_log = []
    page_tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)

        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages; configured limit is {max_pages}."
            )

        for page_index, page in enumerate(pdf.pages):
            page_no = page_index + 1
            tables = _digital_tables(page)
            method = "pdfplumber"
            note = ""

            if not tables and use_ocr:
                try:
                    tables = _extract_scanned_page(pdf_bytes, page_index)
                    method = "img2table + Tesseract OCR"

                    if not tables:
                        note = (
                            "No reliable table was detected by img2table. "
                            "Review the scan and OCR output."
                        )
                except Exception as exc:
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:250]}"

            elif not tables:
                method = "no table detected"
                note = "No digital table found; OCR is disabled."

            page_log.append({
                "page": page_no,
                "method": method,
                "tables": len(tables),
                "note": note,
            })

            for table_no, rows in enumerate(tables, start=1):
                try:
                    normalized = normalize_table(rows)
                except Exception as exc:
                    page_log[-1]["note"] = (
                        f"Normalization failed: {str(exc)[:180]}"
                    )
                    continue

                if not normalized:
                    page_log[-1]["note"] = (
                        "Table was detected but could not be normalized."
                    )
                    continue

                page_tables.append({
                    "page": page_no,
                    "table_no": table_no,
                    "columns": normalized["columns"],
                    "data": normalized["data"],
                    "method": method,
                })

    try:
        stitched = stitch_tables(page_tables)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to combine tables across pages: {exc}"
        ) from exc

    if not stitched:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": (
                "No usable tables were extracted. Check the OCR dependency, "
                "scan quality, and table layout."
            ),
        })

    return stitched, page_log
