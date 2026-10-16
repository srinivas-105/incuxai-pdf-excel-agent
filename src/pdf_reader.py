"""PDF table extraction for the IncuxAI PDF-to-Excel Agent."""

import io
import re
from bisect import bisect_right

import pdfplumber
import pytesseract
import numpy as np
from PIL import Image

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from src.table_stitcher import normalize_table, stitch_tables


def _clean(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_rows(rows):
    rows = [[_clean(c) for c in row] for row in rows if row]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    # Remove completely empty columns.
    keep = [
        i for i in range(width)
        if any(row[i] for row in rows)
    ]
    return [[row[i] for i in keep] for row in rows]


def _is_table(rows, minimum_rows=2):
    if not rows or len(rows) < minimum_rows:
        return False

    width = max(len(row) for row in rows)
    if width < 2:
        return False

    total = sum(len(row) for row in rows) * 1.0
    filled = sum(bool(cell) for row in rows for cell in row)

    return total > 0 and filled / total >= 0.25


def _digital_tables(page):
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
            found = page.extract_tables(table_settings=settings) or []
        except Exception:
            continue

        tables = []
        for table in found:
            rows = _clean_rows(table)
            if _is_table(rows):
                tables.append(rows)

        if tables:
            return tables

    return []


def _render_page(document, index):
    page = document[index]
    scale = 300 / 72
    max_side = 4500
    longest = max(page.rect.width, page.rect.height) * scale

    if longest > max_side:
        scale *= max_side / longest

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )

    image = np.frombuffer(pix.samples, dtype=np.uint8)
    image = image.reshape(pix.height, pix.stride)[:, :pix.width]
    return Image.fromarray(image)


def _ocr_rows(image, lang="eng"):
    """Extract OCR words and reconstruct rows and approximate columns."""
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    for i, value in enumerate(data["text"]):
        value = _clean(value)
        if not value:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < 20:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        if width <= 0 or height <= 0:
            continue

        words.append({
            "text": value,
            "x0": x,
            "x1": x + width,
            "xc": x + width / 2,
            "y0": y,
            "y1": y + height,
            "yc": y + height / 2,
            "height": height,
        })

    if not words:
        return []

    median_height = float(np.median([w["height"] for w in words]))
    row_tolerance = max(4, min(8, median_height * 0.45))

    # Group words by their vertical positions using a fixed row anchor.
    grouped = []
    current = []
    anchor = None

    for word in sorted(words, key=lambda w: (w["yc"], w["x0"])):
        if current and abs(word["yc"] - anchor) <= row_tolerance:
            current.append(word)
            anchor = float(np.median([w["yc"] for w in current]))
        else:
            if current:
                grouped.append(sorted(current, key=lambda w: w["x0"]))
            current = [word]
            anchor = word["yc"]

    if current:
        grouped.append(sorted(current, key=lambda w: w["x0"]))

    # Find persistent vertical whitespace between columns.
    page_width = image.width
    coverage = np.zeros(page_width, dtype=np.int32)

    for word in words:
        left = max(0, word["x0"])
        right = min(page_width, word["x1"])
        coverage[left:right] += 1

    threshold = max(1, int(len(grouped) * 0.03))
    whitespace = coverage <= threshold

    occupied = np.flatnonzero(coverage > 0)
    edges = []

    if occupied.size:
        start = None
        min_gap = max(10, int(median_height * 1.0))

        for x in range(int(occupied[0]), int(occupied[-1]) + 1):
            if whitespace[x]:
                if start is None:
                    start = x
            elif start is not None:
                if x - start >= min_gap:
                    edges.append((start + x - 1) / 2)
                start = None

    # Assign words to columns based on their center positions.
    rows = []
    for group in grouped:
        cells = [[] for _ in range(len(edges) + 1)]

        for word in group:
            column = bisect_right(edges, word["xc"])
            cells[column].append(word)

        row = [
            " ".join(w["text"] for w in sorted(cell, key=lambda w: w["x0"]))
            for cell in cells
        ]
        rows.append(row)

    rows = _clean_rows(rows)
    return rows if _is_table(rows, minimum_rows=3) else []


def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50, lang="eng"):
    """Return (tables, page_log), as expected by app.py."""
    page_log = []
    extracted = []
    ocr_document = None

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Could not open this PDF: {exc}") from exc

    try:
        if len(pdf.pages) > max_pages:
            raise ValueError(
                f"PDF has {len(pdf.pages)} pages; the limit is {max_pages}."
            )

        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            method = "pdfplumber"
            note = ""

            try:
                tables = _digital_tables(page)
            except Exception as exc:
                tables = []
                note = f"Digital extraction failed: {type(exc).__name__}"

            if not tables and use_ocr:
                try:
                    if ocr_document is None:
                        ocr_document = pymupdf.open(
                            stream=pdf_bytes,
                            filetype="pdf",
                        )

                    image = _render_page(ocr_document, page_index)
                    rows = _ocr_rows(image, lang=lang)
                    tables = [rows] if rows else []
                    method = "Tesseract OCR"

                    if not tables:
                        note = "No table was detected on this page."
                except Exception as exc:
                    tables = []
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:200]}"

            elif not tables:
                method = "no table detected"
                note = note or "No digital table found; OCR is disabled."

            page_log.append({
                "page": str(page_number),
                "method": method,
                "tables": len(tables),
                "note": note,
            })

            for table_number, rows in enumerate(tables, start=1):
                rows = _clean_rows(rows)

                if not _is_table(rows):
                    continue

                try:
                    normalized = normalize_table(rows)
                    if not normalized:
                        continue

                    extracted.append({
                        "page": page_number,
                        "table_no": table_number,
                        "columns": normalized["columns"],
                        "data": normalized["data"],
                        "method": method,
                    })
                except Exception as exc:
                    page_log[-1]["note"] = (
                        f"Table normalization failed: {str(exc)[:160]}"
                    )

            try:
                page.flush_cache()
            except Exception:
                pass

    finally:
        pdf.close()
        if ocr_document is not None:
            ocr_document.close()

    try:
        tables = stitch_tables(extracted)
    except Exception as exc:
        raise RuntimeError(f"Could not combine extracted tables: {exc}") from exc

    if not tables:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": "No usable tables were extracted. Check the PDF and OCR setup.",
        })

    return tables, page_log
