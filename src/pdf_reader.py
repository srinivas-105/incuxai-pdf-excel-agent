
import io
import re
import statistics

import fitz
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image, ImageOps, ImageEnhance

from src.table_stitcher import normalize_table, stitch_tables


def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _extract_digital_page(page):
    """Extract tables directly from a digital PDF page."""
    found = []

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
            raw_tables = []

        for raw in raw_tables:
            rows = [
                [_clean_cell(cell) for cell in row]
                for row in raw
                if row and any(_clean_cell(cell) for cell in row)
            ]

            if (
                len(rows) >= 2
                and max((len(row) for row in rows), default=0) >= 2
            ):
                found.append(rows)

        if found:
            break

    return found


def _group_words_into_lines(words):
    """Group OCR words into visual lines using their vertical positions."""
    if not words:
        return []

    words = sorted(words, key=lambda word: (word["cy"], word["x1"]))
    lines = []

    for word in words:
        best_line = None
        best_distance = float("inf")

        # Compare with recent lines, not every line on the page.
        for line in lines[-4:]:
            center = statistics.mean(item["cy"] for item in line)
            median_height = statistics.median(
                item["height"] for item in line
            )
            tolerance = max(8, median_height * 0.65)
            distance = abs(word["cy"] - center)

            if distance <= tolerance and distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append([word])
        else:
            best_line.append(word)

    for line in lines:
        line.sort(key=lambda word: word["x1"])

    return sorted(
        lines,
        key=lambda line: statistics.mean(word["cy"] for word in line),
    )


def _line_to_cells(line):
    """Split a visual line at sufficiently large horizontal gaps."""
    if not line:
        return []

    heights = [word["height"] for word in line if word["height"] > 0]
    median_height = statistics.median(heights) if heights else 10

    # A heuristic: narrow gaps join words; wider gaps may separate cells.
    gap_threshold = max(14, median_height * 1.5)

    cells = []
    current_text = line[0]["text"]
    previous_right = line[0]["x2"]

    for word in line[1:]:
        gap = word["x1"] - previous_right

        if gap > gap_threshold:
            cells.append(current_text.strip())
            current_text = word["text"]
        else:
            current_text += " " + word["text"]

        previous_right = max(previous_right, word["x2"])

    cells.append(current_text.strip())
    return cells


def _ocr_page(pdf_bytes, page_index, dpi=300):
    """
    OCR a scanned PDF page and estimate table rows from word coordinates.

    Note: coordinate-based cell splitting is a heuristic. It cannot guarantee
    correct column alignment for every scanned table.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False,
        )
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    finally:
        doc.close()

    # Improve contrast before OCR.
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)

    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []

    for i, raw_text in enumerate(data["text"]):
        text = _clean_cell(raw_text)

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not text or confidence < 20:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        words.append({
            "text": text,
            "x1": left,
            "x2": left + width,
            "cy": top + height / 2,
            "height": max(1, height),
        })

    if not words:
        return []

    lines = _group_words_into_lines(words)
    rows = [_line_to_cells(line) for line in lines]

    # Remove empty rows.
    rows = [
        [_clean_cell(cell) for cell in row]
        for row in rows
        if row and any(_clean_cell(cell) for cell in row)
    ]

    # Avoid treating a few isolated words as a table.
    if len(rows) < 3:
        return []

    # Retain rows with at least two recognized cells.
    return [row for row in rows if len(row) >= 2]


def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50):
    """
    Extract tables from digital or scanned PDFs.

    Returns:
        stitched_tables: tables normalized and joined across pages
        page_log: per-page extraction status and diagnostic notes
    """
    page_log = []
    page_tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)

        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages; configured limit is "
                f"{max_pages}."
            )

        for page_no, page in enumerate(pdf.pages, start=1):
            tables = _extract_digital_page(page)
            method = "pdfplumber"
            note = ""

            if not tables and use_ocr:
                try:
                    ocr_rows = _ocr_page(pdf_bytes, page_no - 1)

                    if ocr_rows:
                        tables = [ocr_rows]
                        method = "Tesseract OCR (coordinate heuristic)"
                    else:
                        method = "OCR found no table-like rows"
                        note = (
                            "OCR did not produce enough usable rows. "
                            "Check scan quality and column alignment."
                        )

                except Exception as exc:
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:250]}"

            elif not tables:
                method = "no table detected"
                note = "Digital extraction found no tables; OCR is disabled."

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
                        page_log[-1]["note"]
                        + f" Normalization failed: {str(exc)[:150]}"
                    ).strip()
                    continue

                if not normalized:
                    page_log[-1]["note"] = (
                        page_log[-1]["note"]
                        + " Extracted rows could not be normalized."
                    ).strip()
                    continue

                page_tables.append({
                    "page": page_no,
                    "table_no": table_no,
                    "columns": normalized["columns"],
                    "data": normalized["data"],
                    "method": method,
                })

    try:
        stitched_tables = stitch_tables(page_tables)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to combine extracted tables: {exc}"
        ) from exc

    # Make total extraction failure visible in the processing log.
    if not stitched_tables:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": (
                "No usable tables were extracted from the PDF. "
                "Review scan quality, OCR availability, and table layout."
            ),
        })

    return stitched_tables, page_log
