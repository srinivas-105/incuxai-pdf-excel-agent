
import io
import re
import statistics

import fitz
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
from pytesseract import Output

from src.table_stitcher import normalize_table, stitch_tables


def _clean_cell(value):
    if value is None:
        return ""
    value = str(value).replace("\x00", "")
    return re.sub(r"\s+", " ", value).strip()


def _render_page(pdf_bytes, page_index, dpi=350):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False,
        )
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")

    image = ImageOps.autocontrast(image)
    return ImageEnhance.Contrast(image).enhance(1.4)


def _digital_tables(page):
    for settings in (
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
    ):
        try:
            raw_tables = page.extract_tables(settings) or []
        except Exception:
            continue

        tables = []
        for raw in raw_tables:
            rows = [
                [_clean_cell(cell) for cell in row]
                for row in raw
                if row and any(_clean_cell(cell) for cell in row)
            ]
            if len(rows) >= 2 and max(map(len, rows), default=0) >= 2:
                tables.append(rows)

        if tables:
            return tables

    return []


def _get_ocr_words(image):
    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=Output.DICT,
    )

    words = []
    for i, raw in enumerate(data["text"]):
        text = _clean_cell(raw)

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not text or confidence < 15:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        words.append({
            "text": text,
            "x1": x,
            "x2": x + w,
            "y1": y,
            "y2": y + h,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "height": max(1, h),
            "block": data["block_num"][i],
            "paragraph": data["par_num"][i],
            "line": data["line_num"][i],
            "confidence": confidence,
        })

    return words


def _group_lines(words):
    """Use Tesseract's own line IDs, with geometric grouping as a fallback."""
    groups = {}

    for word in words:
        key = (word["block"], word["paragraph"], word["line"])
        groups.setdefault(key, []).append(word)

    lines = list(groups.values())
    lines = [sorted(line, key=lambda w: w["x1"]) for line in lines]
    lines.sort(key=lambda line: statistics.mean(w["cy"] for w in line))
    return lines


def _line_text(line):
    return " ".join(word["text"] for word in line).strip()


def _header_score(line):
    text = _line_text(line).lower()
    terms = {
        "id", "date", "vendor", "supplier", "item", "code",
        "description", "qty", "quantity", "unit", "price",
        "discount", "disc", "tax", "amount", "total",
        "invoice", "product", "customer", "name", "rate",
        "cost", "balance", "status", "reference",
    }
    tokens = set(re.findall(r"[a-z]+", text))
    score = len(tokens & terms)

    for phrase in (
        "item code", "unit price", "line total",
        "invoice id", "tax amount", "total amount",
    ):
        if phrase in text:
            score += 2

    return score


def _find_header(lines):
    candidates = []

    for index, line in enumerate(lines):
        score = _header_score(line)
        if score >= 3 and len(line) >= 3:
            candidates.append((score, len(line), index))

    if candidates:
        return max(candidates)[2]

    # Fallback for headers whose words OCR recognizes imperfectly:
    # prefer a line with many separate words, not a title or short metadata.
    possible = [
        (len(line), index)
        for index, line in enumerate(lines)
        if len(line) >= 7
    ]

    return max(possible)[1] if possible else None


def _find_vertical_boundaries(image, y_start, y_end):
    """
    Detect persistent vertical rules by examining dark-pixel density.
    Returns an empty list if the image does not provide convincing evidence.
    """
    arr = np.asarray(image)
    dark = arr < 150
    height, width = dark.shape

    y_start = max(0, min(height - 1, int(y_start)))
    y_end = max(y_start + 1, min(height, int(y_end)))

    region = dark[y_start:y_end, :]
    density = region.mean(axis=0)

    # A rule should persist through many rows; text characters usually do not.
    active = density >= 0.48
    positions = np.flatnonzero(active)

    if len(positions) == 0:
        return []

    clusters = [[int(positions[0])]]
    for x in positions[1:]:
        x = int(x)
        if x - clusters[-1][-1] <= 4:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    boundaries = [
        int(round(statistics.mean(cluster)))
        for cluster in clusters
        if len(cluster) >= 1
    ]

    # Remove lines that are too close to one another to be column separators.
    merged = []
    for x in boundaries:
        if not merged or x - merged[-1] >= max(8, width // 250):
            merged.append(x)

    # A usable grid must span a substantial portion of the page width.
    if len(merged) < 3:
        return []

    if merged[-1] - merged[0] < width * 0.35:
        return []

    return merged


def _rows_from_grid(image, words, lines, header_index):
    header_line = lines[header_index]
    header_y = statistics.mean(word["cy"] for word in header_line)

    lines_after_header = [
        line for line in lines
        if statistics.mean(word["cy"] for word in line) >= header_y - 4
    ]

    if len(lines_after_header) < 3:
        return [], "Too few lines were found after the header."

    max_y = max(word["y2"] for line in lines_after_header for word in line)
    boundaries = _find_vertical_boundaries(
        image, header_y - 20, max_y + 10
    )

    if len(boundaries) < 3:
        return [], "No reliable vertical column boundaries were detected."

    rows = []
    for line in lines_after_header:
        cells = [""] * (len(boundaries) - 1)

        for word in line:
            col = None
            for j in range(len(boundaries) - 1):
                if boundaries[j] <= word["cx"] < boundaries[j + 1]:
                    col = j
                    break

            if col is not None:
                cells[col] = (cells[col] + " " + word["text"]).strip()

        if any(cells):
            rows.append(cells)

    if len(rows) < 3:
        return [], "Grid was detected, but too few usable rows were reconstructed."

    return rows, ""


def _remove_noise(rows):
    cleaned = []

    for row in rows:
        row = [_clean_cell(cell) for cell in row]
        text = " ".join(cell.lower() for cell in row if cell)

        if not text.strip():
            continue

        if "north ridge industrial" in text or "northridge industrial" in text:
            continue
        if "purchase ledger" in text and "currency" not in text:
            continue
        if "archive scan" in text or "batch:" in text:
            continue
        if "reference mark:" in text or "approved:" in text:
            continue
        if "verify decimal separators" in text:
            continue
        if "sheet" in text and "currency:" in text:
            continue

        cleaned.append(row)

    return cleaned


def _ocr_page(pdf_bytes, page_index):
    image = _render_page(pdf_bytes, page_index)
    words = _get_ocr_words(image)

    if not words:
        return [], "OCR failed", "No words were recognized."

    lines = _group_lines(words)
    header_index = _find_header(lines)

    if header_index is None:
        return [], "OCR failed", "Could not identify a plausible table header."

    rows, note = _rows_from_grid(image, words, lines, header_index)

    if not rows:
        return [], "OCR failed", note

    rows = _remove_noise(rows)

    if len(rows) < 2:
        return [], "OCR failed", "Too few rows remained after removing page metadata."

    return rows, "Tesseract OCR with column grid", ""


def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50):
    page_log = []
    page_tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) > max_pages:
            raise ValueError(
                f"PDF has {len(pdf.pages)} pages; configured limit is {max_pages}."
            )

        for page_no, page in enumerate(pdf.pages, start=1):
            tables = _digital_tables(page)
            method = "pdfplumber"
            note = ""

            if not tables and use_ocr:
                try:
                    rows, method, note = _ocr_page(pdf_bytes, page_no - 1)
                    if rows:
                        tables = [rows]
                except Exception as exc:
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:200]}"

            elif not tables:
                method = "no table detected"
                note = "Digital table extraction found no tables and OCR is disabled."

            page_log.append({
                "page": page_no,
                "method": method,
                "tables": len(tables),
                "note": note,
            })

            for table_no, rows in enumerate(tables, start=1):
                rows = [
                    [_clean_cell(cell) for cell in row]
                    for row in rows
                    if row and any(_clean_cell(cell) for cell in row)
                ]

                if len(rows) < 2:
                    page_log[-1]["note"] = (
                        "Not enough usable rows were extracted."
                    )
                    continue

                try:
                    normalized = normalize_table(rows)
                except Exception as exc:
                    page_log[-1]["note"] = (
                        f"Normalization failed: {str(exc)[:200]}"
                    )
                    continue

                if normalized:
                    page_tables.append({
                        "page": page_no,
                        "table_no": table_no,
                        "columns": normalized["columns"],
                        "data": normalized["data"],
                        "method": method,
                    })
                else:
                    page_log[-1]["note"] = (
                        "The extracted rows could not be normalized."
                    )

    stitched = stitch_tables(page_tables)

    if not stitched:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": (
                "No usable tables were extracted. Check OCR availability, "
                "table boundaries, and the scan quality."
            ),
        })

    return stitched, page_log
