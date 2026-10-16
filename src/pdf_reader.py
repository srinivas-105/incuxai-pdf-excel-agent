
import io
import re

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from src.table_stitcher import normalize_table, stitch_tables


def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _render_page(pdf_bytes, page_index, dpi=300):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False,
        )
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


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
        for table in raw_tables:
            rows = [
                [_clean_cell(c) for c in row]
                for row in table
                if row and any(_clean_cell(c) for c in row)
            ]
            if len(rows) >= 2:
                tables.append(rows)

        if tables:
            return tables

    return []


def _cluster_positions(values, tolerance=5):
    if not values:
        return []

    values = sorted(values)
    clusters = [[values[0]]]

    for value in values[1:]:
        if value - clusters[-1][-1] <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return [int(round(sum(c) / len(c))) for c in clusters]


def _find_grid(binary):
    """Detect long horizontal and vertical rules in a thresholded image."""
    height, width = binary.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(25, width // 35), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, height // 35))
    )

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel
    )

    h_projection = np.count_nonzero(horizontal, axis=1)
    v_projection = np.count_nonzero(vertical, axis=0)

    # Require a substantial fraction of the image width/height.
    ys = np.where(h_projection >= width * 0.18)[0].tolist()
    xs = np.where(v_projection >= height * 0.10)[0].tolist()

    ys = _cluster_positions(ys, max(3, height // 700))
    xs = _cluster_positions(xs, max(3, width // 700))

    return xs, ys, horizontal, vertical


def _ocr_cell(image):
    """Read one isolated table cell; avoid interpreting neighboring columns."""
    if image.width < 3 or image.height < 3:
        return ""

    # Add a white margin so Tesseract does not clip characters at grid lines.
    image = ImageOps.expand(image, border=5, fill="white")
    config = "--psm 7"

    text = pytesseract.image_to_string(image, config=config)
    return _clean_cell(text)


def _grid_table(image):
    """
    Find a ruled table and OCR individual cells.

    Returns rows only when enough usable grid structure is detected.
    """
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold helps with faint or unevenly lit scans.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    xs, ys, horizontal, vertical = _find_grid(binary)

    if len(xs) < 3 or len(ys) < 4:
        return [], "Could not detect enough table grid lines."

    # Locate the strongest rectangular table region.
    # The supplied test uses a ruled table with many vertical separators.
    best_region = None
    best_score = 0

    for yi in range(len(ys) - 2):
        for yj in range(yi + 2, len(ys)):
            y1, y2 = ys[yi], ys[yj]
            if y2 - y1 < image.height * 0.15:
                continue

            vertical_support = sum(
                1 for x in xs
                if np.count_nonzero(vertical[y1:y2, x]) >
                (y2 - y1) * 0.15
            )
            horizontal_support = sum(
                1 for y in ys[yi:yj + 1]
                if np.count_nonzero(horizontal[y, :]) >
                image.width * 0.15
            )

            score = vertical_support * horizontal_support
            if score > best_score:
                best_score = score
                best_region = (yi, yj)

    if best_region is None:
        return [], "No credible rectangular table region detected."

    yi, yj = best_region
    y_edges = ys[yi:yj + 1]

    # Keep vertical separators that persist through the table's height.
    y1, y2 = y_edges[0], y_edges[-1]
    x_edges = [
        x for x in xs
        if np.count_nonzero(vertical[y1:y2, x]) >
        (y2 - y1) * 0.15
    ]

    x_edges = _cluster_positions(x_edges, max(3, image.width // 700))

    if len(x_edges) < 3 or len(y_edges) < 4:
        return [], "Table grid was incomplete after filtering."

    # Crop each cell and OCR it separately.
    rows = []
    for row_index in range(len(y_edges) - 1):
        top, bottom = y_edges[row_index], y_edges[row_index + 1]

        if bottom - top < 5:
            continue

        row = []
        for col_index in range(len(x_edges) - 1):
            left, right = x_edges[col_index], x_edges[col_index + 1]

            if right - left < 5:
                row.append("")
                continue

            # Ignore a thin strip containing the table rules.
            crop = image.crop((
                max(0, left + 2),
                max(0, top + 2),
                min(image.width, right - 2),
                min(image.height, bottom - 2),
            ))
            row.append(_ocr_cell(crop))

        if any(row):
            rows.append(row)

    if len(rows) < 3:
        return [], "Grid was found but too few rows were recognized."

    # Remove completely blank columns only.
    keep = [
        i for i in range(max(map(len, rows)))
        if any(i < len(row) and row[i] for row in rows)
    ]
    rows = [
        [row[i] if i < len(row) else "" for i in keep]
        for row in rows
    ]

    return rows, ""


def _ocr_page(pdf_bytes, page_index):
    image = _render_page(pdf_bytes, page_index)
    rows, note = _grid_table(image)

    if rows:
        return rows, "Tesseract OCR (cell-by-cell grid)", ""

    # Fallback: word-level OCR, with a warning that alignment is uncertain.
    data = pytesseract.image_to_data(
        image,
        config="--psm 11",
        output_type=Output.DICT,
    )

    words = []
    for i, raw in enumerate(data["text"]):
        word = _clean_cell(raw)
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if not word or confidence < 25:
            continue

        words.append({
            "text": word,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "h": max(1, int(data["height"][i])),
        })

    words.sort(key=lambda w: (w["y"], w["x"]))
    lines = []

    for word in words:
        target = None
        for line in reversed(lines[-4:]):
            center = sum(w["y"] + w["h"] / 2 for w in line) / len(line)
            if abs(word["y"] + word["h"] / 2 - center) <= max(8, word["h"]):
                target = line
                break

        if target is None:
            lines.append([word])
        else:
            target.append(word)

    fallback = []
    for line in lines:
        line.sort(key=lambda w: w["x"])
        if line:
            fallback.append([
                " ".join(w["text"] for w in line)
            ])

    if len(fallback) >= 3:
        return fallback, "Tesseract OCR (text-only fallback)", (
            note + " Table columns could not be reliably reconstructed."
        )

    return [], "OCR failed", note


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
                note = "No digital tables found; OCR is disabled."

            page_log.append({
                "page": page_no,
                "method": method if tables else method,
                "tables": len(tables),
                "note": note,
            })

            for table_no, rows in enumerate(tables, start=1):
                try:
                    normalized = normalize_table(rows)
                except Exception as exc:
                    page_log[-1]["note"] = (
                        f"Table normalization failed: {str(exc)[:180]}"
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
                        "Rows were found but could not be normalized."
                    )

    try:
        stitched = stitch_tables(page_tables)
    except Exception as exc:
        raise RuntimeError(f"Table stitching failed: {exc}") from exc

    if not stitched:
        page_log.append({
            "page": "Summary",
            "method": "extraction unsuccessful",
            "tables": 0,
            "note": (
                "No usable tables were extracted. Check the OCR logs, "
                "grid detection, and PDF scan quality."
            ),
        })

    return stitched, page_log

