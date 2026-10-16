"""PDF table extraction.

Digital PDFs : pdfplumber (ruled tables first, then text-aligned tables).
Scanned PDFs : PyMuPDF render -> OpenCV (orientation, deskew, ruling-line
               detection) -> Tesseract word boxes -> grid reconstruction.

Public API (unchanged): extract_pdf(pdf_bytes, use_ocr=True, max_pages=50)
returns (stitched_tables, page_log).
"""
import io
import re
from bisect import bisect_right

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

from src.table_stitcher import normalize_table, stitch_tables

OCR_METHOD = "PyMuPDF + OpenCV + Tesseract OCR"
RENDER_DPI = 300
MAX_RENDER_SIDE = 5000
MIN_WORD_CONF = 30


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def _clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _tidy(rows):
    """Clean cells, pad to equal width, drop empty rows and empty columns."""
    rows = [[_clean_cell(c) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _looks_like_table(rows, min_rows=2, min_fill=0.35, max_avg_len=None):
    if len(rows) < min_rows or not rows or len(rows[0]) < 2:
        return False
    cells = [c for r in rows for c in r]
    filled = [c for c in cells if c]
    if not filled or len(filled) / len(cells) < min_fill:
        return False
    if max_avg_len and sum(map(len, filled)) / len(filled) > max_avg_len:
        return False
    return True


# --------------------------------------------------------------------------
# Digital PDFs (selectable text)
# --------------------------------------------------------------------------
def _digital_tables(page):
    strategies = [
        (
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            },
            False,
        ),
        (
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
            },
            True,
        ),
    ]

    for settings, text_based in strategies:
        try:
            raw_tables = page.extract_tables(settings) or []
        except Exception:
            continue

        result = []
        for table in raw_tables:
            rows = _tidy(table)
            # Text-based detection easily mistakes paragraphs for tables,
            # so it must pass stricter checks.
            if text_based:
                ok = _looks_like_table(
                    rows, min_rows=3, min_fill=0.5, max_avg_len=60
                )
            else:
                ok = _looks_like_table(rows, min_rows=2, min_fill=0.3)
            if ok:
                result.append(rows)

        if result:
            return result

    return []


# --------------------------------------------------------------------------
# Scanned PDFs: rendering and image preparation
# --------------------------------------------------------------------------
def _ensure_tesseract():
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract binary not found. Add 'tesseract-ocr' and "
            "'tesseract-ocr-eng' to packages.txt."
        ) from exc


def _render_gray(doc, page_index):
    page = doc[page_index]
    scale = RENDER_DPI / 72.0
    longest = max(page.rect.width, page.rect.height) * scale
    if longest > MAX_RENDER_SIDE:
        scale *= MAX_RENDER_SIDE / longest
    pix = pymupdf.Pixmap  # noqa: F841 (keeps linters quiet about the import)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width]
    return np.ascontiguousarray(arr)


def _fix_orientation(gray):
    """Rotate 90/180/270 degree scans upright using Tesseract OSD."""
    try:
        h, w = gray.shape
        s = 1500 / max(h, w)
        small = (
            cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if s < 1
            else gray
        )
        osd = pytesseract.image_to_osd(
            Image.fromarray(small),
            config="--psm 0",
            output_type=pytesseract.Output.DICT,
        )
        angle = int(osd.get("rotate", 0)) % 360
        conf = float(osd.get("orientation_conf", 0))
        if angle and conf >= 2.0:
            k = {90: 3, 180: 2, 270: 1}.get(angle, 0)
            if k:
                return np.ascontiguousarray(np.rot90(gray, k))
    except Exception:
        pass
    return gray


def _deskew(gray):
    """Correct small skew (+-5 degrees) via horizontal projection profile."""
    h, w = gray.shape
    s = 800 / max(h, w)
    small = (
        cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if s < 1
        else gray
    )
    _, bw = cv2.threshold(
        small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    sh, sw = bw.shape

    def score(angle):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(profile) ** 2))

    coarse = max(np.arange(-5.0, 5.01, 0.5), key=score)
    best = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
    if abs(best) < 0.3:
        return gray

    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(best), 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _line_masks(gray):
    """Return (horizontal_lines, vertical_lines) binary masks."""
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 30)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    return horiz, vert


# --------------------------------------------------------------------------
# OCR word extraction
# --------------------------------------------------------------------------
def _parse_words(data):
    words = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text or re.fullmatch(r"[|_]+", text):
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < MIN_WORD_CONF:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        words.append({
            "text": text,
            "x0": x, "x1": x + w,
            "y0": y, "y1": y + h,
            "xc": x + w / 2.0,
            "yc": y + h / 2.0,
            "h": h,
        })
    return words


def _ocr_words(gray, lang):
    image = Image.fromarray(gray)
    words = []
    for psm in (6, 11):
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=f"--oem 3 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
        words = _parse_words(data)
        if len(words) >= 4:
            break
    return words


# --------------------------------------------------------------------------
# Grid reconstruction from word boxes
# --------------------------------------------------------------------------
def _cluster_rows(words, med_h):
    """Group words into visual rows by vertical centre."""
    if not words:
        return []
    tol = max(0.6 * med_h, 6)
    groups = []
    current = []
    mean_y = 0.0
    for wd in sorted(words, key=lambda d: d["yc"]):
        if current and abs(wd["yc"] - mean_y) <= tol:
            current.append(wd)
            mean_y = sum(d["yc"] for d in current) / len(current)
        else:
            if current:
                groups.append(current)
            current = [wd]
            mean_y = wd["yc"]
    if current:
        groups.append(current)
    return [sorted(g, key=lambda d: d["x0"]) for g in groups]


def _segments(group, min_gap):
    count = 1
    for a, b in zip(group, group[1:]):
        if b["x0"] - a["x1"] > min_gap:
            count += 1
    return count


def _column_edges(groups, width, min_gap):
    """Find column separators as vertical gaps free of words."""
    if not groups:
        return []
    coverage = np.zeros(width + 2, dtype=np.int32)
    for g in groups:
        mask = np.zeros(width + 2, dtype=np.int32)
        for wd in g:
            mask[max(wd["x0"], 0): min(wd["x1"], width) + 1] = 1
        coverage += mask

    covered = np.flatnonzero(coverage > 0)
    if covered.size == 0:
        return []

    n = len(groups)
    threshold = 0 if n < 6 else int(round(0.05 * n))
    low = coverage <= threshold
    lo, hi = int(covered[0]), int(covered[-1])

    edges = []
    start = None
    for x in range(lo, hi + 1):
        if low[x]:
            if start is None:
                start = x
        elif start is not None:
            if x - start >= min_gap:
                edges.append((start + x - 1) / 2.0)
            start = None
    return edges


def _assign(groups, edges):
    rows = []
    for g in groups:
        cells = [[] for _ in range(len(edges) + 1)]
        for wd in g:
            cells[bisect_right(edges, wd["xc"])].append(wd)
        rows.append([
            " ".join(d["text"] for d in sorted(c, key=lambda d: d["x0"]))
            for c in cells
        ])
    return rows


def _line_positions(mask, axis, min_cover):
    profile = (mask > 0).sum(axis=axis)
    idx = np.flatnonzero(profile >= min_cover)
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > 4:
            out.append((start + prev) // 2)
            start = i
        prev = i
    out.append((start + prev) // 2)
    return out


def _ruled_tables(horiz, vert, words, width, med_h, min_gap):
    """Tables with drawn ruling lines. Returns (tables, used_boxes)."""
    h, w = horiz.shape
    lines = cv2.bitwise_or(horiz, vert)
    joined = cv2.dilate(lines, np.ones((5, 5), np.uint8), iterations=2)
    found = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2]

    tables, used = [], []
    pad = 5

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < w * 0.12 or bh < med_h * 2:
            continue

        hl = [v + y for v in _line_positions(
            horiz[y:y + bh, x:x + bw], 1, 0.6 * bw)]
        vl = [v + x for v in _line_positions(
            vert[y:y + bh, x:x + bw], 0, 0.6 * bh)]
        use_h, use_v = len(hl) >= 3, len(vl) >= 3
        if not (use_h or use_v):
            continue

        ws = [
            d for d in words
            if x - pad <= d["xc"] <= x + bw + pad
            and y - pad <= d["yc"] <= y + bh + pad
        ]
        if not ws:
            continue

        groups = _cluster_rows(ws, med_h)

        if use_h:
            inner_r = hl[1:-1]
            row_idx = {id(d): bisect_right(inner_r, d["yc"]) for d in ws}
            nrows = len(inner_r) + 1
        else:
            row_idx = {id(d): i for i, g in enumerate(groups) for d in g}
            nrows = len(groups)

        inner_c = vl[1:-1] if use_v else _column_edges(groups, width, min_gap)
        ncols = len(inner_c) + 1

        cells = [[[] for _ in range(ncols)] for _ in range(nrows)]
        for d in ws:
            cells[row_idx[id(d)]][bisect_right(inner_c, d["xc"])].append(d)

        rows = []
        for row in cells:
            out = []
            for cell in row:
                ordered = [d for g in _cluster_rows(cell, med_h) for d in g]
                out.append(" ".join(d["text"] for d in ordered))
            rows.append(out)

        rows = _tidy(rows)
        if _looks_like_table(rows, min_rows=2, min_fill=0.25):
            tables.append((y, rows))
            used.append((x - pad, y - pad, x + bw + pad, y + bh + pad))

    return tables, used


def _borderless_tables(words, width, med_h, min_gap):
    """Tables without ruling lines, found by aligned columns of words."""
    if not words:
        return []

    groups = _cluster_rows(words, med_h)
    runs, current, pending = [], [], []
    prev_bottom = None

    def flush():
        nonlocal current, pending
        if current:
            runs.append(current)
        current, pending = [], []

    for g in groups:
        top = min(d["y0"] for d in g)
        bottom = max(d["y1"] for d in g)

        if prev_bottom is not None and top - prev_bottom > 2.0 * med_h:
            flush()

        if _segments(g, min_gap) >= 2:
            current.extend(pending)
            pending = []
            current.append(g)
        elif current:
            pending.append(g)
            if len(pending) > 1:
                flush()
        prev_bottom = bottom
    flush()

    tables = []
    for run in runs:
        if len(run) < 3:
            continue
        edges = _column_edges(run, width, min_gap)
        rows = _tidy(_assign(run, edges))
        if _looks_like_table(rows, min_rows=3, min_fill=0.4):
            top = min(d["y0"] for g in run for d in g)
            tables.append((top, rows))
    return tables


def _extract_scanned_page(doc, page_index, lang="eng"):
    _ensure_tesseract()

    gray = _render_gray(doc, page_index)
    if np.count_nonzero(gray < 160) < 0.0005 * gray.size:
        return []  # blank page

    gray = _fix_orientation(gray)
    gray = _deskew(gray)
    height, width = gray.shape

    horiz, vert = _line_masks(gray)

    # Erase ruling lines so they do not disturb OCR.
    line_mask = cv2.dilate(
        cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8), iterations=1
    )
    clean = gray.copy()
    clean[line_mask > 0] = 255

    words = _ocr_words(clean, lang)
    if not words:
        return []

    med_h = float(np.median([d["h"] for d in words]))
    min_gap = max(int(med_h), 15)

    ruled, used = _ruled_tables(horiz, vert, words, width, med_h, min_gap)

    remaining = [
        d for d in words
        if not any(
            x0 <= d["xc"] <= x1 and y0 <= d["yc"] <= y1
            for x0, y0, x1, y1 in used
        )
    ]
    borderless = _borderless_tables(remaining, width, med_h, min_gap)

    combined = sorted(ruled + borderless, key=lambda t: t[0])
    return [rows for _, rows in combined]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def extract_pdf(pdf_bytes, use_ocr=True, max_pages=50, lang="eng"):
    """Extract tables from a PDF, log each page, and stitch across pages."""
    page_log = []
    page_tables = []
    ocr_doc = None

    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        total_pages = len(pdf.pages)
        if total_pages > max_pages:
            raise ValueError(
                f"PDF has {total_pages} pages; configured limit is {max_pages}."
            )

        for page_index, page in enumerate(pdf.pages):
            page_no = page_index + 1
            method = "pdfplumber"
            note = ""

            try:
                tables = _digital_tables(page)
            except Exception as exc:
                tables = []
                note = f"pdfplumber failed: {type(exc).__name__}"

            if not tables and use_ocr:
                try:
                    if ocr_doc is None:
                        ocr_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    tables = _extract_scanned_page(ocr_doc, page_index, lang)
                    method = OCR_METHOD
                    note = "" if tables else (
                        "No reliable table was detected. "
                        "Review the scan quality and table layout."
                    )
                except Exception as exc:
                    tables = []
                    method = "OCR failed"
                    note = f"{type(exc).__name__}: {str(exc)[:250]}"
            elif not tables:
                method = "no table detected"
                note = note or "No digital table found; OCR is disabled."

            # "page" is always a string so Streamlit/Arrow can serialise
            # the log even when it contains the "Summary" row.
            entry = {
                "page": str(page_no),
                "method": method,
                "tables": len(tables),
                "note": note,
            }
            page_log.append(entry)

            for table_no, rows in enumerate(tables, start=1):
                try:
                    normalized = normalize_table(rows)
                except Exception as exc:
                    entry["note"] = f"Normalization failed: {str(exc)[:180]}"
                    continue

                if not normalized:
                    entry["note"] = (
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

            # Release per-page caches (pdfplumber keeps them otherwise).
            try:
                page.flush_cache()
            except Exception:
                pass
    finally:
        pdf.close()
        if ocr_doc is not None:
            ocr_doc.close()

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
